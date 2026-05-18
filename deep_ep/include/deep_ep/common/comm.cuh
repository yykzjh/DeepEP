#pragma once  // Elastic communication helper 被多个 JIT kernel include，防止重复定义。

#include <cooperative_groups.h>  // grid 级 cooperative sync。
#include <nccl.h>                // NCCL host/device 基础类型。
#include <nccl_device.h>         // ncclGin、team tag、device communicator 等类型。

#include <deep_ep/common/handle.cuh>  // NCCLGin wrapper。
#include <deep_ep/common/ptx.cuh>     // PTX load/store/trap/TMA helper。
#include <deep_ep/common/layout.cuh>  // WorkspaceLayout。

namespace deep_ep::elastic::comm {  // elastic 通信同步辅助命名空间。
// Elastic/V2 通信同步辅助命名空间。

static constexpr int64_t kNumOneSecCycles = 2000000000;  // 约 2000MHz GPU 上 1 秒的 cycle 数。
// 超时后额外等待约 1 秒，让其它线程也有机会打印诊断信息。

// Some reserved tags
static constexpr int kDeviceBarrierTag = 0;  // 默认设备 barrier tag。
// 通用设备 barrier tag。
static constexpr int kKernelBarrierTag = 1;  // 独立 barrier kernel tag。
// 独立 barrier kernel 使用的 tag。
static constexpr int kDispatchTag0 = 2;  // dispatch prologue 前/入口同步 tag。
// dispatch 第一组同步 tag。
static constexpr int kDispatchTag1 = 3;  // dispatch payload 完成同步 tag。
// dispatch 第二组同步 tag。
static constexpr int kCombineTag0 = 4;  // combine 入口同步 tag。
// combine 第一组同步 tag。
static constexpr int kCombineTag1 = 5;  // combine 收尾同步 tag。
// combine 第二组同步 tag。
static constexpr int kHybridDispatchTag0 = 6;  // hybrid dispatch 入口同步 tag。
// hybrid dispatch 第一组同步 tag。
static constexpr int kHybridDispatchTag1 = 7;  // hybrid dispatch epilogue 前同步 tag。
// hybrid dispatch 第二组同步 tag。
static constexpr int kHybridCombineTag0 = 8;  // hybrid combine 入口同步 tag。
// hybrid combine 第一组同步 tag。
static constexpr int kHybridCombineTag1 = 9;  // hybrid combine 预留/收尾同步 tag。
// hybrid combine 第二组同步 tag。

// Some reserved count
static constexpr int kFlushAllAllocatedQPs = -1;  // flush 所有已分配 QP 的哨兵值。
// 特殊值：要求 barrier flush communicator 中所有已分配 Gin QP。

template <int64_t kNumTimeoutCycles, typename func_t>  // 超时周期和轮询回调类型。
__device__ __forceinline__ void timeout_while(const bool& condition, const func_t& func,
                                              int64_t start_clock = 0) {  // 可选共享起始 clock。
    // device 端等待不能无界自旋，否则某个 rank 掉队会让整个 kernel 永久占住 GPU。
    // func(is_last_check) 返回 true 表示等待条件满足；当超时时，会先给所有线程 1 秒打印诊断，
    // 再触发 ptx::trap，方便定位卡在哪个 barrier/tag/rank。
    // User may share a start clock for multiple waits
    if (start_clock == 0)  // 未传外部起点时本地初始化。
        // 未传共享起始时间时，以当前 clock64 作为等待起点。
        start_clock = clock64();

    while (condition) {  // condition 为 false 时直接跳过等待。
        // condition 通常是 true；保留参数是为了外部可以快速禁用等待循环。
        const bool timeout = clock64() - start_clock >= kNumTimeoutCycles;  // 是否到达超时阈值。
        // 判断是否超过调用方配置的超时周期数。
        if (func(timeout))  // 回调返回 true 表示条件满足。
            // func 返回 true 表示等待条件满足，立即退出。
            break;

        if (timeout) {  // 到达超时阈值后进入诊断窗口。
            // 超时后给线程额外打印窗口，然后 trap。
            // Wait another 1 second to let all threads print information and trap
            start_clock = clock64();
            // 重置起点用于额外 1 秒等待。
            while (clock64() - start_clock < kNumOneSecCycles) {}
            // 空转等待其它线程打印。
            ptx::trap();
            // 触发设备端 trap，暴露卡死位置。
        }
    }
}

template <int64_t kNumTimeoutCycles, typename func_t>  // 简化重载：只传回调。
__device__ __forceinline__ void timeout_while(const func_t& func, const int64_t& start_clock = 0) {  // condition 固定为 true。
    // 简化重载：默认 condition=true，循环直到 func 返回 true 或超时。
    timeout_while<kNumTimeoutCycles, func_t>(true, func, start_clock);
}

template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>  // SM/QP/channel 配置。
__device__ __forceinline__ std::pair<int, ncclGinResourceSharingMode> get_qp_mode(
    const int& sm_idx, const int& channel_in_sm_idx, const bool& is_notify_warp = false) {  // 当前 SM、SM 内 channel 和 notify 标记。
    // 把 SM/channel 映射到 Gin QP。设计目标是：
    // - QP 充足时，每个 SM 尽量独占若干 QP，减少 doorbell/队列竞争；
    // - QP 不足时，所有 SM 的 channel 轮询共享 QP；
    // - notify warp 固定使用 QP0，避免和 payload channel 的资源分配互相影响。
    constexpr auto kSharingCTA = NCCL_GIN_RESOURCE_SHARING_CTA;  // CTA 级共享模式。
    // CTA sharing 表示该 QP 只被一个 CTA/SM 的一组 channel 使用。
    constexpr auto kSharingGrid = kNumSMs == 1 ? NCCL_GIN_RESOURCE_SHARING_CTA : NCCL_GIN_RESOURCE_SHARING_GPU;  // 多 SM 时使用 GPU 级共享。
    // 多 SM 共享 QP 时使用 GPU-level sharing；单 SM 时 CTA sharing 足够。

    // Only one QP
    if constexpr (kNumQPs == 1)
        // 只有一个 QP 时所有 SM/channel 共享 QP0。
        return {0, kSharingGrid};

    // The notify warp always use 1 SM and 1 QP
    if (is_notify_warp)  // notify warp 固定走 QP0。
        // notify warp 固定 QP0，并用 CTA sharing 降低和数据通道的耦合。
        return {0, kSharingCTA};

    // Data channels
    constexpr int kQPStartIdx = static_cast<int>(kWithNotifyWarps);  // 数据 QP 起始下标。
    // 如果有 notify warp，数据 QP 从 1 开始，跳过 QP0。
    constexpr int kNumAvailableQPs = kNumQPs - kQPStartIdx;  // payload 可用 QP 数。
    // 数据通道可用 QP 数。
    if constexpr (kNumSMs <= kNumAvailableQPs) {
        // A single SM uses an entire QP
        // e.g., 3 SMs and 10 QPs
        // 中文：下面三行展示 QP 交错分配后的结果，SM0 会拿到 0/3/6/9。
        // SM 0: 0 3 6 9
        // SM 1: 1 4 7
        // SM 2: 2 5 8
        const int num_qps_in_sm = (kNumAvailableQPs / kNumSMs) + (sm_idx < (kNumAvailableQPs % kNumSMs));  // 当前 SM 分到的 QP 数。
        // 当前 SM 分到的 QP 数，前 remainder 个 SM 多一个 QP。
        return {kQPStartIdx + sm_idx + (channel_in_sm_idx % num_qps_in_sm) * kNumSMs, kSharingCTA};
        // QP 按 SM 交错分配，使每个 SM 的 QP 尽量分散。
    } else {
        // All SMs share all QPs
        const auto global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;  // 展平 channel id。
        // 全局 channel 编号。
        return {kQPStartIdx + (global_channel_idx % kNumAvailableQPs), kSharingGrid};
        // channel 轮询映射到可用 QP，使用 GPU sharing。
    }
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag>  // NVLink barrier 静态配置。
__forceinline__ __device__ void nvlink_barrier_wo_local_sync(
    const handle::NCCLGin& gin,  // Gin handle，用于获取 LSA symmetric pointer。
    const layout::WorkspaceLayout& workspace,  // barrier signal/counter 所在 workspace。
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {  // 当前 rank、SM 和线程。
    // NVLink barrier 使用 LSA symmetric pointer 和 workspace 中的 signal/counter 完成。
    // This barrier only uses 1 SM
    if (kNumSMs > 1 and sm_idx > 0)  // 只有 sm0 执行 NVLink barrier。
        // NVLink barrier 只用 sm0，其它 SM 直接返回，外层 gpu_barrier 会负责 grid sync。
        return;

    // Read the current barrier phase first
    const int status = static_cast<int>((*workspace.get_nvl_barrier_counter_ptr()) & 3);  // 读取低两位 phase/sign。
    // counter 低两位编码 phase/sign，避免每轮清零 signal。
    const int phase = status & 1, sign = status >> 1;  // phase 选择 signal 槽，sign 选择加/减。
    // phase 选择双缓冲 signal 槽；sign 决定本轮加一还是减一。

    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    // 每个 rank 需要一个线程发 signal。
    if (thread_idx < kNumRanks) {  // 每个有效线程通知一个 peer rank。
        // 前 kNumRanks 个线程分别通知一个 rank。
        const auto dst_ptr =  // 目标 rank 的 phase signal 地址。
            gin.get_sym_ptr<ncclTeamTagLsa>(workspace.get_nvl_barrier_signal_ptr(phase), thread_idx);
        // 获取目标 LSA peer 上同一 phase signal 的对称地址。
        ptx::red_add_rel_sys(dst_ptr, sign ? -1 : 1);
        // release 语义地对远端 signal 加/减一。
    }
    __syncthreads();
    // 等待所有通知线程完成 red_add。

    // NOTES: we need `2^64 / 1e6 / 3600 / 24 / 365 = 571000` years to make the counter overflow (1 barrier per us)
    // 中文：64 位 counter 在实际运行时间内可视为不会溢出，因此不用额外处理 wrap-around。
    // Add the phase counter
    if (thread_idx == 0)
        // thread0 推进本地 barrier counter，进入下一轮 phase/sign。
        atomicAdd(workspace.get_nvl_barrier_counter_ptr(), 1);

    // Check timeout
    const auto target = sign ? 0 : kNumRanks;  // 当前轮等待的 signal 目标值。
    // 本轮 signal 目标值：加法轮等到 kNumRanks，减法轮等到 0。
    timeout_while<kNumTimeoutCycles>(thread_idx == 0, [=](const bool& is_last_check) {  // thread0 等待所有 peer 到达。
        // 只让 thread0 轮询本地 signal，减少全局内存压力。
        const auto signal = ptx::ld_acquire_sys<int>(workspace.get_nvl_barrier_signal_ptr(phase));  // 当前 phase signal。
        // acquire load 确保看到 peer 的 release red_add。
        if (signal == target)
            // 所有 rank 都到达本轮 barrier。
            return true;

        if (is_last_check) {
            // 超时时打印 tag/rank/thread/phase/counter 方便定位卡在哪一轮。
            printf("DeepEP NVLink barrier timeout, tag: %d, nvl: %d, thread: %d, "
                   "status: %d, signal: %d, phase: %d, target: %d, counter: %llu\n",
                   kTag, rank_idx, thread_idx, status, signal, phase, target,
                   // 中文：counter 直接从 workspace 读取，反映本地 barrier 已推进的轮次。
                   *workspace.get_nvl_barrier_counter_ptr());
        }
        return false;
        // 未达到 target，继续等待。
    });
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles,  // Gin barrier 静态配置。
          typename team_t, int kTag = kDeviceBarrierTag,  // world/rail team 和日志 tag。
          bool kFlushStores = true,  // barrier 前是否 flush 所有 QP。
          int kNumWarps = kNumThreads / 32>  // 每 block warp 数。
__forceinline__ __device__ void gin_barrier_wo_local_sync(
    const ncclDevComm_t& nccl_dev_comm,  // device communicator。
    const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 当前二维 rank 坐标。
    const int& sm_idx, const int& thread_idx) {  // 当前 SM 和线程。
    // Gin barrier 使用 NCCL Gin signal 在 world 或 rail team 内同步。
    const auto global_warp_idx = sm_idx * kNumWarps + (thread_idx / 32);  // 展平 warp id。
    // 全局 warp 编号，用于并行 flush QP。
    const int& rank_idx = (std::is_same_v<team_t, ncclTeamTagWorld>) ? scaleup_rank_idx : scaleout_rank_idx;  // 当前 team 内 rank id。
    // world team barrier 用 scaleup_rank_idx，rail team barrier 用 scaleout_rank_idx。
    const int num_qps = kNumQPs == kFlushAllAllocatedQPs ? nccl_dev_comm.ginContextCount : kNumQPs;  // 需要 flush 的 QP 数。
    // kFlushAllAllocatedQPs 表示读取 communicator 中实际 Gin context 数。

    // Flush all QPs by all SMs (only needed for release semantics)
    if constexpr (kFlushStores) {
        // barrier 前 flush，保证此前通过 Gin 发出的 store/put 在对端可见。
        // 这里 flush 的是 QP 维度，而不是线程维度：同一 kernel 中不同 SM/warp 可能使用不同 Gin context，
        // 所以用 global_warp_idx 把 num_qps 分摊给所有参与 warp，避免只 flush QP0 漏掉 payload 通道。
        for (int i = global_warp_idx; i < num_qps; i += kNumSMs * kNumWarps) {  // 所有 warp 分摊 QP flush。
            // 所有 warp 分摊 flush QP。
            ncclGin(nccl_dev_comm, i, NCCL_GIN_RESOURCE_SHARING_CTA).flush(ncclCoopWarp());
            // 使用 cooperative warp flush 指定 QP。
        }
        // NOTES: we can not use `kNumSMs` to judge, as maybe only part of the SMs will call this function
        // cooperative grid sync 的判断用 gridDim.x，而不是模板 kNumSMs：某些调用只让部分 SM 进入 barrier，
        // 但 launch grid 仍可能更大；这里必须匹配实际参与 cooperative launch 的同步语义。
        (gridDim.x > 1) ? cooperative_groups::this_grid().sync() : __syncthreads();
        // flush 后同步所有参与线程，确保 barrier signal 不早于数据可见。
    }

    if (sm_idx == 0) {  // sm0 负责发 signal 和等待 signal。
        // 只有 sm0 发 barrier signal 和等待所有 rank signal。
        // Use QP 0 to do barrier
        const auto team = (std::is_same_v<team_t, ncclTeamTagWorld>) ?  // 根据模板选择 NCCL team。
            ncclTeamWorld(nccl_dev_comm) : ncclTeamRail(nccl_dev_comm);
        // 根据 team_t 选择 world team 或 rail team。
        const ncclGin gin(nccl_dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);  // barrier 固定使用 QP0。
        // barrier signal 固定使用 QP0。
        for (int i = thread_idx; i < kNumRanks; i += kNumThreads)  // sm0 线程分摊目标 rank。
            // sm0 内线程分摊向所有 rank 发 signal。
            gin.signal(team, i, ncclGin_SignalInc{static_cast<ncclGinSignal_t>(rank_idx)});
            // 对目标 rank 的 signal[rank_idx] 加一，表示本 rank 到达。

        // TODO(NCCL): Using the official NCCL wait signal API, after they added timeout check.
        for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {
            // 每个线程等待若干 peer rank 给自己的 signal。
            const auto signal_idx = static_cast<ncclGinSignal_t>(i);  // peer signal 槽编号。
            // 要等待的 peer signal 槽。
            const auto shadow_ptr = gin.getSignalShadowPtr(signal_idx);  // 本地 shadow 期望值。
            // shadow 记录本地期望的下一次 signal 值。
            // shadow 在本地递增而非清零远端 signal，避免每轮 barrier 都做额外清理；
            // signal 表是单调递增计数，target 就是“本 rank 下一次应该观察到的 peer 到达次数”。
            const auto target = ++(*shadow_ptr);  // 本轮要等待的 signal 值。
            // 当前 barrier 目标值为 shadow 自增后的值。

            const auto gdaki = static_cast<struct ncclGinGdakiGPUContext*>(gin._ginHandle) + gin.contextId;  // 当前 Gin context 状态。
            // 取得 Gin context 的 GDAKI 状态。
            const auto signal_ptr = reinterpret_cast<uint64_t*>(__ldg(reinterpret_cast<uint64_t*>(&gdaki->signals_table.buffer))) + signal_idx;  // signal table 地址。
            // 远端 signal 到达后会更新本地 signals_table 中对应计数。
            timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {  // 等待 peer signal 达到 target。
                // 轮询直到 signal >= target。
                const auto signal = ptx::ld_acquire_sys<uint64_t>(signal_ptr);  // 当前 signal 值。
                // acquire load 确保 peer signal 之前的数据可见。
                if (signal >= target)
                    // peer 已到达本轮 barrier。
                    return true;

                if (is_last_check) {
                    // 超时时打印当前 tag/坐标/signal/target。
                    printf("DeepEP Gin barrier timeout, tag: %d, scaleout: %d, scaleup: %d, thread: %d, "
                           // 中文：signal 是当前观察值，target 是本轮 barrier 要达到的计数。
                           "signal: %lu, target: %lu\n", kTag, scaleout_rank_idx, scaleup_rank_idx, thread_idx, signal, target);
                }
                return false;
                // 未到达，继续等待。
            });
        }  // 中文：结束当前线程负责的 peer signal 等待。
    }  // 中文：结束 Gin barrier 的所有 peer 遍历。
}

template <bool kIsScaleupNVLink,  // scaleup 域是否是 NVLink/LSA，可决定 barrier 走 workspace signal 还是 Gin world。
          int kNumRanks,          // scaleup 域 rank 数。
          int kNumSMs,            // 参与 barrier 的 SM 数。
          int kNumThreads,        // 每个 block/SM 的线程数。
          int kNumQPs,            // Gin QP 数，非 NVLink 路径需要。
          // 中文：模板常量让 barrier 在 JIT 编译期固化拓扑和超时策略，减少运行期分支。
          int64_t kNumTimeoutCycles, // 设备端等待超时周期。
          int kTag = kDeviceBarrierTag, bool kFlushStores = true>  // tag 用于日志定位，kFlushStores 控制 barrier 前是否 flush。
__forceinline__ __device__ void scaleup_barrier_wo_local_sync(
    const handle::NCCLGin& gin,  // Gin handle。
    const layout::WorkspaceLayout& workspace,  // workspace 控制面。
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {  // 当前 rank/SM/thread。
    // scaleup barrier 根据 scaleup 是否 NVLink 选择 LSA/NVLink 或 Gin world team。
    if constexpr (kIsScaleupNVLink) {
        // NVLink 域内优先使用 symmetric pointer + atomic signal。
        nvlink_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumTimeoutCycles, kTag>(
            gin, workspace, rank_idx, sm_idx, thread_idx);
    } else {
        // 非 NVLink scaleup 退回 Gin world team barrier。
        gin_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, ncclTeamTagWorld, kTag, kFlushStores>(
            gin.nccl_dev_comm, 1, rank_idx, sm_idx, thread_idx);
    }  // 中文：结束 scaleup barrier 的 NVLink/Gin 路径选择。
}

template <int kNumRanks,      // scaleout 域 rank 数。
          int kNumSMs,        // 参与 scaleout barrier 的 SM 数。
          int kNumThreads,    // 每个 SM/block 线程数。
          int kNumQPs,        // 可用于 flush/signal 的 Gin QP 数。
          int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true>  // scaleout barrier 默认在发 signal 前 flush 前序 Gin store。
__forceinline__ __device__ void scaleout_barrier_wo_local_sync(
    const handle::NCCLGin& gin,  // Gin handle。
    const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 当前二维 rank 坐标。
    const int& sm_idx, const int& thread_idx) {  // 当前 SM/thread。
    // scaleout barrier 总是使用 rail team，对应跨节点/跨 scaleout 维通信。
    gin_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, ncclTeamTagRail, kTag, kFlushStores>(
        gin.nccl_dev_comm, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
}

template <bool kIsScaleupNVLink,  // scaleup barrier 是否可用 NVLink/LSA 快路径。
          int kNumScaleoutRanks, int kNumScaleupRanks,  // 二维拓扑大小；为 1 的维度会被自动跳过。
          int kNumSMs, int kNumThreads, int kNumQPs,    // 参与同步的 SM/线程/QP 静态配置。
          int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true, bool kSyncAtStart = true, bool kSyncAtEnd = true>  // 是否 flush 与首尾 grid sync。
__forceinline__ __device__ void gpu_barrier(const handle::NCCLGin& gin,  // Gin handle。
                                            const layout::WorkspaceLayout& workspace,  // workspace 控制面。
                                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 当前二维 rank。
                                            const int& sm_idx, const int& thread_idx,  // 当前 SM/thread。
                                            bool do_scaleout = true, bool do_scaleup = true) {  // 运行时是否执行两类 barrier。
    // gpu_barrier 是 Elastic kernel 中的统一同步入口，可选择只做 scaleup、只做 scaleout 或两者都做。
    // scaleup 通常表示节点内/同 scaleout 维，scaleout 通常表示跨节点/rail 维。
    // A general TMA store wait to prevent proxy memory issues
    if constexpr (kFlushStores) {
        // barrier 前提交并等待所有 TMA store，避免 proxy memory 可见性问题。
        // TMA 使用 proxy memory path，如果不 wait，远端 peer 可能看到 signal 但还看不到 payload。
        // 这一步和 Gin QP flush 是两层不同的可见性：TMA wait 保证本 SM 的 shared->global store 完成，
        // 后续 gin_barrier_wo_local_sync 的 QP flush 则保证 Gin 发出的网络写完成。
        ptx::tma_store_commit();
        // 提交当前线程/warp 未提交的 TMA store。
        ptx::tma_store_wait();
        // 等待 TMA store 完成。
        __syncwarp();
        // 同步 warp 内线程，保证 flush 前状态一致。
    }

    // All the SMs should wait
    if constexpr (kSyncAtStart) {
        // barrier 开始前做 grid sync，确保所有 SM 都已完成前序本地工作。
        cooperative_groups::this_grid().sync();
    } else {
        // 不做起始同步时也不能 flush store，否则其它 SM 可能尚未完成写入。
        EP_STATIC_ASSERT(not kFlushStores, "No data to be flushed");
    }

    do_scaleout &= kNumScaleoutRanks > 1;  // 单 rank scaleout 自动跳过。
    // scaleout rank 只有 1 时无需跨 scaleout barrier。
    // 运行时 do_scaleout=false 可用于某些分支只同步节点内流量。
    do_scaleup &= kNumScaleupRanks > 1;  // 单 rank scaleup 自动跳过。
    // scaleup rank 只有 1 时无需 scaleup barrier。
    // 编译期 rank 数为 1 时强制跳过，可以让无通信维度不产生等待。
    if (do_scaleup and do_scaleout) {
        // 同时需要 scaleup/scaleout 时并行执行两个 barrier，降低总同步延迟。
        // Do scaleup and scaleout barrier in parallel
        EP_DEVICE_ASSERT(kNumSMs >= 2 and "At least 2 SMs for a hybrid barrier");
        // 至少需要两个 SM：sm0 做 scaleup，其余 SM 做 scaleout。
        // 这种拆分要求 launch 至少覆盖 kNumSMs 个 block/SM，否则某个 barrier 分支没人执行。
        // 更隐蔽的是 cooperative_groups::this_grid().sync 必须被所有 block 以相同次数执行；
        // 因此 scaleup 分支虽然只用 sm0，也要在需要时补一个额外 grid sync 来匹配 scaleout 分支。
        if (sm_idx == 0) {  // sm0 负责 scaleup barrier。
            // First SM do the scaleup barrier
            // scaleup barrier 不做本地 grid sync，所以外层负责首尾 cooperative_groups sync。
            scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
                gin, workspace, scaleup_rank_idx, sm_idx, thread_idx);  // scaleup_rank_idx 是 scaleup 域内 rank id。
            // sm0 负责 scaleup barrier。

            // We need an extra grid sync, as the scaleout barrier will do a sync after flush, before the barrier
            // NOTES: this is kind of hacky
            if constexpr (kFlushStores)  // 匹配 scaleout 分支 flush 后的 grid sync。
                // scaleout 分支 flush 后会 grid sync，scaleup 分支也需要匹配一次，避免分支不同步。
                // 这条额外 sync 是为了让两个分支拥有相同数量的 grid sync，避免 cooperative grid 死锁。
                cooperative_groups::this_grid().sync();
        } else {  // 其余 SM 负责 scaleout barrier。
            // The remaining SMs do the scaleout barrier
            // scaleout 分支使用 kNumSMs-1，因为 sm0 已经被 scaleup 分支占用。
            scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs - 1, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
                gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx - 1, thread_idx);  // sm0 已给 scaleup 使用，scaleout 看到的 SM 从 0 重编号。
            // sm_idx-1 让 scaleout barrier 看到的 SM 编号从 0 开始。
        }
    } else if (do_scaleup) {
        // Scaleup only
        // 只有 scaleup 通信时，所有 SM 都可投入同一个 barrier，提高 signal 并行度。
        scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
            gin, workspace, scaleup_rank_idx, sm_idx, thread_idx);  // 单维 scaleup 同步，所有 SM 都参与。
        // 只做节点内/scaleup 同步。
    } else if (do_scaleout) {
        // Scaleout only
        // 只有 scaleout 通信时，rail team barrier 覆盖跨节点 rank，scaleup 维不会参与 signal。
        scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
            gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);  // 单维 scaleout 同步，rail team 覆盖跨节点 rank。
        // 只做跨 scaleout 同步。
    }

    // All the SMs should wait
    if constexpr (kSyncAtEnd)  // 需要首尾成对 grid sync 时执行收尾同步。
        // barrier 结束后再 grid sync，确保所有 SM 在同一时刻离开同步区。
        // 这能防止某些 SM 先进入下一阶段并复用 workspace signal。
        cooperative_groups::this_grid().sync();
}

}  // namespace deep_ep::elastic::comm
// Elastic communication helper 命名空间结束。
