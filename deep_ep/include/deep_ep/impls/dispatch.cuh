#pragma once  // 单层 elastic dispatch device 实现由 JIT include，防止重复定义。

// nccl.h/nccl_device.h 提供 device-side NCCL/Gin 通信类型与 team tag。
#include <nccl.h>
#include <nccl_device.h>

// comm.cuh 封装 GPU barrier、QP 选择、scale-up/scale-out 同步协议。
#include <deep_ep/common/comm.cuh>
// compiled.cuh 提供 topk_idx_t、sf_pack_t 等编译期选择出来的类型。
#include <deep_ep/common/compiled.cuh>
// exception.cuh 提供 EP_DEVICE_ASSERT/EP_STATIC_ASSERT。
#include <deep_ep/common/exception.cuh>
// handle.cuh 封装 NCCL Gin 的 put/get/signal 与 NVLink 可达性判断。
#include <deep_ep/common/handle.cuh>
// layout.cuh 把线性 buffer/workspace 解释为 token、rank、expert 等结构化布局。
#include <deep_ep/common/layout.cuh>
// math.cuh 提供 constexpr 对齐、指针偏移、正数编码等工具函数。
#include <deep_ep/common/math.cuh>
// ptx.cuh 提供 TMA、mbarrier、named barrier、warp shuffle 等 PTX 辅助函数。
#include <deep_ep/common/ptx.cuh>


// elastic dispatch 的主 kernel 位于 deep_ep::elastic 命名空间。
namespace deep_ep::elastic {  // elastic 单层 dispatch kernel 命名空间。

// kIsScaleupNVLink 表示 scale-up rank 之间是否可以走 NVLink/LSA 直接访存。
template <bool kIsScaleupNVLink,  // scaleup rank 是否可通过 NVLink/LSA 直接访问。
          // kDoCPUSync 控制是否把计数写到 mapped host workspace，供 CPU 立即读取。
          bool kDoCPUSync,  // 是否把计数写到 mapped host workspace。
          // kReuseSlotIndices 表示 slot 已由 deterministic prologue 计算，dispatch 不再 atomic 分配。
          bool kReuseSlotIndices,  // 是否复用 deterministic prologue 计算好的 slot。
          // kNumSMs 是 cooperative launch 的 block 数，每个 block 近似对应一个 SM 工作单元。
          int kNumSMs,  // cooperative launch block 数。
          // notify warps 负责计数/通知，dispatch warps 负责搬运 payload。
          int kNumNotifyWarps, int kNumDispatchWarps,  // notify 和 payload dispatch warp 数。
          // kNumRanks 是 scale-up rank 数；该文件只处理一层 scale-up dispatch。
          int kNumRanks,  // scale-up rank 数。
          // kNumHiddenBytes 是单 token hidden 字节数；kNumSFPacks 是 FP8 scale pack 个数。
          int kNumHiddenBytes, int kNumSFPacks,  // hidden 字节数和 scale factor pack 数。
          // 每个目标 rank 的 recv buffer 最大 token 容量。
          int kNumMaxTokensPerRank,  // 每个目标 rank 最大 token 容量。
          // expert/top-k 形状与本地 expert 对齐粒度。
          int kNumExperts, int kNumTopk, int kExpertAlignment,  // expert/top-k 形状和 expert 对齐粒度。
          // kNumQPs/kNumTimeoutCycles 控制 Gin QP 分配与 barrier/notify 超时保护。
          int kNumQPs, int64_t kNumTimeoutCycles,  // Gin QP 数和超时周期。
          // notify 线程数只覆盖 notify warps。
          int kNumNotifyThreads = kNumNotifyWarps * 32,  // notify 线程数。
          // dispatch 线程数只覆盖 payload 搬运 warps。
          int kNumDispatchThreads = kNumDispatchWarps * 32,  // dispatch 线程数。
          // block 总线程数是两类 warp 的总和。
          int kNumThreads = kNumNotifyThreads + kNumDispatchThreads,  // block 总线程数。
          // NVLink 模式使用 LSA team；非 NVLink 使用 world team 走更通用路径。
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>  // 根据互联选择 Gin team。
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_impl(
    // x 是输入 token hidden，逻辑形状 [num_tokens, hidden]。
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,  // 输入 token、scale factor、top-k id 和权重。
    // copied_topk_idx 可选，用于把 topk_idx 复制到输出侧以便上层复用。
    topk_idx_t* copied_topk_idx,  // 可选 topk_idx 输出副本。
    // cumulative_local_expert_recv_stats 可选，累计本 rank 各本地 expert 接收量。
    int* cumulative_local_expert_recv_stats,  // 本地 expert 累计接收统计。
    // psum_num_recv_tokens_per_scaleup_rank 输出 rank 维前缀和，epilogue 用它整理 recv_x。
    int* psum_num_recv_tokens_per_scaleup_rank,  // 输出 rank 维接收 token 前缀和。
    // psum_num_recv_tokens_per_expert 输出本地 expert 维 exclusive prefix，expanded 模式用。
    int* psum_num_recv_tokens_per_expert,  // 输出 expert 维前缀和。
    // dst_buffer_slot_idx 输出/复用每个 token top-k lane 的目标 buffer slot。
    int* dst_buffer_slot_idx,  // 每个 token/top-k 的目标 slot。
    // num_tokens 是当前本 rank 输入 token 数。
    const int num_tokens,  // 输入 token 数。
    // FP8 scale tensor 的 token/hidden stride，由 PyTorch tensor 布局传入。
    const int sf_token_stride, const int sf_hidden_stride,  // FP8 scale factor stride。
    // device-side NCCL communicator/window，用于 Gin put 和 symmetric pointer 解析。
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* buffer,  // communicator/window 和 payload buffer。
    // workspace 是 GPU 控制面；mapped_host_workspace 是 CPU 可见的 pinned/mapped 控制面。
    void* workspace, void* mapped_host_workspace,  // GPU workspace 和 host mapped workspace。
    // rank_idx 是当前 GPU 在 scale-up 组内的 rank。
    const int rank_idx  // 当前 scale-up rank。
) {
    // 每个 rank 管辖连续 expert 区间；expert->rank 通过除以 kNumExpertsPerRank 得到。
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;  // 每 rank expert 数。
    // expert 必须能均匀切到 rank，否则上面的整数映射不成立。
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    // notify warp 数要求 4 的倍数，匹配后续 named barrier/warpgroup 调度约束。
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");

    // 运行示例：
    //   kNumRanks=8、kNumExperts=64 时，每个 rank 负责 8 个 expert。
    //   若某 token 的 topk_idx=[3, 9, 10, -1, ...]，expert 3 属于 rank0，
    //   expert 9/10 属于 rank1；rank 级计数需要去重为 rank0、rank1 各一次，
    //   expert 级计数则分别累加 expert3、expert9、expert10。
    // 这解释了下面为什么同时维护 rank_count 和 expert_count，并且 rank_count 要调用
    // ptx::deduplicate 防止同一 token 的多个 expert 落在同一 rank 时重复发送。

    // Utils：当前 block/线程坐标。
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);  // block/线程坐标。
    // warp_idx/lane_idx 用于把 block 内线程划分成 notify 与 dispatch 两个角色组。
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();  // warp/lane 坐标。

    // Workspaces
    // workspace_layout 指向 GPU symmetric window 中的控制面区域；host_workspace_layout 指向
    // pinned mapped host workspace。前者用于 rank/expert 计数、barrier、远端通知；后者用于
    // do_cpu_sync 时把最终计数暴露给 CPU，Python/C++ 才能准确分配 recv_x 大小。
    // scale-out 维度这里固定为 1，因此 WorkspaceLayout 的第一个 rank 维传 1。
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);  // GPU 控制面布局。
    // host_workspace_layout 与 workspace_layout 字段布局一致，只是内存位于 mapped host 区域。
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, 1, kNumRanks, kNumExperts);  // host mapped 控制面布局。

    // The kernel uses a fixed space of dynamic shared memory (no static shared memory)
    // 动态 shared memory 同时服务 notify 计数区和 dispatch TMA staging 区。
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];  // 动态 shared：notify 计数 + TMA staging。
    // notify 区需要容纳 rank_count[kNumRanks]+expert_count[kNumExperts]，并按 notify 线程数对齐。
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?  // notify shared 区字节数。
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    // dispatch TMA staging 紧跟 notify 区，notify 区大小必须满足 TMA 对齐。
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");

    // Named barrier indices：notify warps 专用 barrier id，避免与普通 __syncthreads 混用。
    constexpr int kNotifyBarrierIndex = 1;  // notify named barrier id。

    // Gin handle
    // We treat each warp as a "channel"
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumDispatchWarps, (kNumNotifyWarps > 0)>(  // 当前 warp 的 QP 模式。
        sm_idx, warp_idx - kNumNotifyWarps, warp_idx < kNumNotifyWarps);
    // gin 把 NCCL device communicator/window 封装成统一 put/put_value API。
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);  // Gin 通信句柄。

    // Barrier without TMA store flush, without prologue grid sync
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,  // dispatch 入口团队同步。
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag0, false, false, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // Different warp roles
    // warp 角色分两类：
    // - notify warps：扫描 topk_idx，统计 rank/expert 计数，通知各目标 rank；
    // - dispatch warps：根据已经确定的 slot，把 x/sf/topk payload 写入远端 buffer。
    // 这种拆分让接收方能尽早知道布局，同时 payload 复制可以与通知流程并行推进。
    if (warp_idx < kNumNotifyWarps) {  // notify 分支：统计和发布计数。
        // Assign shared memory：notify 只使用 smem 开头的计数区。
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);  // notify shared 区 int 数。
        // rank_expert_count 是 rank_count 与 expert_count 的连续拼接。
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);  // rank/expert shared 计数起点。

        // Clean initial counts
        // NOTES: if you want to change the order of different warp roles, please take care of the `thread_idx`
        // rank_count 长度 kNumRanks；expert_count 长度 kNumExperts，紧跟在 rank_count 后。
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;  // rank_count 和 expert_count 两段。
        #pragma unroll
        // notify 线程分摊清零 aligned 后的计数区，保证后续 atomicAdd_block 从 0 开始。
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)  // 每个 notify 线程清理若干元素。
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        // notify warp 内的命名 barrier；dispatch warps 不参与，避免整个 block 等待。
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Atomic add on shared memory
        // 每个 lane 对应一个 top-k 槽，因此要求 kNumTopk <= 32。对于一个 token，
        // expert_count 逐 expert 累加；rank_count 先把 expert 映射到 rank，再用
        // ptx::deduplicate 在 warp 内去重。
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        // global_warp_idx 把 notify warp 按 SM 交错，均匀覆盖 num_tokens。
        const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;  // 全局 notify warp id。
        // 每个 notify warp 以 stride=kNumNotifyWarps*kNumSMs 扫描 token。
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {  // 网格步进扫描 token。
            // Expert choice can not be redundant
            // NOTES: no assertions here as they are expensive
            // lane 0..kNumTopk-1 分别处理一个 top-k expert，其余 lane 置 -1。
            const auto dst_expert_idx = lane_idx < kNumTopk ?  // 当前 lane 的目标 expert。
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            // expert_count 统计每个全局 expert 被选择的次数。
            if (dst_expert_idx >= 0)  // 有效 expert 才计数。
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // Rank choice should do deduplication here：同一 token 发往同一 rank 只需要一份 payload。
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;  // expert 映射到目标 rank。
            // deduplicate 在 warp 内相同 rank 只保留一个 lane，避免 rank_count 过大。
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)  // 同 token 同 rank 去重。
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        // 等待 notify warps 的 shared 计数写完，再做跨 SM 归约。
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Do full-grid reduction
        // 每个 SM 只统计了自己负责的 token 子集，需要把所有 SM 的局部计数归约到 workspace。
        // counter 高 32 位存“已有多少个 SM 到达”，低 32 位存累计计数；SM0 后续等待高位
        // 达到 kNumSMs，说明全 grid 的计数已经完整。
        #pragma unroll
        // 每个 notify 线程负责若干 rank/expert 计数项。
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            // counter 高 32 位加 1 表示当前 SM 到达，低 32 位携带该 SM 局部计数。
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];  // arrival_count=1 + 局部计数。
            // 对全局 workspace 做 64-bit red.add，所有 SM 的局部计数会累加到同一槽。
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // Do the remaining work by SM 0：只有 SM0 负责等待归约完成、通知 peers、生成 prefix sum。
        if (sm_idx == 0) {  // SM0 汇总全 grid 并通知 peers。
            // Reduce all SM's count
            // Wait all SMs' arrival
            #pragma unroll
            // SM0 的 notify 线程分摊等待每个 rank/expert 的全 grid 归约结果。
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                // timeout_while 避免通信死锁时无限挂住，最后一次检查会打印诊断。
                comm::timeout_while<kNumTimeoutCycles>(true, [=](const bool& is_last_check) {  // 等待所有 SM 归约到达。
                    // status 高 32 位是已到达 SM 数，低 32 位是累计计数。
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);  // 高位 arrival，低位 count。
                    // 当所有 SM 都写入后，计数完整可用。
                    if ((status >> 32) == kNumSMs) {  // 全部 SM 到达。
                        // Write into shared memory
                        // Write into send buffer if with RDMA
                        // encode_decode_positive 把非负计数编码成“ready”格式，便于对端轮询。
                        const auto encoded =
                            math::encode_decode_positive(static_cast<int>(status & 0xffffffffll));
                        // 写回 shared，后续 prefix sum 和 peer 通知都会从这里读取。
                        rank_expert_count[i] = encoded;
                        // 非 NVLink/RDMA 模式需要把计数放到 symmetric workspace，供 bulk copy 给 peers。
                        if constexpr (not kIsScaleupNVLink)
                            workspace_layout.get_scaleup_rank_expert_count_ptr<true>()[i] = encoded;

                        // Clean for the next usage：清理归约槽，下一轮 dispatch 可复用。
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    // 最后一次超时检查打印 rank、线程、当前到达 SM 数和期望值。
                    if (is_last_check) {
                        // 中文：status 高 32 位是到达 SM 数，低 32 位是本字段归约后的计数值。
                        printf("DeepEP notify (GPU reduction) timeout, rank: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               rank_idx, kNumRanks, thread_idx,
                               // 中文：status 的高/低半字分别用于计数同步和归约值诊断。
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }  // 中文：只在最后一次轮询打印，避免 timeout 前刷屏。
                    return false;
                });
            }  // 中文：结束 GPU reduction 方式的 notify counter 等待。
            // 等待所有 rank/expert 归约结果都写回 shared。
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // TODO: for further optimization, we can fuse rank and expert counters
            // Issue scaleup rank count writes to peers
            // rank_count 会写到目标 peer 的 workspace，使目标 rank 知道“我会从源 rank 收到多少 token”。
            // 对 NVLink 可达 peer 使用 Gin put_value；对 RDMA peer 也走同一抽象，但底层 QP/共享模式不同。
            for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {  // 分摊通知每个目标 rank。
                // Rank counters
                // 目标 peer 的 scaleup_rank_count[rank_idx] 表示“源 rank_idx 会发给我多少 token”。
                const auto dst_rank_counter =  // 目标 rank workspace 中属于源 rank_idx 的 rank-count 槽。
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + rank_idx;
                // rank_count[i] 是本 rank 要发往目标 rank i 的 token 数。
                gin.put_value<team_t>(dst_rank_counter, static_cast<int64_t>(rank_count[i]), i,
                                      ncclGinOptFlagsAggregateRequests);
            }
            // 确保 rank count 的 put_value 请求都发出，再继续 expert count 通知。
            __syncwarp();

            // Issue scaleup expert count writes to peers
            if constexpr (kIsScaleupNVLink) {  // NVLink 直达时逐 expert 直接写目标 peer。
                // NVLink per-element copy
                // We don't use TMA as the dtype of shared memory and global is different
                // NVLink 模式下每个 expert count 可以直接 put_value 到拥有该 expert 的目标 rank。
                for (int i = thread_idx; i < kNumExperts; i += kNumNotifyThreads) {  // 遍历全局 expert 计数。
                    // idx 是目标 peer workspace 中“当前源 rank 对某个本地 expert 的计数”槽位。
                    const auto idx = kNumExpertsPerRank * rank_idx + (i % kNumExpertsPerRank);  // 目标 workspace 中源 rank 的 expert 槽。
                    // i/kNumExpertsPerRank 得到全局 expert i 所属的目标 rank。
                    gin.put_value<team_t>(
                        workspace_layout.get_scaleup_expert_count_ptr<false>() + idx,
                        static_cast<int64_t>(expert_count[i]), i / kNumExpertsPerRank);
                }  // 中文：所有 expert count 写请求已按线程分片发起。
            } else {  // 非 NVLink 路径按 rank 批量 RDMA 搬 expert count。
                // RDMA bulk copy
                // RDMA 模式下先把所有 expert_count 编码后放到本地 symmetric workspace，
                // 再按目标 rank 批量 put 当前 rank 负责的 kNumExpertsPerRank 个 expert 计数。
                for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {  // 每个目标 rank 一次 bulk put。
                    // src_ptr 指向“发往目标 rank i 的那段 expert 计数”。
                    const auto src_ptr = workspace_layout.get_scaleup_expert_count_ptr<true>() + kNumExpertsPerRank * i;  // 本地 send expert 切片。
                    // dst_ptr 指向目标 rank workspace 中记录源 rank_idx 的位置。
                    const auto dst_ptr = workspace_layout.get_scaleup_expert_count_ptr<false>() + kNumExpertsPerRank * rank_idx;  // 远端 recv 中源 rank 切片。
                    // 一次 RDMA put 搬一整段本地 expert 计数，减少逐元素请求数。
                    gin.put<team_t>(dst_ptr, src_ptr, kNumExpertsPerRank * sizeof(int64_t), i);
                }
            }

            // This is necessary, as the waited results will rewrite the shared memory
            // 接下来会把 peer 发来的 rank/expert count 写回 rank_expert_count，因此先确保通知请求都发完。
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Wait for rank and expert count
            // 所有 peer 写入本 rank workspace 后，SM0 在这里轮询等待 ready 编码。
            const auto start_clock = clock64();  // rank/expert count 等待的统一起始时钟。
            // 前 kNumRanks 项是各源 rank 发给本 rank 的 token 数，后 kNumExperts 项是 expert 计数矩阵。
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {  // 等待每个 peer 写入字段。
                // 使用同一个 start_clock，让不同字段共享一个超时窗口。
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {  // 轮询 ready 编码。
                    // NOTES: the global memory type has 64 bits
                    // 远端写入的是 64-bit 槽，这里转回 int 后再 decode ready 状态。
                    const auto count = static_cast<int>(  // 读取 64-bit workspace 槽的低 32-bit count 编码。
                        ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + i));
                    // encode_decode_positive 是对称操作：对 ready 编码再次调用可得到原始非负 count。
                    const auto decoded = math::encode_decode_positive(count);  // 解码 ready 正数。
                    // decoded positive ready 表示 peer 已经完成该字段写入。
                    if (math::is_decoded_positive_ready(decoded)) {  // peer 已完成该字段写入。
                        // 清理 peer 写入槽，避免下一轮 dispatch 误读旧 ready。
                        workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] = 0;
                        // 写入 shared，后续 prefix/reduce 直接读取 rank_expert_count。
                        rank_expert_count[i] = decoded;
                        return true;
                    }

                    // 超时打印字段 i，方便定位是 rank count 还是 expert count 未到。
                    if (is_last_check)
                        // 中文：decoded 是当前观察到的 counter 值，常用于判断远端是否只写了一半。
                        printf("DeepEP notify timeout, rank: %d, thread: %d, count: %d\n", rank_idx, i, decoded);
                    return false;
                }, start_clock);
            }  // 中文：结束 RDMA/NVLink peer count 等待分支。
            // 等待所有 peer count 写回 shared。
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Reduce expert count and add stats
            // 当前 rank 只关心自己本地 kNumExpertsPerRank 个 expert 的总接收量。
            for (int i = thread_idx; i < kNumExpertsPerRank; i += kNumNotifyThreads) {  // 汇总本地 expert 计数。
                // sum 汇总所有源 rank 发给本地 expert i 的 token 数。
                int sum = 0;  // 本地 expert i 的所有源 rank 计数和。
                #pragma unroll
                // expert_count 被解释为 [source_rank, local_expert] 的矩阵。
                for (int j = 0; j < kNumRanks; ++ j)  // 沿 source rank 维求和。
                    sum += expert_count[j * kNumExpertsPerRank + i];
                // 对齐到 kExpertAlignment，方便后续 expanded 输出按对齐后的 expert 区间排布。
                expert_count[i] = math::align(sum, kExpertAlignment);

                // Update statistics counters
                // 统计值使用未对齐的真实 sum，避免监控指标受 padding 影响。
                if (cumulative_local_expert_recv_stats != nullptr)  // 调用方需要累计统计时才写。
                    atomicAdd(cumulative_local_expert_recv_stats + i, sum);
            }
            // 等待本地 expert_count 对齐值全部写回。
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Write host workspace
            if constexpr (kDoCPUSync) {  // CPU sync 路径写 host mapped count。
                // CPU 同步路径把 rank_count 和本地 expert_count 写到 mapped host workspace。
                for (int i = thread_idx; i < kNumRanks + kNumExpertsPerRank; i += kNumNotifyThreads) {  // 写 rank count 和本地 expert count。
                    // host 侧同样读取 ready 编码，保证 CPU 能判断 GPU 是否写完。
                    host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] =
                        math::encode_decode_positive(rank_expert_count[i]);
                }
                // 同 warp 同步，确保 host workspace 写请求已由所有 lane 发出。
                __syncwarp();
            }

            // Do prefix sum by the warps
            // NOTES: we may have fast implementation with `cub::BlockScan`, but it is too heavy to use
            // do_psum 用单个 warp 做小规模前缀和；n 通常是 rank 数或本地 expert 数，不值得引入 CUB。
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {  // 小数组 warp prefix helper。
                // psum 保存前面完整 32-lane 批次的累计和。
                int psum = 0;  // 已完成批次累计值。
                #pragma unroll
                // 若 exclusive，需要多输出一个起点 0，所以长度为 n+1。
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {  // 每轮处理 32 个 prefix 项。
                    // idx 是输出位置；每个 lane 对应一个位置。
                    const auto idx = i * 32 + lane_idx;  // 输出索引。
                    // exclusive 模式下 out[0]=0，因此输入索引向左偏移 1。
                    const auto mem_idx = idx - is_exclusive;  // exclusive 模式输入索引。
                    // 越界 lane 贡献 0，避免影响 warp prefix。
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;  // 越界贡献零。
                    // 当前批次的 inclusive prefix，再加上之前批次 psum。
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);  // 当前 prefix 值。

                    // Store into global memory
                    // inclusive(rank) 输出长度 n；exclusive(expert) 输出长度 n+1。
                    if (idx < n + is_exclusive)  // 有效输出项才写。
                        out[idx] = sum;

                    // Update `psum` by using the last lane's value
                    // lane31 持有当前批次总和，广播给所有 lane 作为下一批基础。
                    psum = ptx::exchange(sum, 31);
                }
            };
            // warp0 生成 rank 维 inclusive 前缀，epilogue 用最后一项得到总接收 token 数。
            if (warp_idx == 0) {
                // Inclusive prefix sum
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumRanks, 0);
            // warp1 生成 expert 维 exclusive 前缀，expanded epilogue 用它分配每个 expert 的输出区间。
            } else if (warp_idx == 1) {
                // Exclusive prefix sum for later expanding
                // 中文：exclusive prefix 给 expanded layout 留出每个 expert 的起始 offset。
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }  // 中文：notify warp prefix scan 完成。
        }  // 中文：结束 notify 子团队。
    } else {
        // dispatch_warp_idx 把全局 warp_idx 转成 payload 搬运组内的 0-based 下标。
        const int dispatch_warp_idx = warp_idx - kNumNotifyWarps;  // dispatch 子团队内 warp id。

        // Buffer layouts
        // token_layout 描述一个待发送 token：hidden、可选 SF、top-k metadata、mbarrier/源 token 字段。
        const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);  // 通信 token layout。
        // dispatch TMA staging 区位于 notify smem 之后，并按 dispatch_warp_idx 分配给每个 dispatch warp。
        const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumDispatchWarps, 1,  // 每个 dispatch warp 一个 shared staging。
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(dispatch_warp_idx).get_token_buffer(0);
        // recv_buffer 是目标 rank 的远端接收区，逻辑形状 [rank, kNumMaxTokensPerRank]。
        auto recv_buffer = layout::BufferLayout<false>(token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);  // 远端接收 buffer。
        // send_buffer 位于 recv_buffer 之后；非 NVLink/RDMA 路径先把 token 暂存到这里再 put。
        auto send_buffer = layout::BufferLayout<false>(token_layout, 1, kNumMaxTokensPerRank, recv_buffer.get_buffer_end_ptr());  // 本地 RDMA 发送 staging。
        // 当前 rank 的 recv_buffer 行用于生成 symmetric 指针基址，再通过 gin 映射到目标 rank。
        recv_buffer = recv_buffer.get_rank_buffer(rank_idx);

        // Init TMA
        // phase 跟踪 shared mbarrier 的当前 phase，TMA load 每轮等待后翻转。
        ptx::arrival_phase phase = 0;  // 当前 TMA mbarrier phase。
        // 当前 dispatch warp 独享的 mbarrier，放在 token staging buffer 内。
        const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();  // 当前 warp staging mbarrier。
        // 只由一个 lane 初始化 mbarrier，避免重复初始化同一 shared 地址。
        if (ptx::elect_one_sync())  // leader lane 初始化 mbarrier。
            ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        // 等待 mbarrier 初始化对整个 warp 可见。
        __syncwarp();

        // Iterate all tokens：dispatch warps 只负责 payload 搬运，不再参与 notify 计数。
        // token_start 让不同 SM/dispatch_warp 交错领取 token，减小单个 SM 的输入偏斜。
        const auto token_start = dispatch_warp_idx * kNumSMs + sm_idx;  // 当前 warp 的首个 token。
        // token_stride 是全局 dispatch warp 数。
        const auto token_stride = kNumDispatchWarps * kNumSMs;  // 当前 warp 的 token stride。
        // 每个 dispatch warp 顺序处理它负责的 token 子序列。
        for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {  // 遍历当前 warp 负责的 token。
            // token_i64_idx 用于字节偏移，避免 num_tokens 大时 32-bit 乘法溢出。
            const auto token_i64_idx = static_cast<int64_t>(token_idx);  // 64-bit token index 用于字节偏移。

            // Wait TMA store arrivals：复用 tma_buffer 前等待上一轮 TMA store 完成。
            ptx::tma_store_wait();
            // 整个 warp 同步后再写 shared staging buffer。
            __syncwarp();

            // Issue data TMA：把输入 hidden 从 global x 异步搬到 shared tma_buffer.hidden。
            if (ptx::elect_one_sync()) {  // leader lane 发起 hidden TMA load。
                // 一个 lane 发起 TMA load，mbarrier 后面等待 hidden bytes 到达。
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            // 允许其他 lane 继续准备 SF/top-k metadata，但 TMA 发射本身需要先同步。
            __syncwarp();

            // Issue SF TMA or cp.async：FP8 scale factor 较小，用 lane 级 cp.async 搬运即可。
            if constexpr (kNumSFPacks > 0) {
                // sf_pack_t 必须至少 4 字节对齐，cp.async_ca 按 32-bit 粒度更安全。
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                // gmem_src_ptr 是当前 token 的 SF 起点，stride 单位是 sf_pack_t 元素。
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));  // SF 全局地址。
                // smem_dst_ptr 是 token staging buffer 中 SF 区的起点。
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();  // SF shared 目标地址。

                // 完整轮每个 lane 搬一个 pack，共 32 个 pack。
                constexpr auto kNumFullIters = kNumSFPacks / 32;  // 32-lane 完整搬运轮数。
                #pragma unroll
                // k 轮覆盖 [k*32, k*32+31] 的 SF pack。
                for (int k = 0; k < kNumFullIters; ++ k) {  // 每轮搬 32 个 SF pack。
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                // 尾部不足 32 个 pack 时，只让有效 lane 发 cp.async。
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {  // 处理尾部不足 32 的 pack。
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                // 将 cp.async 也挂到同一个 mbarrier，后续一次 wait 同时等 hidden 和 SF。
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                // 等待所有 lane 发完自己的 cp.async。
                __syncwarp();
            }

            // Load top-k indices and weights：metadata 直接由 lane 写入 shared staging buffer。
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            // stored_dst_rank_idx 是当前 lane 对应的目标 rank，后续用于 dedup 和远端指针解析。
            int stored_dst_rank_idx = -1;  // 当前 lane 的目标 rank。
            // 前 kNumTopk 个 lane 分别处理一个 top-k 槽。
            if (lane_idx < kNumTopk) {  // top-k lane 读取 metadata。
                // 保留原始 topk_idx 类型，便于 copied_topk_idx 无损复制。
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);  // 原始 expert id。
                // 转 int 方便除法映射 rank 和写入 token metadata。
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);  // 转 int 后计算 rank。
                // expert=-1 表示无效 top-k lane，目标 rank 也置 -1。
                stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
                // 把全局 expert id 写入 token buffer，copy epilogue 会再转换成本地 expert id。
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                // topk_weights 可选；forward combine 需要，某些 backward/expanded 路径可为空。
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                // copied_topk_idx 是给上层返回的辅助副本，不参与通信协议。
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            // 等待所有 metadata lane 写完 staging buffer。
            __syncwarp();

            // Add source metadata (rank index and token index)
            // Please ensure no TMA buffer shared memory writes after this part
            // src_token_global_idx 编码为 rank_idx*kNumMaxTokensPerRank+token_idx，combine 用它恢复原源位置。
            if (ptx::elect_one_sync())  // leader lane 写源 token 全局 id。
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            // fence 确保后面对 tma_buffer 的 TMA store 能看到所有 shared metadata 写入。
            ptx::tma_store_fence();
            // staging buffer 写入结束，等待全 warp 进入 slot 分配。
            __syncwarp();

            // Deduplicate ranks and assign slots：为每个目标 rank 分配接收 buffer slot。
            // stored_dst_slot_idx 是目标 rank 内局部 slot，下游通过 rank+slot 找到 token。
            int stored_dst_slot_idx = -1;  // 目标 rank 内 slot。
            // deterministic prologue 已经预先写好 slot 时，直接复用它，避免 atomic 顺序不确定。
            if constexpr (kReuseSlotIndices) {
                // 每个 top-k lane 读取自己的全局 slot；无效 lane 为 -1。
                if (lane_idx < kNumTopk)  // top-k lane 读取复用 slot。
                    stored_dst_slot_idx = __ldg(dst_buffer_slot_idx + token_idx * kNumTopk + lane_idx);
                // 存储格式是 rank_idx*kNumMaxTokensPerRank+slot，这里还原成本 rank 内 slot。
                stored_dst_slot_idx = stored_dst_slot_idx >= 0 ?
                    (stored_dst_slot_idx - rank_idx * kNumMaxTokensPerRank) : -1;
                // cached/prologue 写入的是全局 token slot，减去本 rank base 后才是 recv_buffer 局部 slot。
            } else {
                // 非 deterministic 路径由去重后的 rank lane 对目标 rank 的 sender counter 做 atomicAdd。
                if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)  // 同目标 rank 只分配一次 slot。
                    stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);
                // 写回 dst_buffer_slot_idx，后续 combine 会依赖这个 slot 做反向路由。
                if (lane_idx < kNumTopk) {  // 每个 top-k lane 记录自己的全局 slot。
                    // value 使用全局 slot 编码；无效 lane 写 -1。
                    const auto value = stored_dst_slot_idx >= 0 ?  // 全局 slot 编码或 -1。
                        rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
                    // 用全局 slot 编码可以让 combine 不依赖当前 rank 局部上下文还原源位置。
                    dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = value;
                }
            }
            // 等待所有 lane 完成 slot 读取/写入，再真正发 payload。
            __syncwarp();

            // Wait TMA load arrival
            // NOTES: this arrive must be after the `ptx::cp_async_mbarrier_arrive`
            // 这里把 hidden 的 TMA transaction 字节数登记到 mbarrier，并等待 hidden/SF 全部到达。
            if (ptx::elect_one_sync()) {  // leader lane 等待 TMA/cp.async 到达。
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
            }
            // shared staging buffer 至此完整可读，可以发 TMA store/put。
            __syncwarp();

            // TMA store to send buffer：非 NVLink 模式需要本地 send buffer 作为 RDMA 源地址。
            auto send_buffer_ptr = send_buffer.get_token_buffer(token_idx).get_base_ptr();  // 本 token 的本地 send staging 地址。
            if constexpr (not kIsScaleupNVLink) {
                // 一个 lane 把完整 token 从 shared staging 写到本地 send_buffer[token_idx]。
                if (ptx::elect_one_sync())  // leader lane 写 send buffer。
                    ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                // commit 发送 store；稍后 tma_store_wait<1> 等待这一次 store 到达。
                ptx::tma_store_commit();
                // 所有 lane 等待 store 请求发出后再判断 NVLink bypass。
                __syncwarp();
            }

            // Issue TMA NVLink stores：如果目标 rank 可直接访问，就直接写远端 recv_buffer。
            EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid top-k selection");
            // dst_ptr 为 nullptr 表示当前 lane 无有效 slot 或目标 rank 不能 NVLink 直达。
            const auto dst_ptr = stored_dst_slot_idx >= 0 ?  // NVLink/LSA 可达目标地址，失败为 nullptr。
                gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), stored_dst_rank_idx) :
                nullptr;
            // NVLink 可达时 dst_ptr 是远端 symmetric pointer，直接 TMA store 完整 token。
            // 如果目标就是本 rank，get_sym_ptr 也会返回本地指针，路径保持一致。
            if (dst_ptr != nullptr)  // 可直接访问时走 TMA store。
                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            // 即使没有 dst_ptr 也 commit，保持 warp 内 TMA store 队列语义简单一致。
            ptx::tma_store_commit();
            // 等待所有 lane 发完直写路径。
            __syncwarp();

            // Issue RDMA put：非 NVLink 模式下，对不能直达的目标 rank 走 Gin put。
            if constexpr (not kIsScaleupNVLink) {
                // Wait the send buffer store to arrive
                // 等待本轮写到 send_buffer 的 TMA store 完成，保证 RDMA 源数据已经落地。
                ptx::tma_store_wait<1>();
                // 所有 lane 同步后再由有效 lane 发 RDMA。
                __syncwarp();

                // NOTES: we should skip the NVLink accessible ranks
                // dst_ptr!=nullptr 的 rank 已经走 NVLink 直写；这里只处理真正需要 RDMA 的目标。
                if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {  // 只有非直达目标需要 RDMA。
                    // 远端地址是目标 rank 的 recv_buffer[slot]，源地址是本地 send_buffer[token_idx]。
                    gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                    // 中文：tma_buffer 字节数覆盖 hidden 与可选 scale/topk metadata 的 token layout。
                                    send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
                }
                // 等待 RDMA 请求发起完成，再处理下一个 token。
                __syncwarp();
            }  // 中文：结束当前 dispatch warp 的 token 循环。
        }  // 中文：结束 payload dispatch 子团队。
    }  // 中文：结束 dispatch 主体，随后进入全局到达 barrier。

    // Barrier to ensure data arrival：确保所有 payload 到达后，copy epilogue 才能读取 buffer。
    // kFlushStores=true 会等待所有 TMA store/RDMA 相关写入发布，再通过 barrier 与其它 rank 对齐。
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,  // payload 完成后的团队同步。
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag1, true, true, false>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // Trigger the copy epilogue kernel：PDL 通知后续 epilogue kernel 可以开始整理输出。
    // Programmatic Dependent Launch 让 epilogue 可以提前排队，在这里收到完成信号后继续执行。
    cudaTriggerProgrammaticLaunchCompletion();  // 触发 copy epilogue 的 programmatic launch。

    // Clean atomic counters：非 deterministic 路径的 sender counter 在本 kernel 末尾清零。
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    // 只让 SM0 的前 kNumRanks 个线程清理，避免重复写。
    // reuse_slot_indices=true 时没有使用 atomic sender counter，因此无需清理。
    if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumRanks)  // 新分配路径清理 sender counter。
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace deep_ep::elastic
