#pragma once  // PP send/recv device 实现由 JIT include，防止重复定义。

#include <cooperative_groups.h>       // 使用 cooperative_groups::this_grid().sync 做跨 block 同步。
#include <deep_ep/common/comm.cuh>    // timeout_while、NCCLGin 等通信辅助。
#include <deep_ep/common/layout.cuh>  // WorkspaceLayout 提供 PP send/recv 计数槽地址。
#include <deep_ep/common/ptx.cuh>     // TMA、mbarrier、warp elect 等 PTX helper。


namespace deep_ep::elastic {  // elastic PP send/recv device kernel 命名空间。
// Elastic/V2 pipeline parallel send/recv device kernel 命名空间。

template <int kNumRanks>  // PP ring 中参与的 rank 数。
__device__ __forceinline__ std::pair<int, int> get_buffer_offset(
    const int& src_rank_idx, const int& dst_rank_idx) {  // 源 rank 和目标 rank。
    // PP 通信只在相邻 stage 间进行。对于 rank i，发送给 (i+1)%N 使用本地槽 0、
    // 反向方向使用槽 1；返回值分别是“对方视角下本 rank 的槽”和“本地视角下对方槽”。
    const auto next_rank_idx = (src_rank_idx + 1) % kNumRanks;  // pipeline ring 的 next rank。
    // next_rank_idx 是 ring 中 src 的下一个 rank，典型 pipeline forward 方向。
    // 非 next 方向被视为 prev 方向；调用方需保证 dst_rank_idx 只可能是相邻 rank。
    return dst_rank_idx == next_rank_idx ? std::make_pair(0, 1) : std::make_pair(1, 0);
    // 返回 pair(local_idx_in_dst, dst_idx_in_local)：
    // forward 方向使用槽 0/1，反向方向使用槽 1/0，双方计算必须完全对称。
}

template <int64_t kNumTimeoutCycles, typename timeout_print_t>  // 超时周期和超时打印回调类型。
__device__ __forceinline__ void check_signal(
    const handle::NCCLGin& gin,  // Gin handle，用于访问 signal table。
    const ncclGinSignal_t& signal_idx,  // 要轮询的 signal 编号。
    const int64_t& target,  // 期望 signal 至少达到的值。
    const timeout_print_t& timeout_print) {  // 超时时执行的打印回调。
    // check_signal 在设备端轮询 GIN signal，直到 signal >= target 或超时。
    const auto gdaki = static_cast<struct ncclGinGdakiGPUContext*>(gin.gin._ginHandle) + gin.gin.contextId;  // 当前 Gin context 的 GDAKI 状态。
    // 从 Gin handle 中取当前 context 对应的 GDAKI GPU context。
    const auto signal_ptr = reinterpret_cast<int64_t*>(  // signal table 中目标计数器地址。
        __ldg(reinterpret_cast<int64_t*>(&gdaki->signals_table.buffer))) + signal_idx;
    // signals_table.buffer 是 signal 数组基址，signal_idx 选择具体计数器。
    // signal 是单调递增计数器。send 等待“对端至少释放到某个序号”，recv 等待
    // “对端至少发送到某个序号”。超时打印 lambda 会在最后一次检查时执行。
    // 这里用 >= 而不是 ==，允许信号因为多次操作已经前进到更大的序号。
    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
        // timeout_while 会反复调用 lambda，最后一次检查时传 is_last_check=true。
        const auto signal = ptx::ld_acquire_sys<int64_t>(signal_ptr);
        // system-scope acquire load，确保看到对端 signal 写入及其之前的数据写入。
        if (signal >= target)
            // signal 是单调递增计数器，达到 target 即表示目标事件发生。
            return true;

        if (is_last_check)
            // 只有最后一次检查打印，避免超时期间刷屏。
            timeout_print();
        return false;
        // 尚未达到 target，继续轮询。
    });
}

template <int kNumSMs,  // cooperative kernel 使用的 block/SM 数。
          int kNumSmemBytes,  // 动态 shared memory 字节数。
          int kNumStages = 2,  // TMA pipeline stage 数，默认双缓冲。
          int kNumTMABytesPerStage = math::constexpr_align<int, false>(  // 每个 stage 可用数据字节数。
              (kNumSmemBytes - kNumStages * sizeof(ptx::mbarrier)) / kNumStages, ptx::kNumTMAAlignBytes),
          int kNumTMABlocksPerStage = kNumTMABytesPerStage / ptx::kNumTMAAlignBytes>  // 每 stage 容纳的 TMA 对齐块数。
__device__ __forceinline__ void tma_copy(
    void* src_ptr, void* dst_ptr,  // 源地址和目标地址。
    const int64_t& num_bytes, const int& sm_idx) {  // 拷贝字节数和当前 SM/block id。
    // tma_copy 使用多个 SM 并行把连续字节流从 src_ptr 拷到 dst_ptr。
    // PP send 用它从用户 tensor 拷到 staging buffer，PP recv 用它从 staging 拷回用户 tensor。
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];  // TMA staging 动态 shared memory。
    // 动态 shared memory 按 TMA 对齐，前半存 stage buffer，尾部存 mbarrier。
    const auto tma_buffers = smem;  // stage 数据区起点。
    // tma_buffers 是 kNumStages 个连续 stage 数据区。
    const auto mbarriers = reinterpret_cast<ptx::mbarrier*>(smem + kNumStages * kNumTMABytesPerStage);  // stage mbarrier 区起点。
    // mbarriers 紧跟所有 stage buffer 之后。
    EP_STATIC_ASSERT(kNumTMABytesPerStage > 0, "Invalid shared memory bytes");
    // 每个 stage 至少要有一个 TMA 对齐块。
    EP_STATIC_ASSERT(kNumStages >= 2, "Need at least 2 stages for pipelining");
    // 至少双缓冲才能让 load/store 重叠。

    // Init mbarriers
    // shared memory 被切成 kNumStages 个 TMA stage，每个 stage 后面配一个 mbarrier；
    // 双缓冲能让下一段 TMA load 和上一段 TMA store 重叠。
    ptx::arrival_phase phases[kNumStages];  // 每个 stage 的当前 phase。
    // 每个 stage 的 mbarrier phase，wait 后会翻转。
    #pragma unroll
    for (int s = 0; s < kNumStages; ++ s)  // 初始化所有 stage barrier。
        // 初始化每个 stage 的 barrier，expected arrivals 为 1。
        phases[s] = 0, ptx::mbarrier_init_with_fence(mbarriers + s, 1);

    // 每个 SM 负责一段连续 TMA block。num_bytes 必须按 16B 对齐，因为底层 TMA
    // 以 ptx::kNumTMAAlignBytes 为最小单位；调用方会按最大 tensor 字节做对齐。
    EP_DEVICE_ASSERT(num_bytes % ptx::kNumTMAAlignBytes == 0);
    // TMA 拷贝长度必须满足硬件对齐要求。
    const auto num_tma_blocks = num_bytes / ptx::kNumTMAAlignBytes;  // 总 TMA 块数。
    // 总 TMA block 数。
    const auto num_tma_blocks_per_sm = math::ceil_div<int64_t>(num_tma_blocks, kNumSMs);  // 每 SM 负责块数。
    // 每个 SM 负责的 TMA block 数，最后一个 SM 可能不足。
    const auto start_block_idx = sm_idx * num_tma_blocks_per_sm;  // 当前 SM 起始块。
    // 当前 SM 的起始 TMA block。
    const auto end_block_idx = std::min(start_block_idx + num_tma_blocks_per_sm, num_tma_blocks);  // 当前 SM 结束块。
    // 当前 SM 的结束 TMA block，半开区间。
    const auto num_iterations = math::ceil_div<int64_t>(end_block_idx - start_block_idx, kNumTMABlocksPerStage);  // pipeline 迭代次数。
    // 当前 SM 需要多少个 stage 轮次完成自己的片段。

    auto get_iter_info = [&](const int64_t& iter_idx) {  // 计算某轮的全局 offset 和字节数。
        // 计算第 iter_idx 轮在整个 tensor 中的字节 offset 和本轮字节数。
        const auto i = start_block_idx + iter_idx * kNumTMABlocksPerStage;  // 本轮首个 TMA block。
        // 本轮第一个 TMA block 编号。
        const auto offset = i * ptx::kNumTMAAlignBytes;  // 本轮字节偏移。
        // 转成字节偏移。
        const auto num_transaction_bytes =  // 本轮实际事务字节数。
            std::min<int>(kNumTMABlocksPerStage, end_block_idx - i) * ptx::kNumTMAAlignBytes;
        // 本轮实际拷贝字节数，最后一轮可能小于 stage 容量。
        return std::make_pair(offset, num_transaction_bytes);
        // 返回 (offset, bytes)。
    };

    for (int64_t iter_idx = 0; iter_idx < num_iterations; ++ iter_idx) {  // 主 TMA pipeline 循环。
        // 主 pipeline 循环，每轮消费一个 stage。
        const auto stage_idx = static_cast<int>(iter_idx % kNumStages);  // 当前使用的环形 stage。
        // stage 环形编号。
        const auto [store_offset, num_store_bytes] = get_iter_info(iter_idx);  // 当前 store 的 offset/bytes。
        // 当前要写回 dst 的 offset/bytes。

        // Fill pipeline: issue loads for the first kNumStages iterations
        if (iter_idx < kNumStages) {  // pipeline 填充阶段。
            // pipeline 填充阶段：前 kNumStages 轮先发起对应 load。
            ptx::tma_load_1d(
                tma_buffers + stage_idx * kNumTMABytesPerStage,
                // 中文：src_ptr/dst_ptr 按字节偏移推进，PP copy 不关心 tensor dtype。
                math::advance_ptr(src_ptr, store_offset),
                mbarriers + stage_idx, num_store_bytes);
            // 从 src_ptr+offset 异步加载到当前 shared stage。
            ptx::mbarrier_arrive_and_set_tx(mbarriers + stage_idx, num_store_bytes);
            // 告诉 mbarrier 本次 TMA 预期传输字节数。
        }

        // Wait for this stage's load, then store
        ptx::mbarrier_wait_and_flip_phase(mbarriers + stage_idx, phases[stage_idx]);
        // 等待当前 stage load 完成，并翻转 phase。
        ptx::tma_store_1d(
            // 中文：store_offset 与前面的 load offset 相同，保证字节块原位复制。
            math::advance_ptr(dst_ptr, store_offset),
            tma_buffers + stage_idx * kNumTMABytesPerStage,
            num_store_bytes);
        // 把 shared stage 异步 store 到 dst_ptr+offset。
        ptx::tma_store_commit();
        // 提交当前 TMA store。

        // Prefetch: wait until this stage's buffer is safe to reuse, then issue next load
        const auto next_iter_idx = iter_idx + kNumStages;  // 当前 stage 下一次对应的迭代。
        // 当前 stage 下一次被复用时对应的迭代编号。
        if (next_iter_idx < num_iterations) {  // 还有未来数据需要预取。
            // 如果还有后续数据，复用当前 stage 发起下一次 load。
            ptx::tma_store_wait<kNumStages - 1>();
            // 等待 store pipeline 至少空出当前 stage 可安全复用的深度。
            const auto [load_offset, num_load_bytes] = get_iter_info(next_iter_idx);  // 下一次 load 的 offset/bytes。
            // 下一次 load 的 offset/bytes。
            ptx::tma_load_1d(
                tma_buffers + stage_idx * kNumTMABytesPerStage,
                // 中文：预取下一段时复用同一 stage 的 shared buffer 和 mbarrier。
                math::advance_ptr(src_ptr, load_offset),
                mbarriers + stage_idx, num_load_bytes);
            // 发起下一段 global->shared load。
            ptx::mbarrier_arrive_and_set_tx(mbarriers + stage_idx, num_load_bytes);
            // 设置下一段 load 的 expected bytes。
        }
    }

    // Drain all outstanding stores
    ptx::tma_store_wait();
    // 等待所有 TMA store 完成，确保 dst_ptr 上数据可见。
}

template <int kNumSMs,  // send kernel block/SM 数。
          int kNumRanks,  // PP ring rank 数。
          int kNumSmemBytes,  // TMA copy 使用的 shared memory 字节数。
          int64_t kNumTimeoutCycles>  // signal 等待超时周期。
__global__ void __launch_bounds__(32, 1)
pp_send_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,  // device communicator/window。
             void* x, const int64_t num_x_bytes,  // 待发送 tensor 地址和真实字节数。
             void* buffer, void* workspace,  // PP payload buffer 和控制面 workspace。
             const int rank_idx, const int dst_rank_idx,  // 当前 rank 和目标 rank。
             const int64_t num_max_tensor_bytes,  // 每个 inflight slot 的容量。
             const int num_max_inflight_tensors) {  // 环形队列槽数。
    // pp_send_impl 把用户 tensor x 发送到 dst_rank_idx 的 PP 环形 buffer。
    const auto sm_idx = static_cast<int>(blockIdx.x);  // 当前 block/SM id。
    // 每个 block/SM 负责 tensor 的一段 TMA copy。
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);  // PP 控制面布局。
    // PP 只使用 scaleup 维 rank，scaleout=1，num_experts=0。
    const auto [local_idx_in_dst, dst_idx_in_local] = get_buffer_offset<kNumRanks>(rank_idx, dst_rank_idx);  // 双方方向槽编号。
    // local_idx_in_dst 是对端视角下本 rank 的方向槽，dst_idx_in_local 是本地视角下目标 rank 的方向槽。

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);  // CTA 级共享 Gin context。
    // PP send/recv 当前使用 Gin context 0。

    // buffer 布局分四组方向槽：前两组是对端写入本地的 recv 区，后两组是本地发送
    // 前的 staging 区。slot_idx = count % inflight 形成环形队列。
    // 方向槽 0/1 对应 forward/backward 两个邻居方向；+2 后的 2/3 是 send staging。
    const auto send_count_ptr = workspace_layout.get_pp_send_count_ptr(dst_idx_in_local);  // 本地方向槽 send counter。
    // 本地记录“发往该方向槽的第几次 send”。
    const auto send_count = __ldg(send_count_ptr);  // 当前 send 序号。
    // send_count 单调递增，用于选择环形 slot 和等待释放。
    const auto slot_idx = send_count % num_max_inflight_tensors;  // 当前环形 slot。
    // 环形槽编号，允许最多 num_max_inflight_tensors 个 tensor 在途。
    auto send_buffer_ptr = math::advance_ptr(  // 本地 send staging slot。
        buffer, ((dst_idx_in_local + 2) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);
    // 本地 staging 区：方向槽 +2 表示 send staging，不和 recv 区重叠。
    auto recv_buffer_ptr = math::advance_ptr(  // 远端 recv slot。
        buffer, ((local_idx_in_dst + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);
    // 远端接收区：对端会用 local_idx_in_dst +0 找到同一个 slot。

    // Wait buffer slot release and do TMA
    if (ptx::elect_one_sync()) {  // 每个 SM 只有一个 lane 执行等待和 TMA copy。
        // 每个 block 只有一个 elected lane 等待 signal 并启动 TMA copy。
        check_signal<kNumTimeoutCycles>(
            gin,
            static_cast<ncclGinSignal_t>(kNumRanks + dst_idx_in_local + 2),
            // release signal 编号：前 kNumRanks 是数据到达信号，后面 +2 方向槽是释放信号。
            send_count - num_max_inflight_tensors + 1,
            // 只有当对端至少释放到这个序号时，当前 slot 才安全复用。
            // 当 send_count < inflight 时 target <= 0，初始 signal=0 会立刻通过，表示还有空槽。
            // TODO: print more info, and control the SM who prints it
            // 中文：超时通常说明接收端没有及时 pp_recv，导致发送 ring 的 inflight 槽耗尽。
            []() { printf("DeepEP PP send timeout, recv buffer is full"); }
        );
        tma_copy<kNumSMs, kNumSmemBytes>(x, send_buffer_ptr, num_x_bytes, sm_idx);
        // 把用户 tensor 的当前 SM 分片拷贝到本地 staging buffer。
    }
    cooperative_groups::this_grid().sync();
    // 等待所有 SM 完成 staging copy，然后 sm0 才能发起一次完整 RDMA put。

    // 第 0 个 SM 提交一次 RDMA put 即可，所有 SM 已通过 cooperative grid sync
    // 完成本地 staging 拷贝。remote_action 会把接收端 signal 加 1，通知 recv 可读。
    if (sm_idx == 0 and ptx::elect_one_sync()) {  // 只由 SM0 发起一次 RDMA put。
        // 只有 sm0 的 elected lane 发 RDMA put，避免重复发送同一个 tensor。
        gin.put<ncclTeamTagWorld>(
            recv_buffer_ptr,
            // 远端目标 buffer 地址。
            send_buffer_ptr,
            // 本地 staging buffer 地址。
            num_x_bytes, dst_rank_idx,
            // 发送真实 tensor 字节数，可能小于槽位容量 num_max_tensor_bytes。
            0,
            // 使用 Gin context/qp 0。
            // TODO: is this signal highly optimized?
            ncclGin_SignalInc(static_cast<ncclGinSignal_t>(local_idx_in_dst + kNumRanks)));
            // RDMA 完成后递增远端“数据到达”signal，通知 recv 可读。
            // 数据到达 signal 使用对端视角的 local_idx_in_dst，保证 recv 用同一公式等待。
        *send_count_ptr += 1;
        // 本地 send_count 前进，下一次 send 使用下一个环形槽。
    }
}

template <int kNumSMs,  // recv kernel block/SM 数。
          int kNumRanks,  // PP ring rank 数。
          int kNumSmemBytes,  // TMA copy shared memory 字节数。
          int64_t kNumTimeoutCycles>  // signal 等待超时周期。
__global__ void __launch_bounds__(32, 1)
pp_recv_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,  // device communicator/window。
             void* x, int64_t num_x_bytes,  // 输出 tensor 地址和真实字节数。
             void* buffer, void* workspace,  // PP payload buffer 和 workspace。
             const int rank_idx, const int src_rank_idx,  // 当前 rank 和源 rank。
             const int64_t num_max_tensor_bytes,  // 每 slot 容量。
             const int num_max_inflight_tensors) {  // 环形队列槽数。
    // pp_recv_impl 从 src_rank_idx 对应的 PP 环形槽接收一个 tensor 到 x。
    const auto sm_idx = static_cast<int>(blockIdx.x);  // 当前 block/SM id。
    // 每个 block/SM 负责输出 tensor 的一段 TMA copy。
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);  // PP 控制面布局。
    // PP workspace 使用 rank 维计数，不使用 expert 维。
    const auto [src_idx_in_local, local_idx_in_src] = get_buffer_offset<kNumRanks>(src_rank_idx, rank_idx);  // 本地/源端方向槽。
    // src_idx_in_local 是本地视角下源 rank 方向槽，local_idx_in_src 是源 rank 视角下本 rank 方向槽。

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);  // CTA 共享 Gin context。
    // 使用 Gin context 0 轮询信号并发送释放 signal。

    // Buffer offsets
    const auto recv_count_ptr = workspace_layout.get_pp_recv_count_ptr(src_idx_in_local);  // 本地方向槽 recv counter。
    // 本地记录从该方向槽接收了多少个 tensor。
    const auto recv_count = __ldg(recv_count_ptr);  // 当前 recv 序号。
    // recv_count 用来决定当前要读取哪个环形 slot。
    const auto slot_idx = recv_count % num_max_inflight_tensors;  // 当前环形 slot。
    // 当前接收槽位。
    const auto recv_buffer_ptr = math::advance_ptr(  // 本地 recv slot 地址。
        buffer, ((src_idx_in_local + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);
    // 本地接收区地址，发送方 RDMA put 的目标就是这里。

    // recv 先等待对应 signal 到达，再从环形槽拷到用户输出 x。x 的字节数可以小于
    // num_max_tensor_bytes；后者只决定槽位容量和 offset。
    // slot_idx 和 send 侧相同，都由单调 count % inflight 计算，因此两端无需传 slot id。
    if (ptx::elect_one_sync()) {  // 每个 SM 一个 lane 等待 signal 并执行 TMA copy。
        // 每个 SM 只有一个 elected lane 等待数据到达并执行本 SM 的 TMA copy。
        check_signal<kNumTimeoutCycles>(
            gin,
            static_cast<ncclGinSignal_t>(src_idx_in_local + kNumRanks),
            // 数据到达 signal 编号：kNumRanks 偏移之后的方向槽 signal。
            recv_count + 1,
            // 等待发送方把该方向槽 signal 递增到下一次接收序号。
            // TODO: print more info, and control the SM who prints it
            // 中文：超时通常说明发送端没有 pp_send，或 send/recv 的方向 rank 不匹配。
            []() { printf("DeepEP PP recv timeout, recv buffer is empty\n"); }
        );
        tma_copy<kNumSMs, kNumSmemBytes>(recv_buffer_ptr, x, num_x_bytes, sm_idx);
        // 从环形 recv buffer 拷贝到用户输出 x。
    }
    cooperative_groups::this_grid().sync();
    // 等待所有 SM 完成从 recv buffer 到 x 的拷贝，再释放槽位给发送方。

    // 拷贝完成后给发送端回一个释放 signal，表示这个 inflight 槽可以再次写入；
    // 同时推进本地 recv_count，使下一次 recv 使用下一个环形槽。
    if (sm_idx == 0 and ptx::elect_one_sync()) {  // 只由 SM0 发送释放 signal。
        // 只有 sm0 elected lane 发送释放 signal 并推进本地计数。
        gin.signal<ncclTeamTagWorld>(
            src_rank_idx, ncclGin_SignalInc(static_cast<ncclGinSignal_t>(kNumRanks + local_idx_in_src + 2))
        );
        // 向源 rank 发送释放 signal：该 inflight slot 已消费，可再次写入。
        // release signal 使用源 rank 视角的 local_idx_in_src，和 send 侧等待的 signal_idx 对齐。
        *recv_count_ptr += 1;
        // 本地 recv_count 前进，下一次 recv 读取下一个环形槽。
    }
}

} // namespace deep_ep::elastic
// Elastic PP send/recv device kernel 命名空间结束。
