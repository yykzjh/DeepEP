#include "compiled.cuh"  // legacy 编译期常量、类型别名、低延迟 phase 标志和设备工具。
// compiled.cuh 汇总了编译期常量、CUDA/NVSHMEM 辅助宏和设备端工具函数，
// 本文件依赖其中的 LEGACY_LOW_LATENCY_* phase 标志、FP8 scale 工具等。
#include "ibgda_device.cuh"  // 设备端 IBGDA/NVSHMEM verbs 封装。
// ibgda_device.cuh 提供设备端 IBGDA/NVSHMEM verbs 封装，本文件直接在 kernel
// 内发起 RDMA put/amo，因此需要这些 device API。
#include "launch.cuh"  // host 侧 LAUNCH_KERNEL/SETUP_LAUNCH_CONFIG 封装。
// launch.cuh 封装 host 侧 kernel launch 配置，dispatch/combine wrapper 统一用它。

namespace deep_ep::legacy {  // legacy EP kernel 命名空间。
// legacy 命名空间对应 Python 侧 deep_ep.buffers.legacy.Buffer 暴露的原始 EP 路径。

namespace internode_ll {  // 跨节点 low-latency decode kernel 子命名空间。
// internode_ll 专门存放跨节点 low-latency decode 路径，区别于 legacy::internode 的
// 普通 dispatch/combine：这里更强调单 token/小 batch 的低控制开销。

// low-latency internode kernel 为 decode/serving 场景设计：直接按 expert 维打包
// RDMA 消息，减少 legacy internode 两级 prefix 的控制开销。dispatch 和 combine
// 都支持 phase 拆分，调用方可只运行 SEND 或 RECV 阶段以和计算重叠。

template <bool use_warp_sync = false>  // use_warp_sync=true 时一 warp 只读一次 mask 再广播。
__forceinline__ __device__ bool is_rank_masked(int* mask_buffer_ptr, int rank) {  // 查询 rank 是否已被容错 mask。
    // rank mask 用于容忍超时 peer。mask_buffer_ptr 为空时表示不允许降级，超时直接 trap；
    // 非空时把目标 rank 标记为 masked，后续 RDMA send/recv 会跳过它。
    if (mask_buffer_ptr == nullptr) {  // 未提供 mask buffer 时表示不启用 mask 机制。
        // 没有 mask buffer 表示调用方不启用 shrink/容错；所有 rank 都被视为可通信。
        return false;  // 所有 rank 都视为未 mask。
    }
    if constexpr (use_warp_sync) {  // warp 级调用路径。
        // warp 级调用时只让 lane0 做 acquire load，再广播到整个 warp，避免 32 个 lane
        // 重复访问同一个全局地址；例如发送 warp 判断 dst_rank 是否已经超时时会走这里。
        return __shfl_sync(0xffffffff, ld_acquire_global(mask_buffer_ptr + rank), 0) != 0;  // lane0 读后广播。
    } else {
        // 普通线程调用直接 acquire load，保证看到其他线程/SM 写入的最新 mask 状态。
        return ld_acquire_global(mask_buffer_ptr + rank) != 0;  // 普通线程直接 acquire 读取。
    }
}

template <int kNumThreads>  // kNumThreads 至少覆盖 num_ranks，便于一线程通知一个 rank。
__forceinline__ __device__ void barrier(int thread_id, int rank, int num_ranks, int* mask_buffer_ptr, int* sync_buffer_ptr) {  // 支持 mask 的低延迟 barrier。
    // 这个 barrier 用于 low-latency buffer 清理前后，它支持 mask 掉超时 rank，
    // 因此不能直接等价替换成 nvshmemx_barrier_all_block。
    EP_DEVICE_ASSERT(kNumThreads >= num_ranks);  // 至少要有足够线程覆盖所有 rank 通知。

    // barrier 前先 quiet 所有 QP，保证之前发出的 IBGDA WR 都已完成，否则清理或复用
    // low-latency buffer 可能覆盖未被远端消费的数据。
    auto qps_per_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;  // 每 peer 的 RC QP 总数。
    // 每个远端 rank 可能有 num_rc_per_pe * 本地 device 数量 条 RC QP；
    // quiet 时需要遍历所有 QP，避免某条队列里仍有未完成 WR。

    for (int i = thread_id; i < qps_per_rank * (num_ranks - 1); i += kNumThreads) {  // 全 CTA 并行 quiet 所有远端 QP。
        // i 的高位选择第几个远端 rank，低位选择该 rank 的第几个 QP。
        auto dst_rank = (rank + 1 + i / qps_per_rank) % num_ranks;  // 跳过本 rank 后的目标 rank。
        // 从 rank+1 开始轮转遍历，跳过自己；取模后覆盖所有其他 rank。
        auto qp_id = i % qps_per_rank;  // 目标 rank 内的 QP 编号。
        // 当前线程负责 quiet 的 QP 编号。
        nvshmemi_ibgda_quiet(dst_rank, qp_id);  // 等该 QP 上已提交操作完成。
        // 等待该 QP 上此前提交的 IBGDA 操作在远端可见。
    }

    // sync_buffer_ptr[rank] 是本 rank 的轮次计数。每次 barrier 递减一次，并把新值
    // 写到其他 rank 的 sync_buffer_ptr[rank] 位置。
    if (thread_id == 0)  // 只由一个线程更新本 rank barrier 轮次。
        // 轮次计数采用递减设计，避免和初始零值/清零后的状态混淆。
        atomicAdd(sync_buffer_ptr + rank, -1);  // 轮次递减，形成新的 cnt。
    __syncthreads();  // 等待本地 cnt 更新完成。
    // 确保所有线程看到 thread0 写入后的本地 cnt。

    int cnt = sync_buffer_ptr[rank];  // 本轮 barrier 目标计数值。
    // Update remote counter and wait for local counter to be updated
    if (thread_id < num_ranks && thread_id != rank) {  // 一线程负责一个非本地 rank。
        // 前 num_ranks 个线程一人负责通知一个 rank；本 rank 不需要通知自己。
        const auto dst_rank = thread_id;  // 当前线程通知的目标 rank。
        // thread_id 直接映射目标 rank，便于并行发通知。
        const auto dst_ptr = reinterpret_cast<uint64_t>(sync_buffer_ptr + rank);  // 远端同 offset 的 sync 槽地址。
        // 远端也用相同虚拟地址布局，dst_ptr 指向“远端 buffer 中记录本 rank 轮次”的槽位。
        const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);  // 可 P2P 时返回远端映射地址。
        // 若两张 GPU 可 P2P 访问，则返回远端地址在本进程可写的映射；否则返回 0 走 RDMA。

        if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {  // 已 mask peer 不参与本轮通知。
            // 已经被 mask 的 rank 不再参与 barrier，避免一个故障 peer 拖住后续请求。
            if (dst_p2p_ptr == 0) {  // 无 P2P 映射，走 IBGDA 写远端。
                // 跨节点或无 P2P 映射时，通过 IBGDA remote put 写入远端计数。
                nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), cnt, dst_rank, 0);  // remote put cnt。
            } else {
                // 同节点 P2P 路径直接 system-scope release store，远端 acquire load 可见。
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), cnt);  // P2P release store cnt。
            }

            auto start_time = clock64();  // 等待起始时间。
            // clock64 统计等待轮次，用于超时判定；单位是 GPU cycle。
            uint64_t wait_recv_cost = 0;  // 等待耗时，单位 GPU cycle。
            // wait_recv_cost 同时作为 while 条件中更新的等待成本。
            while (ld_acquire_sys_global(sync_buffer_ptr + dst_rank) != cnt            // remote is not ready
                   && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES  // not timeout
            )
                // 等待远端把它自己的 cnt 写回本地 sync_buffer_ptr[dst_rank]。
                ;
            // Mask rank if timeout
            if (wait_recv_cost > LEGACY_NUM_TIMEOUT_CYCLES) {  // 超过阈值则进入 mask/trap 分支。
                // 超时说明对应 peer 没有完成本轮 barrier，后续通信将跳过该 rank。
                printf("Warning: DeepEP timeout for barrier, rank %d, dst_rank %d\n", rank, dst_rank);
                if (mask_buffer_ptr == nullptr)
                    // 未启用容错时无法继续安全运行，直接 trap 暴露错误。
                    trap();  // 无 mask buffer 时只能终止 kernel。
                atomicExch(mask_buffer_ptr + dst_rank, 1);  // 标记目标 rank 为故障/跳过。
                // 标记为 masked，其他调用 is_rank_masked 的位置会感知该状态。
            }
        }
    }
    __syncthreads();  // 等待所有线程完成 rank 通知与超时处理。
    // 保证所有线程完成通知/等待后再进入清理或后续 kernel 逻辑。
}

template <int kNumThreads>  // 清理 kernel 固定一个 CTA，模板化线程数方便 launch_bounds。
__launch_bounds__(kNumThreads, 1) __global__ void clean_low_latency_buffer(int* clean_0,  // 第一段待清理 int buffer。
                                                                           int num_clean_int_0,  // clean_0 的 int 元素数。
                                                                           int* clean_1,  // 第二段待清理 int buffer。
                                                                           int num_clean_int_1,  // clean_1 的 int 元素数。
                                                                           int rank,  // 当前 rank。
                                                                           int num_ranks,  // 全局 rank 数。
                                                                           int* mask_buffer_ptr,  // 可选 mask buffer。
                                                                           int* sync_buffer_ptr) {  // 可选自定义 barrier sync buffer。
    auto thread_id = static_cast<int>(threadIdx.x);  // 当前 CTA 内线程编号。
    // clean kernel 只有一个 block，thread_id 覆盖 [0, kNumThreads) 的清理分片。

    // Barrier before cleaning (in case of unfinished chunked EP)
    if (sync_buffer_ptr == nullptr)  // 无自定义 barrier 时使用 NVSHMEM barrier。
        // 没有自定义 sync buffer 时退回 NVSHMEM 全局 block barrier。
        nvshmemx_barrier_all_block();  // block 级 NVSHMEM 全局 barrier。
    else  // 有 sync buffer 时使用支持 mask 的自定义 barrier。
        // 有 sync buffer 时使用上面的 shrink-aware barrier，能够跳过超时 rank。
        barrier<kNumThreads>(thread_id, rank, num_ranks, mask_buffer_ptr, sync_buffer_ptr);  // shrink-aware barrier。

    // Clean
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)  // 并行清零第一段。
        // clean_0 通常对应当前 low-latency buffer 的计数/标志区，按 int 粒度清零。
        clean_0[i] = 0;  // 写 0 清理状态。
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)  // 并行清零第二段。
        // clean_1 通常对应下一轮 buffer 或辅助计数区，调用方传入具体范围。
        clean_1[i] = 0;  // 写 0 清理状态。

    // Barrier after cleaning (make sure the low-latency mode works fine)
    if (sync_buffer_ptr == nullptr)  // 清理后再全局同步。
        // 清理后再次 barrier，确保所有 rank 在进入下轮 dispatch/combine 前看到零状态。
        nvshmemx_barrier_all_block();  // 确保所有 rank 清理完成。
    else  // 使用自定义 barrier 同步清理完成。
        // 自定义 barrier 同样会 quiet QP，避免清理后的 buffer 被旧 WR 覆盖。
        barrier<kNumThreads>(thread_id, rank, num_ranks, mask_buffer_ptr, sync_buffer_ptr);  // 支持跳过 masked rank。
}

void clean_low_latency_buffer(int* clean_0,  // 第一段需要清零的 low-latency 控制区。
                              int num_clean_int_0,  // clean_0 的 int 元素数。
                              int* clean_1,  // 第二段需要清零的控制区，可为空或指向下一轮辅助区。
                              int num_clean_int_1,  // clean_1 的 int 元素数。
                              int rank,  // 当前 rank id。
                              int num_ranks,  // 全局 rank 数。
                              int* mask_buffer_ptr,  // 可选故障 mask buffer，非空时 barrier 可跳过超时 rank。
                              int* sync_buffer_ptr,  // 可选自定义 sync buffer，非空时使用 shrink-aware barrier。
                              cudaStream_t stream) {  // 清理 kernel 所在 CUDA stream。
    constexpr int kNumThreads = 256;
    // 单 block 256 线程足够覆盖计数区清零，同时满足 barrier 中 kNumThreads >= num_ranks。

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    // 清理 kernel 固定一个 CTA，跨 rank 同步在 kernel 内部完成。

    LAUNCH_KERNEL(&cfg,
                  // Host wrapper 只负责透传两个清理区间和可选 shrink-aware barrier 参数。
                  clean_low_latency_buffer<kNumThreads>,
                  clean_0,  // 第一段待清理控制区。
                  num_clean_int_0,  // 第一段 int 元素数。
                  clean_1,  // 第二段待清理控制区。
                  num_clean_int_1,  // 第二段 int 元素数。
                  rank,  // 当前 rank。
                  num_ranks,  // 全局 rank 数。
                  mask_buffer_ptr,  // 可选故障 mask。
                  sync_buffer_ptr);  // 可选自定义 barrier sync buffer。
}

template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch(void* packed_recv_x,  // 输出 packed expert token，逻辑形状 [local_expert, rank, Tmax, hidden]。
                                                    void* packed_recv_x_scales,  // 输出 FP8 scale，UE8M0 时为 uint8 packed，否则 float scale；BF16 路径可为空。
                                                    int* packed_recv_src_info,  // 输出源信息，记录每个 packed token 来自哪个 rank/token/topk lane。
                                                    int64_t* packed_recv_layout_range,  // 输出 layout range，每个 expert/rank 一项 packed(offset,count)。
                                                    int* packed_recv_count,  // 输出每个 local expert/rank 实际收到的 token 数。
                                                    int* mask_buffer_ptr,  // 可选故障 rank mask，barrier 超时时标记并跳过 peer。
                                                    int* cumulative_local_expert_recv_stats,  // 输出 local expert 累计接收统计，Python 可用于调试/性能日志。
                                                    int64_t* dispatch_wait_recv_cost_stats,  // 输出等待 RDMA 接收耗时统计，单位为 GPU cycle。
                                                    void* rdma_recv_x,  // 本 rank 暴露给其它 rank 写入的 RDMA 接收 staging buffer。
                                                    int* rdma_recv_count,  // RDMA 接收计数，按 expert/rank 或 phase 组织。
                                                    void* rdma_x,  // 本 rank 写给远端的 RDMA send staging buffer。
                                                    const void* x,  // 输入 token hidden，形状 [num_tokens, hidden]。
                                                    const topk_idx_t* topk_idx,  // 输入 top-k expert id，形状 [num_tokens, num_topk]。
                                                    int* atomic_counter_per_expert,  // workspace: 每个 expert 的 slot 分配原子计数。
                                                    int* atomic_finish_counter_per_expert,  // workspace: 每个 expert 的发送完成计数。
                                                    int* next_clean,  // 下一轮需要清零的控制区起点。
                                                    int num_next_clean_int,  // next_clean 中需要清零的 int 元素数。
                                                    int num_tokens,  // 当前 rank 输入 token 数。
                                                    int num_max_dispatch_tokens_per_rank,  // 每个 rank/expert 预留容量 Tmax。
                                                    int num_topk,  // 每个 token 路由到的 expert 数。
                                                    int num_experts,  // 全局 expert 总数，需能被 num_ranks 整除。
                                                    int rank,  // 当前全局 rank。
                                                    int num_ranks,  // 全局 rank 数。
                                                    int num_warp_groups,  // 每个 SM 内 expert warp group 数。
                                                    int num_warps_per_group,  // 每个 expert group 使用的 warp 数。
                                                    bool round_scale,  // FP8 scale 是否取整为 2 的幂。
                                                    int phases) {  // low-latency 双缓冲 phase 数。
    // dispatch 输出的是按 local expert 分组的 packed_recv_x：
    // [num_local_experts, num_ranks, num_max_dispatch_tokens_per_rank, hidden]。
    // packed_recv_src_info 保存源 token/rank 信息，packed_recv_layout_range 保存每个
    // expert/rank 的 (offset, count)，combine 根据它反向发送结果。
    const auto sm_id = static_cast<int>(blockIdx.x);
    // sm_id 是当前 CTA/SM 的编号；本 kernel 通常按 num_device_sms 个 block 启动。
    const auto thread_id = static_cast<int>(threadIdx.x);
    // thread_id 是 block 内线程编号，dispatch 使用最多 1024 线程即 32 个 warp。
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    // warp_id 定位当前 warp，lane_id 定位 warp 内 0..31 的 lane。
    const auto num_sms = static_cast<int>(gridDim.x);
    // num_sms 是 grid 中 block 数，用于按 token 或 expert 做 grid-stride 分片。
    const auto num_warps = num_warp_groups * num_warps_per_group;
    // 有效 warp 数 = expert 组数 * 每组 warp 数；可能小于 32，剩余 warp 不参与核心逻辑。
    const auto num_local_experts = num_experts / num_ranks;
    // 每个 rank 持有 num_local_experts 个 expert，要求 expert 在 rank 间均匀切分。
    const auto warp_group_id = warp_id / num_warps_per_group;
    // warp_group_id 表示当前 warp group 负责哪个 expert 分片。
    const auto sub_warp_id = warp_id % num_warps_per_group;
    // sub_warp_id 是 group 内 warp 编号，receive 阶段用不同 sub_warp 重叠等待和拷贝。
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
    // 当前 warp group 负责的全局 expert id；例如 sm_id=2、group=3 时负责 2*num_warp_groups+3。

    // FP8 路径可选择把 scale 存成 UE8M0：scale_t 是单个 scale 元素类型，
    // packed_t 是一次向 RDMA 包里写入的打包类型。
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    // scale_t 是 packed_recv_x_scales 中单个 scale 的存储类型：UE8M0 为 1B，否则 float 为 4B。
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    // packed_t 描述 scale 的打包粒度；UE8M0 用 uint32_t 一次容纳 4 个 uint8 scale。
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");
    // 确保一个 pack 内能整除多个 scale 元素，后面计算 pack_idx/elem_idx 依赖这个性质。

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    // FP8 scale 每 128 个 hidden channel 一个，和 Python/Triton 侧量化布局保持一致。
    const int num_scales = kHidden / kNumPerChannels;
    // 每个 token 需要的 scale 数；例如 hidden=7168 时 num_scales=56。
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    // payload 中 hidden 向量字节数：FP8 为 hidden*1B，BF16 为 hidden*2B。
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);
    // packed_recv_x 以 int4 向量搬运，所以这里记录每个 token 的 int4 数量。

    // RDMA message 包格式：int4 头部（源 token index + 3 个预留 int）、
    // hidden payload、可选 FP8 scales。num_bytes_per_msg 向 int4 对齐，便于 IBGDA 写。
    using vec_t = std::conditional_t<kUseFP8, int2, int4>;
    // FP8 每次从 BF16 int4 转成 int2 写出；BF16 路径保持 int4 原样拷贝。
    const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
    // 单条 RDMA message = 16B header + hidden payload + FP8 scale 数组；BF16 没有 scale。
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    // P2P 快速路径按 int4 向量复制整条 message。
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);
    // message 必须 16B 对齐，否则 int4 copy 和 IBGDA payload 长度都会不安全。

    // Expert counts
    constexpr int kNumMaxWarpGroups = 32;
    // 一个 block 最多 32 个 warp group，对应 shared 数组固定上限。
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];
    // shared_num_tokens_sent_per_expert[group] 保存当前 SM 对该 expert 统计出的发送 token 数。

    // Sending phase
    if ((phases & LEGACY_LOW_LATENCY_SEND_PHASE) == 0)
        // 调用方可只运行 RECV 阶段；例如拆成两个 kernel 和 expert 计算重叠时会跳过发送。
        goto LOW_LATENCY_DISPATCH_RECV;

    // 发送阶段 warp 分两类：
    // 1. 前 num_warps-1 个 warp 做 BF16->FP8(可选) 和 RDMA payload 写入；
    // 2. 最后一个 warp 扫 topk_idx，维护每个 expert 的计数和完成标记。
    if (warp_id < num_warps - 1) {
        // 数据发送 warp：每个 warp 可以处理一个 top-k 目标 expert 的 RDMA 写入。
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        // 每次从 BF16 x 读取一个 int4，等价 8 个 nv_bfloat16 元素。
        EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
        // 一个 warp 按 lane 分摊 hidden，要求 hidden 能被 32*8 整除。
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        // 每个 warp 读一轮正好覆盖 256 个 BF16 元素，包含 2 个 128-channel scale group。
        const auto num_threads = (num_warps - 1) * 32;
        // 所有数据发送 warp 的总线程数；最后一个计数 warp 不参与 payload cast。
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;
        // 输入 x 的每个 token 有多少个 int4，仍按 BF16 原始布局计算。

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            // 每个 SM 处理 token_idx = sm_id, sm_id+num_sms...，避免不同 SM 重复处理同 token。
            const auto x_int4 = static_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            // 指向输入 x[token_idx, :] 的 int4 起点；x 逻辑形状为 [num_tokens, hidden]。
            const auto rdma_x_src_idx = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            // rdma_x 是本地 staging buffer，message 开头 4 个 int 中第一个存源 token_idx。
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            // rdma_x_vec 跳过 16B header 后指向 hidden payload 起点。
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);
            // FP8 scale 紧跟 hidden payload；BF16 路径不会使用这个指针。

            // Overlap top-k index read and source token index writes
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            // 每个发送 warp 对应一个 top-k 位置；超过 num_topk 的 warp 不发数据。
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;
            // 只用 block 内 thread0 写 header 的源 token index，后续接收方写入 packed_recv_src_info。

            // FP8 cast
            EP_STATIC_ASSERT(hidden_bf16_int4 % 32 == 0, "Must use the full warp to reduce");
            // FP8 scale 需要 warp reduce，因此每个 warp 的 lane 都必须有均匀工作。
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                // 所有数据发送线程按 int4 粒度覆盖一个 token 的 hidden 向量。
                // Read
                auto int4_value = __ldg(x_int4 + i);
                // __ldg 走只读缓存读取 BF16 输入，int4_value 内含 8 个 BF16。

                if constexpr (kUseFP8) {
                    // FP8 路径：先计算每 128 channel 的 amax，再写 E4M3 payload 和 float scale。
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                    // 将 int4 的 16B 临时解释为 8 个 BF16，不改变底层字节。
                    float fp32_values[kNumElemsPerRead];
                    // fp32_values 暂存当前线程读到的 8 个元素，供 scale 后转换 FP8。
                    float amax = kFP8Margin, scale, scale_inv;
                    // amax 初值带 margin，scale/scale_inv 由 calculate_fp8_scales 输出。
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++j) {
                        // 遍历 int4 中的 8 个 BF16，求本线程局部最大绝对值。
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        // BF16 转 FP32，后面乘 scale 再量化到 FP8。
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                        // 更新当前 8 个元素的局部 amax。
                    }

                    // Reduce amax and scale
                    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                    // 一个 warp 一轮覆盖 256 channel，因此 lane 0/16 分别对应两个 128-channel scale。
                    amax = warp_reduce_max<16>(amax);
                    // 以 16 lane 为一组做 max reduce，得到每 128 channel 的 amax。
                    calculate_fp8_scales(amax, scale, scale_inv, round_scale);
                    // 根据 amax 生成正向 scale 和反向 scale_inv；round_scale 控制是否舍入 scale。
                    if (lane_id == 0 or lane_id == 16)
                        // 每个 16-lane 子组只需一个 lane 写 scale_inv。
                        rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                    // Cast into send buffer
                    vec_t int2_value;
                    // FP8 payload 8 个元素只需要 8B，因此用 int2 承载。
                    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                    // 每次写两个 FP8，4 次填满 int2。
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; j += 2) {
                        // 两个 FP32 组成 float2，调用 CUDA intrinsic 一次转换成 fp8x2。
                        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                        // 乘以 scale 后映射到 E4M3 可表示范围。
                        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                        // 饱和转换，避免溢出产生非有限值。
                    }
                    rdma_x_vec[i] = int2_value;
                    // 将当前 8 个 channel 的 FP8 结果写入 staging buffer。
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                    // BF16 路径不做量化，直接把输入 int4 payload 写到 RDMA staging buffer。
                }
            }
            asm volatile("bar.sync 1, %0;" ::"r"(num_threads));
            // 数据发送线程局部 barrier，保证 header/payload/scale 都写完后再发 RDMA。

            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                // top-k 中无效位置为 -1；有效 expert 才需要发送。
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                // lane0 为目标 expert 分配一个 slot；slot 是该 expert 从本 rank 收到的第几个 token。
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                // slot_idx 广播给整个 warp，所有 lane 用同一个目的地址。
                const auto dst_rank = dst_expert_idx / num_local_experts;
                // 全局 expert id 映射到拥有该 expert 的目标 rank。
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                // 目标 rank 内的 local expert 编号。
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                // RDMA 源地址指向本地 staging message 开头。
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                    dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                    rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg + slot_idx * num_bytes_per_msg;
                // 远端 rdma_recv_x 逻辑维度：
                // [num_local_experts, num_ranks, num_max_dispatch_tokens_per_rank, num_bytes_per_msg]。
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                // 同节点可 P2P 时直接拷贝；跨节点返回 0 走 IBGDA put。
                if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank)) {
                    // 若目标 rank 已经超时被 mask，则跳过 payload 发送和完成计数。
                    if (dst_p2p_ptr == 0) {
                        // 跨节点路径：整个 warp 协作发一条 non-blocking put。
                        nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                    } else {
                        // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                        // 中文：7K hidden 在 int4 向量和 8 次展开下只需两轮 load/store，P2P 路径很短。
                        const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                        // P2P 源指针按 int4 读取完整 message。
                        const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                        // P2P 目标指针按 int4 写入远端映射地址。
                        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                        // 展开拷贝降低循环开销；例如 7K hidden 的 message 每 lane 只需少量迭代。
                    }
                }

                // Increase counter after finishing
                __syncwarp();
                // 确保 warp 内 payload 写入/put 提交都完成后再更新 finish counter。
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
                // finish counter 记录本地已经为该 expert 完成了多少 token 的发送提交。
            }
        }
    } else if (warp_id == num_warps - 1) {
        // 最后一个 warp 不搬运 payload，负责统计每个 expert 应收到多少 token 并发计数通知。
        EP_DEVICE_ASSERT(num_sms > 1);
        // 至少需要一个以上 SM，否则 sm0 既做全局清理又做所有 expert 统计会破坏并行假设。
        if (sm_id == 0) {
            // sm0 的最后一个 warp 额外承担全局初始化/通知职责。
            // The first SM is also responsible for checking QPs
            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);
            // 每个 local expert 至少需要一条 RC QP，后续 put 使用 dst_expert_local_idx 选 QP。

            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += 32)
                // next_clean 指向下一轮需要归零的计数区，提前清理以减少独立清理 kernel 开销。
                next_clean[i] = 0;

            // Notify before executing `int_p`
            __syncwarp();
            // 确保 next_clean 清理完成后再写 finish counter 的基准 tag。
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                // 给所有 expert 的 finish counter 预加 LEGACY_FINISHED_SUM_TAG。
                atomic_add_release_global(atomic_finish_counter_per_expert + i, LEGACY_FINISHED_SUM_TAG);
        }

        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kNumMaxWarpGroups] = {0};
        // expert_count[group] 是当前 lane 对本 SM 负责 expert 的局部 token 计数。
        const auto expert_begin_idx = sm_id * num_warp_groups;
        // 当前 SM 负责统计的第一个全局 expert。
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);
        // 当前 SM 负责统计的末尾 expert，最后一个 SM 可能不足 num_warp_groups 个。

        // Per lane count
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            // warp 内 32 个 lane 分摊扫描 topk_idx，逻辑形状为 [num_tokens, num_topk]。
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            // idx 是某个 token 的目标全局 expert id。
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                // 只统计当前 SM 负责的 expert，其他 expert 由对应 SM 的最后一个 warp 统计。
                expert_count[idx - expert_begin_idx]++;
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
            // 对当前 SM 负责的每个 expert 做一次 warp 内计数求和。
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            // sum 是所有 lane 对 expert i 扫描得到的 token 总数。
            if (lane_id == 0) {
                // lane0 写 shared 和 finish counter，避免重复写。
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                // shared 数组供后面的 count send 阶段读取。
                atomic_add_release_global(atomic_finish_counter_per_expert + i, LEGACY_FINISHED_SUM_TAG - sum);
                // finish counter 设计：
                // sm0 预加 TAG，本 warp 再加 TAG-sum，payload warp 每发一个加 1；
                // 中文：用这种编码避免额外数组保存每个 expert 的 expected count。
                // 当 payload 全完成时值恰好等于 TAG*2。
            }
        }
    }
    // 中文：同步点把 shared 统计、finish counter 初始化和后续 count send 阶段串起来。
    __syncthreads();
    // 确保 shared_num_tokens_sent_per_expert 和 finish counter 更新对 count send 可见。

    // Issue count sends
    // 中文：payload 发送完成后才发送 count，接收侧以 count 非零作为该 expert/source rank ready 信号。
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        // 每个 expert 只用其负责 warp group 的 sub_warp0 lane0 发送一次计数。
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        // 该 expert 所在的目标 rank。
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        // 目标 rank 内的 local expert 编号。
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];
        // 当前 rank 将发送给该 expert 的 token 数，接收方据此知道何时停止等待。

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != LEGACY_FINISHED_SUM_TAG * 2)
            // 等待 payload warp 已经为该 expert 提交了所有 token 发送。
            ;
        auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
        // 远端 rdma_recv_count 逻辑维度为 [num_local_experts, num_ranks]，
        // 写入位置表示“来自当前 rank 的 token 数”。
        auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
        // 同节点可走 P2P store，跨节点走 IBGDA AMO。
        if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
            // 已 mask 的目标 rank 不再发送 count，接收方也会跳过等待。
            if (dst_p2p_ptr == 0) {
                // 跨节点路径用 AMO add 写负数，和初始 0 区分“尚未到达”。
                nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
            } else {
                // P2P 路径直接 release store；值为 -(count+1)，接收方再取反还原。
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), -num_tokens_sent - 1);
            }
        }

        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        // slot 分配计数清零，供下一次 dispatch 重用 workspace。
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;
        // 完成计数清零，避免下一轮等待条件误判。

        // Clean `packed_recv_count`
        if (dst_rank == 0)
            // 每个 local expert 的 packed_recv_count 只需清一次；dst_rank==0 的 responsible expert
            // 刚好覆盖 local_expert_idx 0..num_local_experts-1。
            packed_recv_count[dst_expert_local_idx] = 0;
    }
    __syncwarp();
    // 这里保证当前 warp 内 count send/清理完成，之后可能跳入接收阶段。

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
    // goto 标签用于支持只运行 RECV phase；SEND phase 关闭时会直接跳到这里。
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0)
        // 调用方只需要 SEND 时可在发送计数后返回，让另一次 launch 专门做接收。
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LEGACY_LOW_LATENCY_SEND_PHASE)
        // 同一个 kernel 同时做 SEND+RECV 时，必须 grid sync 后才能安全读取前面清零的 packed_recv_count。
        cg::this_grid().sync();

    // Receiving and packing
    if (responsible_expert_idx < num_experts) {
        // 一个 warp group 负责一个全局 expert 的某个源 rank 接收槽位。
        const auto src_rank = responsible_expert_idx / num_local_experts;
        // responsible_expert_idx 在接收阶段被解释为“源 rank + local expert”的组合索引。
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        // 当前 rank 本地 expert id，范围 [0, num_local_experts)。
        const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        // rdma_recv_x 的逻辑维度：
        // [num_local_experts, num_ranks, num_max_dispatch_tokens_per_rank, message_bytes]。
        const auto recv_x_int4 =
            static_cast<int4*>(packed_recv_x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        // packed_recv_x 的逻辑维度：
        // [num_local_experts, num_ranks*num_max_dispatch_tokens_per_rank, hidden_int4]。
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        // packed_recv_src_info 与 packed_recv_x 的 token 维对齐，记录源 token_idx。
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        // recv_range[src_rank] 打包保存 (num_recv_tokens, recv_token_begin_idx)，combine 依赖它回传。
        const auto num_aligned_scales = align_up<int>(num_scales, sizeof(float) / sizeof(scale_t));
        // scale 存储按 packed_t 对齐；UE8M0 需要把 scale 数向 4 的倍数补齐。
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_aligned_scales;
        // packed_recv_x_scales 逻辑近似为：
        // [num_local_experts, packed_scale, num_ranks*num_max_dispatch_tokens_per_rank, elems_per_pack]。

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];
        // 每个 warp group 用一格 shared 在 sub_warp1 和其他 sub_warp 间传递接收 token 数和写入起点。

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens = 0, recv_token_begin_idx;
        // num_recv_tokens 接收 count 还原后的 token 数；recv_token_begin_idx 是 packed buffer 中起点。
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
        // 至少需要两个 warp/sub-warp：一个等 count，一个或多个搬数据；group 数也需给 bar.sync id 留空间。
        if (sub_warp_id == 1 and lane_id == 0) {
            // sub_warp1 的 lane0 专门等待来自 src_rank 的 count，减少和数据搬运 warp 的冲突。
            auto start_time = clock64();
            // 记录等待开始时间，用于统计和超时 mask。
            uint64_t wait_recv_cost = 0;
            // 等待 rdma_recv_count 非零花费的 cycle 数。
            if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
                // 已 mask 的源 rank 不等待，直接当作没有 token。
                while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) ==
                           0                                                               // data not arrived
                       // 中文：等待条件同时检查“计数未到”和“尚未超过超时周期”。
                       && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES  // not timeout
                )
                    // 中文：条件里顺带更新 wait_recv_cost，避免循环体额外读 clock。
                    // rdma_recv_count 初始为 0；发送方写 -(count+1)，所以非零表示计数已到。
                    ;
            }  // 中文：结束对当前 src_rank 的 count 等待。
            // Do not receive tokens if rank timeout or masked
            if (num_recv_tokens == 0)
                // 超时或已 mask 时保持 0，会被转成 -1 后得到 0 个 token。
                num_recv_tokens = -1;
            // Mask rank if timeout
            // 中文：启用 shrink/mask 后，后续轮次会跳过该 src_rank，避免反复卡死。
            if (wait_recv_cost > LEGACY_NUM_TIMEOUT_CYCLES) {
                // 等不到 count 表示源 rank 可能卡住，后续通信跳过该 rank。
                printf("Warning: DeepEP timeout for dispatch receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank,
                       // 中文：local_expert_idx/src_rank 可以定位是哪一个 expert 等哪个远端 rank 的 count。
                       local_expert_idx,
                       src_rank);
                if (mask_buffer_ptr == nullptr)
                    // 未提供 mask buffer 时无法恢复，只能 trap 暴露通信错误。
                    trap();
                atomicExch(mask_buffer_ptr + src_rank, 1);
                // 标记源 rank，后续发送/接收都会通过 is_rank_masked 跳过。
            }

            num_recv_tokens = -num_recv_tokens - 1;
            // 发送方编码为 -(count+1)，这里还原成 count；例如 -6 -> 5 个 token。
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            // 多个 src_rank 并发写同一 local_expert 的 packed buffer，用 atomicAdd 分配连续区间。
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            // 把 count 广播给同 group 其他 sub_warp。
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            // 把 packed buffer 起点广播给同 group 其他 sub_warp。
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
            // layout range 是 combine 的关键索引：知道每个 src_rank 的 token 区间才能按源 rank 回传。

            // Add stats for diagnosis
            if (cumulative_local_expert_recv_stats != nullptr)
                // 可选累计统计：每个 local expert 累计接收 token 数，便于负载诊断。
                atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
            if (dispatch_wait_recv_cost_stats != nullptr)
                // 可选等待统计：按 src_rank 累计等待 cycle，便于定位慢 peer。
                atomicAdd(reinterpret_cast<unsigned long long*>(dispatch_wait_recv_cost_stats + src_rank), wait_recv_cost);
        }
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
        // warp group 内 barrier：等待 sub_warp1 写 shared 后，其余 sub_warp 才能搬数据。
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        // 从 shared 读取本 group 的 token 数。
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];
        // 从 shared 读取本 group 在 packed buffer 中的起点。

        // Copy tokens
        EP_DEVICE_ASSERT(num_scales <= 64);
        // scale copy 只处理 lane_id 和 lane_id+32 两段，因此最多支持 64 个 scale。
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            // group 内多个 warp 按 token 维分片复制，每个 warp 负责若干 token。
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            // 当前 RDMA message 的 header 指针，header[0] 是源 token_idx。
            if (lane_id == 0)
                // 只有 lane0 需要拷贝标量 source token index。
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();
            // 确保 source info 写入顺序与后续 payload copy 在 warp 内一致。

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            // 跳过 16B header 后得到 hidden payload 的 int4 起点。
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            // packed_recv_x 中当前 token 的 hidden 写入位置。
            UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);
            // warp 协作复制 hidden payload；st_na_global 避免污染缓存。

            // Copy scales
            if constexpr (kUseFP8) {
                // FP8 接收路径额外复制 scale，并可能从 float 转成 UE8M0 存储。
                // Equivalent CuTe layout:
                //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                // scale 数组紧跟 FP8 hidden payload，发送端始终用 float scale_inv 写入。
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                // 一个 packed_t 内包含多少个 scale 元素；float 为 1，UE8M0 为 4。
                const auto token_idx = recv_token_begin_idx + i;
                // 当前 token 在 local expert packed buffer 中的全局 token 槽位。
                const auto token_stride = num_elems_per_pack;
                // token 连续维 stride，和 CuTe 布局中的 elems_per_pack 对齐。
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
                // 不同 scale pack 之间跨过所有 rank/token 槽位。
                if (lane_id < num_scales) {
                    // lane0..num_scales-1 拷贝前 32 个以内的 scale。
                    const auto pack_idx = lane_id / num_elems_per_pack;
                    // scale 所属 pack 编号。
                    const auto elem_idx = lane_id % num_elems_per_pack;
                    // scale 在 pack 内的位置。
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id));
                    // 根据目标格式保持 float 或压缩为 UE8M0。
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                    // 写入布局让后续 GEMM/专家计算按 token 和 scale pack 高效读取。
                }
                if (lane_id + 32 < num_scales) {
                    // hidden 很大时 scale 数可能超过 32，用同一 warp 的 lane 再处理后半段。
                    const auto pack_idx = (lane_id + 32) / num_elems_per_pack;
                    // 后半段 scale 的 pack 编号。
                    const auto elem_idx = (lane_id + 32) % num_elems_per_pack;
                    // 后半段 scale 在 pack 内的位置。
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id + 32));
                    // 读取并转换第 lane_id+32 个 scale。
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                    // 写入后半段 scale。
                }  // 中文：结束后半段 scale 写入。
            }  // 中文：结束当前 token 的 scale pack 遍历。
        }  // 中文：结束当前 expert/rank 的接收循环。
    }  // 中文：结束低延迟 dispatch 接收端 kernel。
}

// 中文：上面的 device kernel 只做低延迟 dispatch，下面进入 host 侧模板选择/launch 包装。
// 中文：host 侧 dispatch wrapper，按 hidden/FP8/UE8M0 选择模板并启动低延迟 IBGDA kernel。
void dispatch(void* packed_recv_x,  // 输出接收 payload；BF16 或 FP8 data，形状 [local_experts, R*Tmax, H]。
              void* packed_recv_x_scales,  // 输出 FP8 scale buffer；BF16 路径为空。
              int* packed_recv_src_info,  // 输出源 token 信息，combine 用于回传。
              int64_t* packed_recv_layout_range,  // 输出每个 expert 的有效 slot 范围。
              int* packed_recv_count,  // 输出每个 local expert 实际收到的 token 数。
              int* mask_buffer_ptr,  // 可选 rank mask buffer，容错/shrink 模式使用。
              int* cumulative_local_expert_recv_stats,  // 可选累计 expert 统计。
              int64_t* dispatch_wait_recv_cost_stats,  // 可选等待接收耗时统计 [num_ranks,num_ranks]。
              void* rdma_recv_x,  // RDMA 接收 staging buffer。
              int* rdma_recv_count,  // RDMA 接收计数 buffer。
              void* rdma_x,  // RDMA 发送/双缓冲 payload 区。
              const void* x,  // 输入 token payload [num_tokens, hidden]。
              const topk_idx_t* topk_idx,  // 输入路由 [num_tokens,num_topk]。
              int* next_clean,  // 下一轮需要清理的 int 区。
              int num_next_clean_int,  // next_clean 的 int 元素数。
              int num_tokens,  // 当前 rank 输入 token 数。
              int hidden,  // hidden 维度。
              int num_max_dispatch_tokens_per_rank,  // 每 rank 固定容量 Tmax。
              int num_topk,  // top-k 路数。
              int num_experts,  // 全局 expert 数。
              int rank,  // 当前 rank。
              int num_ranks,  // 全局 rank 数。
              bool use_fp8,  // 是否将接收 payload 量化成 FP8。
              bool round_scale,  // 是否把 scale round 到 2 的幂。
              bool use_ue8m0,  // 是否使用 UE8M0 压缩 scale。
              void* workspace,  // 临时计数 workspace。
              int num_device_sms,  // kernel 可使用的 SM 数。
              cudaStream_t stream,  // CUDA launch stream。
              int phases) {  // 当前低延迟双缓冲 phase。
    constexpr int kNumMaxTopK = 11;
    // low-latency dispatch 最多支持 top-k=11；还需要额外一个 warp 给计数逻辑。
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    // 每个 SM 负责 ceil(num_experts / num_sms) 个 expert group。
    const int num_warps_per_group = 32 / num_warp_groups;
    // 32 个 warp 按 expert group 均分；每组至少 2 个 warp 才能重叠 count wait 与 copy。
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    // 防止 expert 数或 SM 数配置导致无法分配 warp group。
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);
    // 数据发送需要最多 top-k 个 warp，另一个 warp 用于计数，所以总 warp 数必须足够。

    const auto num_warps = num_warp_groups * num_warps_per_group;
    // 实际启用 warp 数，作为 launch block 线程数的一部分。
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    // grid block 数，确保所有 expert 都被 responsible_expert_idx 覆盖。
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);
    // 当前请求的 top-k 不能超过 kernel 编译期支持上限。

    // Workspace checks
    auto atomic_counter_per_expert = static_cast<int*>(workspace);
    // workspace 前半段是每个 expert 的 slot 分配计数。
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    // workspace 后半段是每个 expert 的发送完成计数。
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= LEGACY_NUM_WORKSPACE_BYTES);
    // 保证上述两个 int[num_experts] 数组不超过固定 workspace 大小。

    // FP8 checks
    if (use_ue8m0)
        // UE8M0 scale 格式要求 scale 经过 round_scale 处理，否则量化语义不匹配。
        EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(hidden)                         \
    /* hidden 是模板常量，编译器可完全展开 hidden 维循环。 */ \
    {                                                        \
        auto dispatch_func = dispatch<false, false, hidden>; \
        /* 默认 BF16 路径，不需要 scale buffer。 */           \
        if (use_fp8 and not use_ue8m0)                       \
            /* FP8 + float scale 路径。 */                    \
            dispatch_func = dispatch<true, false, hidden>;   \
        if (use_fp8 and use_ue8m0)                           \
            /* FP8 + UE8M0 scale 路径。 */                    \
            dispatch_func = dispatch<true, true, hidden>;    \
        LAUNCH_KERNEL(&cfg,                                  \
                      dispatch_func,                         \
                      /* packed_recv_* 是最终给 expert 计算消费的按 expert/rank 分组输出。 */ \
                      packed_recv_x,                         \
                      packed_recv_x_scales,                  \
                      /* src_info/layout_range/count 是 combine 与 Python 侧读取有效槽的控制面。 */ \
                      packed_recv_src_info,                  \
                      packed_recv_layout_range,              \
                      packed_recv_count,                     \
                      /* mask/stats 用于 low-latency 容错和等待耗时观测，不参与数学结果。 */ \
                      mask_buffer_ptr,                       \
                      cumulative_local_expert_recv_stats,    \
                      dispatch_wait_recv_cost_stats,         \
                      /* rdma_recv_x/rdma_x/rdma_recv_count 是跨 rank 传输的中转区。 */ \
                      rdma_recv_x,                           \
                      rdma_recv_count,                       \
                      rdma_x,                                \
                      /* x/topk_idx 是当前 rank 的输入 token 与路由表。 */ \
                      x,                                     \
                      topk_idx,                              \
                      /* 两个 atomic counter 数组来自 workspace，kernel 内按 expert 分配 slot。 */ \
                      atomic_counter_per_expert,             \
                      atomic_finish_counter_per_expert,      \
                      /* next_clean 描述下一轮进入 dispatch 前需要清理的控制区。 */ \
                      next_clean,                            /* 下一轮清理区指针。 */ \
                      num_next_clean_int,                    /* 下一轮清理区 int 元素数。 */ \
                      num_tokens,                            /* 当前 rank 输入 token 数。 */ \
                      num_max_dispatch_tokens_per_rank,      /* 每 rank/expert 最大 dispatch slot 数。 */ \
                      num_topk,                              /* 每 token top-k 数。 */ \
                      num_experts,                           /* 全局 expert 总数。 */ \
                      rank,                                  /* 当前 rank。 */ \
                      num_ranks,                             /* 全局 rank 数。 */ \
                      /* warp group 参数必须和 SETUP_LAUNCH_CONFIG 的 blockDim 对齐。 */ \
                      num_warp_groups,                       \
                      num_warps_per_group,                   \
                      /* round_scale/phases 控制 FP8 scale 舍入和低延迟双缓冲 phase。 */ \
                      round_scale,                           \
                      phases);                               \
    }                                                        \
    /* 中文：每个 switch case 只实例化一个模板组合，break 防止继续落入后续 hidden/topk 分支。 */ \
    break

    // 中文：blockDim 由 num_warps 决定，内部再划分 dispatch/count/receive 等 sub-warp 角色。
    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    // blockDim = num_warps*32，和 kernel 内 warp_group/sub_warp 划分保持一致。
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
    // 根据运行时 hidden 选择预编译模板实例；未支持的 hidden 会在宏里触发断言/错误。
#undef DISPATCH_LAUNCH_CASE
    // 宏只在 dispatch wrapper 内有效，避免污染后面的 combine wrapper。
}

template <int kNumSendUnrolls>  // 发送端每 lane 预取/压缩的 int4 个数，只支持 2 或 4。
__forceinline__ __device__ int logfmt_encode(void* buffer, nv_bfloat162* shared_amaxmin, const int& lane_id) {
    // logfmt_encode 尝试把每个 128-channel BF16 分组压缩为 LogFMT-10。
    // 输入 buffer 是当前 lane 负责的连续 BF16 分片，输出仍写回同一 buffer 前缀。
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    // 一个 int4 包含 8 个 BF16。
    constexpr float kLogThreshold = 0;
    // 只有 amax < 1 时才尝试对数格式；大值直接保留 BF16 更稳。
    constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`
    // log_amin 最多比 log_amax 小 32，避免极小值导致编码步长过大。
    constexpr int kNumBits = 10;
    // LogFMT 使用 10 bit，其中最高有效编码空间留给 9-bit magnitude 和单独 sign。
    constexpr int kNumValues = 1 << (kNumBits - 1);
    // magnitude 编码值数量为 512；0 表示精确 0，1..511 表示对数区间。

    int4 int4_values[kNumSendUnrolls];
    // 每个 lane 预取 kNumSendUnrolls 个 int4，例如 2 或 4 个 int4。
    const auto uint32_values = reinterpret_cast<uint32_t*>(int4_values);
    // 用 uint32_t 视图提取两个 BF16 的符号位并清符号。
    const auto bf162_values = reinterpret_cast<nv_bfloat162*>(int4_values);
    // 用 bf162 视图做半精度 pair 的 min/max。

    // Calculate lane offset
    const auto ld_buffer = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + lane_id * (kNumSendUnrolls * sizeof(int4)));
    // 每个 lane 从 buffer 中读取自己负责的 kNumSendUnrolls*16B 原始 BF16。
    const auto st_buffer =
        reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + lane_id * (kNumSendUnrolls * sizeof(int4) * 10 / 16));
    // 压缩后每个 BF16 从 16bit 变成 10bit，因此写入起点按 10/16 比例前移。

    // Local log amax
    auto bf162_amax = __nv_bfloat162(CUDART_ZERO_BF16, CUDART_ZERO_BF16);
    // 局部最大绝对值，先以 BF16 pair 形式维护。
    auto bf162_amin = __nv_bfloat162(CUDART_INF_BF16, CUDART_INF_BF16);
    // 局部最小非负值，用于确定对数编码下界。
    uint32_t local_signs = 0;
    // 当前 lane 负责的最多 32 个 BF16 的符号位，按 bit 打包保存。
    #pragma unroll
    for (int k = 0; k < kNumSendUnrolls * kNumElemsPerInt4 / 2; ++k) {
        // 一个 uint32_t 包含两个 BF16，因此循环次数是 BF16 元素数的一半。
        // TODO: eliminate bank conflicts
        uint32_values[k] = ld_buffer[k];
        // 从 shared/TMA buffer 读取两个 BF16 的原始位。
        local_signs |= ((uint32_values[k] >> 15) & 1) << (k * 2);
        // 提取低 16bit BF16 的 sign bit，写入 local_signs 的偶数位。
        local_signs |= ((uint32_values[k] >> 31) & 1) << (k * 2 + 1);
        // 提取高 16bit BF16 的 sign bit，写入 local_signs 的奇数位。
        uint32_values[k] &= 0x7fff7fff;
        // 清除两个 BF16 的符号位，后续 min/max 和 log 都基于绝对值。

        bf162_amax = __hmax2(bf162_amax, bf162_values[k]);
        // 更新两个 lane-local BF16 位置的最大绝对值。
        bf162_amin = __hmin2(bf162_amin, bf162_values[k]);
        // 更新两个 lane-local BF16 位置的最小绝对值。
    }

    // Reduce per 128 channels
    // TODO: figure out how hardware do 2-byte min/max
    auto amax = std::max(static_cast<float>(bf162_amax.x), static_cast<float>(bf162_amax.y));
    // 把 pair 内两个最大值合并成单个 FP32 amax。
    auto amin = std::min(static_cast<float>(bf162_amin.x), static_cast<float>(bf162_amin.y));
    // 把 pair 内两个最小值合并成单个 FP32 amin。
    constexpr static int kNumLanesToReduce = 128 * sizeof(nv_bfloat16) / (kNumSendUnrolls * sizeof(int4));
    // 每个 lane 处理 kNumSendUnrolls*16B，凑满 128 个 BF16 需要的 lane 数。
    amax = warp_reduce_max<kNumLanesToReduce>(amax);
    // 在对应 lane 子组内求 128-channel amax。
    amin = warp_reduce_min<kNumLanesToReduce>(amin);
    // 在对应 lane 子组内求 128-channel amin。

    // Write min/max into the shared memory
    if (shared_amaxmin != nullptr)
        // 每个 128-channel 分组的 leader 写元数据，接收端决定是否按 LogFMT 解码。
        *shared_amaxmin = __nv_bfloat162(amax, amin);
    __syncwarp();
    // 确保 meta_buffers 中 amax/amin 对后续 TMA store 可见。

    // Calculate log amin/amax float
    const auto log_amax = log2f_approx(amax);
    // 对数上界，使用近似 log2 以降低设备端开销。
    const auto log_amin = fmaxf(log2f_approx(amin), log_amax - kMinClip);
    // 对数下界被裁剪到 [log_amax-32, log_amax]。
    const bool enable_cast = warp_reduce_and<kNumLanesToReduce, true>(log_amax < kLogThreshold and log_amin < log_amax);
    // 子组内所有 lane 都满足动态范围条件才压缩；否则整个 128-channel 保留 BF16。

    // Case into LogFMT-10 if satisfied
    if (enable_cast) {
        // LogFMT-10 压缩路径：把 16 个 BF16 的 magnitude 编码成 16 个 9-bit 值 + 16-bit sign。
        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        // 对数区间被 510 个非零编码均匀划分。
        const auto step_inv = 1.0f / step;
        // 预计算倒数，用乘法替代除法。
        const auto rounding = 2.0f - log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * step_inv;
        // rounding 调整编码边界，让对数中心更接近原值。
        const auto fused_rounding = rounding - log_amin * step_inv;
        // 合并 -log_amin/step 和 rounding，编码时只需 log2(x)*step_inv+fused_rounding。

        // Pack every 256 bits into 160 bits
        EP_STATIC_ASSERT(kNumSendUnrolls == 2 or kNumSendUnrolls == 4, "kNumSendUnrolls == 2 or 4 only");
        // 当前位打包逻辑只覆盖 2/4 个 int4 两种展开因子。
        uint32_t encoded[kNumElemsPerInt4 * 2];
        // 一次处理 16 个 BF16 的 9-bit magnitude 编码。
        #pragma unroll 1
        for (int i = 0; i < kNumSendUnrolls / 2; ++i) {
            // 每两个 int4 = 16 个 BF16，压成 5 个 uint32_t = 160 bit。
            #pragma unroll
            for (int k = 0; k < kNumElemsPerInt4; ++k) {
                // kNumElemsPerInt4=8，每次从 bf162_values 中取两个 BF16。
                const auto [x, y] = __bfloat1622float2(bf162_values[i * kNumElemsPerInt4 + k]);
                // 将绝对值 BF16 pair 转成 float2 便于计算 log。
                encoded[k * 2 + 0] = __float2uint_rd(fmaxf(log2f_approx(x) * step_inv + fused_rounding, 0));
                // 第一个 BF16 的 magnitude 编码，0 保留给原值 0。
                encoded[k * 2 + 1] = __float2uint_rd(fmaxf(log2f_approx(y) * step_inv + fused_rounding, 0));
                // 第二个 BF16 的 magnitude 编码。
            }
            st_buffer[i * 5 + 0] = (encoded[0] >> 0) | (encoded[1] << 9) | (encoded[2] << 18) | (encoded[3] << 27);
            // 第 0 个 32bit 字拼接 encoded[0..3] 的低位片段。
            st_buffer[i * 5 + 1] = (encoded[3] >> 5) | (encoded[4] << 4) | (encoded[5] << 13) | (encoded[6] << 22) | (encoded[7] << 31);
            // 第 1 个 32bit 字承接 encoded[3] 剩余位并继续拼 encoded[4..7]。
            st_buffer[i * 5 + 2] = (encoded[7] >> 1) | (encoded[8] << 8) | (encoded[9] << 17) | (encoded[10] << 26);
            // 第 2 个 32bit 字继续按 9-bit 边界拼接。
            st_buffer[i * 5 + 3] =
                (encoded[10] >> 6) | (encoded[11] << 3) | (encoded[12] << 12) | (encoded[13] << 21) | (encoded[14] << 30);
            // 第 3 个 32bit 字承接 encoded[10] 并拼接到 encoded[14]。
            st_buffer[i * 5 + 4] = (encoded[14] >> 2) | (encoded[15] << 7) | ((i == 0) ? (local_signs << 16) : (local_signs & 0xffff0000u));
            // 第 4 个 32bit 字保存 encoded[14..15] 末尾和 16 个 sign bit。
        }
        tma_store_fence();
        // 确保压缩写回 shared buffer 后，后续 TMA store 能读到完整数据。
        __syncwarp();
        // 同步 warp，避免某些 lane 仍在写 st_buffer。
    }

    // Return TMA copy bytes
    return enable_cast ? (32 * (kNumSendUnrolls * sizeof(int4) * 8 * 10 / 16 / 8)) : (32 * (kNumSendUnrolls * sizeof(int4)));
    // 返回整个 warp 需要 TMA store 的字节数：LogFMT 为 BF16 大小的 10/16，非压缩为原 BF16。
}

template <int kNumLanes, int kNumSendUnrolls, int kNumRecvUnrolls>  // LogFMT 接收端按 division/lane/unroll 还原变长布局。
__forceinline__ __device__ void logfmt_check_amaxmin(
    uint8_t* meta_buffer, float2* shared_log_amax, float2* shared_log_amin, int* shared_cast_info, const int lane_id) {
    // logfmt_check_amaxmin 在接收端读取发送端保存的 amax/amin 元数据，
    // 判断每个 decode warp 对应的分片是否真的以 LogFMT 存储。
    constexpr float kLogThreshold = 0;
    // 与 encode 端一致，只有 log_amax < 0 的分片才可能是 LogFMT。
    constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`
    // 与 encode 端一致的下界裁剪范围。

    bool enable_cast = true;
    // 当前 lane 负责的两个 128-channel 分组是否可按 LogFMT 解码。
    if (lane_id < kNumLanes) {
        // kNumLanes = kNumDivisions/2，每个 lane 读取两个 division 的 amax/amin。
        // Calculate log amin/amax float
        auto amaxmin2 = reinterpret_cast<uint64_t*>(meta_buffer)[lane_id];
        // meta_buffer 每个 lane 读取 64bit，里面包含两个 nv_bfloat162 元数据。
        const auto bf162_amaxmin = reinterpret_cast<__nv_bfloat162*>(&amaxmin2);
        // 把 64bit 临时解释为两个 (amax, amin) pair。
        float log_amax[2], log_amin[2];
        // 两个 division 的 log 上下界。
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            // 每个 lane 处理两个 128-channel division。
            auto amax = static_cast<float>(bf162_amaxmin[i].x);
            // 第 i 个 division 的 amax。
            auto amin = static_cast<float>(bf162_amaxmin[i].y);
            // 第 i 个 division 的 amin。
            log_amax[i] = log2f_approx(amax);
            // 还原 encode 端使用的 log_amax。
            log_amin[i] = amin == 0 ? log_amax[i] - kMinClip : fmaxf(log2f_approx(amin), log_amax[i] - kMinClip);
            // amin 为 0 时无法 log，按最小裁剪值处理。
            enable_cast = enable_cast and log_amax[i] < kLogThreshold and log_amin[i] < log_amax[i];
            // 两个 division 都满足 encode 条件时，当前 lane 才认为可压缩。
        }
        shared_log_amax[lane_id] = make_float2(log_amax[0], log_amax[1]);
        // 写入 shared，decode warp 后续按 division 读取。
        shared_log_amin[lane_id] = make_float2(log_amin[0], log_amin[1]);
        // 写入对应的 log_amin。
    }

    const auto casted = warp_reduce_and<kNumSendUnrolls>(enable_cast) ? 1u << (lane_id / kNumRecvUnrolls) : 0u;
    // 每 kNumSendUnrolls 个 lane 对应 encode 端一个压缩单元；全 true 才设置该 decode warp 的 bit。
    const auto num_casted_prefix = __popc(warp_reduce_or<kNumRecvUnrolls, true>(casted) & ((1u << (lane_id / kNumRecvUnrolls)) - 1));
    // 统计当前 decode warp 之前有多少分片是 LogFMT，用于计算变长 payload 的偏移。

    if (lane_id < kNumLanes and lane_id % kNumRecvUnrolls == 0)
        // 每个 decode warp 的首 lane 写 cast_info：低 bit 表示是否 cast，高位是前缀压缩块数。
        shared_cast_info[lane_id / kNumRecvUnrolls] = (num_casted_prefix << 1) | (casted ? 1u : 0u);
    __syncwarp();
    // 确保 cast_info/log_amax/log_amin 写完后 TMA load warp 再计算 payload 长度。
}

template <int kNumRecvUnrolls>  // 接收规约端每 lane 处理的 int4 个数。
__forceinline__ __device__ void decode_and_accumulate(
    uint32_t* ld_buffer, float* accum, const float& log_amax, const float& log_amin, const bool& enable_cast, const float& weight) {
    // decode_and_accumulate 把一个 top-k 专家返回的 hidden 分片累加到 accum。
    // enable_cast=true 时输入是 LogFMT-10，否则输入是原始 BF16。
    if (enable_cast) {
        // LogFMT 解码路径。
        constexpr int kNumBits = 10;
        // 编码总位宽为 10bit，其中 magnitude 使用 9bit。
        constexpr int kNumValues = 1 << (kNumBits - 1);
        // 9bit magnitude 的取值空间。

        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        // 与 encode 端相同的对数步长。
        auto decode = [=](const uint32_t& encoded, const uint32_t& sign) {
            // 小 lambda 根据 magnitude 和 sign 还原 FP32 值。
            const auto decoded = encoded == 0 ? .0f : exp2f_approx((encoded - 1) * step + log_amin);
            // encoded=0 表示精确 0；非零编码映射回 exp2(log_amin + (encoded-1)*step)。
            return sign ? -decoded : decoded;
            // 重新附加符号位。
        };

        EP_STATIC_ASSERT(kNumRecvUnrolls == 2 or kNumRecvUnrolls == 4, "kNumRecvUnrolls == 2 or 4 only");
        // 位解包逻辑只支持当前两种接收展开。
        #pragma unroll
        for (int i = 0; i < kNumRecvUnrolls / 2; ++i) {
            // 每两个 int4 对应 encode 端 16 个 BF16 -> 5 个 uint32_t。
            uint32_t concat[6];
            // concat 用来把跨 uint32_t 边界的 9-bit 编码重新对齐。
            concat[0] = ld_buffer[i * 5];
            // 第一个 32bit 直接作为对齐起点。
            #pragma unroll
            for (int k = 1; k < 5; ++k)
                // 后续 word 通过右移前一个 word 和左移当前 word 拼接，形成连续 9-bit 流。
                concat[k] = (ld_buffer[i * 5 + k - 1] >> (32 - k * 5)) | (ld_buffer[i * 5 + k] << (k * 5));
            concat[5] = ld_buffer[i * 5 + 4] >> 7;
            // 第 16 个 encoded 只需要最后 word 的高位片段。

            const uint32_t local_signs = ld_buffer[i * 5 + 4] >> 16;
            // 16 个 sign bit 存在第 5 个 word 的高 16bit。
            #pragma unroll
            for (int k = 0; k < 5; ++k) {
                // 前 15 个 encoded 每个 concat word 提供 3 个 9-bit 值。
                accum[i * 16 + k * 3 + 0] += decode((concat[k] >> 0) & 0x1ff, (local_signs >> (k * 3 + 0)) & 1) * weight;
                // 第 k 组第 0 个值乘 top-k 权重后累加。
                accum[i * 16 + k * 3 + 1] += decode((concat[k] >> 9) & 0x1ff, (local_signs >> (k * 3 + 1)) & 1) * weight;
                // 第 k 组第 1 个值乘 top-k 权重后累加。
                accum[i * 16 + k * 3 + 2] += decode((concat[k] >> 18) & 0x1ff, (local_signs >> (k * 3 + 2)) & 1) * weight;
                // 第 k 组第 2 个值乘 top-k 权重后累加。
            }
            accum[i * 16 + 15] += decode(concat[5] & 0x1ff, (local_signs >> 15) & 1) * weight;
            // 第 16 个值单独从 concat[5] 解码。
        }
    } else {
        // 非 LogFMT 路径：ld_buffer 中是原始 BF16，直接转换 FP32 后加权累加。
        #pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
            // uint32_t 中包含两个 BF16，所以循环次数是接收 int4 覆盖 BF16 pair 数。
            auto bf16_pack = *reinterpret_cast<__nv_bfloat162*>(ld_buffer + k);
            // 从两个 BF16 组成的 pack 中取 x/y。
            accum[k * 2 + 0] += static_cast<float>(bf16_pack.x) * weight;
            // 第一个 BF16 转 FP32 后乘专家权重累加。
            accum[k * 2 + 1] += static_cast<float>(bf16_pack.y) * weight;
            // 第二个 BF16 转 FP32 后乘专家权重累加。
        }
    }
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls>  // combine 主 kernel 的压缩开关、hidden 和上限参数。
__global__ __launch_bounds__(1024, 1) void combine(void* combined_x,  // 输出 combined activation，逻辑形状 [num_combined_tokens, hidden]。
                                                   void* rdma_recv_x,  // 本 rank 接收其它 rank combine 结果的 RDMA buffer。
                                                   int* rdma_recv_flag,  // 每个返回 slot 的到达标志，接收端按 token/top-k 等待。
                                                   void* rdma_send_x,  // 本 rank 发送给源 rank 的 staging buffer。
                                                   const void* x,  // expert 计算后的输入 activation，按 dispatch packed layout 排列。
                                                   const topk_idx_t* topk_idx,  // 原始路由表，形状 [num_combined_tokens, num_topk]。
                                                   const float* topk_weights,  // top-k 权重，形状 [num_combined_tokens, num_topk]。
                                                   const int* src_info,  // dispatch 输出的源 token/rank 信息，按 expert/rank slot 排列。
                                                   const int64_t* layout_range,  // 每个 expert/rank 的 packed(offset,count)，用于定位专家输出区间。
                                                   int* mask_buffer_ptr,  // 可选故障 rank mask，masked rank 的返回值会被跳过。
                                                   int64_t* combine_wait_recv_cost_stats,  // 输出等待返回结果的耗时统计，单位 GPU cycle。
                                                   int* next_clean,  // 下一轮 combine/dispatch 前需要清理的控制区。
                                                   int num_next_clean_int,  // next_clean 的 int 元素数。
                                                   int* atomic_clean_flag,  // combine 清理阶段的原子完成标志。
                                                   int num_combined_tokens,  // 需要还原到源 token 顺序的 token 数。
                                                   int hidden,  // 运行时 hidden 大小，需与模板 kHidden 一致。
                                                   int num_topk,  // 每个 token 的 top-k 数。
                                                   int num_max_dispatch_tokens_per_rank,  // dispatch 阶段每 expert/rank 预留的最大 slot 数。
                                                   int num_experts,  // 全局 expert 总数。
                                                   int rank,  // 当前全局 rank。
                                                   int num_ranks,  // 全局 rank 数。
                                                   int num_warp_groups,  // 每个 SM 内参与 combine 的 warp group 数。
                                                   int num_warps_per_group,  // 每个 warp group 的 warp 数。
                                                   int phases,  // combine SEND/RECV phase 位掩码。
                                                   bool zero_copy) {  // true 表示可直接从/向用户 buffer 做部分零拷贝。
    // low-latency combine 反向读取 dispatch 生成的 layout_range/src_info：
    // 每个 expert/rank 的专家输出先通过 RDMA 发送回源 rank，接收端按 topk_weights
    // 对同一 token 的多个 expert 结果做加权求和，写入 combined_x [num_combined_tokens, hidden]。
    const auto sm_id = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    // blockIdx.x 广播到整个 warp，避免每个 lane 单独读特殊寄存器。
    const auto num_sms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    // gridDim.x 表示参与 combine 的 CTA 数，发送/接收阶段都用它做 grid-stride。
    const auto thread_id = static_cast<int>(threadIdx.x);
    // block 内线程编号，最多 1024 个线程。
    const auto num_threads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    // blockDim.x 广播后用于接收阶段重新划分 warp group。
    const auto warp_id = __shfl_sync(0xffffffff, thread_id / 32, 0), lane_id = get_lane_id();
    // warp_id/lane_id 是所有 warp 级协作的基本索引。
    const auto num_local_experts = num_experts / num_ranks;
    // 当前 rank 持有的 local expert 数。
    const auto warp_group_id = warp_id / num_warps_per_group;
    // 发送阶段：warp_group_id 对应当前 SM 负责的 expert group。
    const auto sub_warp_id = warp_id % num_warps_per_group;
    // 发送阶段：group 内多个 warp 分摊同一 expert 的 token。
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
    // 发送/等待 flag 阶段负责的全局 expert/rank 组合索引。

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    // combine 使用动态 shared memory：发送阶段放 TMA pipeline buffer，接收阶段重分配为 load/reduce buffer。

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    // 一个 int4 覆盖 8 个 BF16。
    constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;
    // 每个 token 的 hidden 向量按 int4 计数。

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    // 发送端优先用 4 个 int4 展开；hidden 不满足整除时退到 2。
    constexpr int kNumRecvUnrolls = 2;
    // 接收规约端固定每 lane 处理 2 个 int4，便于 TMA buffer 与 LogFMT 位解包配合。
    constexpr int hidden_bf16_int4_pad = align_up(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
    // 发送 TMA pipeline 按 warp*unroll 对齐，不足部分 padding 到整轮。
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    // 接收端固定展开要求 hidden 能被 32 lanes * 2 int4 * 8 BF16 整除。
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    // wrapper 计算 shared memory 时按 kNumMaxUnrolls 预留空间。
    EP_STATIC_ASSERT(hidden_bf16_int4 % kNumSendUnrolls == 0, "Invalid hidden");
    // 发送端每 lane 展开次数必须整除每 token 的 int4 数。
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");
    // LogFMT 编码/解码的分片前缀计算假设发送展开不小于接收展开。

    // combine 的 RDMA slot 包含 BF16 hidden 和可选 LogFMT 元数据。LogFMT 元数据按
    // hidden/128 个 nv_bfloat162 存储，用于低带宽格式的解码规约。
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    // LogFMT 元数据每 128 channel 一组，因此 hidden 必须整除 128。
    constexpr int kNumDivisions = kHidden / 128;
    // 每个 token 的 128-channel division 数。
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
    // 每个 division 存一个 nv_bfloat162(amax, amin)，所以元数据大小为 divisions*4B。
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
    // rdma_recv_x/rdma_send_x 每个 slot 预留 BF16 hidden + LogFMT metadata。
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");
    // slot 必须 16B 对齐，方便 IBGDA put 和 TMA/int4 访问。

    // Sending phase
    if ((phases & LEGACY_LOW_LATENCY_SEND_PHASE) == 0)
        // combine 也支持只运行接收规约阶段；发送阶段被跳过时直接等待 flag/规约。
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        // 仅 sm0/group0/subwarp0 负责清理下一轮 buffer 和初始化 clean flag。
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            // next_clean 通常是下一轮 rdma_recv_flag/计数区，按 int 清零。
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        // 确保 next_clean 清理完成后再释放 atomic_clean_flag。
        if (lane_id == 0)
            // atomic_clean_flag 初始加 num_experts，后面每个 expert 发送完 flag 后减 1。
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // 发送阶段按 global expert 分配 warp group。layout_range[local_expert, dst_rank]
    // 打包了该 expert 需要发回 dst_rank 的 offset/count。
    if (responsible_expert_idx < num_experts) {
        // 每个 responsible_expert_idx 表示要把当前 rank 的 local expert 输出发回某个 dst_rank。
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        // 目标 rank，即 dispatch 时该批 token 的源 rank。
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        // 当前 rank 内的 local expert 编号。
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        // 当前 local expert 对应的全局 expert id，接收方用它定位 rdma_recv_x slot。
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        // layout_range[local_expert, dst_rank] 由 dispatch 接收阶段写入，打包 count/offset。
        const auto local_x =
            static_cast<const int4*>(x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        // x 是专家计算输出，逻辑维度：
        // [num_local_experts, num_ranks*num_max_dispatch_tokens_per_rank, hidden]。
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        // src_info 与 x 的 token 槽位对齐，保存原始 token_idx，作为回传目标 slot。
        const auto rdma_send_x_vec =
            static_cast<uint8_t*>(rdma_send_x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;
        // rdma_send_x 是发送 staging buffer，按 local expert/rank/token/slot 字节布局。

        // Unpack layout
        int offset, num_tokens_to_send;
        // offset 是当前 dst_rank 在 local_x 中的起始 token 槽；num_tokens_to_send 是长度。
        unpack2(layout, num_tokens_to_send, offset);
        // pack2/unpack2 与 dispatch 阶段的 recv_range 写法严格配对。

        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        // 一个 TMA stage 容纳一个 warp 轮次的数据：32 lanes * kNumSendUnrolls int4。
        constexpr int kNumStages = 3;
        // 三阶段 pipeline：当前处理、下一次预取、再下一次等待。
        constexpr int kNumPrefetch = 1;
        // 始终保持一个 stage 预取。
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");
        // 后面 tma_store_wait 的模板参数按这个固定配置写死。

        auto smem_ptr = smem_buffer + warp_id * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);
        // 发送阶段为每个 warp 分配独立 shared 区域：stage buffer + barrier + LogFMT meta。
        uint32_t tma_phase = 0;
        // mbarrier phase bit，随 stage 轮转在 mbarrier_wait 中使用。
        auto tma_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16)); });
        // tma_buffers[i] 是第 i 个 stage 的数据区。
        auto full_barriers = PatternVisitor(
            [=](const int& i) { return reinterpret_cast<uint64_t*>(smem_ptr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        // full_barriers[i] 紧跟数据区后面，表示第 i 个 stage 的 TMA load 完成。
        auto meta_buffers = kUseLogFMT ? reinterpret_cast<nv_bfloat162*>(smem_ptr + kNumStages * (kNumTMABufferBytes + 16)) : nullptr;
        // LogFMT 元数据区只在 kUseLogFMT=true 时使用。
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");
        // 限制每 warp shared buffer 大小，避免超出动态 shared memory 预算。

        // Initialize m-barriers
        if (lane_id < kNumStages) {
            // 前 kNumStages 个 lane 分别初始化一个 full barrier。
            mbarrier_init(full_barriers[lane_id], 1);
            // 每个 TMA load 只有一个 producer，因此 expected arrival count 为 1。
            fence_barrier_init();
            // fence 确保 barrier 初始化对后续 TMA 操作可见。
        }
        __syncwarp();
        // 确保所有 stage 的 barrier 初始化完成。

        constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumSendUnrolls);
        // 一个 token 的 hidden 需要多少个 warp TMA 轮次。
        auto tma_load_and_arrive = [&](const int& stage_idx, const int4* gmem_ptr, const int& num_bytes) {
            // 小 lambda 封装一次 global->shared TMA load 并设置 mbarrier 期望字节数。
            tma_load_1d(tma_buffers[stage_idx], gmem_ptr, full_barriers[stage_idx], num_bytes);
            // 发起 1D TMA load，将 gmem_ptr 的 num_bytes 拷到当前 stage。
            mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_bytes);
            // 通知 barrier 本 stage 预期接收 num_bytes。
        };
        auto get_num_tma_bytes = [&](const int& offset_int4) {
            // 计算从 offset_int4 开始本轮最多能读取多少字节，最后一轮可能不足整 stage。
            return min(kNumTMABufferBytes, static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
        };

        // Issue IBGDA send
        if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank)) {
            // 目标 rank 被 mask 时跳过整个 expert 的回传。
            for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
                // group 内多个 sub_warp 按 token 槽分片发送。
                const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
                // 当前专家输出 token 的 BF16 hidden 起点。
                const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
                // 当前 token 的 staging slot 起点；名字保留历史语义，实际第一个 int 不再存 type。
                const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);
                // 字节视图用于 TMA/IBGDA 地址计算。

                // Copy directly to local rank, or copy to buffer and issue RDMA
                const auto src_idx = __shfl_sync(0xffffffff, __ldg(local_src_info + token_idx), 0);
                // 源 token index，lane0 读取后广播；接收方 slot 按原 token_idx 放置，便于 topk 规约。
                const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
                // staging buffer 地址，跨节点 RDMA put 的源地址。
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                    (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
                // 目标 rdma_recv_x 逻辑维度：[num_experts, num_max_dispatch_tokens_per_rank, slot_bytes]。
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                // 同节点 P2P 直接写远端 slot，跨节点则先写 staging 再 RDMA put。
                int num_send_bytes = hidden * sizeof(nv_bfloat16);
                // 默认发送完整 BF16 hidden；LogFMT 启用时会根据压缩后字节数覆盖。

                if (not zero_copy or dst_p2p_ptr != 0) {
                    // zero_copy 且跨节点时假设数据已经在 rdma_send_x，中间拷贝可省略。
                    // Read from `cpy_src_int4_ptr` and copy into `cpy_dst_int4_ptr`
                    const auto cpy_src_int4_ptr = zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
                    // zero_copy 从 staging 读，否则从专家输出 x 读。
                    const auto cpy_dst_int4_ptr =
                        dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr) : reinterpret_cast<int4*>(dst_p2p_ptr);
                    // 跨节点先写 staging，P2P 直接写远端目标。

                    // Prefetch
                    if (elect_one_sync())
                        // 每个 warp 只需一个 elected lane 发起 TMA load。
                        tma_load_and_arrive(0, cpy_src_int4_ptr, get_num_tma_bytes(0));
                    __syncwarp();
                    // 确保第一个 prefetch 已提交后进入 pipeline。

                    int tma_offset_bytes = kNumMetaBytes;
                    // LogFMT 路径把元数据放在 slot 开头，payload 从 kNumMetaBytes 后开始。
                    #pragma unroll
                    for (int i = lane_id * kNumSendUnrolls, iter_idx = 0; i < hidden_bf16_int4_pad; i += 32 * kNumSendUnrolls, ++iter_idx) {
                        // 每个 lane 处理 hidden 的一段，iter_idx 控制三 stage 轮转。
                        // Load the next iteration
                        const int stage_idx = iter_idx % kNumStages;
                        // 当前处理的 stage。
                        const int next_stage_idx = (iter_idx + 1) % kNumStages;
                        // 下一轮预取要写入的 stage。
                        if (iter_idx + 1 < kNumIters and elect_one_sync()) {
                            // 还有下一轮时，提前发起下一 stage 的 TMA load。
                            tma_store_wait<kNumStages - kNumPrefetch - 1>();
                            // 防止 store pipeline 超前太多，保持最多 kNumPrefetch 个未完成。
                            const auto offset_int4 = i + 32 * kNumSendUnrolls;
                            // 下一轮从 hidden 中读取的 int4 偏移。
                            tma_load_and_arrive(next_stage_idx, cpy_src_int4_ptr + offset_int4, get_num_tma_bytes(offset_int4));
                            // 预取下一段 hidden 到 shared。
                        }
                        __syncwarp();
                        // 确保下一段预取提交和当前 stage 状态对 warp 内所有 lane 一致。

                        // Wait the current TMA arrival
                        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
                        // mbarrier_wait 的 phase bit 使用 32bit 掩码，stage 数必须小于 32。
                        mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                        // 等待当前 stage 的 TMA load 完成。
                        if constexpr (kUseLogFMT) {
                            // LogFMT 发送路径：尝试把当前 stage 的 BF16 压成对数格式。
                            // Cast if possible
                            // 中文：只有当前 division 的 leader lane 写 meta，其余 lane 只参与编码计算。
                            constexpr int kNumInt4PerDivision = 128 / kNumElemsPerInt4;
                            // 一个 128-channel division 对应多少个 int4。
                            int num_tma_bytes = logfmt_encode<kNumSendUnrolls>(
                                tma_buffers[stage_idx],
                                // NOTES: only the leader lane will write the result
                                // 中文：非 leader lane 传 nullptr，避免多个 lane 竞争写同一个 meta 槽。
                                (i % kNumInt4PerDivision == 0) ? meta_buffers + i / kNumInt4PerDivision : nullptr,
                                lane_id);
                            // logfmt_encode 返回当前 warp 实际需要写出的 payload 字节数。
                            if (elect_one_sync())
                                // elected lane 发起 shared->global/P2P 目标的 TMA store。
                                tma_store_1d(
                                    tma_buffers[stage_idx], reinterpret_cast<uint8_t*>(cpy_dst_int4_ptr) + tma_offset_bytes, num_tma_bytes);
                            tma_offset_bytes += num_tma_bytes;
                            // payload 是变长的，下一段紧跟当前压缩段后面。
                        } else {
                            // BF16 original values
                            if (elect_one_sync())
                                // 非 LogFMT 路径直接把 BF16 stage 原样写到目标。
                                tma_store_1d(tma_buffers[stage_idx], cpy_dst_int4_ptr + i, get_num_tma_bytes(i));
                        }
                        __syncwarp();
                        // 确保当前 stage 的 encode/store 提交后再进入下一轮。
                    }

                    // Store metadata (min/max values) for LogFMT
                    if constexpr (kUseLogFMT) {
                        // payload 全部写完后，再把 meta_buffers 写到 slot 开头。
                        num_send_bytes = tma_offset_bytes;
                        // 最终 RDMA put 字节数 = meta + 所有变长 payload。
                        if (elect_one_sync())
                            // 一个 lane 发起 metadata 的 TMA store。
                            tma_store_1d(meta_buffers, cpy_dst_int4_ptr, kNumMetaBytes);
                    }

                    // Flush all stores
                    tma_store_wait<0>();
                    // 等待所有 TMA store 完成，保证 RDMA put 或 flag 写入前数据已落地。
                    __syncwarp();
                    // warp 内同步，防止某些 lane 提前发 finishing flag。
                }

                // Issue RDMA
                // NOTES: for zero-copy mode, we assume the data is already in the send buffer
                if (dst_p2p_ptr == 0)
                    // 跨节点路径在 staging buffer 准备好后，通过 IBGDA put 发给目标 rank。
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, lane_id, token_idx - offset);
            }
        }

        // Put the finishing flag
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        // 需要 sub_warp1 写 flag，且 bar.sync id 为 warp_group_id+1，故 group 数小于 16。
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        // 等待同一 expert group 内所有 sub_warp 的 token RDMA 都提交完。
        if (sub_warp_id == 1 and lane_id == 0) {
            // 每个 expert group 只由 sub_warp1/lane0 发一次 finishing flag。
            while (ld_acquire_global(atomic_clean_flag) == 0)
                // 等待 sm0 初始化 clean flag，避免下一轮清理与本轮 flag 发送乱序。
                ;
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx);
            // 目标 rank 的 rdma_recv_flag[global_expert_idx] 表示该 expert 输出已经到齐。
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            // flag 同样支持 P2P store 或跨节点 AMO。
            if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
                // 目标 rank 已 mask 时不再发送 flag。
                if (dst_p2p_ptr == 0) {
                    // 跨节点用 non-fetch AMO add 写 flag。
                    nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
                } else {
                    // P2P 路径直接 release store 1。
                    st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
                }
            }  // 中文：结束当前 src_rank 的 flag 发布或跳过。
            atomic_add_release_global(atomic_clean_flag, -1);
            // 本 expert 的 flag 处理结束，clean flag 递减；下一轮可据此判断是否全部完成。
        }
        // 中文：warp 内所有 lane 对 flag/clean 处理完成后才能进入下一轮 expert。
        __syncwarp();
        // 中文：所有 lane 等待 flag 写入路径发起，随后进入接收/等待阶段。
        // 确保 flag 发送和 atomic_clean_flag 更新结束。

        // Destroy m-barriers
        if (lane_id < kNumStages) {
            // 前 kNumStages 个 lane 销毁各自 stage 的 mbarrier。
            mbarrier_inval(full_barriers[lane_id]);
            // 避免 shared memory barrier 状态污染下一次 kernel/同 kernel 后续复用。
            fence_barrier_init();
            // fence 确保 invalidation 完成。
        }
        __syncwarp();
        // warp 内同步后离开发送阶段。
    }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
    // combine 接收阶段从这里开始；发送阶段关闭时会直接跳到该标签。
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0)
        // 只运行 SEND 阶段时，flag 发完即可返回。
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        // 每个 responsible_expert_idx 等待一个 src_rank/global_expert 的完成 flag。
        EP_DEVICE_ASSERT(num_warps_per_group > 1);
        // sub_warp0 等 flag，其他 warp 后续参与规约。
        if (sub_warp_id == 0 and lane_id == 0) {
            // 每个 expert group 只需要一个 lane 轮询 flag。
            const auto src_rank = responsible_expert_idx / num_local_experts;
            // 该 global expert 所在的源 rank。
            auto start_time = clock64();
            // 记录等待开始时间，用于超时诊断。
            uint64_t wait_recv_cost = 0;
            // 等待 flag 的 cycle 数。
            if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
                // 已 mask 的源 rank 直接跳过等待。
                while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0  // recv not ready
                       && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES   // not timeout
                )
                    // 等待发送方写 rdma_recv_flag[responsible_expert_idx]=1。
                    ;
            }  // 中文：结束当前 src_rank/responsible_expert 的 flag 等待。
            // Mask rank if timeout
            if (wait_recv_cost > LEGACY_NUM_TIMEOUT_CYCLES) {
                // 等不到完成 flag 表示该源 rank 的 expert 输出未到齐。
                // 中文：combine 等待的是“该 expert 所有 payload 到齐”的 flag，不是 token count。
                printf("Warning: DeepEP timeout for combine receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank,
                       // 中文：responsible_expert_idx 取模后得到本地 expert 下标。
                       responsible_expert_idx % num_local_experts,
                       src_rank);
                if (mask_buffer_ptr == nullptr)
                    // 未启用 mask 容错时直接 trap。
                    trap();
                atomicExch(mask_buffer_ptr + src_rank, 1);
                // 标记该源 rank，后面的 top-k 规约会跳过它。
            }

            if (combine_wait_recv_cost_stats != nullptr) {
                // 可选统计：按 src_rank 累加等待 flag 的耗时。
                atomicAdd(reinterpret_cast<unsigned long long*>(combine_wait_recv_cost_stats + src_rank), wait_recv_cost);
            }  // 中文：结束可选等待耗时统计更新。
        }
    }
    // 中文：grid sync 确保所有等待 flag 的线程组都完成 mask/统计更新。
    cg::this_grid().sync();
    // 确保所有 expert/rank 的 flag 等待结束后，再开始读取 rdma_recv_x 做规约。

    // Reassign warp groups
    constexpr int kMaxNumGroups = 2;
    // 接收规约阶段一个 CTA 最多并行处理两个 token group。
    const int num_decode_warps = hidden_bf16_int4_pad / (kNumRecvUnrolls * 32);
    // 规约一个 token 的 hidden 需要多少个 decode warp；每个 warp 覆盖 kNumRecvUnrolls*32 int4。
    const int num_groups = min(kMaxNumGroups, (num_threads / 32) / (num_decode_warps + 1));
    // 每个 group 包含 num_decode_warps 个 reduction warp + 1 个 TMA load warp。
    const int decode_warp_idx = __shfl_sync(0xffffffff, warp_id % (num_decode_warps + 1), 0);
    // group 内 warp 编号；等于 num_decode_warps 的 warp 是 TMA load warp。
    const int group_idx = __shfl_sync(0xffffffff, warp_id / (num_decode_warps + 1), 0);
    // 当前 warp 属于第几个 token group。
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    // hidden 至少要能被一个 warp 的 int4 向量覆盖整除。
    EP_DEVICE_ASSERT(num_topk <= 32);
    // topk_idx_by_lane 使用一个 warp 的 lane 存 top-k，因此 top-k 最大 32。
    EP_DEVICE_ASSERT(num_groups > 0);
    // 至少需要一个 group 才能处理 token。

    if (group_idx < num_groups) {
        // 多余 warp 不参与接收规约。
        constexpr int kNumStages = 3;
        // TMA load/reduce 使用三 stage pipeline。
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        // 每个 stage：两个 barrier 的 16B 空间 + 一个 token 的 BF16 payload 最大空间。
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2;
        // 一个 decode warp 原始 BF16 分片的字节数。
        constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes / 16 * 10;
        // 一个 decode warp 对应 LogFMT 压缩后的字节数，按 10/16 比例。
        constexpr int kNumDivisionBytes = kNumDivisions * sizeof(uint32_t);
        // log_amax/log_amin/cast_info 每个 stage 各需要 kNumDivisions 个 4B 槽。
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * 2 + kNumStages * kNumDivisionBytes * 3;
        // 一个 group 的 shared memory 总需求：stage buffer + store buffer + LogFMT 辅助数组。

        // Reallocate shared memory
        const auto smem_group_buffer = smem_buffer + kNumBytesPerGroup * group_idx;
        // 接收阶段按 group_idx 重新切分 smem_buffer。
        auto full_barriers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes); });
        // full_barriers[i] 表示第 i 个 stage 的 TMA load 已完成。
        auto empty_barriers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes + 8); });
        // empty_barriers[i] 表示第 i 个 stage 已被所有 reduction warp 消费，可重新装载。
        auto tma_ld_buffers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint8_t*>(smem_group_buffer + i * kNumTMABufferBytes + 16); });
        // tma_ld_buffers[i] 是第 i 个 stage 的 RDMA slot payload 缓冲区。
        auto tma_st_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint32_t*>(smem_group_buffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes);
        });
        // tma_st_buffers[decode_warp_idx] 暂存该 decode warp 规约后的 BF16 输出。

        // Redundant when logfmt is disabled
        const auto smem_group_ptr = smem_group_buffer + kNumStages * kNumTMABufferBytes + kHidden * 2;
        // LogFMT 辅助数组起点；非 LogFMT 时这些空间虽预留但不使用。
        auto log_amax_buffers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_group_ptr + i * kNumDivisionBytes); });
        // log_amax_buffers[stage][division] 保存对应 division 的 log_amax。
        auto log_amin_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<float*>(smem_group_ptr + kNumStages * kNumDivisionBytes + i * kNumDivisionBytes);
        });
        // log_amin_buffers[stage][division] 保存对应 division 的 log_amin。
        auto cast_info_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<int*>(smem_group_ptr + kNumStages * kNumDivisionBytes * 2 + i * kNumDivisionBytes);
        });
        // cast_info_buffers[stage][decode_warp] 的低 bit 表示该分片是否 LogFMT，高位是压缩前缀数。

        uint32_t tma_phase = 0;
        // reduction warp 初始 phase 为 0。
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        // barrier phase 掩码需要 stage 数小于 32。
        if (decode_warp_idx == num_decode_warps)
            // TMA load warp 初始认为所有 empty barrier 都已到达，便于第一次 load 直接开始。
            tma_phase = (1 << kNumStages) - 1;

        // Initialize m-barriers
        if (decode_warp_idx == num_decode_warps and lane_id < kNumStages) {
            // 由 TMA load warp 初始化每个 stage 的 full/empty barrier。
            mbarrier_init(full_barriers[lane_id], 1);
            // full barrier 只有一个 producer，即 TMA load warp。
            mbarrier_init(empty_barriers[lane_id], num_decode_warps);
            // empty barrier 需要所有 reduction warp 消费后才算 empty。
        }
        asm volatile("bar.sync %0, %1;" ::"r"(group_idx + 1), "r"((num_decode_warps + 1) * 32));
        // group 内 barrier：等待 mbarrier 初始化完成。

        int stage_idx = 0, topk_idx_by_lane = 0;
        // stage_idx 在 top-k 维轮转；topk_idx_by_lane 暂存当前 token 的 top-k expert id。
        EP_STATIC_ASSERT(kNumMaxTopk <= 32, "Invalid number of topks");
        // 一个 warp lane 数足以保存 top-k 列。
        if (decode_warp_idx == num_decode_warps) {
            // group 内最后一个 warp 专门负责 TMA load，每轮装载一个 top-k expert 的输出。
            // TMA load warp
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                // 不同 SM/group 按 token_idx 交错处理 combined token。
                if (lane_id < num_topk)
                    // lane0..num_topk-1 读取当前 token 的 top-k expert id。
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                for (int i = 0; i < num_topk; ++i) {
                    // 按 top-k 顺序逐个装载专家输出，reduction warp 同步累加。
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    // lane i 持有的 expert id 广播给整个 TMA load warp。
                    if (topk_idx_reg < 0)
                        // 负数表示 padding/无效 expert，直接跳过。
                        continue;
                    if (is_rank_masked(mask_buffer_ptr, topk_idx_reg / num_local_experts))
                        // 该 expert 所在 rank 已 mask 时跳过，避免读未到达的数据。
                        continue;

                    mbarrier_wait<true>(empty_barriers[stage_idx], tma_phase, stage_idx);
                    // 等待当前 stage 被上一轮 reduction warp 消费完。
                    auto buffer = static_cast<uint8_t*>(rdma_recv_x) +
                        (topk_idx_reg * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot;
                    // rdma_recv_x 按 [global_expert, src_token_idx, slot_bytes] 布局。
                    if constexpr (kUseLogFMT) {
                        // LogFMT 模式下先读取 slot 开头的 amax/amin 元数据，计算变长 payload 布局。
                        logfmt_check_amaxmin<kNumDivisions / 2, kNumSendUnrolls, kNumRecvUnrolls>(
                            buffer,  // 当前 expert/token slot 的 RDMA 接收 buffer 起点。
                            reinterpret_cast<float2*>(log_amax_buffers[stage_idx]),  // 当前 stage 的 amax 输出。
                            reinterpret_cast<float2*>(log_amin_buffers[stage_idx]),  // 当前 stage 的 amin 输出。
                            cast_info_buffers[stage_idx],  // 当前 stage 的压缩分片信息。
                            lane_id);  // 当前 warp lane，用于分片读取元数据。
                    }
                    if (elect_one_sync()) {
                        // 一个 elected lane 计算当前 slot 需要 TMA load 的字节数并发起 load。
                        int num_casted = 0;
                        // num_casted 是前 num_decode_warps 个分片中采用 LogFMT 的数量。
                        if constexpr (kUseLogFMT) {
                            // 从最后一个 decode warp 的 cast_info 读出总压缩分片数。
                            const auto info = cast_info_buffers[stage_idx][num_decode_warps - 1];
                            // info 高位为此前缀数，低 bit 为自己是否压缩。
                            num_casted = (info >> 1) + (info & 1);
                            // 总压缩数 = 前缀数 + 当前分片是否压缩。
                        }
                        int num_tma_bytes = num_casted * kNumLogFMTPerWarpBytes + (num_decode_warps - num_casted) * kNumBF16PerWarpBytes;
                        // 变长 payload 总字节数 = 压缩分片字节 + 未压缩 BF16 分片字节。
                        tma_load_1d(
                            tma_ld_buffers[stage_idx], buffer + (kUseLogFMT ? kNumMetaBytes : 0), full_barriers[stage_idx], num_tma_bytes);
                        // 跳过 metadata 后把 payload 加载到 shared stage。
                        mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_tma_bytes);
                        // 标记 full barrier 的期望传输字节数。
                    }
                    __syncwarp();
                    // 确保 stage load 已提交后轮转 stage。
                    stage_idx = (stage_idx + 1) % kNumStages;
                    // 下一 top-k 使用下一个 stage，形成三阶段流水。
                }
            }
        } else {
            // reduction warp：等待 TMA load，把各 top-k expert 输出加权累加并写 combined_x。
            // Reduction warps
            float topk_weights_by_lane;
            // lane0..num_topk-1 对应当前 token 的 top-k weight。
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                // 与 TMA load warp 使用相同 token 分配方式，保证同步顺序一致。
                if (lane_id < num_topk) {
                    // 每个 lane 读取一个 top-k expert id 和 weight。
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                    // topk_idx 逻辑形状为 [num_combined_tokens, num_topk]。
                    topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
                    // topk_weights 与 topk_idx 同形状，combine 时用于加权求和。
                }
                __syncwarp();
                // 确保 top-k id/weight 已在 warp 内可广播。

                float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                // 当前 lane 负责的 hidden 分片累加器，元素数 = 8 BF16/int4 * kNumRecvUnrolls。
                for (int i = 0; i < num_topk; ++i) {
                    // 依次消费每个 top-k expert 的输出。
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    // 从第 i 个 lane 广播 expert id。
                    if (topk_idx_reg < 0)
                        // 无效 expert 跳过。
                        continue;
                    if (is_rank_masked(mask_buffer_ptr, topk_idx_reg / num_local_experts))
                        // expert 所属 rank 已 mask 时不累加。
                        continue;
                    const auto topk_weight = __shfl_sync(0xffffffff, topk_weights_by_lane, i);
                    // 从第 i 个 lane 广播 expert weight。

                    mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                    // 等待 TMA load warp 把第 i 个 expert 的 payload 放入当前 stage。
                    if constexpr (kUseLogFMT) {
                        // LogFMT 模式需根据 cast_info 选择压缩/原始解码路径。
                        const auto info = cast_info_buffers[stage_idx][decode_warp_idx];
                        // 当前 decode warp 对应分片的压缩状态和前缀偏移。
                        bool enable_cast = info & 1;
                        // 低 bit 为 1 表示该分片采用 LogFMT。
                        int num_casted_prefix = info >> 1;
                        // 当前 decode warp 之前有多少分片是 LogFMT。
                        int tma_offset =
                            kNumLogFMTPerWarpBytes * num_casted_prefix + kNumBF16PerWarpBytes * (decode_warp_idx - num_casted_prefix);
                        // 变长 payload 中当前 decode warp 分片的起始偏移。
                        int division_idx = decode_warp_idx * (kNumRecvUnrolls * 2) + lane_id * kNumRecvUnrolls / 16;
                        // 当前 lane 对应的 128-channel division，读取相应 log_amax/log_amin。
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset +
                                                        (enable_cast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / 32 * lane_id),  // 当前 lane 的 payload 分片地址。
                            combined_values,  // 当前 token 的局部累加寄存器数组。
                            log_amax_buffers[stage_idx][division_idx],  // LogFMT 解码所需 amax。
                            log_amin_buffers[stage_idx][division_idx],  // LogFMT 解码所需 amin。
                            enable_cast,  // 当前分片是否采用 LogFMT 压缩。
                            topk_weight);  // 当前 expert 对该 token 的 top-k 权重。
                        // 解码当前 expert 的分片并乘 topk_weight 累加到 combined_values。
                    } else {
                        // 非 LogFMT 模式直接按 BF16 分片读取。
                        int tma_offset = kNumBF16PerWarpBytes * decode_warp_idx;
                        // 当前 decode warp 在 fixed-size payload 中的偏移。
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset + kNumBF16PerWarpBytes / 32 * lane_id),
                            combined_values,
                            // 中文：BF16 路径没有 log scale，上下界参数传 0 且 enable_cast=false。
                            0,
                            // 中文：第二个 0 是 log_amin 占位，BF16 不需要动态范围。
                            0,
                            false,
                            topk_weight);
                        // BF16 解码不需要 log_amax/log_amin。
                    }  // 中文：结束 LogFMT/BF16 解码分支。

                    if (elect_one_sync())
                        // 当前 stage 已被本 decode warp 消费，通知 empty barrier。
                        mbarrier_arrive(empty_barriers[stage_idx]);
                    stage_idx = (stage_idx + 1) % kNumStages;
                    // 规约 warp 和 TMA load warp 使用同样 stage 轮转顺序。
                }
                tma_store_wait<0>();
                // 等待此前写 combined_x 的 TMA store 完成，避免复用 store buffer 时冲突。

                #pragma unroll
                for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
                    // 将 FP32 累加结果压回 BF16 pair 存入 shared store buffer。
                    auto combined_pack = __nv_bfloat162(combined_values[k * 2], combined_values[k * 2 + 1]);
                    // 两个 FP32 转成一个 BF16 pair。
                    tma_st_buffers[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] = *reinterpret_cast<uint32_t*>(&combined_pack);
                    // 按 lane 连续写入 shared，后续由 TMA store 写回 combined_x。
                }
                tma_store_fence();
                // 保证 shared store buffer 写完后 TMA store 可读。
                if (elect_one_sync()) {
                    // 每个 decode warp 一个 elected lane 发起 shared->global store。
                    tma_store_1d(tma_st_buffers[decode_warp_idx],
                                 static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 + decode_warp_idx * kNumRecvUnrolls * 32,
                                 kNumBF16PerWarpBytes);
                    // 写回 combined_x[token_idx, decode_warp_idx 对应 hidden 分片]。
                }  // 中文：结束当前 token 的所有 decode warp 写回。
                __syncwarp();
                // 确保 store 提交后再处理下一个 token。
            }  // 中文：结束当前 decode warp 的 token 循环。
        }  // 中文：结束当前 warp group 的 combine 计算。
    }  // 中文：结束低延迟 combine device kernel。
}

void combine(void* combined_x,  // 输出 combine 后 BF16 token [num_combined_tokens, hidden]。
             void* rdma_recv_x,  // RDMA 接收 buffer，存放来自各 rank 的 expert 输出。
             int* rdma_recv_flag,  // RDMA 接收完成 flag。
             void* rdma_send_x,  // RDMA 发送 buffer；zero_copy=false 时由 x 拷入。
             const void* x,  // 本地 expert 输出，形状 [local_experts, R*Tmax, hidden]。
             const topk_idx_t* topk_idx,  // 原 token 路由 [num_combined_tokens,num_topk]。
             const float* topk_weights,  // 原 token 权重 [num_combined_tokens,num_topk]。
             const int* src_info,  // dispatch 返回的源 token 信息。
             const int64_t* layout_range,  // dispatch 返回的每 expert 有效 slot 范围。
             int* mask_buffer_ptr,  // 可选 rank mask buffer。
             int64_t* combine_wait_recv_cost_stats,  // 可选 combine 等待耗时统计。
             int* next_clean,  // 下一轮需要清理的 int 区。
             int num_next_clean_int,  // next_clean 元素数。
             int num_combined_tokens,  // 要还原的原 token 数。
             int hidden,  // hidden 维度。
             int num_max_dispatch_tokens_per_rank,  // dispatch 固定容量 Tmax。
             int num_topk,  // top-k 路数。
             int num_experts,  // 全局 expert 数。
             int rank,  // 当前 rank。
             int num_ranks,  // 全局 rank 数。
             bool use_logfmt,  // 是否对发送 payload 使用 LogFMT 压缩。
             void* workspace,  // combine 临时 workspace。
             int num_device_sms,  // kernel 可使用 SM 数。
             cudaStream_t stream,  // CUDA launch stream。
             int phases,  // 当前双缓冲 phase。
             bool zero_copy) {  // true 表示 x 已经位于下一块 RDMA send buffer。
    // Host wrapper 负责把运行时 hidden/use_logfmt/phase 组合映射到具体模板实例。
    // 与 dispatch wrapper 一样，真正的数据布局仍由 device kernel 解释；这里重点检查
    // top-k 上限、workspace、shared memory 预算，并把每个参数按固定顺序透传给 kernel。
    // 典型 low-latency decode 会先调用 SEND phase，把 expert 输出回传到源 rank，
    // 再调用 RECV phase 在源 rank 上按 topk_weights 做规约；也可以一次性传 SEND|RECV。
    constexpr int kNumMaxTopk = 11;
    // combine 与 dispatch 使用相同 top-k 上限，避免模板实例爆炸。
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    // 发送/flag 等待阶段每个 SM 负责的 expert group 数。
    const int num_warps_per_group = 32 / num_warp_groups;
    // 每个 expert group 可用的 warp 数。
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    // 接收规约阶段估算每个 SM 处理多少 token，用于确定 grid size。
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm >= 0);
    // 检查 expert/SM 分配合法。

    const auto num_warps = num_warp_groups * num_warps_per_group;
    // 启动的 warp 数；blockDim = num_warps*32。
    const auto num_sms =
        max(ceil_div(num_experts, num_warp_groups), num_recv_per_sm == 0 ? 1 : ceil_div(num_combined_tokens, num_recv_per_sm));
    // grid size 同时覆盖发送阶段所有 expert 和接收阶段所有 token。

    // Check workspace
    auto atomic_clean_flag = static_cast<int*>(workspace);
    // workspace 只需要一个 int 作为 combine 发送阶段的清理/完成协调 flag。
    EP_HOST_ASSERT(sizeof(int) <= LEGACY_NUM_WORKSPACE_BYTES);
    // 固定 workspace 必须至少容纳该 flag。
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);
    // 运行时 top-k 不能超过编译期模板上限。

    // Online cast cannot use zero-copy
    EP_HOST_ASSERT(not(zero_copy and use_logfmt));
    // LogFMT 需要在线编码写 staging/目标 buffer，不能和“数据已在 send buffer”的 zero_copy 假设同时成立。

    constexpr int kNumStages = 3;
    // host 侧 shared memory 估算与 kernel 中 TMA stage 数一致。
    constexpr int kNumMaxUnrolls = 4;
    // 发送端最大展开因子，shared memory 按最大情况预留。
    constexpr int kMaxNumGroups = 2;
    // 接收规约每 CTA 最多两个 group，与 kernel 内 kMaxNumGroups 一致。

    // Send buffer size
    const int num_meta_bytes = hidden / 128 * 4;
    // 每 128 hidden channel 一个 nv_bfloat162(4B) 元数据。
    const int num_send_tma_bytes = 32 * sizeof(int4) * kNumMaxUnrolls + 16;
    // 发送阶段每 stage 的最大数据区 + 16B barrier 空间。
    const int smem_send_size = num_warps * (kNumStages * num_send_tma_bytes + num_meta_bytes);
    // 发送阶段为每个 warp 独占一份 TMA stage 和 meta buffer。

    // Receive buffer size
    const int num_recv_tma_bytes = 16 + hidden * 2;
    // 接收阶段每 stage 的 barrier 空间 + 一个 token 的 BF16 payload 最大空间。
    const int smem_recv_size = kMaxNumGroups * (kNumStages * num_recv_tma_bytes + hidden * 2 + kNumStages * num_meta_bytes * 3);
    // 接收阶段每 group 需要 load stages、store buffer、log_amax/log_amin/cast_info 三组元数据。

    // Total requirement
    const int smem_size = max(smem_send_size, smem_recv_size);
    // 动态 shared memory 取发送/接收两阶段需求的最大值。

#define COMBINE_LAUNCH_CASE(hidden)                                                                                                \
    /* hidden 是模板常量，便于展开 TMA/LogFMT 的 hidden 维循环。 */                                                                  \
    {                                                                                                                              \
        auto combine_func =                                                                                                        \
            use_logfmt ? combine<true, hidden, kNumMaxTopk, kNumMaxUnrolls> : combine<false, hidden, kNumMaxTopk, kNumMaxUnrolls>; \
        /* 根据 use_logfmt 选择是否启用在线 LogFMT 压缩/解码。 */                                                                    \
        SET_SHARED_MEMORY_FOR_TMA(combine_func);                                                                                   \
        /* 为使用 TMA 的 kernel 设置动态 shared memory 属性。 */                                                                     \
        LAUNCH_KERNEL(&cfg,                                                                                                        \
                      combine_func,                                                                                                 \
                      /* combined_x: [num_combined_tokens, hidden]，最终写回源 rank 的 BF16 输出。 */                         \
                      combined_x,                                                                                                   \
                      /* rdma_recv_x/flag: 接收其它 rank combine 返回 payload 与到达标志。 */                                \
                      rdma_recv_x,                                                                                                  \
                      rdma_recv_flag,                                                                                               \
                      /* rdma_send_x: 本 rank 发送 expert 输出前使用的 RDMA staging buffer。 */                               \
                      rdma_send_x,                                                                                                  \
                      /* x: 本地 expert 计算后的固定容量布局；zero_copy=true 时可能已在 RDMA buffer 中。 */                  \
                      x,                                                                                                            \
                      /* topk_idx/topk_weights: 原 token 路由和权重，用于接收端加权规约。 */                                 \
                      topk_idx,                                                                                                     \
                      topk_weights,                                                                                                 \
                      /* src_info/layout_range: low_latency_dispatch 返回的 handle，描述每个 expert slot 的来源。 */         \
                      src_info,                                                                                                     \
                      layout_range,                                                                                                 \
                      /* mask_buffer_ptr: shrink/超时容错时跳过被 mask 的 rank。 */                                          \
                      mask_buffer_ptr,                                                                                              \
                      /* combine_wait_recv_cost_stats: 可选 [num_ranks,num_ranks] 等待耗时统计。 */                          \
                      combine_wait_recv_cost_stats,                                                                                 \
                      /* next_clean/atomic_clean_flag: 下一轮低延迟 buffer 清理协作状态。 */                                 \
                      next_clean,                                                                                                   \
                      num_next_clean_int,                                                                                           \
                      atomic_clean_flag,                                                                                            \
                      /* num_combined_tokens/hidden/num_topk 决定输出和路由矩阵维度。 */                                     \
                      num_combined_tokens,                                                                                          \
                      hidden,                                                                                                       \
                      num_topk,                                                                                                     \
                      /* Tmax/num_experts/rank/num_ranks 描述固定容量布局和全局拓扑。 */                                     \
                      num_max_dispatch_tokens_per_rank,                                                                             \
                      num_experts,                                                                                                  \
                      /* rank/num_ranks 共同决定原 token rank 与本地 expert 分片。 */                                           \
                      rank,                                                                                                         \
                      num_ranks,                                                                                                    \
                      /* warp group 配置控制发送/接收/规约工作划分。 */                                                     \
                      num_warp_groups,                                                                                              \
                      num_warps_per_group,                                                                                          \
                      /* phases 选择低延迟双缓冲相位，zero_copy 控制是否跳过本地复制。 */                                    \
                      phases,                                                                                                       \
                      zero_copy);                                                                                                   \
    }                                                                                                                              \
    /* 中文：每个 hidden/topk/format case 只启动一个模板实例，随后退出 switch。 */                                             \
    break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    // num_sms 和 num_warps 在上面已经同时满足发送和接收阶段的覆盖需求。
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
    // 按运行时 hidden 选择预编译 combine 模板实例。
#undef COMBINE_LAUNCH_CASE
    // 避免宏定义泄漏到后续 mask buffer helper。
}

template <int kNumThreads>  // mask 查询 kernel 的线程数。
__launch_bounds__(kNumThreads, 1) __global__ void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor) {
    // query_mask_buffer 把设备端 mask buffer 复制到用户可观察的 mask_tensor。
    const auto num_sms = static_cast<int>(gridDim.x);
    // 当前 helper 通常只启动一个 block，但这里保留 grid-stride 写法。
    const auto sm_id = static_cast<int>(blockIdx.x);
    // block id。
    const auto num_threads = num_sms * kNumThreads;
    // 全 grid 总线程数。
    const auto thread_id = sm_id * kNumThreads + static_cast<int>(threadIdx.x);
    // 线性线程 id，用于覆盖 rank 维。
    for (int rank_id = thread_id; rank_id < num_ranks; rank_id += num_threads)
        // mask_tensor[rank_id] 为 1 表示该 rank 已被低延迟路径 mask。
        mask_tensor[rank_id] = mask_buffer_ptr[rank_id];
}

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor, cudaStream_t stream) {
    // Host wrapper：查询 mask 状态，供 Python 侧诊断或恢复策略使用。
    constexpr int num_sms = 1;
    // mask buffer 很小，一个 block 足够。
    constexpr int kNumThreads = 1024;
    // 1024 线程覆盖大量 rank 也足够。
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    // 设置查询 kernel 的 launch 配置。
    LAUNCH_KERNEL(&cfg, query_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks, mask_tensor);
    // 异步启动查询 kernel。
}

template <int kNumThreads>  // mask 单点更新 kernel 的线程数。
__launch_bounds__(kNumThreads, 1) __global__ void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask) {
    // update_mask_buffer 手动设置某个 rank 的 mask 状态，用于测试或外部恢复控制。
    const auto sm_id = static_cast<int>(blockIdx.x);
    // 当前 block id。
    const auto thread_id = static_cast<int>(threadIdx.x);
    // block 内线程 id。
    if (sm_id == 0 and thread_id == 0)
        // 只用一个线程执行 atomicExch，避免重复写。
        // atomicExch 让同一 stream 上可能并发的查询/更新看到一致的 0/1 状态。
        atomicExch(mask_buffer_ptr + rank_to_mask, mask ? 1 : 0);
        // mask=true 写 1，mask=false 写 0。
}

void update_mask_buffer(int* mask_buffer_ptr, int rank, bool mask, cudaStream_t stream) {
    // Host wrapper：把 rank 的 mask 状态更新提交到给定 stream。
    constexpr int num_sms = 1;
    // 单 block 足够。
    constexpr int kNumThreads = 32;
    // 一个 warp 足够执行单点更新。
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    // 设置更新 kernel 的 launch 配置。
    LAUNCH_KERNEL(&cfg, update_mask_buffer<kNumThreads>, mask_buffer_ptr, rank, mask);
    // 异步启动更新 kernel。
}

template <int kNumThreads>  // mask 清理 kernel 的线程数。
__launch_bounds__(kNumThreads, 1) __global__ void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks) {
    // clean_mask_buffer 清空所有 rank 的 mask 状态。
    auto thread_id = static_cast<int>(threadIdx.x);
    // 单 block 内线程编号。
    #pragma unroll
    for (int i = thread_id; i < num_ranks; i += kNumThreads)
        // 每个线程按 stride 清零多个 rank 槽。
        mask_buffer_ptr[i] = 0;
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream) {
    // Host wrapper：在 stream 上清空 mask buffer，通常在新一轮任务前调用。
    constexpr int num_sms = 1;
    // mask buffer 只有 num_ranks 个 int，单 block 足够。
    constexpr int kNumThreads = 32;
    // 一个 warp 即可覆盖常见 rank 数。
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    // 设置清理 kernel 的 launch 配置。
    LAUNCH_KERNEL(&cfg, clean_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks);
    // 异步启动清理 kernel。
}

}  // namespace internode_ll
// internode_ll 命名空间结束。

}  // namespace deep_ep::legacy
// deep_ep::legacy 命名空间结束。
