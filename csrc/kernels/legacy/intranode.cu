#include "buffer.cuh"  // 引入 legacy channel buffer 布局辅助函数。
#include "compiled.cuh"  // 引入 legacy 编译期常量、类型和 rank switch 宏。
#include "launch.cuh"  // 引入 CUDA kernel 发射宏。
#include "utils.cuh"  // 引入 barrier、warp reduce、TMA 等工具函数。

namespace deep_ep::legacy {  // legacy 通信实现命名空间。

namespace intranode {  // 单节点 NVLink dispatch/combine kernel 命名空间。

// legacy intranode 路径只处理单节点 NVLink 域。整体流程是：
// layout.cu 先得到 is_token_in_rank/rank_prefix_matrix；
// notify_dispatch 在 peer buffer 中交换每个 rank 要收多少 token；
// dispatch 按 channel 把 x/scales/topk 元数据写到目标 rank；
// combine 再根据 recv_src_idx 和 prefix 矩阵把专家输出加回源 token。
// 与 legacy internode.cu 相比，本文件没有 RDMA 层：buffer_ptrs 都是 CUDA IPC/NVLink
// 可直接访问的同节点 peer buffer；因此 handle 主要保存 rank/channel prefix 和 send_head。
// 与 internode_ll.cu 相比，本文件服务 high-throughput 批量 token，先统计精确接收量再分配
// recv_x；low-latency 路径则使用固定容量专家槽位和 phase 双缓冲，牺牲一部分空间换延迟。

template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {  // 节点内 GPU barrier kernel。
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);  // 使用每个 peer 暴露的 signal 做 block 级同步。
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                  \
/* 中文：根据运行时 rank 数实例化 barrier<ranks>，这样 barrier_block 内循环可编译期展开。 */ \
LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank); \
break

    SETUP_LAUNCH_CONFIG(1, 32, stream);  // 单 block 32 线程足够完成 barrier signal 操作。
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);  // 根据运行时 num_ranks 选择编译期模板 rank 数。
#undef BARRIER_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,  // 输入 [kNumRanks]：本 rank 发往各 rank 的 token 数。
                                int* moe_recv_counter_mapped,  // 输出 mapped host 指针：本 rank 总接收 token 数。
                                const int* num_tokens_per_expert,  // 输入 [num_experts]：本 rank 发往各 expert 的 token 数。
                                int* moe_recv_expert_counter_mapped,  // 输出 mapped host 指针：[num_local_experts]。
                                int num_experts,  // 全局 expert 数。
                                int num_tokens,  // 本 rank 输入 token 数。
                                int num_channels,  // dispatch channel 数。
                                const bool* is_token_in_rank,  // 输入 [num_tokens,kNumRanks] 路由布尔矩阵。
                                int* channel_prefix_matrix,  // 输出 [kNumRanks,num_channels] channel prefix。
                                int* rank_prefix_matrix_copy,  // 输出 [kNumRanks,kNumRanks] rank prefix 副本。
                                int num_memset_int,  // 需要清零的队列控制字 int 数。
                                int expert_alignment,  // expert token 数对齐粒度。
                                void** buffer_ptrs,  // GPU 指针表：每个 peer 的 NVL buffer。
                                int** barrier_signal_ptrs,  // GPU 指针表：每个 peer 的 barrier signal。
                                int rank) {  // 当前节点内 rank。
    auto sm_id = static_cast<int>(blockIdx.x);  // block 0 负责控制面，其余 block 负责 dst rank channel prefix。
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);  // 线程 id 和线程数。
    auto lane_id = thread_id % 32, warp_id = thread_id / 32, num_warps = num_threads / 32;  // warp 内 lane、warp id、warp 数。

    if (sm_id == 0) {
        // 0 号 SM 负责全局控制面：先做一次 peer barrier，保证所有 rank 都已经完成
        // layout 统计并准备好 buffer_ptrs/barrier_signal_ptrs。
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        int *per_rank_buffer, *per_expert_buffer;  // peer buffer 内 rank/expert metadata 两段指针。
        if (thread_id < kNumRanks) {
            per_rank_buffer = static_cast<int*>(buffer_ptrs[thread_id]);  // 指向 peer thread_id 的 buffer 开头。
            per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;  // rank prefix 矩阵之后是 expert 计数矩阵。
        }

        // 写入本 rank 的发送计数后，每个 peer 都能在自己的 buffer 中看到矩阵：
        //   per_rank_buffer[i, j] = rank i 将发送给 rank j 的 token 数
        //   per_expert_buffer[i, e] = rank i 将发送给本地 expert e 的 token 数
        // per_rank_buffer 形状 [kNumRanks, kNumRanks]，per_expert_buffer 形状
        // [kNumRanks, num_experts_per_rank]。
        int num_experts_per_rank = num_experts / kNumRanks;  // 每个 rank 的本地 expert 数。
        if (thread_id < kNumRanks) {
            per_rank_buffer[rank * kNumRanks + thread_id] = num_tokens_per_rank[thread_id];  // 写本 rank 发往 thread_id rank 的 token 数。
            #pragma unroll
            for (int i = 0; i < num_experts_per_rank; ++i)
                per_expert_buffer[rank * num_experts_per_rank + i] = num_tokens_per_expert[thread_id * num_experts_per_rank + i];  // 写本 rank 发往 peer 本地 expert i 的 token 数。
        }

        // Wait for all ranks to be finished
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);  // 等所有 rank 都把自己的计数写入所有 peer buffer。

        // 对每个目标 rank 做前缀和：local_per_rank_buffer[i, j] 变成 rank 0..i
        // 发往 rank j 的累计 token 数。thread_id==rank 写回本 rank 总接收数给 CPU mapped counter。
        auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);  // 当前 rank 自己 buffer 中收集到的 [src_rank,dst_rank] 计数矩阵。
        if (thread_id < kNumRanks) {
            #pragma unroll
            for (int i = 1; i < kNumRanks; ++i)
                local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];  // 对每个 dst_rank 沿 src_rank 做前缀和。
            if (thread_id == rank)
                *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];  // 写回本 rank 总接收 token 数给 CPU。
        }

        // expert 级计数按发送 rank 求和，并按 expert_alignment 向上对齐。输出
        // moe_recv_expert_counter_mapped [num_experts_per_rank]，Python 用它分配每个 expert 的容量。
        auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;  // 当前 rank buffer 中的 [src_rank,local_expert] 计数矩阵。
        if (thread_id < num_experts_per_rank) {
            int sum = 0;  // 当前本地 expert 的接收 token 总数。
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++i)
                sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];  // 汇总所有源 rank 发给该 expert 的 token 数。
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;  // 向上对齐，便于 expert 侧固定 tile/容量。
            moe_recv_expert_counter_mapped[thread_id] = sum;  // 写回 CPU mapped expert 计数。
        }
        __syncthreads();  // 等计数和 prefix 都写完再清理临时区域。

        // Copy rank size prefix matrix to another tensor
        #pragma unroll
        for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
            rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];  // 保存 rank prefix 到输出 tensor，后续 combine/cached dispatch 复用。

        // 清理 per_expert_buffer 后面的队列/计数区域，dispatch kernel 会把它当作
        // send_head/channel head 复用。
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            local_per_expert_buffer[i] = 0;  // 清零后续 dispatch 队列元数据区。

        // Barrier
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);  // 确认所有 peer buffer 控制区清理完成。
    } else {
        int dst_rank = sm_id - 1;  // block 1..kNumRanks 分别负责一个目标 rank。
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;  // 当前 channel 负责扫描的 token 半开区间。
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);  // 均分 token 给 channel。

            // 其他 SM 分别负责一个目标 rank 的 channel prefix。channel_prefix_matrix
            // 形状 [kNumRanks, num_channels]，每项是当前 channel 及之前 channel 中
            // 需要发给 dst_rank 的 token 累计数。
            int count = 0;  // 当前 warp 对该 dst_rank/channel 的 token 计数。
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32)
                count += is_token_in_rank[i * kNumRanks + dst_rank];  // 统计该 token 是否需要发给 dst_rank。
            count = warp_reduce_sum(count);  // warp 内求和得到 channel 计数。
            if (elect_one_sync())
                channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;  // 每个 warp 选一个 lane 写回计数。
        }
        __syncthreads();  // 等所有 channel 计数写入矩阵。

        // 每个 dst_rank 的 channel 内计数转为前缀和，dispatch 时可用
        // prefix[channel-1]..prefix[channel] 定位该 channel 的写入段。
        if (thread_id == 0) {
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                channel_prefix_matrix[dst_rank * num_channels + i] += channel_prefix_matrix[dst_rank * num_channels + i - 1];  // 将 channel count 原地转成前缀和。
        }
    }
}

void notify_dispatch(const int* num_tokens_per_rank,  // 输入 [num_ranks] rank 计数。
                     int* moe_recv_counter_mapped,  // 输出 mapped 总接收计数。
                     int num_ranks,  // 节点内 rank 数。
                     const int* num_tokens_per_expert,  // 输入 [num_experts] expert 计数。
                     int* moe_recv_expert_counter_mapped,  // 输出 mapped 本地 expert 计数。
                     int num_experts,  // 全局 expert 数。
                     int num_tokens,  // 输入 token 数。
                     const bool* is_token_in_rank,  // 输入 token-rank 布尔矩阵。
                     int* channel_prefix_matrix,  // 输出 channel prefix。
                     int* rank_prefix_matrix_copy,  // 输出 rank prefix。
                     int num_memset_int,  // 清零 int 数。
                     int expert_alignment,  // expert 对齐粒度。
                     void** buffer_ptrs,  // peer buffer 指针表。
                     int** barrier_signal_ptrs,  // peer barrier 指针表。
                     int rank,  // 当前 rank。
                     cudaStream_t stream,  // CUDA 发射流。
                     int num_channels) {  // channel 数。
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)        \
    LAUNCH_KERNEL(&cfg,                           \
                  notify_dispatch<ranks>,         \
                  num_tokens_per_rank,            /* rank 级发送计数。 */ \
                  moe_recv_counter_mapped,        /* mapped rank 接收计数输出。 */ \
                  num_tokens_per_expert,          /* expert 级发送计数。 */ \
                  moe_recv_expert_counter_mapped, /* mapped expert 接收计数输出。 */ \
                  num_experts,                    /* 全局 expert 数。 */ \
                  num_tokens,                     /* 输入 token 数。 */ \
                  num_channels,                   /* 通信 channel 数。 */ \
                  is_token_in_rank,               /* token-rank bool 矩阵。 */ \
                  channel_prefix_matrix,          /* channel prefix 输出。 */ \
                  rank_prefix_matrix_copy,        /* rank prefix 副本输出。 */ \
                  num_memset_int,                 /* 需要清零的 int 数。 */ \
                  expert_alignment,               /* expert token 对齐粒度。 */ \
                  buffer_ptrs,                    /* peer buffer 指针表。 */ \
                  barrier_signal_ptrs,            /* peer barrier 信号表。 */ \
                  rank);                          /* 当前 rank。 */ \
    break

    constexpr int kNumThreads = 128;  // notify kernel 使用 128 线程。
    EP_HOST_ASSERT(num_experts % num_ranks == 0);  // expert 必须按 rank 均分。
    EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads and num_ranks <= kNumThreads);  // 本地 expert 数和 rank 数需能由线程直接覆盖。

    SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);  // block0 控制面 + 每个目标 rank 一个 block。
    SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);  // 根据 num_ranks 选择模板实例。
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_dispatch(
    const int* rank_prefix_matrix,  // 输入 cached rank prefix，形状 [kNumRanks,kNumRanks]。
    int num_memset_int,  // 需要清零的队列控制字数量。
    void** buffer_ptrs,  // peer buffer 指针表。
    int** barrier_signal_ptrs,  // peer barrier 指针表。
    int rank) {  // 当前 rank。
    // cached 版本不重新根据 topk 统计 layout，只把调用方保存的 rank_prefix_matrix
    // 拷回 peer buffer，并清理临时队列。适合前向 dispatch 后，反向或下一轮复用相同路由。
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);  // 先做带 release/acquire 语义的 peer barrier。

    // Copy and clean
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);  // 线程 id 与线程数。
    auto ptr = static_cast<int*>(buffer_ptrs[rank]);  // 当前 rank 本地 NVL buffer 的 int 视图。
    #pragma unroll
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
        ptr[i] = rank_prefix_matrix[i];  // 把 cached rank prefix 拷回 buffer 开头。
    #pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads)
        ptr[kNumRanks * kNumRanks + i] = 0;  // 清零 rank prefix 后面的队列控制区。

    // Barrier after cleaning
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);  // 等所有 rank 都完成拷贝和清理。
}

void cached_notify_dispatch(const int* rank_prefix_matrix,  // 输入 cached rank prefix。
                            int num_memset_int,  // 清零 int 数。
                            void** buffer_ptrs,  // peer buffer 指针表。
                            int** barrier_signal_ptrs,  // peer barrier 指针表。
                            int rank,  // 当前 rank。
                            int num_ranks,  // 节点内 rank 数。
                            cudaStream_t stream) {  // CUDA 发射流。
#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                                                   \
    LAUNCH_KERNEL(&cfg, cached_notify_dispatch<ranks>, rank_prefix_matrix, num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 128, stream);  // cached notify 只需要一个 block 完成拷贝和清理。
    SWITCH_RANKS(CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);  // 按 num_ranks 选择模板实例。
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch(int4* recv_x,  // 输出接收 token 特征，按 int4 向量写。
                                                           float* recv_x_scales,  // 输出接收 FP8 scales，可为空。
                                                           int* recv_src_idx,  // 输出源 token index，[num_recv_tokens]。
                                                           topk_idx_t* recv_topk_idx,  // 输出接收 topk_idx，可为空。
                                                           float* recv_topk_weights,  // 输出接收 topk 权重，可为空。
                                                           int* recv_channel_offset,  // 输出接收侧 channel prefix/offset。
                                                           int* send_head,  // 输出发送槽位，[num_tokens,kNumRanks]。
                                                           const int4* x,  // 输入 token 特征，按 int4 读取。
                                                           const float* x_scales,  // 输入 FP8 scales，可为空。
                                                           const topk_idx_t* topk_idx,  // 输入 topk_idx，可为空。
                                                           const float* topk_weights,  // 输入 topk 权重，可为空。
                                                           const bool* is_token_in_rank,  // 输入 token-rank 路由布尔表。
                                                           const int* channel_prefix_matrix,  // 输入 [kNumRanks,num_channels] channel prefix。
                                                           int num_tokens,  // 输入 token 数。
                                                           int num_worst_tokens,  // worst-case token 容量。
                                                           int hidden_int4,  // hidden 以 int4 为单位的长度。
                                                           int num_topk,  // top-k 路数。
                                                           int num_experts,  // 全局 expert 数。
                                                           int num_scales,  // FP8 scale 数。
                                                           int scale_token_stride,  // scale token stride。
                                                           int scale_hidden_stride,  // scale hidden stride。
                                                           void** buffer_ptrs,  // peer buffer 指针表。
                                                           int rank,  // 当前 rank。
                                                           int num_max_send_tokens,  // 每 channel send chunk token 上限。
                                                           int num_recv_buffer_tokens) {  // 每 channel recv chunk token 上限。
    // dispatch 使用成对 SM：偶数 block 是 sender，奇数 block 是 receiver。
    // sender 扫描本 rank 的 token，写入目标 rank 暴露的环形 channel buffer；
    // receiver 读取自己 buffer 中的 token，整理到 recv_x [num_recv_tokens, hidden]。
    const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);  // 总 block 数和当前 block id。
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();  // 线程 id 和 warp lane id。
    const bool is_sender = sm_id % 2 == 0;  // 偶数 block 负责发送，奇数 block 负责接收。
    EP_DEVICE_ASSERT(num_sms % 2 == 0);  // block 数必须成对。

    // Several warps are response for a single rank
    const auto num_threads_per_rank = kNumThreads / kNumRanks;  // 每个 rank 分到的线程数。
    const auto num_channels = num_sms / 2;  // channel 数等于 block 对数。
    const auto responsible_rank = (static_cast<int>(thread_id)) / num_threads_per_rank;  // 当前线程负责的 peer rank。
    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    const auto responsible_channel = sm_id / 2;  // 当前 block 对应的 channel id。

    int num_experts_per_rank = num_experts / kNumRanks;  // 每个 rank 的 expert 数。
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);  // 有 topk 时必须能映射 expert->rank。
    EP_DEVICE_ASSERT(num_topk <= 32);  // top-k 不超过 warp lane 数，便于 warp 内处理。
    EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));  // 输入 topk_idx/weights 必须成对。
    EP_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));  // 输出 topk_idx/weights 必须成对。

    // buffer_ptrs[peer] 指向 peer 暴露的 NVLink buffer。rank_prefix_matrix 占据开头
    // kNumRanks*kNumRanks*sizeof(int)，后续按 channel 元数据和数据 payload 顺序切分。
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[is_sender ? responsible_rank : rank]) +
                                       kNumRanks * kNumRanks * sizeof(int));  // 跳过 rank_prefix_matrix，定位到 channel 元数据区域。
    int target_rank = is_sender ? rank : responsible_rank;  // sender 写入目标 peer buffer 时 target_rank=自己，receiver 读本地 buffer 中的 peer rank。
    auto num_channels_total = num_channels * kNumRanks;  // 所有 channel-rank 组合数。
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;  // 当前 channel/rank 在线性元数据表中的偏移。

    // channel 元数据都存放在接收端 buffer：sender 只推进 tail，receiver 只推进 head。
    // start/end offset 使用 -value-1 编码，value=0 也能和“尚未写入”的 0 区分。
    // `start_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `end_offset`: kNumChannels * kNumRanks * sizeof(int)
    // 中文：start/end offset 描述该 source rank 在当前 channel 的 token 区间。
    // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);  // 当前 channel/rank 的起始 offset 元数据。
    auto channel_end_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);  // 当前 channel/rank 的结束 offset 元数据。
    auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);  // receiver 推进的 ring head。
    auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);  // sender 推进的 ring tail。

    // 每个 channel/rank 有独立环形数据区，数据区维度可理解为
    // [num_channels, kNumRanks, num_recv_buffer_tokens, ...]。num_recv_buffer_tokens
    // 是环形队列容量，num_max_send_tokens 是 sender 每次最多批量写入的 token 数。
    // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
    // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
    // 中文：src_idx 保存原 token 下标，combine 时用它把 expert 输出写回原 batch 行。
    // `topk_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(topk_idx_t)
    // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
    // 中文：这些 metadata ring 与 payload ring 使用相同 token slot，下标必须同步推进。
    // `x_scales_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_scales * sizeof(float)
    auto channel_x_buffers = Buffer<int4>(
        ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);  // token payload 环形 buffer。
    auto channel_src_idx_buffers =
        Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);  // 源 token index 环形 buffer。
    auto channel_topk_idx_buffers = Buffer<topk_idx_t>(
        ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);  // topk_idx 环形 buffer。
    auto channel_topk_weights_buffers =
        Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);  // topk weight 环形 buffer。
    auto channel_x_scales_buffers = Buffer<float>(
        ptr, num_channels_total * num_recv_buffer_tokens * num_scales, channel_rank_offset * num_recv_buffer_tokens * num_scales);  // FP8 scales 环形 buffer。

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];  // SM90 TMA 中转 shared memory。
    auto half_hidden_int4 = hidden_int4 / 2;  // TMA 分两半搬运 hidden，降低单次 shared buffer 需求。
    auto half_hidden_bytes = half_hidden_int4 * static_cast<int>(sizeof(int4));  // 半个 hidden 的字节数。
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;  // 每个 warp 独占一段 TMA shared buffer。
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + half_hidden_bytes);  // mbarrier 放在 TMA buffer 尾部。
    uint32_t tma_phase = 0;  // TMA barrier phase。
    if (elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);  // 每个 warp 初始化自己的 TMA barrier。
        fence_barrier_init();  // 确保 barrier 初始化对后续 TMA 可见。
        EP_DEVICE_ASSERT(hidden_int4 % 2 == 0 and half_hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerWarp);  // TMA buffer 容量检查。
    }
    __syncwarp();  // 等待 warp 内 barrier 初始化完成。
#endif

    if (is_sender) {
        // sender 侧按目标 rank 分 warp 组，每个目标 rank 的 warp 只扫描自己负责的
        // token channel，命中 is_token_in_rank 时才把完整 hidden 向量写入对端 buffer。
        constexpr int num_send_warps = kNumThreads / 32;  // sender block 中 warp 总数。
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;  // 每个目标 rank 分配的 warp 数。
        const auto send_thread_id = thread_id;  // sender 线程 id。
        const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;  // 当前线程所在的 rank 内 warp id。
        EP_DEVICE_ASSERT(kNumRanks <= 32);  // rank 数不能超过 warp lane 可表达范围。
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);  // warp 必须能均分给 rank。

        // start/end offset 先写给 receiver：例如 channel 前缀为 [3, 7]，说明本 channel
        // 需要接收 token 序号 3..6。使用负数编码是为了让 receiver 可以用 0 表示未 ready。
        if (send_warp_id_in_rank == 0 and elect_one_sync()) {
            int value = responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0;  // 当前 channel 起始 token 偏移。
            st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);  // 负数编码后写 start offset。
            value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];  // 当前 channel 结束 token 偏移。
            st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1);  // 负数编码后写 end offset。
        }
        __syncwarp();  // 确保 start/end offset 写入完成后再发送数据。

        // Get tasks
        int token_start_idx, token_end_idx;  // 当前 channel 扫描的 token 范围。
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);  // 按 channel 均分 token。

        // Iterate over all tokens and send by chunks
        int cached_channel_tail_idx = 0;  // sender 本地缓存的 ring tail。
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            // NOTES: the head index received by different warps may not be the same
            auto start_time = clock64();  // 记录等待空槽开始时间。
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());  // 当前 ring 已使用槽数。
                    if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)
                        break;  // 剩余槽足够一个发送 chunk。

                    // Rare cases to loop again
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for dispatch senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();  // 触发设备异常，避免 sender 永久等待。
                    }
                }
            }
            __syncwarp();  // 等待空槽检查完成。

            int chunk_token_idx = 0;  // 当前 chunk 内已处理 token 数。
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                // send_head[token, rank] 记录该 token 发往 responsible_rank 时所在的
                // 环形槽 tail index；combine 使用它反向定位专家输出应该写回哪个源 token。
                if (token_idx % num_send_warps_per_rank == send_warp_id_in_rank and elect_one_sync())
                    send_head[token_idx * kNumRanks + responsible_rank] =
                        is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;  // 命中则记录 tail，否则记 -1。

                // Skip if not selected
                if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;  // 不发给该 rank 的 token 直接跳过。
                    continue;  // 继续扫描下一个 token。
                }

                // Get an empty slot
                int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;  // 分配 ring buffer 槽位并推进 tail。
                if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
                    // Copy data
                    auto shifted_channel_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;  // 目标槽位的 payload 起点。
                    auto shifted_x = x + token_idx * hidden_int4;  // 输入 token 的 payload 起点。
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_channel_x_buffers, shifted_x, __ldg, st_na_global);  // warp 并行拷贝 hidden。

                    // Copy source index
                    if (elect_one_sync())
                        channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);  // 写入源 token index。

                    // topk_idx 从全局 expert id 转成本 rank 内局部 expert id，无效 lane
                    // 写 -1 且 weight 写 0。输出形状 [slot, num_topk]。
                    if (lane_id < num_topk) {
                        // Top-k index
                        int recv_expert_begin = responsible_rank * num_experts_per_rank,
                            recv_expert_end = (responsible_rank + 1) * num_experts_per_rank;  // 目标 rank 拥有的 expert id 范围。
                        auto idx_value = __ldg(topk_idx + token_idx * num_topk + lane_id);  // 读取当前 top-k lane 的全局 expert id。
                        idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ? idx_value - recv_expert_begin : -1;  // 转成本地 expert id。
                        channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;  // 写入 topk_idx buffer。

                        // Top-k weights
                        auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);  // 读取 top-k 权重。
                        weight_value = (idx_value >= 0) ? weight_value : 0.0f;  // 不属于目标 rank 的 lane 权重清零。
                        channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = weight_value;  // 写入 topk 权重 buffer。
                    }

                    // Copy `x_scales`
                    #pragma unroll
                    for (int i = lane_id; i < num_scales; i += 32) {
                        auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;  // 输入 scale 的实际内存偏移。
                        channel_x_scales_buffers[dst_slot_idx * num_scales + i] = __ldg(x_scales + offset);  // 拷贝 scale 到环形 buffer。
                    }
                }

                // Move token index
                chunk_token_idx++, token_idx++;  // 推进 chunk 内计数和全局 token 扫描位置。
            }

            // 本目标 rank 的所有 warp 先通过 named barrier 对齐 cached_channel_tail_idx，
            // 再由一个 warp release 写 tail，receiver 用 acquire load 观察到新数据。
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);  // release 写 tail，让 receiver acquire 后看到 payload。
        }
    } else {
        // receiver 侧从每个 source rank 的 channel buffer 中取 token，写入连续 recv 张量。
        // rank_offset 来自 prefix 矩阵，total_offset 来自 channel prefix，两者相加得到
        // recv_x 中的最终行号。
        constexpr int num_recv_warps = kNumThreads / 32;  // receiver block 中 warp 总数。
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;  // 每个源 rank 分配的 receiver warp 数。
        const auto recv_thread_id = thread_id;  // receiver 线程 id。
        const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;  // rank 内线程 id。
        const auto recv_warp_id_in_rank = recv_thread_id_in_rank / 32;  // rank 内 warp id。
        EP_DEVICE_ASSERT(kNumRanks <= 32);  // rank 数不超过 warp lane 数。
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);  // receiver warp 必须能均分给 rank。

        // Calculate offset first
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);  // 本地 buffer 开头保存 rank prefix 矩阵。
        int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;  // 当前源 rank 在 recv_x 中的起始 offset。

        // 读取 sender 写入的 start/end offset；recv_channel_offset [src_rank, channel]
        // 保存该 channel 的全局起始行号，combine 可复用它反向计算。
        int total_offset, num_tokens_to_recv;  // 当前 channel 的全局起始 offset 和需要接收 token 数。
        if (elect_one_sync()) {
            while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0)
                ;  // 等 sender 写入 start offset。
            while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0)
                ;  // 等 sender 写入 end offset。
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;  // 解码负数 offset。
            if (recv_warp_id_in_rank == 0)
                recv_channel_offset[responsible_rank * num_channels + responsible_channel] = total_offset;  // 保存 channel offset 给 combine/cached 路径。
            num_tokens_to_recv -= total_offset;  // end-start 得到当前 channel token 数。
        }
        total_offset = __shfl_sync(0xffffffff, total_offset, 0);  // 广播 channel 内起始 offset。
        total_offset += rank_offset;  // 加上源 rank 前缀，得到 recv_x 绝对行号。
        num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);  // 广播当前 channel token 数。

        // Shared tail indices for different warps
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];  // 每个源 rank 一份共享 tail，保证 rank 内 warp 看到一致值。

        auto start_time = clock64();  // 接收等待开始时间。
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;  // receiver 本地缓存 head/tail。
        while (num_tokens_to_recv > 0) {
            // NOTES: unlike the sender, the receiver must ensure that the tail indices hold by different warps are the same
            while (recv_thread_id_in_rank == 0) {
                cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());  // acquire 读取 sender release 写入的 tail。

                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx) {
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;  // 将 ready tail 写入 shared，供同 rank 其它 warp 使用。
                    break;  // 有新数据可接收。
                }

                // Timeout check
                if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        // 中文：打印 responsible_channel 便于定位哪个 channel/rank 的 NVLink ring 没有推进。
                        printf("DeepEP timeout for dispatch receivers, rank %d, responsible_channel = %d, tokens remained: %d\n",
                               rank,
                               // 中文：responsible_channel 是当前 receiver block 负责的 channel 编号。
                               responsible_channel,
                               num_tokens_to_recv);
                    trap();  // receiver 超时，触发设备异常。
                }
            }

            // Synchronize queue tail
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];  // rank 内所有 warp 使用同一个 tail。

            // 将环形 buffer 中 [head, tail) 的 token payload 拷入 recv_x；SM90 用 TMA
            // 分两半搬 hidden，旧架构用 warp copy。
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;  // 本轮可接收 token 数。
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens; chunk_idx += num_recv_warps_per_rank) {
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;  // ring buffer 中的槽位。
                auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;  // 源环形 buffer payload。
                auto shifted_recv_x_int4 = recv_x + static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;  // 目标 recv_x 行。
#ifndef DISABLE_SM90_FEATURES
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    tma_store_wait<0>();  // 等前一次 TMA store 完成，避免覆盖 shared buffer。
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer, shifted_buffer_x_int4 + i * half_hidden_int4, tma_mbarrier, half_hidden_bytes);  // 从环形 buffer TMA load 半个 hidden。
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);  // 声明本轮 TMA 传输字节数。
                        mbarrier_wait(tma_mbarrier, tma_phase);  // 等 TMA load 完成。
                        tma_store_1d(tma_buffer, shifted_recv_x_int4 + i * half_hidden_int4, half_hidden_bytes, false);  // TMA store 到 recv_x。
                    }
                }
                __syncwarp();  // warp 内等待 TMA 发射路径一致。
#else
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_int4, shifted_buffer_x_int4, ld_nc_global, st_na_global);  // 非 SM90 用 warp copy 搬运。
#endif
            }

            // Copy `src_idx`
            #pragma unroll 4
            for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank; chunk_idx < cached_channel_tail_idx;
                 // 中文：每个线程跨步 32*num_recv_warps_per_rank，覆盖当前 channel 的全部 metadata 槽。
                 chunk_idx += 32 * num_recv_warps_per_rank)
                // recv_src_idx 是 combine 反向路由的核心：它记录当前接收行来自哪个原始 token。
                // total_offset 是该 rank/channel 在最终 recv_x 中的基址，chunk_idx-head 是当前 channel 内偏移。
                recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] =
                    ld_nc_global(channel_src_idx_buffers.buffer() + chunk_idx % num_recv_buffer_tokens);  // 拷贝源 token index。

            // Copy `topk_idx` and `topk_weights`
            #pragma unroll 4
            for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk; idx += 32 * num_recv_warps_per_rank) {
                int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;  // 展开后的 token 序号和 topk lane。
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;  // ring buffer 槽位。
                auto recv_idx = static_cast<int64_t>(total_offset + chunk_idx) * num_topk + token_topk_idx;  // recv topk 输出线性位置。
                auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;  // ring topk buffer 线性位置。
                recv_topk_idx[recv_idx] = ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);  // 拷贝 topk_idx。
                recv_topk_weights[recv_idx] = ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);  // 拷贝 topk 权重。
            }

            // Copy `x_scales`
            #pragma unroll 4
            for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales; i += 32 * num_recv_warps_per_rank) {
                int chunk_idx = i / num_scales, scales_idx = i % num_scales;  // 展开后的 token 序号和 scale lane。
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;  // ring buffer 槽位。
                recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) * num_scales + scales_idx] =
                    ld_nc_global(channel_x_scales_buffers.buffer() + token_idx_in_buffer * num_scales + scales_idx);  // 拷贝 FP8 scale。
            }

            // Move queue
            cached_channel_head_idx += num_recv_tokens;  // 本地 head 前移到 tail。
            total_offset += num_recv_tokens;  // recv_x 写入 offset 同步前移。
            // responsible_rank named barrier 只同步当前 rank 对应的 receiver 线程组，避免整个 block 等待其它 rank。
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and elect_one_sync())
                // 只有最后一个 receiver warp 发布 head；release 语义不需要，因为前面 payload store 已经通过普通/TMA wait 完成。
                st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);  // 写回 head，通知 sender 槽位已释放。

            // Exit
            num_tokens_to_recv -= num_recv_tokens;  // 更新剩余待接收 token 数。
        }
    }

    // Clean unused `recv_topk_idx` as -1
    if (num_worst_tokens > 0) {
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);  // 读取真实接收 token 数所在的 rank prefix。
        const auto num_recv_tokens = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];  // 真实接收 token 数。
        const auto clean_start = num_recv_tokens * num_topk + sm_id * kNumThreads;  // 当前 block 清理起点。
        const auto clean_end = num_worst_tokens * num_topk;  // worst-case 分配的 topk 总容量。
        const auto clean_stride = num_sms * kNumThreads;  // 全 grid 清理步长。
        #pragma unroll
        for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
            recv_topk_idx[i] = -1;  // 未实际接收的 topk lane 标记为 -1。
    }
}

void dispatch(void* recv_x,  // 输出接收 token 特征。
              float* recv_x_scales,  // 输出接收 scales，可为空。
              int* recv_src_idx,  // 输出源 token index。
              topk_idx_t* recv_topk_idx,  // 输出接收 topk_idx，可为空。
              float* recv_topk_weights,  // 输出接收 topk 权重，可为空。
              int* recv_channel_offset,  // 输出接收 channel offset。
              int* send_head,  // 输出发送槽位。
              const void* x,  // 输入 token 特征。
              const float* x_scales,  // 输入 scales，可为空。
              const topk_idx_t* topk_idx,  // 输入 topk_idx，可为空。
              const float* topk_weights,  // 输入 topk 权重，可为空。
              const bool* is_token_in_rank,  // 输入 token-rank 路由表。
              const int* channel_prefix_matrix,  // 输入 channel prefix。
              int num_tokens,  // 输入 token 数。
              int num_worst_tokens,  // worst-case 容量。
              int hidden_int4,  // hidden 的 int4 长度。
              int num_topk,  // top-k 路数。
              int num_experts,  // 全局 expert 数。
              int num_scales,  // FP8 scale 数。
              int scale_token_stride,  // scale token stride。
              int scale_hidden_stride,  // scale hidden stride。
              void** buffer_ptrs,  // peer buffer 指针表。
              int rank,  // 当前 rank。
              int num_ranks,  // 节点内 rank 数。
              cudaStream_t stream,  // CUDA 发射流。
              int num_sms,  // kernel block/SM 数。
              int num_max_send_tokens,  // send chunk 上限。
              int num_recv_buffer_tokens) {  // ring buffer token 容量。
    constexpr int kNumThreads = 768;  // dispatch kernel 每 block 线程数。
    constexpr int kNumTMABytesPerWarp = 8192;  // SM90 每 warp TMA shared buffer 大小。
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);  // 总 shared memory 大小。
#endif

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());  // scale offset 要能放进 int。

#define DISPATCH_LAUNCH_CASE(ranks)                                      \
    {                                                                    \
        auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \
        /* 中文：每个 ranks 模板实例会生成固定 rank 数的 intranode dispatch kernel。 */ \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                               \
        LAUNCH_KERNEL(&cfg,                                              \
                      kernel,                                            \
                      reinterpret_cast<int4*>(recv_x),                   /* 接收 payload 输出。 */ \
                      recv_x_scales,                                     /* FP8 scale 输出。 */ \
                      recv_src_idx,                                      /* 源 token index 输出。 */ \
                      recv_topk_idx,                                     /* 接收 top-k id 输出。 */ \
                      recv_topk_weights,                                 /* 接收 top-k 权重输出。 */ \
                      recv_channel_offset,                               /* channel offset 输出。 */ \
                      send_head,                                         /* send ring head。 */ \
                      reinterpret_cast<const int4*>(x),                  /* 输入 payload。 */ \
                      x_scales,                                          /* 输入 FP8 scale。 */ \
                      topk_idx,                                          /* top-k 路由矩阵。 */ \
                      topk_weights,                                      /* top-k 权重矩阵。 */ \
                      is_token_in_rank,                                  /* token-rank bool 矩阵。 */ \
                      channel_prefix_matrix,                             /* channel prefix。 */ \
                      num_tokens,                                        /* token 数。 */ \
                      num_worst_tokens,                                  /* CUDA graph 固定容量上界。 */ \
                      hidden_int4,                                       /* hidden 的 int4 数。 */ \
                      num_topk,                                          /* top-k 数。 */ \
                      num_experts,                                       /* expert 数。 */ \
                      num_scales,                                        /* FP8 scale 数。 */ \
                      scale_token_stride,                                /* scale token stride。 */ \
                      scale_hidden_stride,                               /* scale hidden stride。 */ \
                      buffer_ptrs,                                       /* peer buffer 指针表。 */ \
                      rank,                                              /* 当前 rank。 */ \
                      num_max_send_tokens,                               /* send chunk 上限。 */ \
                      num_recv_buffer_tokens);                           /* recv ring 容量。 */ \
    }                                                                    \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(num_sms % 2 == 0);  // block 数必须为偶数，成对分配 sender/receiver。
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);  // 构造 launch config。
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);  // 根据 num_ranks 选择模板实例。
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_combine(
    void** buffer_ptrs,  // peer buffer 指针表。
    int* send_head,  // dispatch 记录的发送槽位矩阵，[num_recv_tokens,kNumRanks]。
    int num_channels,  // channel 数。
    int num_recv_tokens,  // combine 输出 token 数。
    int num_memset_int,  // 需要清零的 head/tail 控制字数量。
    int** barrier_signal_ptrs,  // peer barrier 指针表。
    int rank) {  // 当前 rank。
    const auto sm_id = static_cast<int>(blockIdx.x);  // block0 负责清理，block1..num_channels 负责修正 send_head。
    if (sm_id == 0) {
        // Barrier before cleaning
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);  // 清理前先等所有 rank 结束上一阶段访问。

        // Clean
        auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);  // 线程 id 和 block 线程数。
        auto ptr = static_cast<int*>(buffer_ptrs[rank]);  // 当前 rank 本地 buffer 作为 int 数组。
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            ptr[i] = 0;  // 清零 combine 队列 head/tail 区。

        // Barrier after cleaning
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);  // 清理完成后同步所有 rank。
    } else {
        const auto channel_id = sm_id - 1;  // 当前 block 对应的 channel。
        const auto thread_id = static_cast<int>(threadIdx.x);  // 线程 id。
        const auto rank_id = thread_id / 32;  // 每个 warp 对应一个 rank。
        const auto lane_id = thread_id % 32;  // warp 内 lane。
        if (rank_id >= kNumRanks)
            return;  // 多余 warp 直接退出。

        int token_start_idx, token_end_idx;  // 当前 channel 负责的 token 范围。
        get_channel_task_range(num_recv_tokens, num_channels, channel_id, token_start_idx, token_end_idx);  // 按 channel 切分 token。

        // NOTES: `1 << 25` is a heuristic large number
        // cached combine 需要把 send_head 中“未命中某 rank”的位置修成负数编码。
        // 反向扫描可以让空洞 token 指向后面最近的有效 head：combine 遇到负数就知道不必等待该 rank，
        // 但仍能把释放进度推进到 -expected_head-1。
        int last_head = 1 << 25;  // 从尾向前扫描时记录最近一个有效 head，初始设为大数哨兵。
        #pragma unroll
        for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= 32) {
            int token_idx = token_idx_tail - lane_id, expected_head = 0;  // lane 负责的 token 和要写回的 expected head。
            auto current_head = (token_idx >= token_start_idx) ? __ldg(send_head + token_idx * kNumRanks + rank_id) : -1;  // 读取 dispatch 记录的 head。
            for (int i = 0; i < min(32, token_idx_tail - token_start_idx + 1); ++i) {
                const int head = __shfl_sync(0xffffffff, current_head, i);  // 从第 i 个 lane 广播 head。
                if (head < 0) {
                    if (lane_id == i)
                        // -last_head-1 避免和合法 head=0 混淆；读取端通过 -value-1 还原下一有效 head。
                        expected_head = -last_head - 1;  // 对未发送 token 写负数编码，指向后续最近有效 head。
                } else {
                    last_head = head;  // 更新最近有效 head。
                }
            }
            if (current_head < 0 and token_idx >= token_start_idx)
                send_head[token_idx * kNumRanks + rank_id] = expected_head;  // 写回修正后的负数 expected head。
        }
    }
}

void cached_notify_combine(void** buffer_ptrs,  // peer buffer 指针表。
                           int* send_head,  // send_head 矩阵。
                           int num_channels,  // channel 数。
                           int num_recv_tokens,  // combine 输出 token 数。
                           int num_memset_int,  // 清零 int 数。
                           int** barrier_signal_ptrs,  // barrier 指针表。
                           int rank,  // 当前 rank。
                           int num_ranks,  // 节点内 rank 数。
                           cudaStream_t stream) {  // CUDA 发射流。
#define CACHED_NOTIFY_COMBINE(ranks)            \
    LAUNCH_KERNEL(&cfg,                         \
                  cached_notify_combine<ranks>, \
                  buffer_ptrs,                  /* peer buffer 指针表。 */ \
                  send_head,                    /* cached send head。 */ \
                  num_channels,                 /* channel 数。 */ \
                  num_recv_tokens,              /* 本 rank recv token 数。 */ \
                  num_memset_int,               /* 清零 int 数。 */ \
                  barrier_signal_ptrs,          /* barrier 信号表。 */ \
                  rank);                        /* 当前 rank。 */ \
    break

    const int num_threads = std::max(128, 32 * num_ranks);  // 至少 128 线程，且每 rank 一个 warp。
    EP_HOST_ASSERT(num_ranks <= num_threads);  // 线程数要覆盖所有 rank。
    EP_HOST_ASSERT(num_threads <= 1024);  // CUDA 单 block 线程上限。
    EP_HOST_ASSERT(1 + num_channels <= num_channels * 2);  // cached notify block 数不能超过后续 combine block 数。
    SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);  // block0 清理 + 每 channel 一个修正 block。
    SWITCH_RANKS(CACHED_NOTIFY_COMBINE);  // 按 num_ranks 选择模板实例。
#undef CACHED_NOTIFY_COMBINE
}

template <typename dtype_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) combine(dtype_t* recv_x,  // 输出 combine 后 token 特征。
                                                          float* recv_topk_weights,  // 输出 combine 后 topk 权重，可为空。
                                                          const dtype_t* x,  // 输入 expert 输出 token 特征。
                                                          const float* topk_weights,  // 输入 topk 权重，可为空。
                                                          const dtype_t* bias_0,  // 可选 bias0。
                                                          const dtype_t* bias_1,  // 可选 bias1。
                                                          const int* src_idx,  // dispatch 返回的源 token index。
                                                          const int* rank_prefix_matrix,  // dispatch 返回的 rank prefix。
                                                          const int* channel_prefix_matrix,  // dispatch 返回的 channel prefix。
                                                          int* send_head,  // dispatch/cached notify 生成的 expected head 表。
                                                          int num_tokens,  // expert 输出 token 数。
                                                          int num_recv_tokens,  // combine 输出 token 数。
                                                          int hidden,  // hidden 维度。
                                                          int num_topk,  // top-k 路数。
                                                          void** buffer_ptrs,  // peer buffer 指针表。
                                                          int rank,  // 当前 rank。
                                                          int num_max_send_tokens,  // send chunk 上限。
                                                          int num_recv_buffer_tokens) {  // ring buffer 容量。
    const auto num_sms = static_cast<int>(gridDim.x);  // block 总数。
    const auto thread_id = static_cast<int>(threadIdx.x);  // 线程 id。
    const auto sm_id = static_cast<int>(blockIdx.x), lane_id = get_lane_id();  // block id 和 warp lane。
    const auto num_channels = num_sms / 2;  // channel 数。
    const bool is_sender = sm_id % 2 == 0;  // 偶数 block 发送，奇数 block 接收/归约。
    const int responsible_channel = sm_id / 2;  // 当前 channel。
    EP_DEVICE_ASSERT(num_topk <= 32);  // top-k 不超过 warp lane 数。

    constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);  // 一个 int4 包含的 dtype 元素数。
    int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);  // hidden 的 int4 长度。
    int hidden_int4_aligned = align_down(hidden_int4, 32);  // 对齐到 32 int4，便于 warp 规约主循环。
    auto x_int4 = reinterpret_cast<const int4*>(x);  // 输入 x 的 int4 视图。
    auto bias_0_int4 = reinterpret_cast<const int4*>(bias_0);  // bias0 的 int4 视图，可为空但不解引用。
    auto bias_1_int4 = reinterpret_cast<const int4*>(bias_1);  // bias1 的 int4 视图，可为空但不解引用。
    auto recv_int4 = reinterpret_cast<int4*>(recv_x);  // 输出 recv_x 的 int4 视图。

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];  // SM90 TMA shared memory。
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;  // 每 warp 一段 TMA buffer。
#endif

    if (is_sender) {
        // Workers for sending
        // combine 的 sender 方向和 dispatch 相反：这里读取本 rank 已经算完的 expert 输出 x，
        // 再按“原始 token 属于哪个目标 rank”写入对方暴露的 NVLink ring buffer。
        // Several warps are responsible for a single rank
        constexpr int num_send_warps_per_rank = (kNumThreads / 32) / kNumRanks;  // 每个目标 rank 分配的 sender warp 数。
        constexpr int num_send_warps = num_send_warps_per_rank * kNumRanks;  // 实际参与发送的 warp 总数。
        const auto num_threads_per_rank = num_send_warps_per_rank * 32;  // 每个 rank 对应的线程数。
        const auto send_thread_id = thread_id;  // sender 线程 id。
        const auto send_warp_id = send_thread_id / 32;  // sender warp id。
        // responsible_channel 加到 send_warp_id 上做轮转，避免所有 channel 的 0 号 warp
        // 同时打向同一个 send_rank_id，从而减轻 rank 0/1 等低编号目标的瞬时压力。
        const auto send_rank_id = (responsible_channel + send_warp_id) % kNumRanks;  // 当前 warp 负责发送到的目标 rank。
        const auto send_warp_id_in_rank = send_warp_id / kNumRanks;  // 当前 rank 内的 warp 序号。
        EP_STATIC_ASSERT(num_send_warps * 32 == kNumThreads, "Invalid warp count");  // 确认所有线程都被 warp 分配覆盖。

        // Calculate pointers by the specific layout
        auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[send_rank_id]));  // 目标 rank 暴露的 NVL buffer。
        auto num_channels_total = num_channels * kNumRanks;  // 所有 channel-rank 组合数。
        auto channel_rank_offset = responsible_channel * kNumRanks + rank;  // 当前 channel/源 rank 在线性表中的偏移。

        // Channel meta data
        // 这份 ring 布局与 dispatch receiver 使用的布局同构，但含义反过来：
        // 当前 rank 是“发送 expert 输出的源 rank”，send_rank_id 是“等待 combine 回填的目标 rank”。
        // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
        // 中文：combine sender 用 tail 通知目标 rank，receiver 用 head 释放已消费槽位。
        // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
        // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
        auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);  // 目标 rank receiver 推进的 head。
        auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);  // 当前 sender 推进的 tail。
        auto channel_x_buffers = Buffer<int4>(
            ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);  // combine payload 环形 buffer。
        auto channel_src_idx_buffers =
            Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);  // source index 环形 buffer。
        auto channel_topk_weights_buffers = Buffer<float>(
            ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);  // topk weight 环形 buffer。

        // Get tasks
        // rank_prefix_matrix 形状 [dst_rank, src_rank]，这里按 send_rank_id 取一整段：
        // rank_offset..rank_offset+num_rank_tokens 是“需要回传给 send_rank_id 的 token”。
        // NOTES: `channel_offset` is already shifted
        // channel_prefix_matrix 形状 [dst_rank, channel]，负责把上面的 rank 段继续拆给
        // 不同 channel；例如 channel 2 的区间从 prefix[1] 到 prefix[2]。
        int rank_offset = send_rank_id > 0 ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank] : 0;  // 目标 rank 在本 rank recv token 序列中的起点。
        int num_rank_tokens = rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;  // 需要发回 send_rank_id 的 token 数。
        int channel_offset = channel_prefix_matrix[send_rank_id * num_channels + responsible_channel];  // 当前 channel 在目标 rank token 段内的偏移。
        int num_channel_tokens =
            (responsible_channel == num_channels - 1 ? num_rank_tokens
                                                     : channel_prefix_matrix[send_rank_id * num_channels + responsible_channel + 1]) -
            channel_offset;  // 当前 channel 要发送的 token 数。
        int token_start_idx = rank_offset + channel_offset, token_end_idx = rank_offset + channel_offset + num_channel_tokens;  // 当前 channel token 半开区间。

        // Iterate over all tokens and send by chunks
        int current_channel_tail_idx = 0;  // sender 本地 tail。
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            auto start_time = clock64();  // 等待空槽开始时间。
            int num_round_tokens = min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));  // 本轮 chunk token 数。
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    // 这里用 num_round_tokens 当作最坏需求量：真实每个 sender warp 写多少槽要再做跨 warp
                    // 统计，开销会超过偶发等待的收益；保守等待能保证不覆盖 receiver 尚未释放的槽。
                    int num_used_slots = current_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());  // ring 已用槽数。
                    if (num_recv_buffer_tokens - num_used_slots >= num_round_tokens)
                        break;  // 空槽足够本轮发送。

                    // Rare cases to loop again
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();  // sender 等待空槽超时。
                    }
                }
            }
            __syncwarp();  // 等待空槽检查结果。

            // Send by chunk
            #pragma unroll
            // 中文：每个 send warp 处理 round 内的交错 token，减少单 warp 串行搬运压力。
            for (int i = send_warp_id_in_rank; i < num_round_tokens; i += num_send_warps_per_rank) {
                // Get an empty slot
                // 中文：多个 send warp 交错领取 token，dst_slot_idx 用 tail+i 映射到 ring 槽。
                int dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens;  // 当前 token 对应 ring 槽。

                // Copy data
                auto shifted_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;  // 目标 ring payload。
                auto shifted_x = x_int4 + (token_idx + i) * hidden_int4;  // 输入 expert 输出 token。
                UNROLLED_WARP_COPY(4, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);  // warp 并行拷贝 hidden。

                // Send source index
                if (elect_one_sync())
                    channel_src_idx_buffers[dst_slot_idx] = __ldg(src_idx + token_idx + i);  // 写源 token index，receiver 用它定位输出行。

                // Send `topk_weights`
                if (num_topk > 0 and lane_id < num_topk)
                    channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] =
                        __ldg(topk_weights + (token_idx + i) * num_topk + lane_id);  // 拷贝 topk 权重。
            }
            token_idx += num_round_tokens;  // 推进 token 扫描位置。
            current_channel_tail_idx += num_round_tokens;  // 推进本地 tail。

            // Move tail index
            // named barrier 的 id 使用 send_rank_id，使同一 block 内不同目标 rank 的 sender
            // 线程组互不干扰；只有负责该目标 rank 的 num_threads_per_rank 个线程参与同步。
            asm volatile("bar.sync %0, %1;" ::"r"(send_rank_id), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), current_channel_tail_idx);  // release 写 tail 通知 receiver。
        }  // 中文：结束当前目标 rank 的 chunk send 循环。
    } else {
        // Workers for receiving
        // One warp for moving the queue head, others for reduction
        // 中文：head warp 负责释放 ring 槽，其它 warp 负责把多个源 rank 的数据规约到 recv_x。
        constexpr int num_recv_warps = kNumThreads / 32;  // receiver warp 总数。
        const auto recv_warp_id = thread_id / 32;  // 当前 receiver warp id。
        EP_DEVICE_ASSERT(kNumRanks <= 32 and kNumThreads > 32);  // 第 0 warp 负责 head 更新，其它 warp 做 reduce。
        EP_DEVICE_ASSERT(thread_id >= 0 and kNumThreads % 32 == 0);  // 线程数必须是 warp 整数倍。

        // Shared head, tail and retired flags for receiver warps
        // receiver 第 0 个 warp 不做归约，只负责读取各 source rank 的 tail，并把 reducer
        // warp 已经消费到的最小 head 写回 sender；这样 sender 才知道哪些 ring 槽可以复用。
        __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];  // 每个 reducer warp 对每个 rank 的 head 进度。
        __shared__ volatile int channel_tail_idx[kNumRanks];  // 第 0 warp 维护的每 rank tail。
        __shared__ volatile bool warp_retired[num_recv_warps];  // reducer warp 是否完成。
        if (thread_id < num_recv_warps)
            warp_retired[thread_id] = false;  // 初始化退休标记。
        if (lane_id < kNumRanks)
            warp_channel_head_idx[recv_warp_id][lane_id] = 0;  // 初始化本 warp 的 head 进度。
        if (thread_id < kNumRanks)
            channel_tail_idx[thread_id] = 0;  // 初始化共享 tail。
        asm volatile("bar.sync 0, %0;" ::"r"(kNumThreads));  // 全 block 同步，确保 shared 初始化完成。

        if (thread_id < 32) {
            int* channel_head_idx_ptr = static_cast<int*>(buffer_ptrs[rank]) + responsible_channel * kNumRanks + lane_id;  // 当前 channel/rank 的 head 地址。
            int* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels * kNumRanks;  // 当前 channel/rank 的 tail 地址。

            // Queue head updater
            // lane_id 映射 source rank，因此一个 warp 正好维护 kNumRanks 路 ring。
            // last_head 是上次已经发布给 sender 的 head，避免重复写系统内存。
            int last_head = 0;  // 最近一次写给 sender 的最小 head。
            while (lane_id < kNumRanks) {
                // Check retired
                bool retired = true;  // 是否所有 reducer warp 都完成。
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++i)
                    retired = retired and warp_retired[i];  // reducer warp 全部 retired 后退出。
                if (retired)
                    break;  // 不再需要更新 head。

                // Update queue tail
                channel_tail_idx[lane_id] = ld_acquire_sys_global(channel_tail_idx_ptr);  // acquire 读取 sender 写入的 tail。

                // Update minimum head
                // 对同一个 source rank，多个 reducer warp 会并行消费不同 token；sender 只能复用
                // “所有未完成 reducer warp 都越过”的槽，所以这里必须取最小 head。
                int min_head = std::numeric_limits<int>::max();  // 当前所有活跃 reducer warp 中最小 head。
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++i)
                    if (not warp_retired[i])
                        min_head = min(min_head, warp_channel_head_idx[i][lane_id]);  // 只统计未完成 warp。
                // 中文：head 只能推进到所有活跃 reduce warp 都已消费的位置，避免覆盖未读 token。
                if (min_head != std::numeric_limits<int>::max() and min_head > last_head)
                    st_relaxed_sys_global(channel_head_idx_ptr, last_head = min_head);  // 向 sender 释放已消费槽位。
            }  // 中文：head warp 完成本轮所有 source lane 的 head 推进。
        } else {
            // Receivers
            // Channel metadata
            // 中文：receiver 先构建每个 source rank 的 buffer 视图，再按 lane/source 轮询 tail。
            // All lanes will use data buffer, but only rank lane will use `head/tail/src_idx`
            // 中文：lane_id 对应 source rank，只有该 lane 读取对应 rank 的 head/tail/src_idx 控制面。
            Buffer<int4> channel_x_buffers[kNumRanks];  // 每个源 rank 一个 payload buffer 视图。
            Buffer<float> channel_topk_weights_buffers[kNumRanks];  // 每个源 rank 一个 topk weight buffer 视图。

            // Calculate pointers by the specific layout
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++i) {
                auto channel_rank_offset = responsible_channel * kNumRanks + i;  // 当前 channel/source-rank 线性偏移。
                auto num_channels_total = num_channels * kNumRanks;  // channel-rank 总组合数。
                // `head_idx` & `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
                auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[rank]) + 2 * num_channels * kNumRanks * sizeof(int));  // 跳过 head/tail 元数据。

                // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
                channel_x_buffers[i] = Buffer<int4>(ptr,
                                                    num_channels_total * num_recv_buffer_tokens * hidden_int4,
                                                    channel_rank_offset * num_recv_buffer_tokens * hidden_int4);  // 建立源 rank i 的 payload ring 视图。

                // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
                ptr = reinterpret_cast<void*>(static_cast<int8_t*>(ptr) + num_channels_total * num_recv_buffer_tokens * sizeof(int));  // combine receiver 不读 src_idx，直接跳过。

                // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
                channel_topk_weights_buffers[i] = Buffer<float>(
                    ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);  // 建立源 rank i 的 topk weight ring 视图。
            }

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;  // 当前 channel 负责的 combine 输出 token 范围。
            get_channel_task_range(num_recv_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);  // 按 channel 均分 combine token。

            // Iterate over all tokens and combine
            for (int64_t token_idx = token_start_idx + recv_warp_id - 1; token_idx < token_end_idx; token_idx += num_recv_warps - 1) {
                // Read expected head
                // send_head[token, rank] 来自 dispatch 阶段/cached_notify_combine：非负值表示
                // 本 token 需要从该 source rank 的 ring 槽读取；负数则编码“这个 rank 无贡献，
                // 但 receiver 的 head 至少可以推进到 -value-1”。
                int expected_head = -1;  // 当前 lane/source-rank 期望消费的 head。
                if (lane_id < kNumRanks)
                    expected_head = ld_nc_global(send_head + token_idx * kNumRanks + lane_id);  // 从 send_head 读取该 token 是否来自 source rank。

                auto start_time = clock64();  // 等待 source rank 数据到达的开始时间。
                while (__any_sync(0xffffffff, channel_tail_idx[lane_id] <= expected_head and expected_head >= 0)) {
                    // 只有 expected_head >= 0 的 lane 才需要等待 tail；负数 lane 表示本 token
                    // 不读该 source rank，只在最后用编码值推进 head，避免 ring 永久占用。
                    // Timeout check
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        // 中文：expect 是当前 lane/source-rank 等待的 ring head，可反推出卡住的源 rank。
                        printf("DeepEP timeout for combine receivers, rank %d, responsible_channel = %d, expect = %d\n",
                               rank,
                               // 中文：responsible_channel 对应 block/channel，expected_head 是该 source ring 期待的 tail。
                               responsible_channel,
                               expected_head);
                        trap();  // 等待某个 source rank tail 超时。
                    }
                }
                __syncwarp();  // 确保所有 lane 都观察到可用 tail。

                // Broadcast current heads
                // 先把有效 source rank 压缩成 topk_ranks/slot_indices，后续 hidden 维循环只遍历
                // 实际有贡献的 rank；典型 top-k 小于 kNumRanks，这比每个 hidden 元素扫全 rank 更省。
                int num_topk_ranks = 0, topk_ranks[kNumRanks], slot_indices[kNumRanks];  // 本 token 实际来自哪些 rank 以及对应 ring 槽位。
                #pragma unroll
                for (int i = 0; i < kNumRanks; ++i) {
                    auto expected_head_i = __shfl_sync(0xffffffff, expected_head, i);  // 从第 i 个 lane 获取该 rank 的 expected head。
                    if (expected_head_i >= 0) {
                        slot_indices[num_topk_ranks] = expected_head_i % num_recv_buffer_tokens;  // 对应 ring 槽位。
                        topk_ranks[num_topk_ranks++] = i;  // 记录有效 source rank。
                    }
                }

                // Wait shared memory release
#ifndef DISABLE_SM90_FEATURES
                tma_store_wait<0>();  // 等前一轮 TMA store 完成。
                __syncwarp();  // warp 内同步 TMA 状态。
#endif

                // Reduce data with pipeline
                constexpr int kNumStages = 8;  // TMA store pipeline stage 数。
                // 中文：每个 stage 覆盖最多 32 个 int4，8 stage 用于隐藏 shared reduce 和 TMA store 延迟。
                EP_STATIC_ASSERT(kNumStages * 32 * sizeof(int4) <= kNumTMABytesPerWarp, "Invalid count");  // shared buffer 必须容纳所有 stage。
                #pragma unroll
                // 中文：每个 lane 以 warp stride 处理 hidden 向量，保证完整覆盖 hidden_int4。
                for (int i = lane_id; i < hidden_int4; i += 32) {
                    // Read bias
                    // TODO: make it as a template
                    // 中文：bias 暂时运行期判空，后续可模板化以去掉分支并让编译器常量折叠。
                    int4 bias_0_value_int4 =
                        bias_0_int4 != nullptr ? __ldg(bias_0_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);  // 读取 bias0 或零。
                    int4 bias_1_value_int4 =
                        bias_1_int4 != nullptr ? __ldg(bias_1_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);  // 读取 bias1 或零。

                    // Read buffers
                    int4 recv_value_int4[kNumRanks];  // 当前 hidden int4 位置上来自各 source rank 的值。
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j)
                        recv_value_int4[j] = ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);  // 读取 source rank 的专家输出。

                    // Reduce bias
                    float values[kDtypePerInt4];  // 用 float 累加，降低 BF16/FP16 累加误差。
                    auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);  // bias0 的 dtype 视图。
                    auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);  // bias1 的 dtype 视图。
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++j)
                        values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);  // bias 初值相加。

                    // Reduce all-to-all results
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j) {
                        auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);  // source rank 值的 dtype 视图。
                        #pragma unroll
                        for (int k = 0; k < kDtypePerInt4; ++k)
                            values[k] += static_cast<float>(recv_value_dtypes[k]);  // 对所有 source rank 的贡献求和。
                    }

                    // Cast back to `dtype_t`
                    int4 out_int4;  // 输出 int4 打包值。
                    auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);  // 输出 dtype 视图。
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++j)
                        out_dtypes[j] = static_cast<dtype_t>(values[j]);  // float 累加结果转回原 dtype。

#ifndef DISABLE_SM90_FEATURES
                    if (i < hidden_int4_aligned) {
                        // 对齐主段使用 TMA store：每个 lane 产生一个 int4，先写入 shared stage，
                        // 再由单 lane 发起 1D TMA 写回 recv_x。kNumStages=8 让上一批 TMA
                        // 写回和下一批 hidden 归约重叠。
                        // Wait TMA arrival
                        tma_store_wait<kNumStages - 1>();  // pipeline 保护：等待最老 stage 可复用。
                        __syncwarp();  // warp 内同步。

                        // Write into TMA buffer
                        auto tma_stage_idx = (i / 32) % kNumStages;  // 当前 hidden 块对应的 TMA stage。
                        reinterpret_cast<int4*>(tma_buffer)[tma_stage_idx * 32 + lane_id] = out_int4;  // 写入 shared TMA stage。

                        // Issue TMA
                        tma_store_fence();  // 保证 shared 写入对 TMA store 可见。
                        __syncwarp();  // 等所有 lane 完成 shared 写。
                        if (elect_one_sync()) {
                            auto tma_bytes = min(32, hidden_int4 - i) * static_cast<int>(sizeof(int4));  // 本次 TMA store 字节数。
                            // 中文：尾部不足 32 个 int4 时按实际字节数 store，避免越界写 recv_x。
                            tma_store_1d(reinterpret_cast<int4*>(tma_buffer) + tma_stage_idx * 32,
                                         recv_int4 + token_idx * hidden_int4 + i,
                                         tma_bytes,
                                         false);  // 将 stage 中的数据写回 recv_x。
                        }
                        __syncwarp();  // 等 TMA 发射路径完成。
                    } else {
#endif
                        recv_int4[token_idx * hidden_int4 + i] = out_int4;  // 尾部未对齐区域直接普通写。
#ifndef DISABLE_SM90_FEATURES
                    }
#endif
                }  // 中文：结束当前 token hidden payload 的规约/写回。

                // Reduce `topk_weights`
                // 中文：topk 权重规约和 hidden payload 规约使用相同 source token/rank 集合。
                if (lane_id < num_topk) {
                    float value = 0;  // 当前 topk lane 的权重累加器。
                    #pragma unroll
                    for (int i = 0; i < num_topk_ranks; ++i)
                        value += ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk + lane_id);  // 累加各 source rank 权重。
                    recv_topk_weights[token_idx * num_topk + lane_id] = value;  // 写回 combine 后权重。
                }

                // Update head
                if (lane_id < kNumRanks)
                    // 非负 expected_head 表示刚消费了该槽，head 推到 expected_head+1；
                    // 负数 expected_head 表示本 token 没读这个 rank，但 cached notify 已经告诉
                    // receiver 下一段有效 head，所以解码为 -expected_head-1。
                    warp_channel_head_idx[recv_warp_id][lane_id] = (expected_head < 0) ? -expected_head - 1 : expected_head + 1;  // 更新本 warp 对该 source rank 的消费进度。
            }

            // Retired
            __syncwarp();  // 等 warp 内最后一次 head 更新完成。
            if (elect_one_sync())
                warp_retired[recv_warp_id] = true;  // 标记该 reducer warp 完成。
        }
    }
}

void combine(cudaDataType_t type,  // 输入/输出 dtype 的 CUDA enum。
             void* recv_x,  // 输出 combine token 特征。
             float* recv_topk_weights,  // 输出 combine topk 权重，可为空。
             const void* x,  // 输入 expert 输出 token 特征。
             const float* topk_weights,  // 输入 topk 权重，可为空。
             const void* bias_0,  // 可选 bias0。
             const void* bias_1,  // 可选 bias1。
             const int* src_idx,  // dispatch 源 token index。
             const int* rank_prefix_matrix,  // rank prefix。
             const int* channel_prefix_matrix,  // channel prefix。
             int* send_head,  // send head/expected head 矩阵。
             int num_tokens,  // 输入 token 数。
             int num_recv_tokens,  // 输出 token 数。
             int hidden,  // hidden 维度。
             int num_topk,  // top-k 路数。
             void** buffer_ptrs,  // peer buffer 指针表。
             int rank,  // 当前 rank。
             int num_ranks,  // 节点内 rank 数。
             cudaStream_t stream,  // CUDA 发射流。
             int num_sms,  // block/SM 数。
             int num_max_send_tokens,  // send chunk 上限。
             int num_recv_buffer_tokens) {  // ring buffer token 容量。
    constexpr int kNumThreads = 768;  // combine kernel 每 block 线程数。
    constexpr int kNumTMABytesPerWarp = 4096;  // SM90 每 warp TMA shared buffer 大小。
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);  // 总 shared memory 大小。
#endif

#define COMBINE_LAUNCH_CASE(dtype, ranks)                                      \
    {                                                                          \
        /* dtype/ranks 固化后，device 端可以展开 rank warp 和 TMA 搬运循环。 */ \
        auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \
        /* 中文：TMA shared memory 大小由 hidden 对齐后的 kNumTMABytesPerWarp 决定。 */ \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                     \
        LAUNCH_KERNEL(&cfg,                                                    \
                      kernel,                                                  \
                      /* recv_x: [num_recv_tokens, hidden]，写回原 token rank 的 combine 输出。 */ \
                      reinterpret_cast<dtype*>(recv_x),                        \
                      /* recv_topk_weights: 可选 [num_recv_tokens,num_topk]，combine 后的权重输出。 */ \
                      recv_topk_weights,                                       \
                      /* x/topk_weights: expert 计算后的输入和对应 top-k 权重。 */ \
                      reinterpret_cast<const dtype*>(x),                       \
                      topk_weights,                                            \
                      /* bias_0/bias_1: 可选 epilogue bias，dtype 与 x 相同。 */ \
                      reinterpret_cast<const dtype*>(bias_0),                  \
                      reinterpret_cast<const dtype*>(bias_1),                  \
                      /* src_idx: dispatch 输出的源 token 索引，combine 按它把结果送回原位置。 */ \
                      src_idx,                                                 \
                      /* rank/channel prefix: dispatch handle 中保存的反向路由布局。 */ \
                      rank_prefix_matrix,                                      \
                      channel_prefix_matrix,                                   \
                      /* send_head: per rank/channel ring head，控制 sender/receiver 进度。 */ \
                      send_head,                                               \
                      /* num_tokens/num_recv_tokens: expert 输出 token 数和最终 combine 输出 token 数。 */ \
                      num_tokens,                                              \
                      num_recv_tokens,                                         \
                      /* hidden/num_topk: payload stride 和权重矩阵宽度。 */     \
                      hidden,                                                  \
                      num_topk,                                                \
                      /* buffer_ptrs/rank: NVLink peer buffer 表和当前 rank。 */ \
                      buffer_ptrs,                                             \
                      rank,                                                    \
                      /* chunk/ring 容量决定发送端切片和接收端环形缓冲大小。 */ \
                      num_max_send_tokens,                                     \
                      num_recv_buffer_tokens);                                 \
    }                                                                          \
    /* 中文：结束当前 dtype/rank combine launch case。 */                      \
    break
// 中文：dtype switch 外层只决定 payload 类型，rank switch 内层决定模板展开的 rank 数。
#define COMBINE_DTYPE_LAUNCH_CASE(dtype)                 \
    /* 中文：combine 先按 dtype 分派 BF16/FP8 payload，再按 rank 数选择模板实例。 */ \
    SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_LAUNCH_CASE); \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving
    EP_HOST_ASSERT(num_sms % 2 == 0);  // sender/receiver block 成对。
    EP_HOST_ASSERT(kNumThreads >= num_ranks * 32);  // 至少保证每个 rank 一个 warp。
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);  // 构造 launch config。
    SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);  // 先按 dtype 分发，再按 rank 数分发模板。
#undef COMBINE_DTYPE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE
}

}  // namespace intranode  // intranode kernel 命名空间结束。

}  // namespace deep_ep::legacy  // legacy 命名空间结束。
