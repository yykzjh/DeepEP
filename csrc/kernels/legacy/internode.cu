#include <functional>  // host wrapper 中使用 std::function 风格工具类型。
#include <optional>  // host wrapper 中使用 std::optional 表达可选参数。

#include "buffer.cuh"  // 引入 SymBuffer/AsymBuffer/Buffer 等 RDMA/NVL buffer 布局封装。
#include "compiled.cuh"  // 引入 legacy 常量、类型和模板分发宏。
#include "ibgda_device.cuh"  // 引入 NVSHMEM IBGDA device 侧 put/quiet/amo 接口。
#include "launch.cuh"  // 引入 CUDA kernel launch 宏。
#include "utils.cuh"  // 引入 barrier、warp copy、TMA、锁等工具函数。

namespace deep_ep {  // DeepEP 顶层命名空间。

namespace nvshmem {  // NVSHMEM 相关全局状态命名空间。

extern nvshmem_team_t cpu_rdma_team;  // Host 初始化好的 RDMA team，device kernel 通过参数接收并同步。

}  // namespace nvshmem  // NVSHMEM 命名空间结束。

namespace legacy {  // legacy 通信实现命名空间。

namespace internode {  // 跨节点 RDMA + 节点内 NVL 通信 kernel 命名空间。

// legacy internode 路径是“两级通信”：
// - RDMA 层在不同节点/rdma_rank 之间交换 token chunk；
// - NVLink 层在同节点 8 个 nvl_rank 之间分发到最终 expert rank。
// low_latency_mode 会把每个 GPU 当作独立 RDMA 目标，减少节点内二次转发延迟。
// 与 internode_ll.cu 的区别：本文件服务 legacy 普通吞吐路径，先通过
// notify_dispatch 交换精确 rank/expert 计数，再按 RDMA chunk 和 NVL chunk
// 分片搬运；输出 recv_x 的 token 数是动态计算出的精确值。internode_ll.cu
// 面向 decode 小 batch，使用固定容量 [local_expert, rank*Tmax, hidden] 与
// phase/hook 减少 CPU 同步和动态分配。因此本文件适合学习“两级 prefix +
// chunked ring buffer”，LL 文件更适合学习“固定槽位 + GPU 端等待/超时 mask”。

struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;  // 来源 RDMA rank，以及来源节点内 NVL rank 的 8-bit mask。

    EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");  // bit mask 编码依赖 8 个 NVL peer。

    __forceinline__ SourceMeta() = default;  // 默认构造用于局部变量占位，之后再赋值。

    // SourceMeta 被写进每个 RDMA token 包，src_rdma_rank 记录来源节点，
    // is_token_in_nvl_rank_bits 是 8-bit mask，表示这个 token 在来源节点内哪些
    // NVLink rank 需要参与 combine。例：mask=0b00000101 表示 local rank 0 和 2。
    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;  // 保存来源 RDMA rank。
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];  // bit0 表示来源节点内 nvl_rank 0 是否命中。
        #pragma unroll
        for (int i = 1; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;  // 将第 i 个 bool 压到第 i 个 bit。
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const { return (is_token_in_nvl_rank_bits >> nvl_rank) & 1; }  // 查询某个 NVL rank 是否需要该 token。
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");  // SourceMeta 需要按 int 对齐，便于 staging buffer 计算。

int get_source_meta_bytes() {
    return sizeof(SourceMeta);  // 返回 C++/Python 分配 recv_src_meta 所需字节数。
}

__host__ __device__ __forceinline__ int get_num_bytes_per_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    // RDMA/NVL staging buffer 中单个 token 包含：
    // hidden payload + SourceMeta + FP8 scales + topk_idx + topk_weights，并向 int4 对齐。
    return static_cast<int>(align_up(hidden_int4 * sizeof(int4) + sizeof(SourceMeta) + num_scales * sizeof(float) +
                                         num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float),
                                     sizeof(int4)));  // 返回 int4 对齐后的 token 消息字节数。
}

__host__ __device__ __forceinline__ std::pair<int, int> get_rdma_clean_meta(int hidden_int4,  // hidden 的 int4 长度。
                                                                            int num_scales,  // FP8 scale 个数。
                                                                            int num_topk_idx,  // token 消息内 topk_idx 个数。
                                                                            int num_topk_weights,  // token 消息内 topk_weights 个数。
                                                                            int num_rdma_ranks,  // RDMA rank 数。
                                                                            int num_rdma_recv_buffer_tokens,  // 每 channel RDMA 接收 ring 容量。
                                                                            int num_channels) {  // channel 数。
    // 返回需要清零的 int32 区间：offset 跳过 token payload 区，count 覆盖每个
    // RDMA channel 的 head/tail/signal 等控制元数据。
    return {(get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens *
             num_rdma_ranks * 2 * num_channels) /
                sizeof(int),  // 清理起点：跳过 dispatch/combine 两套 RDMA token payload 区。
            (LEGACY_NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels};  // 清理数量：每 RDMA rank/channel 的 meta/head/tail。
}

__host__ __device__ __forceinline__ std::pair<int, int> get_nvl_clean_meta(int hidden_int4,  // hidden 的 int4 长度。
                                                                           int num_scales,  // FP8 scale 个数。
                                                                           int num_topk_idx,  // topk_idx 个数。
                                                                           int num_topk_weights,  // topk_weights 个数。
                                                                           int num_rdma_ranks,  // RDMA rank 数。
                                                                           int num_nvl_ranks,  // 节点内 NVL rank 数。
                                                                           int num_nvl_recv_buffer_tokens,  // NVL ring 容量。
                                                                           int num_channels,  // channel 数。
                                                                           bool is_dispatch) {  // 当前公式参数保留方向信息，未来可区分 dispatch/combine。
    // NVL buffer 清理区同样跳过 token payload，控制元数据数量与 nvl_ranks、
    // rdma_ranks、channel 数有关。dispatch/combine 方向共用公式。
    EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");  // SourceMeta 必须 int 对齐。
    (void)is_dispatch;  // 当前 dispatch/combine 共用同一清理公式，显式消除未使用参数语义歧义。

    return {
        (num_nvl_recv_buffer_tokens * get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_nvl_ranks *
         num_channels) /
            sizeof(int),  // 清理起点：跳过 NVL token payload 区。
        num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_channels,  // 清理数量：prefix start/end + head/tail。
    };
}

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    // low latency 模式下 RDMA team 的 PE 编号按“节点 * 8 + 本地 GPU”展开；
    // 普通模式只按节点编号通信。
    return kLowLatencyMode ? (dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;  // 返回 NVSHMEM PE id。
}

template <bool kLowLatencyMode>
__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
    kLowLatencyMode ? void(nvshmem_sync(rdma_team)) : nvshmem_sync_all();  // low-latency 只同步同 GPU index team，普通模式同步所有 PE。
}

template <bool kLowLatencyMode, int kNumRDMARanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,  // 输入 [num_ranks]：本 rank 发往每个全局 rank 的 token 数。
                                int* moe_recv_counter_mapped,  // 输出 mapped host：最终接收 token 总数。
                                int num_ranks,  // 全局 rank 数。
                                const int* num_tokens_per_rdma_rank,  // 输入 [kNumRDMARanks]：本 rank 发往每个 RDMA rank 的 token 数。
                                int* moe_recv_rdma_counter_mapped,  // 输出 mapped host：RDMA 层接收 token 总数。
                                const int* num_tokens_per_expert,  // 输入 [num_experts]：本 rank 发往每个 expert 的 token 数。
                                int* moe_recv_expert_counter_mapped,  // 输出 mapped host：[num_local_experts]。
                                int num_experts,  // 全局 expert 数。
                                const bool* is_token_in_rank,  // 输入 [num_tokens,num_ranks] 路由布尔矩阵。
                                int num_tokens,  // 输入 token 数。
                                int num_worst_tokens,  // worst-case 容量，非 0 时跳过 CPU counter 写回。
                                int num_channels,  // channel 数。
                                int expert_alignment,  // expert token 计数对齐粒度。
                                const int rdma_clean_offset,  // RDMA buffer 控制区清理起点，单位 int。
                                const int rdma_num_int_clean,  // RDMA buffer 需要清零的 int 数。
                                const int nvl_clean_offset,  // NVL buffer 控制区清理起点，单位 int。
                                const int nvl_num_int_clean,  // NVL buffer 需要清零的 int 数。
                                int* rdma_channel_prefix_matrix,  // 输出 [kNumRDMARanks,num_channels] RDMA channel prefix。
                                int* recv_rdma_rank_prefix_sum,  // 输出 [kNumRDMARanks] RDMA rank prefix。
                                int* gbl_channel_prefix_matrix,  // 输出 [num_ranks,num_channels] 全局 NVL channel prefix。
                                int* recv_gbl_rank_prefix_sum,  // 输出 [num_ranks] 全局 rank prefix。
                                void* rdma_buffer_ptr,  // NVSHMEM symmetric RDMA buffer。
                                void** buffer_ptrs,  // 节点内 NVL peer buffer 指针表。
                                int** barrier_signal_ptrs,  // 节点内 barrier signal 指针表。
                                int rank,  // 当前全局 rank。
                                const nvshmem_team_t rdma_team) {  // low-latency 模式下同 GPU index 的 RDMA team。
    auto sm_id = static_cast<int>(blockIdx.x);  // block0 处理控制面，其余 block 处理每个目标 RDMA rank 的 channel 统计。
    auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();  // 线程、warp、lane 编号。
    auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;  // block 线程数和 warp 数。

    auto rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS, nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;  // 全局 rank 拆成节点号和节点内 GPU 号。
    auto num_rdma_experts = num_experts / kNumRDMARanks, num_nvl_experts = num_rdma_experts / LEGACY_NUM_MAX_NVL_PEERS;  // RDMA rank/expert 与 NVL rank/expert 数。

    if (sm_id == 0) {
        // 0 号 SM 处理控制面：先 quiet 所有可能未完成的 IBGDA WR，再做跨节点与
        // 节点内 barrier，保证接下来可以安全清理 RDMA/NVL staging buffer。
        EP_DEVICE_ASSERT(num_warps > 1);  // 至少需要多个 warp 才能同时做通信和清理。
        EP_DEVICE_ASSERT(kNumRDMARanks <= num_threads);  // 线程数要覆盖所有 RDMA rank。

        // waiting for all previous inflight wrs to complete,
        // in case of rewriting cleared rdma_buffer
        auto qps_per_rdma_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;  // 每个 RDMA peer 的 QP 数。
        for (int i = thread_id; i < qps_per_rdma_rank * (kNumRDMARanks - 1); i += num_threads) {
            auto dst_rdma_rank = (i / qps_per_rdma_rank + rdma_rank + 1) % kNumRDMARanks;  // 遍历除本地外的目标 RDMA rank。
            auto qp_id = i % qps_per_rdma_rank;  // 当前需要 quiet 的 QP id。
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), qp_id);  // 等待该 QP 上未完成 WR 完成。
        }
        __syncthreads();  // 等所有 QP quiet 完成。

        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);  // 用一个线程执行 RDMA PE 间同步。
        barrier_block<LEGACY_NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);  // 节点内 NVL peers 同步。

        // 第一阶段交换 RDMA 级计数。rdma_recv_num_tokens_mixed 逻辑形状为
        // [kNumRDMARanks, 8 + num_rdma_experts + 1]，分别保存每个 NVL rank 的
        // token 数、每个 RDMA expert 的 token 数、以及该 RDMA rank 总 token 数。
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);  // RDMA buffer 的 int 视图，用于清理控制区。
        auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr, LEGACY_NUM_MAX_NVL_PEERS + num_rdma_experts + 1, kNumRDMARanks);  // RDMA 计数交换 buffer。

        // 在发送计数的同时清理后续数据 dispatch 要用的 RDMA 控制区；quiet+barrier
        // 已保证不会覆盖上一次还在使用的数据。
        EP_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));  // 计数交换区不能覆盖后续清理区。
        #pragma unroll
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;  // 清零 RDMA 控制元数据。

        // Copy to send buffer
        #pragma unroll
        for (int i = thread_id; i < num_ranks; i += num_threads)
            rdma_recv_num_tokens_mixed.send_buffer(i / LEGACY_NUM_MAX_NVL_PEERS)[i % LEGACY_NUM_MAX_NVL_PEERS] = num_tokens_per_rank[i];  // 写给目标 RDMA rank 的每 NVL rank 计数。
        #pragma unroll
        for (int i = thread_id; i < num_experts; i += num_threads)
            rdma_recv_num_tokens_mixed.send_buffer(i / num_rdma_experts)[LEGACY_NUM_MAX_NVL_PEERS + i % num_rdma_experts] =
                num_tokens_per_expert[i];  // 写给目标 RDMA rank 的 expert 计数。
        if (thread_id < kNumRDMARanks)
            rdma_recv_num_tokens_mixed.send_buffer(thread_id)[LEGACY_NUM_MAX_NVL_PEERS + num_rdma_experts] = num_tokens_per_rdma_rank[thread_id];  // 写 RDMA rank 总 token 数。
        __syncthreads();  // 等 send buffer 填充完成。

        // 向其他 RDMA rank 发送计数：远端走 IBGDA put，本地 rdma_rank 直接 warp copy。
        // TODO: more light fence or barrier or signaling
        // 中文：这里仍使用较重的同步，后续可优化为更细粒度 fence 或和 NVL 清理重叠。
        // TODO: overlap EP barrier and NVL cleaning
        for (int i = warp_id; i < kNumRDMARanks; i += num_warps) {
            if (i != rdma_rank) {
                // 中文：远端 count message 写到对方 recv_buffer(rdma_rank)，即“对方视角来自本 RDMA rank”的槽。
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.send_buffer(i)),  // 本地 send count buffer。
                                                  (LEGACY_NUM_MAX_NVL_PEERS + num_rdma_experts + 1) * sizeof(int),  // count message 字节数。
                                                  translate_dst_rdma_rank<kLowLatencyMode>(i, nvl_rank),  // 目标 RDMA rank。
                                                  0,  // 使用 QP 0 发送计数控制面。
                                                  lane_id,  // warp lane id。
                                                  0);  // slot index 对计数消息无意义，置 0。
            } else {
                UNROLLED_WARP_COPY(1,  // 本地 RDMA rank 直接 warp copy，避免走 IBGDA。
                                   lane_id,  // 当前 warp lane。
                                  LEGACY_NUM_MAX_NVL_PEERS + num_rdma_experts + 1,  // count message int 数。
                                  rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank),  // 本地 recv count buffer。
                                  rdma_recv_num_tokens_mixed.send_buffer(i),  // 本地 send count buffer。
                                  // 中文：本地路径仍走同一 copy 宏，保持远端/本地 count message 布局一致。
                                  // 中文：下面两个 functor 分别指定 volatile 读和 non-allocating 写，减少缓存污染。
                                  ld_volatile_global,  // volatile load。
                                  st_na_global);  // non-allocating store。
            }  // 中文：结束本地 RDMA rank 的直接 copy 分支。
        }  // 中文：结束向所有 RDMA rank 发布 count message 的循环。
        __syncthreads();  // 等所有计数 put/copy 发起完成。

        // Wait previous operations to be finished
        // 中文：计数控制面必须先全局可见，后续 dispatch 主 kernel 才能安全读取 prefix/counter。
        if (thread_id < kNumRDMARanks and thread_id != rdma_rank)
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank), 0);  // 等每个远端 rank 的计数 put 完成。
        __syncthreads();  // 等 quiet 完成。

        // Barrier
        if (thread_id == 0)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);  // RDMA 层 barrier，确保所有 rank 都收到计数。
        __syncthreads();  // 等同步线程完成。

        // 第二阶段把 RDMA 级计数在节点内 8 个 NVLink rank 间交换/规约，得到每个
        // 本地 expert 最终要接收的 token 数，以及每个全局 rank 的 prefix。
        auto nvl_send_buffer = thread_id < LEGACY_NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;  // 每个线程对应一个 NVL peer 的发送 buffer。
        auto nvl_recv_buffer = buffer_ptrs[nvl_rank];  // 当前 nvl_rank 的本地接收 buffer。
        auto nvl_reduced_num_tokens_per_expert = Buffer<int>(nvl_recv_buffer, num_rdma_experts).advance_also(nvl_send_buffer);  // RDMA expert 计数规约结果。
        auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS);  // 发送给 NVL peer 的 rank 计数。
        auto nvl_send_num_tokens_per_expert = AsymBuffer<int>(nvl_send_buffer, num_nvl_experts, LEGACY_NUM_MAX_NVL_PEERS);  // 发送给 NVL peer 的 expert 计数。
        auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS);  // 从 NVL peer 收到的 rank 计数。
        auto nvl_recv_num_tokens_per_expert = AsymBuffer<int>(nvl_recv_buffer, num_nvl_experts, LEGACY_NUM_MAX_NVL_PEERS);  // 从 NVL peer 收到的 expert 计数。

        // Clean up for later data dispatch
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);  // 本地 NVL buffer 的 int 视图。
        EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes +
                             nvl_send_num_tokens_per_expert.total_bytes <=
                         nvl_clean_offset * sizeof(int));  // 计数交换区不能覆盖后续待清理控制区。
        #pragma unroll
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;  // 清零 NVL 控制元数据。

        // Reduce number of tokens per expert into the NVL send buffer
        // TODO: may use NVSHMEM reduction
        EP_DEVICE_ASSERT(num_rdma_experts <= num_threads);  // 线程数足够覆盖 RDMA expert。
        if (thread_id < num_rdma_experts) {
            int sum = 0;  // 当前 RDMA expert 的跨 RDMA rank token 总数。
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i)
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[LEGACY_NUM_MAX_NVL_PEERS + thread_id];  // 汇总所有 RDMA 来源。
            nvl_reduced_num_tokens_per_expert[thread_id] = sum;  // 写入本地 NVL 共享规约区。
        }
        __syncthreads();  // 等 expert 规约结果写完。

        // Reduce RDMA received tokens
        if (thread_id == 0) {
            int sum = 0;  // RDMA rank prefix 累计和。
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i) {
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[LEGACY_NUM_MAX_NVL_PEERS + num_rdma_experts];  // 累加第 i 个 RDMA rank token 数。
                recv_rdma_rank_prefix_sum[i] = sum;  // 写 RDMA rank prefix。
            }
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1)
                    ;  // 等 host 侧把 mapped counter 置为 -1。
                *moe_recv_rdma_counter_mapped = sum;  // 写回 RDMA 层接收 token 总数。
            }
        }

        // Send numbers of tokens per rank/expert to NVL ranks
        EP_DEVICE_ASSERT(LEGACY_NUM_MAX_NVL_PEERS <= num_threads);  // 线程数要覆盖 8 个 NVL peer。
        if (thread_id < LEGACY_NUM_MAX_NVL_PEERS) {
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i)
                nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] = rdma_recv_num_tokens_mixed.recv_buffer(i)[thread_id];  // 把发给 thread_id NVL rank 的 RDMA 计数写入发送区。
            #pragma unroll
            for (int i = 0; i < num_nvl_experts; ++i)
                nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] = nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i];  // 写该 NVL rank 的本地 expert 计数。
        }
        barrier_block<LEGACY_NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);  // 节点内同步，确保所有 NVL peer 计数可读。

        // Reduce the number of tokens per rank/expert
        EP_DEVICE_ASSERT(num_nvl_experts <= num_threads);  // 线程数要覆盖本地 NVL expert。
        if (thread_id == 0) {
            int sum = 0;  // 全局 rank prefix 累计和。
            #pragma unroll
            for (int i = 0; i < num_ranks; ++i) {
                int src_rdma_rank = i / LEGACY_NUM_MAX_NVL_PEERS, src_nvl_rank = i % LEGACY_NUM_MAX_NVL_PEERS;  // 全局 rank 拆成 RDMA/NVL 坐标。
                sum += nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank];  // 累加该来源 rank 的 token 数。
                recv_gbl_rank_prefix_sum[i] = sum;  // 写全局 rank prefix。
            }
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_counter_mapped) != -1)
                    ;  // 等 host 侧 mapped counter 置为 -1。
                *moe_recv_counter_mapped = sum;  // 写最终接收 token 总数。
            }
        }
        if (thread_id < num_nvl_experts) {
            int sum = 0;  // 当前本地 expert 的 token 总数。
            #pragma unroll
            for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
                sum += nvl_recv_num_tokens_per_expert.buffer(i)[thread_id];  // 汇总节点内所有 NVL peer 的 expert 计数。
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;  // expert 容量按对齐粒度向上取整。
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1)
                    ;  // 等 host 侧该 expert counter 置为 -1。
                moe_recv_expert_counter_mapped[thread_id] = sum;  // 写回本地 expert 接收 token 数。
            }
        }

        // Finally barrier
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);  // 最后 RDMA PE 间同步。
        barrier_block<LEGACY_NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);  // 最后 NVL peer 间同步。
    } else {
        // Calculate meta data
        int dst_rdma_rank = sm_id - 1;  // block1..kNumRDMARanks 分别负责一个目标 RDMA rank 的 channel 统计。
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;  // 当前 channel 负责扫描的 token 范围。
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);  // 按 channel 均分 token。

            // Iterate over tokens
            int total_count = 0, per_nvl_rank_count[LEGACY_NUM_MAX_NVL_PEERS] = {0};  // RDMA 总 token 数与每个 NVL rank token 数。
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
                EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");  // 8 个 bool 可一次按 uint64 读取。
                auto is_token_in_rank_uint64 =
                    *reinterpret_cast<const uint64_t*>(is_token_in_rank + i * num_ranks + dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS);  // 读取该 token 对目标 RDMA rank 下 8 个 NVL rank 的命中 mask。
                auto is_token_in_rank_values = reinterpret_cast<const bool*>(&is_token_in_rank_uint64);  // 将 64-bit mask 解释为 8 个 bool。
                #pragma unroll
                for (int j = 0; j < LEGACY_NUM_MAX_NVL_PEERS; ++j)
                    per_nvl_rank_count[j] += is_token_in_rank_values[j];  // 统计每个 NVL rank 需要接收多少 token。
                total_count += (is_token_in_rank_uint64 != 0);  // 同一 token 只要命中该 RDMA rank 任一 NVL peer，就计为一个 RDMA token。
            }

            // Warp reduce
            total_count = warp_reduce_sum(total_count);  // warp 内规约 RDMA token 数。
            #pragma unroll
            for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
                per_nvl_rank_count[i] = warp_reduce_sum(per_nvl_rank_count[i]);  // warp 内规约每个 NVL rank 的 token 数。

            // Write into channel matrix
            // 中文：channel matrix 的行是目标 rank/peer，列是 channel_id，后续 prefix scan 逐列读取。
            if (elect_one_sync()) {
                #pragma unroll
                for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
                    gbl_channel_prefix_matrix[(dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + i) * num_channels + channel_id] = per_nvl_rank_count[i];  // 写全局 rank/channel 计数。
                // 中文：RDMA prefix 统计 token 去往目标 RDMA rank 的总量，NVL prefix 进一步拆到节点内 rank。
                // per_nvl_rank_count 的第 i 项对应目标 RDMA rank 内第 i 个 NVL peer。
                rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] = total_count;  // 写 RDMA rank/channel 计数。
            }
        }

        // Calculate prefix sum
        __syncthreads();  // 等所有 channel 计数写入矩阵。
        if (thread_id == 0) {
            auto prefix_row = rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;  // 当前目标 RDMA rank 的 channel 行。
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                prefix_row[i] += prefix_row[i - 1];  // RDMA channel count 原地转 prefix。
            // 中文：prefix_row[i] 表示截至 channel i 的累计 token 数，channel 起点由前一项给出。
        }

        EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");  // 一个 warp 可覆盖所有 NVL peer。
        if (thread_id < LEGACY_NUM_MAX_NVL_PEERS) {
            auto prefix_row = gbl_channel_prefix_matrix + (dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + thread_id) * num_channels;  // 当前全局 rank 的 channel 行。
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                prefix_row[i] += prefix_row[i - 1];  // NVL channel count 原地转 prefix。
            // 中文：每个 NVL peer 独立做 channel prefix，forwarder 后续按它切分节点内转发区间。
        }
    }
}

void notify_dispatch(const int* num_tokens_per_rank,  // 输入 [num_ranks] rank 计数。
                     int* moe_recv_counter_mapped,  // 输出 mapped 总接收计数。
                     int num_ranks,  // 全局 rank 数。
                     const int* num_tokens_per_rdma_rank,  // 输入 [num_rdma_ranks] RDMA rank 计数。
                     int* moe_recv_rdma_counter_mapped,  // 输出 mapped RDMA 接收计数。
                     const int* num_tokens_per_expert,  // 输入 [num_experts] expert 计数。
                     int* moe_recv_expert_counter_mapped,  // 输出 mapped expert 计数。
                     int num_experts,  // 全局 expert 数。
                     const bool* is_token_in_rank,  // 输入 token-rank 路由表。
                     int num_tokens,  // token 数。
                     int num_worst_tokens,  // worst-case token 容量。
                     int num_channels,  // channel 数。
                     int hidden_int4,  // hidden 的 int4 长度。
                     int num_scales,  // FP8 scale 数。
                     int num_topk,  // top-k 路数。
                     int expert_alignment,  // expert 对齐粒度。
                     int* rdma_channel_prefix_matrix,  // 输出 RDMA channel prefix。
                     int* recv_rdma_rank_prefix_sum,  // 输出 RDMA rank prefix。
                     int* gbl_channel_prefix_matrix,  // 输出全局 channel prefix。
                     int* recv_gbl_rank_prefix_sum,  // 输出全局 rank prefix。
                     void* rdma_buffer_ptr,  // RDMA symmetric buffer。
                     int num_max_rdma_chunked_recv_tokens,  // RDMA recv ring 容量。
                     void** buffer_ptrs,  // NVL peer buffer 指针表。
                     int num_max_nvl_chunked_recv_tokens,  // NVL recv ring 容量。
                     int** barrier_signal_ptrs,  // NVL barrier 指针表。
                     int rank,  // 当前 rank。
                     cudaStream_t stream,  // CUDA 发射流。
                     int64_t num_rdma_bytes,  // RDMA buffer 总字节数。
                     int64_t num_nvl_bytes,  // NVL buffer 总字节数。
                     bool low_latency_mode) {  // 是否启用 low-latency PE 编号。
#define NOTIFY_DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                                    \
    {                                                                                                                                  \
        /* low_latency_mode 只影响 kernel 模板布尔值，实参布局保持一致，便于 host 端共用一套参数准备逻辑。 */                         \
        auto notify_dispatch_func = low_latency_mode ? notify_dispatch<true, num_rdma_ranks> : notify_dispatch<false, num_rdma_ranks>; \
        /* notify_dispatch 只处理路由统计和 prefix/control 区初始化，不搬运 hidden payload。 */                                       \
        LAUNCH_KERNEL(&cfg,                                                                                                            \
                      /* num_tokens_per_rank: [num_ranks]，每个目标 rank 的接收 token 数。 */                                         \
                      notify_dispatch_func,                                                                                            \
                      num_tokens_per_rank,                                                                                             \
                      /* moe_recv_counter_mapped: rank 级接收计数，Python 侧用于分配输出。 */                                        \
                      moe_recv_counter_mapped,                                                                                         \
                      num_ranks,                                                                                                       \
                      /* num_tokens_per_rdma_rank: [num_rdma_ranks]，按跨节点 RDMA rank 聚合的计数。 */                               \
                      num_tokens_per_rdma_rank,                                                                                        \
                      moe_recv_rdma_counter_mapped,                                                                                    \
                      /* num_tokens_per_expert: [num_experts]，expert 维度计数用于 MoE local expert 排布。 */                         \
                      num_tokens_per_expert,                                                                                           \
                      moe_recv_expert_counter_mapped,                                                                                  \
                      num_experts,                                                                                                     \
                      /* is_token_in_rank: [num_tokens,num_ranks]，dispatch 主 kernel 复用这张布尔路由表。 */                         \
                      is_token_in_rank,                                                                                                \
                      /* num_tokens/num_worst_tokens 决定真实 token 数与 CUDA graph 固定容量上界。 */                                  \
                      num_tokens,                                                                                                      \
                      num_worst_tokens,                                                                                                \
                      /* num_channels/expert_alignment 控制 channel 切片和 expert 输出对齐。 */                                      \
                      num_channels,                                                                                                    \
                      expert_alignment,                                                                                                \
                      /* clean_meta.first/second 描述控制区中需要清零的 int 偏移和长度。 */                                           \
                      rdma_clean_meta.first,                                                                                           \
                      rdma_clean_meta.second,                                                                                          \
                      /* nvl_clean_meta 与 rdma_clean_meta 含义相同，但作用在节点内 peer buffer。 */                                  \
                      nvl_clean_meta.first,                                                                                            \
                      nvl_clean_meta.second,                                                                                           \
                      /* prefix matrix 会被后续 dispatch/combine kernel 作为跨 rank 定位表读取。 */                                  \
                      rdma_channel_prefix_matrix,                                                                                      \
                      recv_rdma_rank_prefix_sum,                                                                                       \
                      /* gbl prefix 是 RDMA 节点内 NVL 转发阶段使用的全局 rank 定位表。 */                                           \
                      gbl_channel_prefix_matrix,                                                                                       \
                      recv_gbl_rank_prefix_sum,                                                                                        \
                      /* rdma_buffer_ptr 是 NVSHMEM symmetric buffer；buffer_ptrs 是同节点 NVL peer 地址表。 */                       \
                      rdma_buffer_ptr,                                                                                                 /* RDMA symmetric buffer。 */ \
                      buffer_ptrs,                                                                                                     /* NVL peer buffer 表。 */ \
                      barrier_signal_ptrs,                                                                                             /* barrier 信号表。 */ \
                      rank,                                                                                                            /* 当前 rank。 */ \
                      nvshmem::cpu_rdma_team);                                                                                         /* CPU RDMA team。 */ \
    }                                                                                                                                  \
    break

    constexpr int kNumThreads = 512;  // notify kernel 每 block 线程数。
    const auto num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;  // RDMA rank 数等于全局 rank / 每节点 GPU 数。

    // Get clean meta
    auto rdma_clean_meta =
        get_rdma_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);  // 计算 RDMA 控制区清理范围。
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4,  // hidden 的 int4 数。
                                             num_scales,  // FP8 scale 数。
                                             num_topk,  // top-k idx 数。
                                             num_topk,  // top-k weight 数。
                                             num_rdma_ranks,  // RDMA rank 数。
                                             LEGACY_NUM_MAX_NVL_PEERS,  // 节点内 NVL rank 上限。
                                             num_max_nvl_chunked_recv_tokens,  // NVL recv chunk 容量。
                                             num_channels,  // channel 数。
                                             true);  // cached_dispatch=true，计算 NVL 控制区清理范围。
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);  // RDMA 清理范围不能越界。
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);  // NVL 清理范围不能越界。
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());  // device kernel 中部分 offset 使用 int。
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());  // NVL buffer 同样要求 int 范围内。

    // Launch kernel
    SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);  // block0 控制面 + 每个 RDMA rank 一个统计 block。
    SWITCH_RDMA_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);  // 根据 RDMA rank 数选择模板实例。
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;  // 单 token 最多广播到 8 个 RDMA rank，限制局部数组大小。
}

template <bool kLowLatencyMode,  // 是否低延迟模式。
          int kNumRDMARanks,  // RDMA rank 数。
          bool kCachedMode,  // 是否 cached dispatch。
          int kNumTMABytesPerWarp,  // 每 warp TMA staging 字节数。
          int kNumDispatchRDMASenderWarps,  // RDMA sender warp 数。
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>  // 单 token 最多命中的 RDMA rank 数。
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + LEGACY_NUM_MAX_NVL_PEERS) * 32), 1)
    dispatch(int4* recv_x,  // 最终输出 token 特征，形状 [num_recv_tokens,hidden] 的 int4 视图。
             float* recv_x_scales,  // 最终输出 FP8 scales，可为空。
             topk_idx_t* recv_topk_idx,  // 最终输出 topk_idx，可为空。
             float* recv_topk_weights,  // 最终输出 topk 权重，可为空。
             SourceMeta* recv_src_meta,  // 非 cached 输出 source meta，combine 需要它回传。
             const int4* x,  // 输入 token 特征，int4 视图。
             const float* x_scales,  // 输入 FP8 scales，可为空。
             const topk_idx_t* topk_idx,  // 输入 topk_idx，可为空。
             const float* topk_weights,  // 输入 topk 权重，可为空。
             int* send_rdma_head,  // 非 cached 输出 RDMA send head，[num_tokens,kNumRDMARanks]。
             int* send_nvl_head,  // 非 cached 输出 NVL send head，[rdma_recv_tokens,8]。
             int* recv_rdma_channel_prefix_matrix,  // 非 cached 输出接收侧 RDMA channel prefix。
             int* recv_gbl_channel_prefix_matrix,  // 非 cached 输出接收侧全局 channel prefix。
             const int* rdma_channel_prefix_matrix,  // 输入 RDMA channel prefix。
             const int* recv_rdma_rank_prefix_sum,  // 输入 RDMA rank prefix。
             const int* gbl_channel_prefix_matrix,  // 输入全局 channel prefix。
             const int* recv_gbl_rank_prefix_sum,  // 输入全局 rank prefix。
             const bool* is_token_in_rank,  // 输入 token-rank 路由表。
             int num_tokens,  // 输入 token 数。
             int num_worst_tokens,  // worst-case 容量。
             int hidden_int4,  // hidden 的 int4 长度。
             int num_scales,  // FP8 scale 数。
             int num_topk,  // top-k 路数。
             int num_experts,  // 全局 expert 数。
             int scale_token_stride,  // scale token stride。
             int scale_hidden_stride,  // scale hidden stride。
             void* rdma_buffer_ptr,  // RDMA symmetric staging buffer。
             int num_max_rdma_chunked_send_tokens,  // RDMA send chunk 上限。
             int num_max_rdma_chunked_recv_tokens,  // RDMA recv ring 容量。
             void** buffer_ptrs,  // NVL peer buffer 指针表。
             int num_max_nvl_chunked_send_tokens,  // NVL send chunk 上限。
             int num_max_nvl_chunked_recv_tokens,  // NVL recv ring 容量。
             int rank,  // 当前全局 rank。
             int num_ranks) {  // 全局 rank 数。
    enum class WarpRole { kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers };  // 一个 block 内不同 warp 的职责。

    const auto num_sms = static_cast<int>(gridDim.x);  // kernel block 数。
    const auto sm_id = static_cast<int>(blockIdx.x);  // 当前 block id。
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;  // 线程数和 warp 数。
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();  // 线程/warp/lane 编号。
    const auto num_channels = num_sms / 2, channel_id = sm_id / 2;  // 偶奇 block 组成一个 channel。
    const bool is_forwarder = sm_id % 2 == 0;  // 偶数 block 做 RDMA->NVL forwarder，奇数 block 做 RDMA sender/NVL receiver。
    const auto rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS, nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;  // 全局 rank 拆成 RDMA/NVL 坐标。

    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_channels or ibgda_get_state()->num_rc_per_pe >= num_sms);  // QP 数要覆盖 channel 或 block。

    const auto role_meta = [=]() -> std::pair<WarpRole, int> {  // 根据 block 奇偶和 warp id 分配角色。
        if (is_forwarder) {
            if (warp_id < LEGACY_NUM_MAX_NVL_PEERS) {
                return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % LEGACY_NUM_MAX_NVL_PEERS};  // 前 8 个 warp 各负责一个目标 NVL rank。
            } else {
                return {WarpRole::kForwarderCoordinator, warp_id - LEGACY_NUM_MAX_NVL_PEERS};  // 剩余 warp 做 forwarder 协调。
            }
        } else if (warp_id < kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASender, -1};  // 奇数 block 前若干 warp 负责把本地 token 打包到 RDMA buffer。
        } else if (warp_id == kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASenderCoordinator, -1};  // 一个 warp 负责批量发起 RDMA put。
        } else {
            return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % LEGACY_NUM_MAX_NVL_PEERS};  // 剩余 warp 从 NVL buffer 收 token。
        }
    }();
    auto warp_role = role_meta.first;  // 当前 warp 的角色。
    auto target_rank = role_meta.second;  // Not applicable for RDMA senders
    EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + LEGACY_NUM_MAX_NVL_PEERS);  // warp 总数必须与角色划分一致。

    // Data checks
    EP_DEVICE_ASSERT(num_topk <= 32);  // top-k lane 数不超过 warp lane 数。

    // RDMA symmetric layout
    EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");  // 8 个 bool 可按 uint64 mask 搬运。
    auto hidden_bytes = hidden_int4 * sizeof(int4);  // hidden payload 字节数。
    auto scale_bytes = num_scales * sizeof(float);  // scale payload 字节数。
    auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_scales, num_topk, num_topk);  // 单 token RDMA/NVL 消息字节数。
    auto rdma_channel_data = SymBuffer<uint8_t>(
        rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);  // 每 RDMA rank/channel 的 token ring data。
    auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, LEGACY_NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);  // RDMA channel start/end meta。
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);  // RDMA ring head。
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);  // RDMA ring tail。

    // NVL buffer layouts
    // NOTES: `rs_wr_buffer_ptr` means "Read for Senders, Write for Receivers", `ws_rr_buffer_ptr` means "Write for Senders, Read for
    // Receivers"
    // 这里的命名是 legacy NVL 双缓冲约定：
    //   rs_wr = sender 视角读、receiver 视角写；
    //   ws_rr = sender 视角写、receiver 视角读。
    // forwarder 和 receiver 对同一块 peer buffer 的读写方向相反，所以必须同时维护“指针”和“rank 维度”的对应关系。
    void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;  // 两个方向的 NVL buffer 指针：读/写角色相反。
    int rs_wr_rank = 0, ws_rr_rank = 0;  // 两个 buffer 对应的 rank 编号。
    if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        // forwarder 读本 rank buffer 中 RDMA 已到达的数据，再写到目标 NVL rank 可读的 buffer。
        rs_wr_buffer_ptr = buffer_ptrs[nvl_rank], ws_rr_buffer_ptr = buffer_ptrs[target_rank], rs_wr_rank = nvl_rank,
        ws_rr_rank = target_rank;  // forwarder 从本地 RDMA/NVL buffer 读，写目标 NVL rank buffer。
    if (warp_role == WarpRole::kNVLReceivers)
        // receiver 的视角刚好反过来：它从负责 target_rank 的写入 buffer 读数据，并把 head 写回给对方观察。
        rs_wr_buffer_ptr = buffer_ptrs[target_rank], ws_rr_buffer_ptr = buffer_ptrs[nvl_rank], rs_wr_rank = target_rank,
        ws_rr_rank = nvl_rank;  // receiver 从目标/source rank buffer 读，写/观察本地 buffer。

    // Allocate buffers
    auto nvl_channel_x = AsymBuffer<uint8_t>(ws_rr_buffer_ptr,  // 写侧/读侧共享的 NVL buffer 指针。
                                             num_max_nvl_chunked_recv_tokens * num_bytes_per_token,  // 每 rank/channel payload 字节数。
                                             LEGACY_NUM_MAX_NVL_PEERS,  // NVL rank 维度。
                                             channel_id,  // 当前 channel。
                                             num_channels,  // channel 总数。
                                             rs_wr_rank)  // source/write rank 维度。
                             .advance_also(rs_wr_buffer_ptr);  // NVL token payload ring，按 channel/source-rank 分片。
    auto nvl_channel_prefix_start =
        AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);  // 每 source RDMA rank 的 channel start prefix。
    auto nvl_channel_prefix_end = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                                      .advance_also(rs_wr_buffer_ptr);  // 每 source RDMA rank 的 channel end prefix。
    auto nvl_channel_head =
        AsymBuffer<int>(rs_wr_buffer_ptr, 1, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, ws_rr_rank).advance_also(ws_rr_buffer_ptr);  // NVL ring head，由 receiver 推进。
    auto nvl_channel_tail =
        AsymBuffer<int>(ws_rr_buffer_ptr, 1, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);  // NVL ring tail，由 forwarder 推进。

    // RDMA sender warp synchronization
    // NOTES: `rdma_send_channel_tail` means the latest released tail
    // NOTES: `rdma_send_channel_window` means the ongoing 32 transactions' status
    // sender warp 会并行打包不同 token，完成顺序可能乱序；window 用 32-bit bitmap 记录最近 32 个 tail 是否完成。
    // coordinator 只推进“连续完成”的 tail，这样 RDMA put 永远发送一段连续 token，避免远端 forwarder 看到洞。
    __shared__ int rdma_send_channel_lock[kNumRDMARanks];  // 每个 RDMA 目标一个 CTA 内锁，保护 tail/window。
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];  // 已完成写入并可 RDMA put 的连续 tail。
    __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];  // 32-slot 完成窗口，处理 sender warp 乱序完成。
    auto sync_rdma_sender_smem = []() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumDispatchRDMASenderWarps + 1) * 32)); };  // RDMA sender 组专用 named barrier。

    // TMA stuffs
    // TMA buffer 被 forwarder/receiver 复用为“payload 暂存区 + mbarrier”。每个 target_rank 独占一段，
    // 这样多个 NVL receiver warp 可以并发 load/store，而不会覆盖彼此正在等待的 barrier。
    extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];  // TMA 中转 shared memory。
    auto tma_buffer = smem_tma_buffer + target_rank * kNumTMABytesPerWarp;  // 每个 target NVL rank 一段 TMA buffer。
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);  // TMA barrier 放在 token buffer 之后。
    uint32_t tma_phase = 0;  // TMA barrier phase。
    if ((warp_role == WarpRole::kRDMAAndNVLForwarder or warp_role == WarpRole::kNVLReceivers) and elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);  // 初始化当前 warp 使用的 TMA barrier。
        fence_barrier_init();  // 确保 barrier 初始化可见。
        EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumTMABytesPerWarp);  // TMA shared buffer 容量检查。
    }
    __syncwarp();  // 等 warp 内 TMA 初始化完成。

    // Forward warp synchronization
    __shared__ volatile int forward_channel_head[LEGACY_NUM_MAX_NVL_PEERS][kNumRDMARanks];  // 每个 forwarder warp 对每个 RDMA rank 的消费进度。
    __shared__ volatile bool forward_channel_retired[LEGACY_NUM_MAX_NVL_PEERS];  // forwarder warp 是否完成。
    auto sync_forwarder_smem = []() { asm volatile("barrier.sync 1, %0;" ::"r"((LEGACY_NUM_MAX_NVL_PEERS + 1) * 32)); };  // forwarder 组 named barrier。

    if (warp_role == WarpRole::kRDMASender) {
        // Get tasks
        int token_start_idx, token_end_idx;  // 当前 channel 负责扫描的 token 范围。
        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);  // 按 channel 均分输入 token。

        // Send number of tokens in this channel by `-value - 1`
        // meta 用负数编码 ready：例如 prefix=0 写成 -1，prefix=7 写成 -8。
        // forwarder 读取到负数后用 -value-1 还原真实 prefix；这样 0 prefix 也能和“未写入的 0”区分开。
        EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr =
                dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);  // 本地写 recv buffer，远端先写 send buffer。
            if (lane_id < LEGACY_NUM_MAX_NVL_PEERS) {
                // lane 0..7 写目标 RDMA rank 内每个 NVL rank 的 start prefix。
                // 举例：channel_id=2 时，它写的是 channel 1 的累计 end，代表 channel 2 的起点。
                dst_ptr[lane_id] =
                    -(channel_id == 0
                          ? 0
                          // 中文：非 0 channel 的起点等于前一 channel 的累计 end prefix。
                          : gbl_channel_prefix_matrix[(dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1]) -
                    1;  // 写给目标 RDMA rank 内每个 NVL rank 的 start prefix，负数编码表示 ready。
            } else if (lane_id < LEGACY_NUM_MAX_NVL_PEERS * 2) {
                // lane 8..15 写对应 NVL rank 的 end prefix；end-start 就是该 channel 应转发到这个 NVL rank 的 token 数。
                dst_ptr[lane_id] =
                    -gbl_channel_prefix_matrix[(dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + lane_id - LEGACY_NUM_MAX_NVL_PEERS) * num_channels +
                                               channel_id] -
                    1;  // 写给目标 RDMA rank 内每个 NVL rank 的 end prefix。
            } else if (lane_id == LEGACY_NUM_MAX_NVL_PEERS * 2) {
                // lane 16 写 RDMA rank 级 start prefix，coordinator/forwarder 用它判断 RDMA ring 中本 channel 起点。
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1;  // 写 RDMA channel start prefix。
            } else if (lane_id == LEGACY_NUM_MAX_NVL_PEERS * 2 + 1) {
                // lane 17 写 RDMA rank 级 end prefix；剩余 lane 空闲，因为 meta 总共只需要 18 个 int。
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;  // 写 RDMA channel end prefix。
            }
            __syncwarp();  // 等 meta 写入 send buffer 完成。

            // Issue RDMA for non-local ranks
            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)), // 远端 meta recv buffer。
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)), // 本地 meta staging buffer。
                                                  sizeof(int) * (LEGACY_NUM_MAX_NVL_PEERS * 2 + 2), // meta 长度：8 个 start、8 个 end、2 个 RDMA prefix。
                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), // 目标 PE。
                                                  channel_id, // channel 对应的 IBGDA 队列。
                                                  lane_id,    // warp lane 协作。
                                                  0);         // 保留参数。
            }
        }
        sync_rdma_sender_smem();  // 等 RDMA sender 和 coordinator 完成 shared 初始化同步。

        // Iterate over tokens and copy into buffer
        int64_t token_idx;  // 当前扫描 token id。
        // global_rdma_tail_idx 是按目标 RDMA rank 独立累计的逻辑 tail；
        // cached_rdma_channel_head 是从远端释放 head 缓存来的值，用于避免覆盖未被 forwarder 消费的 ring 槽。
        int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;  // 本 lane 目标 RDMA rank 的 head 缓存和全局 tail 计数。
        auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);  // lane 对应目标 RDMA rank 的 staging buffer。
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
            // Read RDMA rank existence
            uint64_t is_token_in_rank_uint64 = 0;  // 当前 token 对某个 RDMA rank 下 8 个 NVL rank 的命中 mask。
            if (lane_id < kNumRDMARanks) {
                is_token_in_rank_uint64 =
                    __ldg(reinterpret_cast<const uint64_t*>(is_token_in_rank + token_idx * num_ranks + lane_id * LEGACY_NUM_MAX_NVL_PEERS));  // 读取目标 RDMA rank 的 8-bit NVL mask。
                global_rdma_tail_idx += (is_token_in_rank_uint64 != 0);  // 只要命中任一 NVL rank，就给该 RDMA rank 分配一个 token 槽。
            }
            __syncwarp();  // 等所有 lane 更新各自 RDMA tail。

            // Skip the token which does not belong to this warp
            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;  // 多个 sender warp 轮流打包 token，非本 warp 负责的 token 跳过。
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;  // 当前 token 在 lane 对应 RDMA rank 的 tail 槽。

            // Wait the remote buffer to be released
            auto start_time = clock64();  // 等待 RDMA ring 空槽开始时间。
            // ring 空间判断：tail-head >= capacity 表示目标 RDMA rank 的接收 ring 已满。
            // sender 只在 token 确实命中该 RDMA rank 时等待；未命中时 rdma_tail_idx=-1 直接跳过。
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));  // 读取目标 RDMA rank 已释放 head。

                // Timeout check
                if (clock64() - start_time >= LEGACY_NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id, // 当前 channel。
                           rdma_rank,  // 本 rank 的 RDMA 坐标。
                           nvl_rank,   // 本 rank 的 NVL 坐标。
                           lane_id,    // 当前 lane 对应的目标 RDMA rank。
                           cached_rdma_channel_head, // 已释放 head。
                           rdma_tail_idx); // 当前 token 要写入的 tail。
                    trap();  // RDMA ring 满且长时间不释放，触发异常。
                }
            }
            __syncwarp();  // 等所有 lane 完成空槽检查。

            // Store RDMA head for combine
            if (lane_id < kNumRDMARanks and not kCachedMode)
                send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;  // 保存 token 发往每个 RDMA rank 的槽位。

            // Broadcast tails
            SourceMeta src_meta;  // 当前 token 写入 RDMA buffer 的 source meta。
            int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];  // 当前 token 实际命中的 RDMA rank 列表。
            void* dst_send_buffers[kNumTopkRDMARanks];  // 每个命中 RDMA rank 的目标 staging buffer 指针。
            // topk_ranks/dst_send_buffers 是一个紧凑扇出表，长度为 num_topk_ranks。
            // 第 j 项表示“当前 token 需要写给 topk_ranks[j] 这个 RDMA rank，并写入 dst_send_buffers[j] 指向的 token 消息槽”。
            // 这个表最多 8 项，原因是 get_num_topk_rdma_ranks 限制单 token 最多扇出到 8 个 RDMA rank。
            // 遍历所有 RDMA rank 的 lane 广播结果，把当前 token 需要发送到的 RDMA rank 收集成紧凑列表。
            #pragma unroll
            for (int i = 0, slot_idx; i < kNumRDMARanks; ++i)
                if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
                    slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;  // 映射到 RDMA ring 槽位。
                    topk_ranks[num_topk_ranks] = i;  // 记录第 num_topk_ranks 个目标 RDMA rank。
                    auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);  // 广播目标 RDMA rank 的 NVL mask。
                    auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);  // mask 解释为 8 个 bool。
                    if (lane_id == num_topk_ranks)
                        src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);  // 用前几个 lane 构造对应目标的 SourceMeta。
                    // SourceMeta 被放在每个目标消息里，后续 forwarder/receiver 通过它判断是否要转发到某个 NVL rank。
                    dst_send_buffers[num_topk_ranks++] =
                        reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_token;  // 计算目标 token 消息起点。
                }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);  // 命中 RDMA rank 数不能超过局部数组容量。

            // Copy `x` into symmetric send buffer
            auto st_broadcast = [=](const int key, const int4& value) {
                // 中文：key 是 hidden_int4 内的元素位置，同一个 token 可能写到多个目标 RDMA rank。
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);  // 将同一 hidden int4 写入所有命中 RDMA rank 的 staging buffer。
            };  // 广播写 lambda：一次读取 x，多目标写入。
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);  // warp 并行复制 hidden payload。
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;  // 指针越过 hidden payload。

            // Copy `x_scales` into symmetric send buffer
            #pragma unroll
            for (int i = lane_id; i < num_scales; i += 32) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;  // 输入 scale 实际偏移。
                auto value = ld_nc_global(x_scales + offset);  // 读取 scale。
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);  // 写入每个目标 RDMA rank 的 scale 区。
            }
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;  // 指针越过 scales 区。

            // Copy source metadata into symmetric send buffer
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);  // 每个目标 RDMA rank 写一份 SourceMeta。
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;  // 指针越过 SourceMeta。

            // Copy `topk_idx` and `topk_weights` into symmetric send buffer
            #pragma unroll
            for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
                auto rank_idx = i / num_topk, copy_idx = i % num_topk;  // 展开索引映射到目标 RDMA rank 和 topk lane。
                auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));  // 读取全局 expert id。
                auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);  // 读取 topk 权重。
                st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);  // 写 topk_idx。
                st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);  // 写 topk_weights。
            }
            __syncwarp();  // 等 token 消息所有字段写入 staging buffer。

            // Release the transaction in the window
            if (is_token_in_rank_uint64 != 0) {
                // Acquire lock first
                acquire_lock(rdma_send_channel_lock + lane_id);  // 锁住该目标 RDMA rank 的完成窗口。
                auto latest_tail = rdma_send_channel_tail[lane_id];  // 已连续完成的 tail。
                auto offset = rdma_tail_idx - latest_tail;  // 当前 token 相对连续 tail 的偏移。
                while (offset >= 32) {
                    release_lock(rdma_send_channel_lock + lane_id);  // offset 超过 32-bit window，先释放锁等待 coordinator 推进。
                    acquire_lock(rdma_send_channel_lock + lane_id);  // 重新获取锁。
                    latest_tail = rdma_send_channel_tail[lane_id];  // 重新读取连续 tail。
                    offset = rdma_tail_idx - latest_tail;  // 重新计算 offset。
                }

                // Release the transaction slot
                // Add the bit and move the ones if possible
                auto window = rdma_send_channel_window[lane_id] | (1u << offset);  // 标记当前 offset 的 token 已打包完成。
                if (offset == 0) {
                    auto num_empty_slots = (~window) == 0 ? 32 : __ffs(~window) - 1;  // 从 bit0 起连续完成的槽数。
                    st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots);  // release 更新连续 tail 给 coordinator。
                    window >>= num_empty_slots;  // 移除已经连续完成的窗口位。
                }
                rdma_send_channel_window[lane_id] = window;  // 保存剩余未连续完成窗口。

                // Release lock
                release_lock(rdma_send_channel_lock + lane_id);  // 释放目标 RDMA rank 的窗口锁。
            }
            __syncwarp();  // sender warp 内同步，进入下一个 token。
        }
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        // NOTES: in case of splitting, the issued put at the end of the buffer
        EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);  // send chunk 必须整除 recv ring，避免跨 ring 尾拆包。

        // Clean shared memory
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");  // 一个 warp lane 覆盖所有 RDMA rank。
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;  // 初始化窗口锁。
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;  // 初始化连续 tail。
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;  // 初始化完成窗口。

        // Synchronize shared memory
        sync_rdma_sender_smem();  // 等 sender warps 看到初始化后的 shared 状态。

        // Get number of tokens to send for each RDMA rank
        int num_tokens_to_send = 0;  // 当前 lane/RDMA rank 还需要 RDMA put 的 token 数。
        if (lane_id < kNumRDMARanks) {
            num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];  // 当前 channel 截止 prefix。
            if (channel_id > 0)
                num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];  // 减上一 channel prefix 得到本 channel token 数。
        }

        // Iterate all RDMA ranks
        int last_issued_tail = 0;  // 当前 lane/RDMA rank 已经发起 put 的 tail。
        auto start_time = clock64();  // coordinator 等待开始时间。
        while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
            // Timeout check
            if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                // 中文：coordinator 等待某个目标 RDMA rank 的可发送 token 清零；remaining 非零说明 put/tail 未推进。
                printf("DeepEP RDMA sender coordinator timeout, channel: %d, IB: %d, nvl %d, dst IB: %d, tail: %d, remaining: %d\n",
                       channel_id,  // 当前 channel。
                       rdma_rank,  // 当前 RDMA rank。
                       nvl_rank,  // 当前 NVL rank。
                       lane_id,  // 等待的目标 RDMA rank lane。
                       last_issued_tail,  // 已发起 tail。
                       num_tokens_to_send);  // 剩余 token 数。
                trap();  // coordinator 长时间等不到 sender 完成打包，触发异常。
            }

            // TODO: try thread-level `put_nbi`?
            for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++i) {
                // To mitigate incast congestion, shuffle the starting index of target rank for different ranks and channels
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;  // 打散不同 channel/rank 的目标顺序，缓解 incast。
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank);  // 获取目标 RDMA rank 剩余 token 数。
                if (synced_num_tokens_to_send == 0)
                    continue;  // 该目标 RDMA rank 没有待发送 token。

                // Read the latest progress
                // NOTES: `rdma_send_channel_tail` does not need to be protected by lock
                auto processed_tail =
                    __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);  // sender 已连续打包完成到的 tail。
                auto synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank);  // 该目标已发起 put 的 tail。
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;  // 可发起 put 的连续 token 数。
                if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
                    continue;  // 不到 chunk 上限且不是最后一批，继续等待以减少小包。

                // Issue RDMA send
                auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);  // 本次 put 的 token 数。
                EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);  // put 数量必须合法。
                if (dst_rdma_rank != rdma_rank) {
                    auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;  // RDMA ring 目标槽位。
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);  // 不允许单次 put 跨 ring 尾。
                    const size_t num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue;  // 本次 put 字节数。
                    const auto dst_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_token);  // 远端接收 buffer 地址。
                    const auto src_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_token);  // 本地发送 staging 地址。
                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr,  // 远端 recv buffer 地址。
                                                      src_ptr,   // 本地 send buffer 地址。
                                                      num_bytes_per_msg,  // 本次发送字节数。
                                                      translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),  // 目标 NVSHMEM PE。
                                                      channel_id,  // 使用当前 channel 的 IBGDA 队列。
                                                      lane_id,     // warp lane 参与 put 协作。
                                                      0);          // wr_id/flag 保留为 0。
                } else {
                    // Lighter fence for local RDMA rank
                    memory_fence();  // 本地 RDMA rank 无需 put，但需要 fence 保证 staging 写入可见。
                }
                __syncwarp();  // 等 warp 内 put/fence 发起完成。

                // Update tails
                if (lane_id == dst_rdma_rank) {
                    last_issued_tail += num_tokens_to_issue;  // 更新已发起 put 的 tail。
                    num_tokens_to_send -= num_tokens_to_issue;  // 减少剩余 token 数。
                    // 中文：tail 是给远端 forwarder 的可读 token 通知，必须在 put 发起后再发布。
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
                                                num_tokens_to_issue,
                                                translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                // 中文：tail 更新使用数据 channel_id，和后面的 head 释放 QP 区分开。
                                                channel_id,
                                                    dst_rdma_rank == rdma_rank);  // 远端 tail 原子增加，通知 forwarder 可读。
                }  // 中文：当前 dst_rdma_rank 的 tail 通知已发出。
                __syncwarp();  // 同步 lane 状态后处理下一个目标 RDMA rank。
            }  // 中文：结束一个目标 RDMA rank 的发送循环。
        }  // 中文：结束 RDMA sender 角色。
    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        // RDMA consumers and NVL producers
        // 中文：forwarder 从跨节点 RDMA ring 读取 token，再写入目标节点内 NVL ring。
        const auto dst_nvl_rank = target_rank;  // 当前 forwarder warp 负责转发给的目标 NVL rank。

        // Wait counters to arrive
        // 中文：先等待所有 RDMA source 的计数到达，才能知道每个 source ring 本轮要读多少 token。
        int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;  // 当前 lane/RDMA source 待接收 token 数和全局 prefix。
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);  // 一个 warp lane 覆盖所有 RDMA rank。
        auto start_time = clock64();  // 等待 RDMA meta 到达的起始时间。
        if (lane_id < kNumRDMARanks) {
            while (true) {
                // meta_0/meta_1 是目标 NVL rank 的 prefix 范围，meta_2/meta_3 是源 RDMA rank 的 prefix 范围。
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);  // 该 source RDMA rank 发给 dst_nvl_rank 的 channel start。
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + LEGACY_NUM_MAX_NVL_PEERS + dst_nvl_rank);  // 对应 channel end。
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + LEGACY_NUM_MAX_NVL_PEERS * 2);  // RDMA channel start。
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + LEGACY_NUM_MAX_NVL_PEERS * 2 + 1);  // RDMA channel end。
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    // meta 使用负数作为“已发布”标记，-value-1 解码后得到非负 prefix。
                    // Notify NVL ranks
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;  // 解码目标 NVL rank 的 channel start/end。
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);  // prefix 必须单调合法。
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);  // 将 start prefix 写给 NVL receiver。
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);  // 将 end prefix 写给 NVL receiver。

                    // Save RDMA channel received token count
                    src_rdma_channel_prefix = -meta_2 - 1;  // RDMA channel start。
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;  // RDMA channel end。
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;  // 当前 source RDMA rank 的 RDMA token 数。
                    if (not kCachedMode)
                        recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;  // 保存接收侧 RDMA channel prefix。
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];  // 加 RDMA rank prefix 得到全局 RDMA token 偏移。
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);  // token 数不能为负。
                    break;  // meta 已到达。
                }

                // Timeout check
                if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                    // 中文：meta 四元组分别是 NVL start/end 和 RDMA start/end，负数表示 ready 编码。
                    printf(
                        "DeepEP dispatch forwarder timeout (RDMA meta), channel: %d, RDMA: %d, nvl: %d, src RDMA lane: %d, dst NVL: %d, "
                        "meta: %d, %d, %d, %d\n",
                        channel_id,   // 当前 channel，帮助定位哪个 SM 分片卡住。
                        rdma_rank,    // 本 rank 的 RDMA 坐标。
                        nvl_rank,     // 本 rank 的 NVL 坐标。
                        lane_id,      // 当前 lane 代表的源 RDMA rank。
                        dst_nvl_rank, // 当前 forwarder 的目标 NVL rank。
                        meta_0,       // 目标 NVL prefix start 原始值。
                        meta_1,       // 目标 NVL prefix end 原始值。
                        meta_2,       // 源 RDMA prefix start 原始值。
                        meta_3);      // 源 RDMA prefix end 原始值。
                    trap();  // 等 RDMA meta 超时。
                }
            }
        }
        __syncwarp();  // 等所有 RDMA lane 得到 meta。

        // Shift cached head
        send_nvl_head += src_rdma_channel_prefix * LEGACY_NUM_MAX_NVL_PEERS + dst_nvl_rank;  // 将 send_nvl_head 指到当前 RDMA source/channel 的基准位置。

        // Wait shared memory to be cleaned
        sync_forwarder_smem();  // 等 forwarder coordinator 初始化 shared head/retired。

        // Forward tokens from RDMA buffer
        // NOTES: always start from the local rank
        int src_rdma_rank = sm_id % kNumRDMARanks;  // round-robin 起始 source RDMA rank。
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;  // 当前 source RDMA ring 的 head/tail。
        int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;  // 目标 NVL ring 的 head/tail 以及该 RDMA source 内命中 dst_nvl_rank 的序号。
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            // Check destination queue emptiness, or wait a buffer to be released
            start_time = clock64();  // 等待 NVL ring 空槽开始时间。
            while (true) {
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;  // NVL ring 已用槽数。
                // 中文：forwarder 一次最多写 num_max_nvl_chunked_send_tokens，必须先确认目标 ring 空间足够。
                if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens)
                    break;  // 目标 NVL ring 空槽足够一个 send chunk。
                cached_nvl_channel_head = __shfl_sync(0xffffffffu, ld_volatile_global(nvl_channel_head.buffer()), 0);  // 读取 receiver 已释放 head。

                // Timeout check
                if (elect_one_sync() and clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                    // 中文：NVL check 超时通常表示目标 receiver 没推进 head，ring 被占满。
                    printf(
                        "DeepEP dispatch forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, head: %d, tail: %d\n",
                        channel_id,  // 当前 channel。
                        rdma_rank,  // 当前 RDMA rank。
                        nvl_rank,  // 当前 NVL rank。
                        dst_nvl_rank,  // 目标 NVL rank。
                        ld_volatile_global(nvl_channel_head.buffer()),  // receiver 已释放 head。
                        cached_nvl_channel_tail);  // sender 当前 tail。
                    trap();  // 等 NVL ring 空槽超时。
                }
            }

            // Find next source RDMA rank (round-robin)
            start_time = clock64();  // 等待 RDMA 数据到达开始时间。
            while (true) {
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;  // 轮询下一个 RDMA source，避免固定顺序偏置。
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                    // 只有 source lane 自己维护该 source 的 cached head/tail，其它 lane 通过 shuffle 读取它的判断。
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));  // 读取 source RDMA ring tail。
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank))
                        break;  // 该 source RDMA rank 有可读 token。
                }

                // Timeout check
                if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    // 中文：RDMA check 超时表示某个 source RDMA lane 预期还有 token，但 tail 没有超过 head。
                    printf(
                        "DeepEP dispatch forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, src RDMA lane: %d, "
                        "head: %d, tail: %d, expected: %d\n",
                        channel_id,  // 当前 channel。
                        rdma_rank,   // 本 rank 的 RDMA 坐标。
                        nvl_rank,    // 本 rank 的 NVL 坐标。
                        dst_nvl_rank, // 目标 NVL rank。
                        lane_id,     // 当前 lane/source RDMA rank。
                        cached_rdma_channel_head, // 已消费到的 RDMA head。
                        cached_rdma_channel_tail, // 已观察到的 RDMA tail。
                        num_tokens_to_recv_from_rdma); // 当前 lane 还期望接收的 token 数。
                    trap();  // 等 RDMA 数据到达超时。
                }
            }
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);  // 广播选中 source RDMA rank 的 head。
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);  // 广播选中 source RDMA rank 的 tail。

            // Iterate over every token from the RDMA buffer
            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
                auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;  // RDMA ring 槽位。
                auto shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_token;  // RDMA token 消息起点。
                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));  // 读取 SourceMeta。
                lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;  // 对应 source lane 扣减剩余 RDMA token。
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);  // 该 token 是否需要转发给当前目标 NVL rank。
                if (lane_id == src_rdma_rank) {
                    auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;  // 命中则记录 NVL token 序号，否则 -1。
                    rdma_nvl_token_idx += is_in_dst_nvl_rank;  // 只对命中当前 NVL rank 的 token 递增。
                    if (not kCachedMode)
                        send_nvl_head[i * LEGACY_NUM_MAX_NVL_PEERS] = cached_head;  // 保存 RDMA token 到 NVL ring 的 head 映射。
                }
                if (not is_in_dst_nvl_rank)
                    continue;  // 不属于当前目标 NVL rank 的 token 不转发。

                // Get an empty slot
                int dst_slot_idx = (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;  // 分配 NVL ring 槽位。
                auto dst_shifted = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;  // NVL token 消息目标地址。

                // Copy data
                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, num_bytes_per_token, false);  // 从 RDMA ring TMA load 整个 token 消息。
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, num_bytes_per_token);  // 告知 barrier 本次传输字节数。
                }
                __syncwarp();  // 等 TMA load 发起。
                mbarrier_wait(tma_mbarrier, tma_phase);  // 等 TMA load 完成。
                if (elect_one_sync())
                    tma_store_1d(tma_buffer, dst_shifted, num_bytes_per_token);  // TMA store 到目标 NVL ring。
                __syncwarp();  // 等 TMA store 发起。

                // In case of insufficient NVL buffers, early stopping
                if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens)
                    src_rdma_tail = i + 1;  // 达到本轮 NVL send chunk 上限，提前停止当前 source RDMA rank。

                // Wait TMA to be finished
                tma_store_wait<0>();  // 等 NVL TMA store 完成后再释放 tail。
                __syncwarp();  // warp 内同步。
            }

            // Sync head index
            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);  // 更新该 forwarder 对 source RDMA rank 的消费 head。

            // Move tail index
            __syncwarp();  // 确保所有 token 写入 NVL ring。
            if (elect_one_sync())
                st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail);  // release 更新 NVL tail，通知 receiver 可读。
        }

        // Retired
        __syncwarp();  // 等最后一次 head/tail 更新完成。
        if (elect_one_sync())
            forward_channel_retired[dst_nvl_rank] = true;  // 标记当前目标 NVL rank forwarder 完成。
    } else if (warp_role == WarpRole::kForwarderCoordinator) {
        // Extra warps for forwarder coordinator should exit directly
        if (target_rank > 0)
            return;  // 只保留一个 coordinator warp，其它额外 warp 退出。

        // Forward warp coordinator
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");  // 一个 warp lane 覆盖所有 RDMA rank。

        // Clean shared memory
        EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");  // 一个 warp 覆盖所有 NVL peer。
        #pragma unroll
        for (int i = lane_id; i < kNumRDMARanks * LEGACY_NUM_MAX_NVL_PEERS; i += 32)
            forward_channel_head[i % LEGACY_NUM_MAX_NVL_PEERS][i / LEGACY_NUM_MAX_NVL_PEERS] = 0;  // 初始化每个 forwarder 对每个 RDMA rank 的 head。
        if (lane_id < LEGACY_NUM_MAX_NVL_PEERS)
            forward_channel_retired[lane_id] = false;  // 初始化 retired 标志。
        sync_forwarder_smem();  // 等 forwarder warps 看到初始化后的 shared 状态。

        int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;  // 当前 lane 负责推进的 RDMA source head。
        while (true) {
            // Find minimum head
            int min_head = std::numeric_limits<int>::max();  // 所有未完成 forwarder 中最小消费 head。
            #pragma unroll
            for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
                if (not forward_channel_retired[i])
                    min_head = min(min_head, forward_channel_head[i][target_rdma]);  // 只统计未 retired 的目标 NVL rank。
            if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max()))
                break;  // 所有 forwarder 完成。

            // Update remote head
            // 中文：coordinator 取所有未完成 forwarder 的最小 head，确保不会提前覆盖仍在读取的 RDMA slot。
            if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + num_max_rdma_chunked_send_tokens and
                lane_id < kNumRDMARanks) {
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                min_head - last_head,  // 本次释放的 slot 数。
                                                translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank),  // 目标 RDMA rank。
                                                channel_id + num_channels,  // head 使用的 QP/队列编号。
                                                lane_id == rdma_rank);  // 是否本地 rank，原子增加 RDMA head 释放远端 sender ring。
                last_head = min_head;  // 更新已释放 head。
            }

            // Nanosleep and let other warps work
            __nanosleep(LEGACY_NUM_WAIT_NANOSECONDS);  // 让出执行资源，减少 coordinator 忙等压力。
        }  // 中文：forwarder coordinator 循环持续推进 RDMA head 直到所有源完成。
    } else {
        // NVL consumers
        // Retrieve rank offset from barrier results (each lane's register stores an RDMA rank)
        // 中文：receiver 最终把节点内 NVL ring 的 token 写入 recv_x，并生成 recv_src_meta/topk 信息。
        // 中文：barrier 结果按 lane 保存各 RDMA rank 的 prefix，receiver 用它还原全局接收偏移。
        int src_nvl_rank = target_rank, total_offset = 0;  // 当前 receiver warp 负责读取的源 NVL rank，以及输出起始 offset。
        const int local_expert_begin = rank * (num_experts / num_ranks);  // 当前 rank 拥有的全局 expert 起点。
        const int local_expert_end = local_expert_begin + (num_experts / num_ranks);  // 当前 rank 拥有的全局 expert 终点。

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");  // 一个 warp lane 覆盖所有 RDMA rank。
        if (lane_id < kNumRDMARanks and lane_id * LEGACY_NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            total_offset = recv_gbl_rank_prefix_sum[lane_id * LEGACY_NUM_MAX_NVL_PEERS + src_nvl_rank - 1];  // 每个 RDMA lane 的全局 rank 前缀基址。

        // Receive channel offsets
        int start_offset = 0, end_offset = 0, num_tokens_to_recv;  // 当前 source RDMA lane 的 channel start/end 和总接收数。
        auto start_time = clock64();  // 等待 NVL prefix 到达开始时间。
        while (lane_id < kNumRDMARanks) {
            start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);  // 读取 source RDMA rank 的 channel start。
            end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);  // 读取 source RDMA rank 的 channel end。
            if (start_offset < 0 and end_offset < 0) {
                start_offset = -start_offset - 1, end_offset = -end_offset - 1;  // 解码负数 prefix。
                total_offset += start_offset;  // 加 channel 内偏移得到该 RDMA source 的输出起点。
                break;  // prefix 已 ready。
            }

            // Timeout check
            if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                // 中文：start/end 仍非负说明 forwarder 尚未发布该 channel 对应的 NVL prefix。
                printf(
                    "DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, src nvl: %d, start: %d, end: %d\n",
                    channel_id,  // 当前 channel。
                    rdma_rank,  // 当前 RDMA rank。
                    nvl_rank,  // 当前 NVL rank。
                    lane_id,  // 等待的源 RDMA rank lane。
                    src_nvl_rank,  // 源 NVL rank。
                    start_offset,  // prefix start。
                    end_offset);  // prefix end。
                trap();  // 等 NVL prefix 超时。
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);  // 当前 src_nvl_rank/channel 总共要接收的 token 数。

        // Save for combine usage
        if (lane_id < kNumRDMARanks and not kCachedMode)
            recv_gbl_channel_prefix_matrix[(lane_id * LEGACY_NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;  // 保存接收侧全局 channel prefix。
        __syncwarp();  // 等 prefix 写入完成。

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;  // NVL receiver 本地 head/tail。
        while (num_tokens_to_recv > 0) {
            // Check channel status by lane 0
            start_time = clock64();  // 等待 NVL 数据到达开始时间。
            while (true) {
                // Ready to copy
                // cached_channel_head_idx 是本 receiver 已消费到的位置；tail 是 forwarder 发布的生产位置。
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;  // 有可读 token。
                // tail 由 forwarder release 写入，这里 acquire 读取，保证看到 tail 后 payload 也已经可见。
                cached_channel_tail_idx = __shfl_sync(0xffffffff, ld_acquire_sys_global(nvl_channel_tail.buffer()), 0);  // acquire 读取 forwarder release 写入的 tail。

                // Timeout check
                // elect_one_sync 只让一个 lane 打印，避免 32 个 lane 同时输出导致日志淹没。
                if (elect_one_sync() and clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, head: %d, tail: %d\n",
                           channel_id,  // 当前 channel。
                           rdma_rank,  // 当前 RDMA rank。
                           nvl_rank,  // 当前 NVL rank。
                           src_nvl_rank,  // 源 NVL rank。
                           cached_channel_head_idx,  // receiver 已消费 head。
                           cached_channel_tail_idx);  // sender 已发布 tail。
                    trap();  // 等 NVL token 超时。
                }
            }

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;  // 本轮可接收 token 数。
            // 本循环逐 token 消费 NVL ring；每消费一个 token，num_tokens_to_recv 同步减少。
            for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
                // ring 容量为 num_max_nvl_chunked_recv_tokens，head 递增后用取模映射到物理槽位。
                int token_idx_in_buffer = (cached_channel_head_idx++) % num_max_nvl_chunked_recv_tokens;  // NVL ring 槽位并推进 head。
                auto shifted = nvl_channel_x.buffer() + token_idx_in_buffer * num_bytes_per_token;  // NVL token 消息起点。
                // SourceMeta 紧跟在 hidden 和 scales 后面，描述该 token 最初来自哪个 RDMA/NVL/source token。
                auto meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));  // 读取 SourceMeta。
                // total_offset 在不同 lane 中保存不同 source RDMA rank 的输出写入位置。
                int64_t recv_token_idx = __shfl_sync(0xffffffff, total_offset, meta.src_rdma_rank);  // 根据来源 RDMA rank 选择对应 lane 的输出 offset。
                // 只有负责该 source RDMA rank 的 lane 前进 offset，保持每个 source 的顺序稳定。
                (lane_id == meta.src_rdma_rank) ? (total_offset += 1) : 0;  // 只有来源 RDMA lane 推进自己的输出 offset。

                // scales 和 hidden 紧邻存放；若 scale_bytes 是 16B 对齐，可以和 hidden 一起走 TMA。
                bool scale_aligned = (scale_bytes % 16 == 0);  // scales 若 16B 对齐即可随 TMA 一起搬。
                auto tma_load_bytes = hidden_bytes + (scale_aligned ? scale_bytes : 0);  // TMA load 覆盖 hidden 和可选 aligned scales。

                // Copy data
                if (elect_one_sync()) {
                    // 只需一个 lane 发起 TMA load，warp 内其它 lane 通过 mbarrier 等完成即可。
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, tma_load_bytes);  // 从 NVL ring TMA load payload。
                    // expect_tx 的字节数必须和 TMA load 字节数一致，否则 barrier 永远等不到正确事务数。
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, tma_load_bytes);  // 设置 TMA barrier 期望字节数。
                }
                __syncwarp();  // 等 TMA load 发起。
                mbarrier_wait(tma_mbarrier, tma_phase);  // 等 TMA load 完成。
                if (elect_one_sync()) {
                    // hidden 按 int4 视图存储到 recv_x，recv_token_idx 是接收侧线性 token 位置。
                    tma_store_1d(tma_buffer, recv_x + recv_token_idx * hidden_int4, hidden_bytes, false);  // TMA store hidden 到 recv_x。
                    if (scale_aligned)
                        // aligned scales 和 hidden 一样走 TMA store，减少 per-lane 普通 store 开销。
                        tma_store_1d(tma_buffer + hidden_bytes, recv_x_scales + recv_token_idx * num_scales, scale_bytes, false);  // aligned scales 也用 TMA store。
                }
                __syncwarp();  // 等 store 发起。
                shifted += hidden_bytes;  // 指针移动到 scales 区。

                // Copy scales
                // TODO: make it as templated
                if (not scale_aligned) {
                    // 非 16B 对齐时 TMA store 约束不方便满足，因此退回 warp copy。
                    UNROLLED_WARP_COPY(1,  // 非 16B 对齐 scales 用普通 warp copy。
                                       lane_id,  // 当前 lane。
                                       num_scales,  // scale 元素数。
                                       recv_x_scales + recv_token_idx * num_scales,  // 输出 scale 起点。
                                       reinterpret_cast<float*>(shifted),  // 输入 scale 起点。
                                       ld_nc_global,  // non-coherent load。
                                       st_na_global);  // non-allocating store。
                }
                shifted += scale_bytes;  // 指针移动到 SourceMeta 区。

                // Copy source meta
                if (not kCachedMode and elect_one_sync())
                    // non-cached dispatch 必须保存 SourceMeta；cached dispatch 后续会复用上一次布局。
                    st_na_global(recv_src_meta + recv_token_idx, meta);  // 保存 SourceMeta 给 combine 使用。
                shifted += sizeof(SourceMeta);  // 指针移动到 topk_idx 区。

                // Copy `topk_idx` and `topk_weights`
                if (lane_id < num_topk) {
                    // Read
                    // topk_idx 存的是全局 expert id，topk_weights 与它同一 top-k lane 对齐。
                    auto idx_value = static_cast<topk_idx_t>(ld_nc_global(reinterpret_cast<int*>(shifted) + lane_id));  // 读取全局 expert id。
                    auto weight_value = ld_nc_global(reinterpret_cast<float*>(shifted + sizeof(int) * num_topk) + lane_id);  // 读取 topk 权重。
                    auto recv_idx = recv_token_idx * num_topk + lane_id;  // 输出 topk 线性位置。

                    // Transform and write
                    // 输出给本 rank 的 expert 只需要本地 expert id；非本地 expert 置 -1 并清零权重。
                    idx_value = (idx_value >= local_expert_begin and idx_value < local_expert_end) ? idx_value - local_expert_begin : -1;  // 全局 expert id 转本地 expert id。
                    weight_value = idx_value >= 0 ? weight_value : 0.0f;  // 非本地 expert lane 权重清零。
                    st_na_global(recv_topk_idx + recv_idx, idx_value);  // 写输出 topk_idx。
                    st_na_global(recv_topk_weights + recv_idx, weight_value);  // 写输出 topk 权重。
                }

                // Wait TMA to be finished
                tma_store_wait<0>();  // 等 hidden/scales TMA store 完成。
                __syncwarp();  // warp 内同步后处理下一个 token。
            }

            // Move queue
            if (elect_one_sync())
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);  // 写回 NVL head，释放 ring 槽给 forwarder。
        }
    }

    // Clean unused `recv_topk_idx` as -1
    if (num_worst_tokens > 0) {
        if (is_forwarder)
            return;  // forwarder block 不参与输出清理。
        // get the actual number of num_recv_tokens on the current rank
        int num_recv_tokens = recv_gbl_rank_prefix_sum[num_ranks - 1];  // 真实接收 token 数。
        // some ForwarderCoordinator threads exit early, so we only use non-forwarder in clean-up
        // channel_id * num_threads is the offset of the current non-forwarder sms
        const auto clean_start = num_recv_tokens * num_topk + channel_id * num_threads;  // 当前 non-forwarder block 的清理起点。
        const auto clean_end = num_worst_tokens * num_topk;  // worst-case topk 容量终点。
        const auto clean_stride = num_channels * num_threads;  // 清理步长。
        #pragma unroll
        for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
            recv_topk_idx[i] = -1;  // 未使用 topk lane 标记为 -1。
    }
}

void dispatch(void* recv_x,  // 输出接收 token 特征。
              float* recv_x_scales,  // 输出接收 scales，可为空。
              topk_idx_t* recv_topk_idx,  // 输出接收 topk_idx，可为空。
              float* recv_topk_weights,  // 输出接收 topk_weights，可为空。
              void* recv_src_meta,  // 输出 source meta，可为空或 cached 复用。
              const void* x,  // 输入 token 特征。
              const float* x_scales,  // 输入 scales，可为空。
              const topk_idx_t* topk_idx,  // 输入 topk_idx，可为空。
              const float* topk_weights,  // 输入 topk_weights，可为空。
              int* send_rdma_head,  // 输出 RDMA send head。
              int* send_nvl_head,  // 输出 NVL send head。
              int* recv_rdma_channel_prefix_matrix,  // 输出接收 RDMA channel prefix。
              int* recv_gbl_channel_prefix_matrix,  // 输出接收全局 channel prefix。
              const int* rdma_channel_prefix_matrix,  // 输入 RDMA channel prefix。
              const int* recv_rdma_rank_prefix_sum,  // 输入 RDMA rank prefix。
              const int* gbl_channel_prefix_matrix,  // 输入全局 channel prefix。
              const int* recv_gbl_rank_prefix_sum,  // 输入全局 rank prefix。
              const bool* is_token_in_rank,  // 输入 token-rank 路由表。
              int num_tokens,  // 输入 token 数。
              int num_worst_tokens,  // worst-case 容量。
              int hidden_int4,  // hidden 的 int4 长度。
              int num_scales,  // scale 数。
              int num_topk,  // top-k 路数。
              int num_experts,  // expert 总数。
              int scale_token_stride,  // scale token stride。
              int scale_hidden_stride,  // scale hidden stride。
              void* rdma_buffer_ptr,  // RDMA symmetric buffer。
              int num_max_rdma_chunked_send_tokens,  // RDMA send chunk 上限。
              int num_max_rdma_chunked_recv_tokens,  // RDMA recv ring 容量。
              void** buffer_ptrs,  // NVL peer buffer 指针表。
              int num_max_nvl_chunked_send_tokens,  // NVL send chunk 上限。
              int num_max_nvl_chunked_recv_tokens,  // NVL recv ring 容量。
              int rank,  // 当前 rank。
              int num_ranks,  // 全局 rank 数。
              bool is_cached_dispatch,  // 是否复用 cached prefix/head。
              cudaStream_t stream,  // CUDA 发射流。
              int num_channels,  // channel 数。
              bool low_latency_mode) {  // 是否使用 low-latency PE 编号。
    // host 入口只负责根据 RDMA rank 数、cached/low-latency 标志选择模板实例。
    // kernel 输出：
    //   recv_x: [num_recv_tokens, hidden]
    //   recv_x_scales: FP8 时 [num_recv_tokens, num_scales]
    //   recv_topk_idx/weights: [num_recv_tokens, num_topk]
    //   recv_src_meta: [num_recv_tokens, sizeof(SourceMeta)]，cached 模式可复用旧 meta。
    constexpr int kNumDispatchRDMASenderWarps = 7;  // 每个非 forwarder block 用 7 个 warp 打包 RDMA token。
    constexpr int kNumTMABytesPerWarp = 16384;  // 每个目标 NVL rank 的 TMA shared buffer 大小。
    constexpr int smem_size = kNumTMABytesPerWarp * LEGACY_NUM_MAX_NVL_PEERS;  // dispatch kernel 总 shared memory。

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());  // scale offset 在 kernel 中用 int 表示。

#define DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                                   \
    {                                                                                                                          \
        /* dispatch 模板同时固定 low-latency、RDMA rank 数、cached 模式和 TMA buffer 配置。 */                                 \
        auto dispatch_func = low_latency_mode                                                                                  \
            /* 中文：四个分支只改变模板常量，host 侧实参顺序完全一致。 */                                                     \
            ? (is_cached_dispatch ? dispatch<true, num_rdma_ranks, true, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>     \
                                  /* 中文：cached=true 时复用 head/prefix，只重新搬运 payload。 */                            \
                                  : dispatch<true, num_rdma_ranks, false, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>)   \
            : (is_cached_dispatch ? dispatch<false, num_rdma_ranks, true, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>    \
                                  /* 中文：low_latency=false 时仍保留相同 RDMA rank 模板实例。 */                             \
                                  : dispatch<false, num_rdma_ranks, false, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>); \
        /* dispatch 内部用 TMA 把 hidden/scales 批量搬到 shared staging，再写入 RDMA/NVL ring。 */                            \
        SET_SHARED_MEMORY_FOR_TMA(dispatch_func);                                                                              \
        LAUNCH_KERNEL(&cfg,                                                                                                    \
                      dispatch_func,                                                                                           \
                      /* recv_x/scales/topk/source_meta 是目标 rank 本轮 dispatch 的最终输出区域。 */                         \
                      reinterpret_cast<int4*>(recv_x),                                                                         \
                      recv_x_scales,                                                                                           \
                      /* recv_topk_idx/weights 只有 Python 请求 with_topk 或权重时才非空。 */                                  \
                      recv_topk_idx,                                                                                           \
                      recv_topk_weights,                                                                                       \
                      reinterpret_cast<SourceMeta*>(recv_src_meta),                                                            \
                      /* x/scales/topk/topk_weights 是源 rank 输入，按 token 和 expert 路由拆分发送。 */                      \
                      reinterpret_cast<const int4*>(x),                                                                        \
                      x_scales,                                                                                                \
                      /* topk_idx 是路由主表，topk_weights 会跟随 token 一起传给 combine。 */                                 \
                      topk_idx,                                                                                                \
                      topk_weights,                                                                                            \
                      /* send_*_head 是发送端 ring head，non-cached 模式会写出给 combine 复用。 */                           \
                      send_rdma_head,                                                                                          \
                      send_nvl_head,                                                                                           \
                      /* recv_*_channel_prefix_matrix 是接收侧 prefix，combine 反向路径依赖它定位输出槽位。 */                \
                      recv_rdma_channel_prefix_matrix,                                                                         \
                      recv_gbl_channel_prefix_matrix,                                                                          \
                      /* rdma/gbl prefix 输入来自 notify_dispatch，描述每个 channel/rank 的 token 区间。 */                   \
                      rdma_channel_prefix_matrix,                                                                              \
                      recv_rdma_rank_prefix_sum,                                                                               \
                      gbl_channel_prefix_matrix,                                                                               \
                      /* recv_gbl_rank_prefix_sum 给 receiver 计算最终 recv_x 行偏移。 */                                      \
                      recv_gbl_rank_prefix_sum,                                                                                \
                      /* is_token_in_rank: [num_tokens,num_ranks]，true 表示该 token 至少有一路 expert 属于该 rank。 */       \
                      is_token_in_rank,                                                                                        \
                      num_tokens,                                                                                              \
                      num_worst_tokens,                                                                                        \
                      /* hidden_int4 是 hidden/16 字节；num_scales 是 FP8 scale 元素数。 */                                   \
                      hidden_int4,                                                                                             \
                      num_scales,                                                                                              \
                      /* num_topk/num_experts 决定 SourceMeta 解码和 expert->rank 映射。 */                                   \
                      num_topk,                                                                                                \
                      num_experts,                                                                                             \
                      /* scale stride 描述 x_scales 的二维布局，允许 token stride 与 hidden stride 分离。 */                  \
                      scale_token_stride,                                                                                      \
                      scale_hidden_stride,                                                                                     \
                      /* RDMA/NVL buffer 与 chunk 容量决定 ring 分片大小，必须和 host 侧 get_*_meta 计算一致。 */             \
                      rdma_buffer_ptr,                                                                                         /* RDMA buffer 指针。 */ \
                      num_max_rdma_chunked_send_tokens,                                                                        /* RDMA send chunk 上限。 */ \
                      num_max_rdma_chunked_recv_tokens,                                                                        /* RDMA recv chunk 上限。 */ \
                      buffer_ptrs,                                                                                             /* NVL peer buffer 表。 */ \
                      num_max_nvl_chunked_send_tokens,                                                                         /* NVL send chunk 上限。 */ \
                      num_max_nvl_chunked_recv_tokens,                                                                         /* NVL recv chunk 上限。 */ \
                      rank,                                                                                                    /* 当前 rank。 */ \
                      num_ranks);                                                                                              /* 全局 rank 数。 */ \
    }                                                                                                                          \
    break

    EP_HOST_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));  // 输入 topk_idx/weights 必须成对。
    EP_HOST_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));  // 输出 topk_idx/weights 必须成对。

    SETUP_LAUNCH_CONFIG(num_channels * 2, (kNumDispatchRDMASenderWarps + 1 + LEGACY_NUM_MAX_NVL_PEERS) * 32, stream);  // 每 channel 一对 forwarder/non-forwarder block。
    SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);  // 根据 RDMA rank 数选择模板实例。
#undef DISPATCH_LAUNCH_CASE
}

template <bool kLowLatencyMode, int kNumTMABytesPerWarp>
__global__ void cached_notify(const int rdma_clean_offset,  // RDMA 控制区清理起点。
                              const int rdma_num_int_clean,  // RDMA 控制区清理 int 数。
                              const int nvl_clean_offset,  // NVL 控制区清理起点。
                              const int nvl_num_int_clean,  // NVL 控制区清理 int 数。
                              int* combined_rdma_head,  // combine 模式输入/修正 RDMA head。
                              int num_combined_tokens,  // combine 输出 token 数。
                              int num_channels,  // channel 数。
                              const int* rdma_channel_prefix_matrix,  // combine 模式 RDMA channel prefix。
                              const int* rdma_rank_prefix_sum,  // combine 模式 RDMA rank prefix。
                              int* combined_nvl_head,  // combine 模式输入/修正 NVL head。
                              void* rdma_buffer_ptr,  // RDMA symmetric buffer。
                              void** buffer_ptrs,  // NVL peer buffer 指针表。
                              int** barrier_signal_ptrs,  // NVL barrier 指针表。
                              int rank,  // 当前 rank。
                              int num_ranks,  // 全局 rank 数。
                              bool is_cached_dispatch,  // true 表示 dispatch cached notify；false 表示 combine notify。
                              const nvshmem_team_t rdma_team) {  // RDMA team。
    // cached_notify 是 internode combine/dispatch 的轻量控制面：不重新统计 topk，
    // 只 quiet 旧 RDMA WR、做 RDMA+NVL barrier、清理控制区，并把已有 prefix/head
    // 复制到本轮使用的 staging buffer。
    // block 分工：
    //   sm_id==0: quiet RDMA、barrier、清理 RDMA/NVL 控制区；
    //   sm_id==1: cached combine 时修正/填充 RDMA head；
    //   sm_id>=2: 按 channel 修正 NVL head 或 prefix，供后续主 kernel 直接消费。
    auto sm_id = static_cast<int>(blockIdx.x);  // block id：0 清理/同步，1 修正 RDMA head，其余修正 NVL head。
    auto thread_id = static_cast<int>(threadIdx.x);  // 线程 id。
    auto num_threads = static_cast<int>(blockDim.x);  // block 线程数。
    auto num_warps = num_threads / 32;  // warp 数。
    auto warp_id = thread_id / 32;  // warp id。
    auto lane_id = get_lane_id();  // warp lane id。

    auto nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;  // 当前节点内 rank。
    auto num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;  // RDMA rank 数。
    auto rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS;  // 当前 RDMA rank。

    // 使用两个 SM 分工清理：SM0 负责 RDMA buffer，SM1 负责 NVL buffer，避免单 SM
    // 线性清理大控制区拖慢 cached 路径。
    if (sm_id == 0) {
        auto qps_per_rdma_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;  // 每个 RDMA peer 的 QP 数。
        // cached 路径会复用同一 RDMA buffer，因此清理前必须等待上一轮所有 QP 的 WR 完成。
        for (int i = thread_id; i < qps_per_rdma_rank * (num_rdma_ranks - 1); i += num_threads) {
            auto dst_rdma_rank = (i / qps_per_rdma_rank + rdma_rank + 1) % num_rdma_ranks;  // 遍历非本地 RDMA rank。
            auto qp_id = i % qps_per_rdma_rank;  // 当前 QP id。
            // 中文：quiet 每个非本地 RDMA peer 的每个 QP，保证后续清零不会踩到尚未完成的 WR。
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), qp_id);  // 等旧 WR 完成。
        }
        __syncthreads();  // 等所有 quiet 完成。

        // Barrier for RDMA
        if (thread_id == 32)
            // 只让一个线程触发 RDMA team sync，避免全 block 重复调用 collective。
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);  // RDMA 层同步。

        // Barrier for NVL
        // barrier_block 覆盖节点内 NVL peers，确保本地 peer 都完成上一轮访问。
        barrier_block<LEGACY_NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);  // NVL 层同步。

        // Clean RDMA buffer
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);  // RDMA buffer int 视图。
        #pragma unroll
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            // rdma_clean_offset/rdma_num_int_clean 来自 get_rdma_clean_meta，精确覆盖控制区而非 payload 区。
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;  // 清零 RDMA 控制区。

        // Clean NVL buffer
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);  // 本地 NVL buffer int 视图。
        #pragma unroll
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            // nvl_clean_offset/nvl_num_int_clean 同样只清理 head/tail/prefix 等控制字段。
            nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;  // 清零 NVL 控制区。
        __syncthreads();  // 等控制区清理完成。

        // Barrier again
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);  // RDMA 层再次同步，确保清理可见。
        barrier_block<LEGACY_NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);  // NVL 层再次同步。
    } else if (sm_id == 1) {
        if (is_cached_dispatch)
            return;  // cached dispatch 不需要修正 combine RDMA head。

        EP_DEVICE_ASSERT(num_warps >= num_channels);  // 每个 channel 至少一个 warp。
        EP_DEVICE_ASSERT(num_rdma_ranks <= 32);  // 一个 warp lane 覆盖 RDMA rank。

        // Iterate in reverse order
        if (lane_id < num_rdma_ranks and warp_id < num_channels) {
            int token_start_idx, token_end_idx;  // 当前 channel 的 combine token 范围。
            get_channel_task_range(num_combined_tokens, num_channels, warp_id, token_start_idx, token_end_idx);  // 按 channel 切分。

            // NOTES: `1 << 25` is a heuristic large number
            // cached combine 的 head 表里，未命中某 RDMA rank 的 token 也需要知道“下一次有效 head 在哪里”。
            // 反向扫描可以把空洞 token 写成 -last_head-1，后续 combine 遇到负数就能跳过等待并推进到
            // 下一个有效 slot；这避免每轮重新计算完整路由。
            int last_head = 1 << 25;  // 从尾向前扫描时最近有效 head 的哨兵。
            for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
                auto current_head = __ldg(combined_rdma_head + token_idx * num_rdma_ranks + lane_id);  // 读取 token 对该 RDMA rank 的 head。
                if (current_head < 0) {
                    combined_rdma_head[token_idx * num_rdma_ranks + lane_id] = -last_head - 1;  // 未命中 token 写负数编码，指向下一个有效 head。
                } else {
                    last_head = current_head;  // 更新最近有效 head。
                }  // 中文：结束当前 token 的 RDMA head 修正。
            }  // 中文：结束当前 lane 负责的 token 范围。
        }  // 中文：结束 RDMA head 修正分支。
    } else {
        if (is_cached_dispatch)
            return;  // cached dispatch 不需要修正 combine NVL head。

        // 中文：combine cached notify 需要把原始 head 表补成“未命中槽指向下一个有效 head”的负数链。
        // 中文：该分支面向 internode combine 的 NVL head 表，与上方 RDMA head 修正逻辑对称。
        EP_DEVICE_ASSERT(num_warps >= num_channels);  // 每个 channel 至少一个 warp。
        EP_DEVICE_ASSERT(rdma_channel_prefix_matrix != nullptr and rdma_rank_prefix_sum != nullptr);  // combine notify 必须传 RDMA prefix。
        EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS <= 32, "Too many NVL peers");  // 一个 warp lane 覆盖 NVL peer。

        if (warp_id < num_channels) {
            constexpr int tma_batch_size = kNumTMABytesPerWarp - sizeof(uint64_t);  // 每 warp TMA buffer 扣除 mbarrier 后的 payload 空间。
            constexpr int num_bytes_per_token = sizeof(int) * LEGACY_NUM_MAX_NVL_PEERS;  // combined_nvl_head 每 token 8 个 int。
            constexpr int num_tokens_per_batch = tma_batch_size / num_bytes_per_token;  // 每次 TMA 批量处理的 token 数。
            EP_STATIC_ASSERT(num_bytes_per_token % 16 == 0, "num_bytes_per_token should be divisible by 16");  // TMA 字节数需要 16B 对齐。

            // TMA stuffs
            extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];  // TMA 中转 shared memory。
            auto tma_buffer = smem_tma_buffer + warp_id * kNumTMABytesPerWarp;  // 每个 channel warp 一段 TMA buffer。
            auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + tma_batch_size);  // mbarrier 放在 batch buffer 末尾。
            uint32_t tma_phase = 0;  // TMA phase。
            if (elect_one_sync()) {
                mbarrier_init(tma_mbarrier, 1);  // 初始化 TMA barrier。
                fence_barrier_init();  // 确保 barrier 初始化可见。
            }
            __syncwarp();  // 等 TMA 初始化完成。

            for (int dst_rdma_rank = sm_id - 2; dst_rdma_rank < num_rdma_ranks; dst_rdma_rank += num_channels * 2 - 2) {
                // Iterate in reverse order
                int token_start_idx = warp_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id - 1];  // 当前 RDMA/channel 起点。
                int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id];  // 当前 RDMA/channel 终点。
                int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];  // RDMA rank 全局前缀偏移。
                token_start_idx += shift, token_end_idx += shift;  // 转为 combined_nvl_head 全局 token 范围。

                // NOTES: `1 << 25` is a heuristic large number
                // NVL head 表同样用负数编码空洞。与 RDMA head 不同，combined_nvl_head 每 token 有 8 个 NVL peer lane，
                // 因此这里用 TMA 批量读写一段 [token, nvl_peer] 矩阵，再在 shared 中反向修正。
                int last_head = 1 << 25;  // 从尾向前扫描时最近有效 head。
                for (int batch_end_idx = token_end_idx; batch_end_idx > token_start_idx; batch_end_idx -= num_tokens_per_batch) {
                    auto batch_start_idx = max(token_start_idx, batch_end_idx - num_tokens_per_batch);  // 当前反向批次起点。

                    if (elect_one_sync()) {
                        // 批量 load 一段 combined_nvl_head 到 shared，避免逐 token/lane 直接 global 读写。
                        tma_load_1d(tma_buffer,
                                    combined_nvl_head + batch_start_idx * LEGACY_NUM_MAX_NVL_PEERS,
                                    tma_mbarrier,
                                    (batch_end_idx - batch_start_idx) * num_bytes_per_token);  // 批量 load combined_nvl_head。
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, (batch_end_idx - batch_start_idx) * num_bytes_per_token);  // 设置 TMA 字节数。
                    }
                    mbarrier_wait(tma_mbarrier, tma_phase);  // 等 load 完成。
                    __syncwarp();  // warp 内同步。

                    for (int token_idx = batch_end_idx - 1; token_idx >= batch_start_idx; --token_idx) {
                        if (lane_id < LEGACY_NUM_MAX_NVL_PEERS) {
                            auto current_head =
                                reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * LEGACY_NUM_MAX_NVL_PEERS + lane_id];  // 当前 token/NVL lane 的 head。
                            if (current_head < 0) {
                                reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * LEGACY_NUM_MAX_NVL_PEERS + lane_id] =
                                    -last_head - 1;  // 未命中 token 写负数编码。
                            } else {
                                last_head = current_head;  // 更新最近有效 head。
                            }
                        }
                    }
                    tma_store_fence();  // 确保 shared 修改对 TMA store 可见。
                    __syncwarp();  // 等修改完成。

                    if (elect_one_sync())
                        // 写回修正后的负数编码；combine 主 kernel 后续会直接读取这个表。
                        tma_store_1d(tma_buffer,
                                     combined_nvl_head + batch_start_idx * LEGACY_NUM_MAX_NVL_PEERS,
                                     (batch_end_idx - batch_start_idx) * num_bytes_per_token);  // 将修正后的 head 批量写回。
                    tma_store_wait<0>();  // 等 TMA store 完成。
                    __syncwarp();  // 同步后处理下一批。
                }  // 中文：结束当前 batch 的 NVL head TMA 写回。
            }  // 中文：结束当前 warp 负责的 token batch 循环。
        }  // 中文：结束 NVL head 修正分支。
    }  // 中文：结束 cached_notify device kernel。
}

// 中文：host wrapper 负责按 hidden/topk/scale 布局计算控制区大小，并选择 cached_notify 模板。
// 中文：该入口只处理 combine cached notify 的 head 表修正，不搬运 token payload。
void cached_notify(int hidden_int4,  // hidden 的 int4 长度。
                   int num_scales,  // scale 数。
                   int num_topk_idx,  // token 消息里的 topk_idx 数。
                   int num_topk_weights,  // token 消息里的 topk_weights 数。
                   int num_ranks,  // 全局 rank 数。
                   int num_channels,  // channel 数。
                   int num_combined_tokens,  // combine token 数。
                   // 中文：下面的 head/prefix 参数共同驱动 cached_notify 修正 combine 反向路径。
                   int* combined_rdma_head,  // combine RDMA head。
                   const int* rdma_channel_prefix_matrix,  // RDMA channel prefix。
                   const int* rdma_rank_prefix_sum,  // RDMA rank prefix。
                   // 中文：RDMA prefix 用于修正跨节点 head，NVL head 用于节点内反向转发。
                   int* combined_nvl_head,  // combine NVL head。
                   void* rdma_buffer_ptr,  // RDMA symmetric buffer。
                   int num_max_rdma_chunked_recv_tokens,  // RDMA recv ring 容量。
                   void** buffer_ptrs,  // NVL peer buffer 指针表。
                   int num_max_nvl_chunked_recv_tokens,  // NVL recv ring 容量。
                   int** barrier_signal_ptrs,  // NVL barrier 指针表。
                   int rank,  // 当前 rank。
                   cudaStream_t stream,  // CUDA 发射流。
                   int64_t num_rdma_bytes,  // RDMA buffer 总字节数。
                   int64_t num_nvl_bytes,  // NVL buffer 总字节数。
                   bool is_cached_dispatch,  // 是否 cached dispatch。
                   bool low_latency_mode) {  // 是否 low-latency PE 编号。
    const int num_threads = std::max(128, 32 * num_channels);  // 每 channel 一个 warp，至少 128 线程。
    const int num_warps = num_threads / 32;  // warp 数。
    const auto num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;  // RDMA rank 数。
    const int kNumTMABytesPerWarp = 8192;  // cached notify 修正 head 时每 warp TMA buffer 大小。
    const int smem_size = kNumTMABytesPerWarp * num_warps;  // 总 shared memory。

    // Get clean meta
    auto rdma_clean_meta = get_rdma_clean_meta(
        hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);  // 计算 RDMA 清理范围。
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4,  // hidden 的 int4 数。
                                             num_scales,  // FP8 scale 数。
                                             num_topk_idx,  // top-k id 数。
                                             num_topk_weights,  // top-k weight 数。
                                             num_rdma_ranks,  // RDMA rank 数。
                                             LEGACY_NUM_MAX_NVL_PEERS,  // NVL rank 上限。
                                             num_max_nvl_chunked_recv_tokens,  // NVL recv chunk 容量。
                                             num_channels,  // channel 数。
                                             is_cached_dispatch);  // 是否 cached dispatch，决定清理范围。
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);  // RDMA 清理范围越界检查。
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);  // NVL 清理范围越界检查。
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());  // device offset 使用 int。
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());  // NVL offset 使用 int。
    EP_HOST_ASSERT(num_channels * 2 > 3);  // cached_notify kernel 至少需要 block0/block1/一个 NVL 修正 block。

    // Launch kernel
    auto cached_notify_func = low_latency_mode ? cached_notify<true, kNumTMABytesPerWarp> : cached_notify<false, kNumTMABytesPerWarp>;  // 选择 low-latency/普通 PE 编号模板。
    SETUP_LAUNCH_CONFIG(num_channels * 2, num_threads, stream);  // 每 channel 一对 blocks。
    SET_SHARED_MEMORY_FOR_TMA(cached_notify_func);  // 配置动态 shared memory。
    LAUNCH_KERNEL(&cfg,
                  cached_notify_func,  // cached notify kernel。
                  rdma_clean_meta.first,  // RDMA 清理起点。
                  rdma_clean_meta.second,  // RDMA 清理长度。
                  nvl_clean_meta.first,  // NVL 清理起点。
                  nvl_clean_meta.second,  // NVL 清理长度。
                  combined_rdma_head,  // combine RDMA head。
                  num_combined_tokens,  // combine token 数。
                  num_channels,  // channel 数。
                  rdma_channel_prefix_matrix,  // RDMA channel prefix。
                  rdma_rank_prefix_sum,  // RDMA rank prefix。
                  // 中文：combined_nvl_head 与 RDMA head 一起描述 combine 反向路径的两级 ring 位置。
                  combined_nvl_head,  // combine NVL head。
                  rdma_buffer_ptr,  // RDMA buffer。
                  buffer_ptrs,  // NVL buffer 指针表。
                  barrier_signal_ptrs,  // NVL barrier 指针表。
                  rank,  // 当前 rank。
                  num_ranks,  // 全局 rank 数。
                  is_cached_dispatch,  // cached dispatch 标志。
                  nvshmem::cpu_rdma_team);  // RDMA team。
}

template <int kNumRanks,  // combine 涉及的 rank 数。
          bool kMaybeWithBias,  // 是否可能带 bias。
          typename dtype_t,  // hidden payload 数据类型。
          int kMaxNumRanks,  // 编译期最大 rank 数，用于局部数组上限。
          bool kUseTMA,  // 是否使用 TMA load。
          int kNumStages,  // TMA pipeline stage 数。
          int kNumTMALoadBytes = 0,  // 每次 TMA load 字节数。
          typename GetAddrFn,  // 根据 rank/lane 返回 payload 地址的函数对象。
          typename ReceiveTWFn>  // 接收 top-k weight 的函数对象。
__device__ int combine_token(bool is_token_in_rank,
                             int head_idx,  // 当前 lane/rank 对应的 ring head。
                             int lane_id,  // warp lane id。
                             int hidden_int4,  // hidden 的 int4 长度。
                             int num_topk,  // top-k 路数。
                             int4* combined_row,  // 输出 combine 行。
                             float* combined_topk_weights,  // 输出 combine topk 权重行。
                             const int4* bias_0_int4,  // 可选 bias0 行，int4 视图。
                             const int4* bias_1_int4,  // 可选 bias1 行，int4 视图。
                             int num_max_recv_tokens,  // ring buffer token 容量。
                             const GetAddrFn& get_addr_fn,  // 根据 rank/slot/hidden offset 取输入 payload 地址。
                             const ReceiveTWFn& recv_tw_fn,  // 根据 rank/slot/topk lane 读取 topk weight。
                             uint8_t* smem_ptr,  // TMA 模式使用的 shared memory。
                             uint32_t (&tma_phase)[kNumStages]) {  // TMA stage phase。
    constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);  // 一个 int4 内包含的 dtype 元素数。

    // Broadcast current heads
    // Lane `i` holds the head of rank `i` and `is_token_in_rank`
    EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");  // 一个 warp 最多覆盖 32 个 rank。
    int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];  // 当前 token 实际命中的 rank 及对应 ring slot。
    #pragma unroll
    for (int i = 0; i < kNumRanks; ++i)
        if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {
            slot_indices[num_topk_ranks] = __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;  // 从 lane i 取得 head 并转成 ring slot。
            topk_ranks[num_topk_ranks++] = i;  // 记录有效 rank。
        }
    EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);  // 有效 rank 数不能超过数组容量。
    // TMA 分支用于 forwarder warp 从 NVL buffer 聚合后写 RDMA send buffer；它只处理通信 payload，
    // 不负责最终输出 epilogue，因此 bias 只允许在非 TMA receiver 分支叠加到 combined_row。
    EP_STATIC_ASSERT(not(kUseTMA and kMaybeWithBias), "TMA cannot be used by receiver warps");  // 当前 TMA 路径不处理 bias。
    EP_STATIC_ASSERT(kNumStages == 2, "Only support 2 stages now");  // TMA pipeline 当前固定 2 stages。

    // Reduce data
    if constexpr (kUseTMA) {
        // TMA 路径把 hidden_int4 按 32 个 int4 为一块流水处理：
        // stage0 正在 reduce/store 时，stage1 可以预取下一块。每个 stage 的 shared 布局是：
        // [最多 8 个输入 rank tile][1 个输出 tile][mbarrier]。
        // 这样可以减少普通 global load/store 指令数量，适合 forwarder 的大块连续搬运。
        constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (LEGACY_NUM_MAX_NVL_PEERS + 1) + 16;  // 每 stage 存多个输入 rank 和一个输出 tile。
        EP_DEVICE_ASSERT(hidden_int4 % 32 == 0);  // TMA 路径按 32 个 int4 为一个 tile。

        auto tma_load_buffer = [=](const int& i, const int& j) -> int4* {
            return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + j * kNumTMALoadBytes);  // 第 i 个 stage、第 j 个输入 rank 的 shared tile。
        };  // 输入 TMA load buffer 地址计算。
        auto tma_store_buffer = [=](const int& i) -> int4* {
            return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + LEGACY_NUM_MAX_NVL_PEERS * kNumTMALoadBytes);  // 第 i 个 stage 的输出 tile。
        };  // 输出 TMA store buffer 地址计算。
        auto tma_mbarrier = [=](const int& i) -> uint64_t* {
            return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + (LEGACY_NUM_MAX_NVL_PEERS + 1) * kNumTMALoadBytes);  // 第 i 个 stage 的 mbarrier。
        };  // TMA barrier 地址计算。

        // Prefetch
        if (lane_id < num_topk_ranks)
            // 每个有效 rank 由一个 lane 发起一条 TMA load；lane_id>=num_topk_ranks 的 lane 不加载，
            // 但仍参与 mbarrier arrive，expect_tx=0 保持 warp 内控制流一致。
            tma_load_1d(
                tma_load_buffer(0, lane_id), get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], 0), tma_mbarrier(0), kNumTMALoadBytes);  // 预取第 0 个 hidden tile。
        mbarrier_arrive_and_expect_tx(tma_mbarrier(0), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);  // 设置 stage0 期望传输量。
        __syncwarp();  // 等预取发起。

        for (int shifted = 0, iter = 0; shifted < hidden_int4; shifted += 32, iter += 1) {
            const int stage_idx = iter % kNumStages;  // 当前消费 stage。
            const int next_stage_idx = (iter + 1) % kNumStages;  // 下一预取 stage。

            // Prefetch next stage
            if (shifted + 32 < hidden_int4) {
                if (lane_id < num_topk_ranks)
                    // 下一 stage 的预取提前发起，当前 stage 的 reduce 等待时可以与下一块数据传输重叠。
                    tma_load_1d(tma_load_buffer(next_stage_idx, lane_id),
                                get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], shifted + 32),
                                tma_mbarrier(next_stage_idx),
                                kNumTMALoadBytes);  // 预取下一 hidden tile。
                mbarrier_arrive_and_expect_tx(tma_mbarrier(next_stage_idx), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);  // 设置下一 stage 期望传输量。
                __syncwarp();  // 等下一 stage load 发起。
            }

            mbarrier_wait(tma_mbarrier(stage_idx), tma_phase[stage_idx]);  // 等当前 stage load 完成。
            float values[kDtypePerInt4] = {0};  // float 累加器。
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j) {
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(tma_load_buffer(stage_idx, j) + lane_id);  // 第 j 个 rank 的当前 int4 tile。
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    values[k] += static_cast<float>(recv_value_dtypes[k]);  // 累加该 rank 的贡献。
            }

            // Wait shared memory to be released
            // 允许最多 kNumStages-1 个 store 在途；这里等待最老 stage，防止覆盖还未写回的 shared 输出 tile。
            tma_store_wait<kNumStages - 1>();  // 等待最老 store stage 可复用。

            // Copy into shared and issue TMA
            auto out_dtypes = reinterpret_cast<dtype_t*>(tma_store_buffer(stage_idx) + lane_id);  // 输出 tile 的当前 lane 位置。
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                out_dtypes[j] = static_cast<dtype_t>(values[j]);  // float 累加结果转回 dtype。
            tma_store_fence();  // shared 写入对 TMA store 可见。
            __syncwarp();  // 等所有 lane 写 shared。

            if (elect_one_sync())
                tma_store_1d(tma_store_buffer(stage_idx), combined_row + shifted, kNumTMALoadBytes);  // TMA store 输出 tile。
            __syncwarp();  // 等 store 发起。
        }

        // Flush all writes
        tma_store_wait<0>();  // 等所有 TMA store 完成。
    } else {
        // 非 TMA 路径由每个 lane 处理 hidden_int4 中 stride=32 的若干 int4，
        // 适合 RDMA receiver 直接写最终 combined_x，并能顺便叠加 bias。
        #pragma unroll
        // 中文：hidden_int4 以 int4 为单位，每个 lane 处理间隔 32 的向量片段。
        for (int i = lane_id; i < hidden_int4; i += 32) {
            // Read bias
            // TODO: make it as a finer-grained template
            // 中文：这里仍用运行期指针判空，后续可按 bias 个数细分模板减少分支。
            int4 bias_0_value_int4, bias_1_value_int4;  // 当前 hidden int4 的两个 bias 值。
            if constexpr (kMaybeWithBias) {
                bias_0_value_int4 = bias_0_int4 != nullptr ? ld_nc_global(bias_0_int4 + i) : make_int4(0, 0, 0, 0);  // 读取 bias0 或零。
                bias_1_value_int4 = bias_1_int4 != nullptr ? ld_nc_global(bias_1_int4 + i) : make_int4(0, 0, 0, 0);  // 读取 bias1 或零。
            }

            // Read buffers
            // TODO: maybe too many registers here
            // 这里会把所有有效 rank 的同一个 hidden int4 都先放进寄存器数组。
            // kMaxNumRanks 较小时这样最直接；rank 数变大时寄存器压力会明显上升，所以 TODO 提醒后续优化。
            int4 recv_value_int4[kMaxNumRanks];  // 当前 hidden int4 上来自各 rank 的输入值。
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j)
                recv_value_int4[j] = ld_nc_global(get_addr_fn(topk_ranks[j], slot_indices[j], i));  // 读取第 j 个有效 rank 的 payload。

            // Clean
            // Reduce bias
            float values[kDtypePerInt4] = {0};  // float 累加器。
            if constexpr (kMaybeWithBias) {
                auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);  // bias0 dtype 视图。
                auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);  // bias1 dtype 视图。
                #pragma unroll
                for (int j = 0; j < kDtypePerInt4; ++j)
                    values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);  // bias 作为初始累加值。
            }

            // Reduce all-to-all results
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j) {
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);  // 第 j 个 rank 输入的 dtype 视图。
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    values[k] += static_cast<float>(recv_value_dtypes[k]);  // 累加 rank 贡献。
            }

            // Cast back to `dtype_t` and write
            int4 out_int4;  // 输出 int4 打包值。
            auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);  // 输出 dtype 视图。
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                out_dtypes[j] = static_cast<dtype_t>(values[j]);  // 转回目标 dtype。
            st_na_global(combined_row + i, out_int4);  // 写 combine 输出。
        }
    }

    // Reduce `topk_weights`
    if (lane_id < num_topk) {
        // top-k weight 与 hidden 一样按 rank 贡献相加；每个 lane 负责一个 top-k 列。
        // 如果某个 rank 对当前 token 没有贡献，它不会出现在 topk_ranks/slot_indices 中。
        float value = 0;  // topk 权重累加器。
        #pragma unroll
        for (int i = 0; i < num_topk_ranks; ++i)
            value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);  // 累加各 rank 的 topk 权重。
        st_na_global(combined_topk_weights + lane_id, value);  // 写 combine 后 topk 权重。
    }

    // Return the minimum top-k rank
    // 调用方只需要知道“本 token 至少消费了哪个 rank 的一个 slot”来推进 head。
    // topk_ranks 是按 rank lane 从小到大收集的，因此 topk_ranks[0] 可作为稳定代表。
    return topk_ranks[0];  // 返回第一个有效 rank，调用方用它推进对应 head。
}

template <bool kLowLatencyMode,          // 是否使用 low-latency PE 映射；影响 translate_dst_rdma_rank 的目标 PE。
          int kNumRDMARanks,             // 节点间 RDMA rank 数，等于 num_ranks / LEGACY_NUM_MAX_NVL_PEERS。
          typename dtype_t,              // hidden payload 数据类型，legacy 路径当前由 host wrapper 实例化为 nv_bfloat16。
          int kNumCombineForwarderWarps, // combine forwarder 的最大 warp 数，决定 NVL->RDMA 并行度。
          int kNumTMABytesPerSenderWarp, // NVL sender warp 可用的 TMA shared-memory scratch 字节数。
          int kNumTMABytesPerForwarderWarp, // forwarder warp 可用的 TMA shared-memory scratch 字节数。
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),  // 一个 token 最多涉及的 RDMA source 数。
          int kNumWarpsPerForwarder = (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1, // 每个 RDMA rank 分到的 forwarder warp 数。
          int kNumForwarders = kNumRDMARanks* kNumWarpsPerForwarder, // 实际 forwarder warp 总数。
          int kNumRDMAReceivers = kNumForwarders - LEGACY_NUM_MAX_NVL_PEERS> // 非 forwarder block 中除 NVL sender 外剩余的 receiver warp 数。
__global__ void __launch_bounds__((kNumForwarders + 1) * 32, 1) combine(int4* combined_x,  // 输出 combine 后 hidden，形状 [num_combined_tokens, hidden_int4]。
                                                                        float* combined_topk_weights,  // 输出 combine 后权重，形状 [num_combined_tokens, num_topk]。
                                                                        const bool* is_combined_token_in_rank,  // API 保留路由表，kernel 内主要依赖 head 表。
                                                                        const int4* x,  // 输入 expert 输出，形状 [num_tokens, hidden_int4]。
                                                                        const float* topk_weights,  // 输入权重，形状 [num_tokens, num_topk]。
                                                                        const int4* bias_0,  // 可选 bias0，形状 [num_combined_tokens, hidden_int4] 或空。
                                                                        const int4* bias_1,  // 可选 bias1，形状 [num_combined_tokens, hidden_int4] 或空。
                                                                        const int* combined_rdma_head,  // combine RDMA head 表，描述每个输出 token 对 RDMA slot 的依赖。
                                                                        const int* combined_nvl_head,  // combine NVL head 表，布局 [token, LEGACY_NUM_MAX_NVL_PEERS]。
                                                                        const SourceMeta* src_meta,  // dispatch 阶段保存的源 token 元信息，用于反向路由。
                                                                        const int* rdma_channel_prefix_matrix,  // [kNumRDMARanks, num_channels] 的 RDMA prefix 表。
                                                                        const int* rdma_rank_prefix_sum,  // RDMA rank 维度 prefix，用于把局部 token id 转全局 combine id。
                                                                        const int* gbl_channel_prefix_matrix,  // 全局 channel prefix，NVL sender 计算 token 范围使用。
                                                                        int num_tokens,  // 输入 expert token 总数。
                                                                        int num_combined_tokens,  // combine 输出 token 总数。
                                                                        int hidden,  // hidden 元素数，必须能被 int4/dtype_t 整除。
                                                                        int num_topk,  // token 的 top-k 路数。
                                                                        void* rdma_buffer_ptr,  // NVSHMEM symmetric RDMA 环形 buffer。
                                                                        int num_max_rdma_chunked_send_tokens,  // RDMA send chunk 上限。
                                                                        int num_max_rdma_chunked_recv_tokens,  // RDMA recv ring 槽位数。
                                                                        void** buffer_ptrs,  // 节点内 NVL peer buffer 指针表，长度 LEGACY_NUM_MAX_NVL_PEERS。
                                                                        int num_max_nvl_chunked_send_tokens,  // NVL send chunk 上限。
                                                                        int num_max_nvl_chunked_recv_tokens,  // NVL recv ring 总槽位数，会按 RDMA source 均分。
                                                                        int rank,  // 当前全局 rank。
                                                                        int num_ranks) {  // 全局 rank 数。
    enum class WarpRole { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator };

    const auto sm_id = static_cast<int>(blockIdx.x);  // 当前 block id。
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;  // block 线程数和 warp 数。
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();  // 线程 id 和 warp lane。
    const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;  // combine 也用一对 block 表示一个 channel。
    const bool is_forwarder_sm = sm_id % 2 == 1;  // 奇数 block 负责 NVL->RDMA forwarder，偶数 block 负责 NVL sender/RDMA receiver。

    EP_DEVICE_ASSERT(num_topk <= 32);  // top-k lane 数不超过 warp lane 数。
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);  // hidden 必须能按 int4 打包。
    const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));  // hidden 的 int4 长度。
    const auto hidden_bytes = hidden_int4 * sizeof(int4);  // hidden 字节数。
    const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, 0, 0, num_topk);  // combine token 消息字节数：hidden + SourceMeta + topk_weights。

    // NOTES: we decouple a channel into 2 SMs
    const auto rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS, nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;  // 当前全局 rank 的 RDMA/NVL 坐标。
    auto role_meta = [=]() -> std::pair<WarpRole, int> {
        auto warp_id = thread_id / 32;  // 原始 warp id。
        if (not is_forwarder_sm) {
            if (warp_id < LEGACY_NUM_MAX_NVL_PEERS) {
                auto shuffled_warp_id = warp_id;  // NVL sender 目标 rank。
                shuffled_warp_id = (shuffled_warp_id + channel_id) % LEGACY_NUM_MAX_NVL_PEERS;  // 按 channel 打散目标 NVL rank。
                return {WarpRole::kNVLSender, shuffled_warp_id};  // 前 8 个 warp 负责发给节点内 peers。
            } else if (warp_id < kNumForwarders) {
                return {WarpRole::kRDMAReceiver, warp_id - LEGACY_NUM_MAX_NVL_PEERS};  // 其余部分 warp 从 RDMA ring 收最终结果。
            } else {
                return {WarpRole::kCoordinator, 0};  // 最后一个 warp 协调 head 释放。
            }
        } else {
            if (warp_id < kNumForwarders) {
                auto shuffled_warp_id = (warp_id + channel_id) % kNumForwarders;  // 按 channel 打散 forwarder warp。
                return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};  // 从 NVL 读并写 RDMA staging。
            } else {
                return {WarpRole::kCoordinator, 0};  // forwarder block 的 coordinator。
            }
        }
    }();
    auto warp_role = role_meta.first;  // 当前 warp 角色。
    auto warp_id = role_meta.second;  // 角色内编号。

    EP_DEVICE_ASSERT(num_warps == kNumForwarders + 1);  // warp 数必须等于工作 warp + coordinator。
    auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;  // NVL ring 按 RDMA source 均分容量。

    if (warp_role == WarpRole::kNVLSender) {
        // NVL producers
        const auto dst_nvl_rank = warp_id;  // 当前 warp 负责发送给的目标 NVL rank。

        // NVL layouts
        // NOTES: to avoid deadlocks, we use separate NVL buffers for different RDMA sources
        auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];  // 目标 NVL buffer 和本地 buffer。
        auto nvl_channel_x = AsymBuffer<uint8_t>(dst_buffer_ptr,  // 目标 NVL buffer 指针。
                                                 num_max_nvl_chunked_recv_tokens * num_bytes_per_token,  // payload ring 字节数。
                                                 LEGACY_NUM_MAX_NVL_PEERS,  // NVL rank 维度。
                                                 channel_id,  // 当前 channel。
                                                 num_channels,  // channel 总数。
                                                 nvl_rank)  // source NVL rank。
                                 .advance_also(local_buffer_ptr);  // combine NVL payload ring。
        auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, dst_nvl_rank)
                                    .advance_also(dst_buffer_ptr);  // 目标 receiver 写回的 head。
        auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                                    .advance_also(local_buffer_ptr);  // 当前 sender 写入的 tail。

        // TMA stuffs
        extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];  // TMA shared memory。
        auto tma_buffer = smem_tma_buffer + dst_nvl_rank * kNumTMABytesPerSenderWarp;  // 每个目标 NVL rank 一段 sender TMA buffer。
        auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);  // mbarrier 放在 token 消息后。
        uint32_t tma_phase = 0;  // TMA phase。
        if (elect_one_sync()) {
            mbarrier_init(tma_mbarrier, 1);  // 初始化 sender TMA barrier。
            fence_barrier_init();  // 确保 barrier 初始化可见。
            EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumTMABytesPerSenderWarp);  // TMA buffer 容量检查。
        }
        __syncwarp();  // 等 TMA 初始化完成。

        // Get tasks for each RDMA lane
        int token_start_idx = 0, token_end_idx = 0;  // 当前 RDMA lane 负责的 token 范围。
        if (lane_id < kNumRDMARanks) {
            int prefix_idx = (lane_id * LEGACY_NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;  // 全局 rank/channel prefix 索引。
            token_start_idx = gbl_channel_prefix_matrix[prefix_idx];  // 当前 channel 起点。
            token_end_idx = (prefix_idx == num_channels * num_ranks - 1) ? num_tokens : gbl_channel_prefix_matrix[prefix_idx + 1];  // 当前 channel 终点。
        }
        __syncwarp();  // 等所有 lane 得到范围。

        // NOTES: here the cached value of each lane is only responsible for a single RDMA buffer
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;  // 当前 lane/RDMA source 的 NVL ring head/tail。
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");  // 一个 warp lane 覆盖所有 RDMA source。

        // Iterate over all tokens and send by chunks
        int current_rdma_idx = channel_id % kNumRDMARanks;  // round-robin 起点，按 channel 打散。
        while (true) {
            // Exit if possible
            if (__all_sync(0xffffffff, token_start_idx >= token_end_idx))
                break;  // 所有 RDMA lane 的 token 都发送完成。

            // Decide the next RDMA buffer to send
            bool is_lane_ready = false;  // 当前 lane 是否有 token 且 NVL ring 空间足够。
            auto start_time = clock64();  // 等待 NVL 空槽开始时间。
            while (true) {
                int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;  // 当前 RDMA source 分片的 ring 已用槽。
                is_lane_ready = lane_id < kNumRDMARanks and token_start_idx < token_end_idx and
                    num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;  // 空槽是否足够一个 chunk。
                if (__any_sync(0xffffffff, is_lane_ready))
                    break;  // 至少一个 RDMA source 可以发送。

                // Retry
                if (lane_id < kNumRDMARanks and token_start_idx < token_end_idx)
                    cached_channel_head_idx = ld_volatile_global(nvl_channel_head.buffer() + lane_id);  // 读取 receiver 释放的 head。

                // Timeout check
                if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf(
                        // 中文：该格式串分两行是为了避免单行过长，参数在下一段逐项给出。
                        "DeepEP combine NVL sender timeout, channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, RDMA lane: %d, head: %d, tail: "
                        // 中文：该打印发生在等待目标 NVL receiver 释放 ring 空间的路径。
                        // 中文：head/tail/start/end 能定位哪个 RDMA lane 的 NVL ring 槽位迟迟未释放。
                        "%d, start: %d, end: %d\n",
                        // 中文：下面的坐标组合定位“当前 sender 等哪个目标 NVL rank 的哪个 RDMA lane”。
                        channel_id,  // 当前 channel。
                        rdma_rank,   // 本 rank 的 RDMA 坐标。
                        nvl_rank,    // 本 rank 的 NVL 坐标。
                        dst_nvl_rank, // 当前 NVL sender 的目标 NVL rank。
                        lane_id,     // 当前 lane/RDMA source。
                        ld_volatile_global(nvl_channel_head.buffer() + lane_id), // receiver 已释放 head。
                        cached_channel_tail_idx, // sender 已写 tail。
                        token_start_idx, // 当前 lane 的待发送起点。
                        token_end_idx);  // 当前 lane 的待发送终点。
                    trap();  // 等 NVL 空槽超时。
                }
            }

            // Sync token start index and count
            for (int i = 0; i < kNumRDMARanks; ++i) {
                current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks;  // 轮询下一个 RDMA source。
                if (__shfl_sync(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
                    continue;  // 该 RDMA source 没有 token 或暂不可发送。

                // Sync token start index
                auto token_idx = static_cast<int64_t>(__shfl_sync(0xffffffff, token_start_idx, current_rdma_idx));  // 广播被选中 RDMA source 的 token 起点。
                int num_tokens_in_chunk =
                    __shfl_sync(0xffffffff, min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), current_rdma_idx);  // 本轮发送 token 数。

                // Send by chunk
                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
                    // Get an empty slot
                    int dst_slot_idx = 0;  // NVL ring 槽位。
                    if (lane_id == current_rdma_idx) {
                        dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;  // 当前 RDMA source 分片内槽位。
                        dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;  // 转成整个 NVL ring 的槽位。
                    }
                    dst_slot_idx = __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);  // 广播槽位给整个 warp。

                    // Load data
                    auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;  // NVL token 消息目标地址。
                    auto shifted_x = x + token_idx * hidden_int4;  // 输入 expert 输出 token 地址。
                    tma_store_wait<0>();  // 等前一轮 TMA store 完成。
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer, shifted_x, tma_mbarrier, hidden_bytes);  // TMA load hidden。
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, hidden_bytes);  // 设置 TMA load 字节数。
                    }
                    __syncwarp();  // 等 load 发起。
                    mbarrier_wait(tma_mbarrier, tma_phase);  // 等 load 完成。

                    // Load source meta
                    if (lane_id == num_topk)
                        *reinterpret_cast<SourceMeta*>(tma_buffer + hidden_bytes) = ld_nc_global(src_meta + token_idx);  // 将 SourceMeta 写入 TMA buffer。

                    // Load `topk_weights`
                    if (lane_id < num_topk)
                        *reinterpret_cast<float*>(tma_buffer + hidden_bytes + sizeof(SourceMeta) + lane_id * sizeof(float)) =
                            ld_nc_global(topk_weights + token_idx * num_topk + lane_id);  // 将 topk weight 写入 TMA buffer。

                    // Issue TMA store
                    tma_store_fence();  // 确保 SourceMeta/topk_weights 写入 shared 后再 store。
                    __syncwarp();  // 等 shared 写入完成。
                    if (elect_one_sync())
                        tma_store_1d(tma_buffer, shifted_x_buffers, num_bytes_per_token, false);  // TMA store 整个 token 消息到 NVL ring。
                }
                lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;  // 被选中 lane 更新自己的 token 起点。
            }

            // Move queue tail
            tma_store_wait<0>();  // 等本轮所有 TMA store 完成。
            __syncwarp();  // warp 内同步。
            if (lane_id < kNumRDMARanks and is_lane_ready)
                st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);  // release 更新该 RDMA source 分片的 tail。
        }  // 中文：结束 NVL sender 分片发送循环。
    } else {
        // Combiners and coordinators
        // RDMA symmetric layout
        // 中文：combine 反向路径先从 RDMA ring 收跨节点 payload，再经 NVL ring 回到源 NVL rank。
        // 中文：该分支同时包含 RDMA receiver、NVL forwarder 和最终 combiner/coordinator 角色。
        auto rdma_channel_data = SymBuffer<int8_t>(
            rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);  // combine RDMA payload ring。
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);  // RDMA head，由 receiver/coordinator 释放。
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);  // RDMA tail，由 forwarder/put 写入。

        // NVL layouts
        void* local_nvl_buffer = buffer_ptrs[nvl_rank];  // 本地 NVL buffer。
        void* nvl_buffers[LEGACY_NUM_MAX_NVL_PEERS];  // 节点内所有 NVL peer buffer 指针。
        #pragma unroll
        // 中文：本地数组拷贝让后续 AsymBuffer 构造在编译期展开 peer 数。
        for (int i = 0; i < LEGACY_NUM_MAX_NVL_PEERS; ++i)
            nvl_buffers[i] = buffer_ptrs[i];  // 拷贝指针表到局部数组，供 AsymBuffer 使用。
        // 中文：AsymBuffer 用本地视角组织 peer buffer，advance_also 会同步推进所有 rank 的布局游标。
        auto nvl_channel_x =
            AsymBuffer<uint8_t>(
                // 中文：每个 token slot 的字节数为 hidden+SourceMeta+topk weight 区。
                local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels)
                .advance_also<LEGACY_NUM_MAX_NVL_PEERS>(nvl_buffers);  // combine NVL payload ring。
            // 中文：payload ring 之后紧跟 head/tail 控制区，三者必须用相同 channel_id/num_channels 切片。
        // 中文：下面的 head/tail ring 每个 NVL peer 存 kNumRDMARanks 个分片进度。
        auto nvl_channel_head =
            AsymBuffer<int, LEGACY_NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                .advance_also(local_nvl_buffer);  // NVL head，按 source NVL rank/RDMA 分片。
        // 中文：tail 放在本地 buffer 中，由 sender 更新；head 分布在 peer buffer 中，由 receiver/coordinator 释放。
        auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks, LEGACY_NUM_MAX_NVL_PEERS, channel_id, num_channels)
                                    .advance_also<LEGACY_NUM_MAX_NVL_PEERS>(nvl_buffers);  // NVL tail，sender release 写入。

        // Combiner warp synchronization
        __shared__ volatile int forwarder_nvl_head[kNumForwarders][LEGACY_NUM_MAX_NVL_PEERS];  // forwarder 对每个 NVL source 的消费 head。
        __shared__ volatile bool forwarder_retired[kNumForwarders];  // forwarder warp 是否完成。
        __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];  // RDMA receiver 对每个 RDMA source 的消费 head。
        __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];  // RDMA receiver warp 是否完成。
        auto sync_forwarder_smem = [=]() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumForwarders + 1) * 32)); };  // forwarder 组同步。
        auto sync_rdma_receiver_smem = [=]() { asm volatile("barrier.sync 1, %0;" ::"r"((kNumRDMAReceivers + 1) * 32)); };  // RDMA receiver 组同步。

        if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
            // Receive from NVL ranks and forward to RDMA ranks
            // NOTES: this part is using "large warps" for each RDMA ranks
            // 一个目标 RDMA rank 可能由多个 warp 协作，因此先把角色内 warp_id 拆成目标 rank 和子 warp 编号。
            const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder;  // 当前 large-warp 组负责写入的目标 RDMA rank。
            const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;    // 当前 warp 在目标 RDMA rank 组内的编号。
            auto send_buffer =
                // 目标是本 RDMA rank 时直接写 recv_buffer，目标是远端时先写本地 send_buffer 再用 IBGDA put。
                dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
            auto sync_large_warp = [=]() {
                if (kNumWarpsPerForwarder == 1) {
                    __syncwarp();  // 单 warp forwarder 时，warp 内同步即可。
                } else {
                    // 多 warp forwarder 时使用独立 barrier id；barrier 2 起留给每个 dst RDMA rank 的 large-warp 组。
                    asm volatile("bar.sync %0, %1;" ::"r"(dst_rdma_rank + 2), "r"(kNumWarpsPerForwarder * 32));
                }
            };
            // CUDA CTA barrier id 通常只有 0..15，这里前两个给 forwarder/receiver 共享内存同步，后面给 RDMA rank 组。
            EP_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= 16, "Barriers are not enough");

            // TMA stuffs
            constexpr int kNumStages = 2;  // 双缓冲 TMA stage，隐藏 load/store 与计算之间的延迟。
            constexpr int kNumTMALoadBytes = sizeof(int4) * 32;  // 一个 warp 每次搬 32 个 int4，即 512B hidden tile。
            constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (LEGACY_NUM_MAX_NVL_PEERS + 1) + 16;  // 每个 stage 放各 NVL peer tile、输出 tile 和 mbarrier。
            // 编译期检查 shared memory 切片足够，否则模板实例化时直接失败，避免运行期越界。
            EP_STATIC_ASSERT(kNumTMABufferBytesPerStage * kNumStages <= kNumTMABytesPerForwarderWarp, "TMA buffer is not larger enough");

            extern __shared__ __align__(1024) uint8_t smem_buffer[];  // forwarder block 共享 TMA scratch，由每个 warp 切一段。
            auto smem_ptr = smem_buffer + warp_id * kNumStages * kNumTMABufferBytesPerStage;  // 当前 warp 的 stage 0 起始地址。
            auto tma_mbarrier = [=](const int& i) {
                // 每个 stage 的 mbarrier 放在所有 tile buffer 之后；16B 对齐满足 mbarrier 指令要求。
                return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (LEGACY_NUM_MAX_NVL_PEERS + 1));
            };
            uint32_t tma_phase[kNumStages] = {0};  // 每个 TMA stage 的 phase bit，mbarrier_wait 会按阶段翻转。
            if (lane_id < kNumStages) {
                mbarrier_init(tma_mbarrier(lane_id), 32);  // 用前两个 lane 初始化两个 stage 的 barrier。
                fence_barrier_init();                      // 确保 barrier 初始化对后续 TMA 指令可见。
            }
            __syncwarp();  // 所有 lane 等待 barrier 初始化完成。

            // Advance to the corresponding NVL buffer
            // combine 的 NVL ring 按 RDMA source 分片；当前 forwarder 只消费 dst_rdma_rank 这一片。
            nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_bytes_per_token);  // 指到该 RDMA 分片的 payload 起点。
            nvl_channel_head.advance(dst_rdma_rank);  // 指到该 RDMA 分片的 head 数组。
            nvl_channel_tail.advance(dst_rdma_rank);  // 指到该 RDMA 分片的 tail 数组。

            // Clean shared memory and sync
            EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
            lane_id < LEGACY_NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0) : 0;  // 初始化每个 NVL source 的可释放 head。
            lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;  // 当前 forwarder 尚未完成。
            sync_forwarder_smem();  // 让 coordinator 看到初始化后的 shared 状态。

            // Get count and cached head
            int cached_nvl_channel_tail_idx = 0;  // 当前 lane 缓存的源 NVL tail，减少反复全局内存读取。
            int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];  // 当前 channel 结束 prefix。
            int num_tokens_prefix = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];  // 当前 channel 起始 prefix。
            num_tokens_to_combine -= num_tokens_prefix;  // 当前 channel 在目标 RDMA rank 上要 combine 的 token 数。
            num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];  // 加跨 RDMA rank prefix 得到全局 token 偏移。
            combined_nvl_head += num_tokens_prefix * LEGACY_NUM_MAX_NVL_PEERS;  // 跳到当前 channel 的 [token, nvl_peer] head 表起点。

            // Iterate over all tokens and combine by chunks
            // 按 RDMA send buffer 能承载的 token 数分块；token_start_idx 在下面也被当作生产者 tail 使用。
            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                // Check destination queue emptiness, or wait a buffer to be released
                // 本块的右边界不能超过当前 channel 实际负责的 combine token 总数。
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                auto num_chunked_tokens = token_end_idx - token_start_idx;  // 本轮要写入 RDMA 环形队列的 token 数。
                auto start_time = clock64();                                // 等待目标 RDMA recv 队列释放空间的超时起点。
                while (sub_warp_id == 0 and lane_id == 0) {
                    // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >= num_chunked_tokens`
                    // Here, `token_start_idx` is the actual tail
                    // head 由目标 RDMA receiver 推进；tail 由当前 token_start_idx 表示，因此 tail-head 是队列已占用槽数。
                    int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                    // 只有剩余槽位足够容纳整个 chunk，才允许后续 sub-warp 写 send_buffer，避免覆盖尚未消费的数据。
                    if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
                        break;

                    // Timeout check
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        printf(
                            // 中文：RDMA check timeout 的格式串包含目标 RDMA rank 和 ring head/tail。
                            "DeepEP combine forwarder (RDMA check) timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA: %d, head: %ld, tail: "
                            // 中文：该检查发生在 forwarder 准备把 NVL 汇总结果写入目标 RDMA ring 前。
                            // 中文：chunked 是本次准备发送的 token 数，可判断是否因为 ring 空间不足卡住。
                            "%d, chunked: %d\n",
                            // 中文：该 timeout 说明目标 RDMA ring 没有释放足够空间给 forwarder。
                            channel_id,  // 当前 channel。
                            // 中文：后续参数给出本地 RDMA/NVL 坐标与目标 RDMA ring 进度。
                            rdma_rank,   // 本 rank 的 RDMA 坐标。
                            nvl_rank,    // 本 rank 的 NVL 坐标。
                            dst_rdma_rank, // 当前 forwarder 目标 RDMA rank。
                            ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)), // 目标队列已释放 head。
                            token_start_idx, // 当前生产者 tail。
                            num_chunked_tokens); // 本 chunk 需要的槽位数。
                        trap();  // 队列长期无空间通常表示 receiver 未推进 head，直接 trap 暴露死锁。
                    }
                }
                sync_large_warp();  // 等待负责探测容量的 lane 放行后，所有 forwarder sub-warp 再开始写本 chunk。

                // Combine and write to the RDMA buffer
                // 多个 sub-warp 以 stride 方式瓜分 chunk 内 token，例如 4 个 sub-warp 时分别处理 idx%4 的 token。
                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder) {
                    // Read expected head
                    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                    int expected_head = -1;  // 每个 lane 对应一个 NVL peer，记录该 peer 对当前 token 的期望环形槽位。
                    if (lane_id < LEGACY_NUM_MAX_NVL_PEERS) {
                        // combined_nvl_head 的布局是 [num_tokens_to_combine, LEGACY_NUM_MAX_NVL_PEERS]。
                        // 正数/零表示该 NVL peer 有贡献并位于对应 slot；负数使用 -slot-1 编码，表示该 peer 无贡献但 head 可推进到 slot。
                        expected_head = ld_nc_global(combined_nvl_head + token_idx * LEGACY_NUM_MAX_NVL_PEERS + lane_id);
                        // 中文：负数编码会被转换成可推进 head，正数编码表示需要等待真实 payload。
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head);
                    }

                    // Wait lanes to be ready
                    start_time = clock64();  // 等待每个源 NVL peer 把对应 slot 写完的超时起点。
                    while (cached_nvl_channel_tail_idx <= expected_head) {
                        // acquire 语义确保看到 tail 推进后，也能看到源 NVL sender 在该 slot 中写入的 token 数据。
                        cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

                        // Timeout check
                        if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and lane_id < LEGACY_NUM_MAX_NVL_PEERS) {
                            printf(
                                // 中文：NVL check timeout 的格式串包含源 NVL rank、目标 RDMA rank和 token 分类。
                                "DeepEP combine forwarder (NVL check) timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, dst RDMA: %d, "
                                // 中文：该检查发生在 forwarder 等待源 NVL sender 发布 payload 时。
                                // 中文：sub/large 标记当前是否为分片 token 与大 token 路径，辅助定位转发策略。
                                "tail: %d, waiting: %d, total: %d, sub: %d, large: %d, expected: %d\n",
                                // 中文：该 timeout 说明源 NVL sender 尚未把 expected_head 对应 token 发布到 tail。
                                channel_id,  // 当前 channel。
                                // 中文：tail/waiting/expected 三者能定位是源 NVL 未生产还是 head 编码错误。
                                rdma_rank,   // 本 rank 的 RDMA 坐标。
                                nvl_rank,    // 本 rank 的 NVL 坐标。
                                lane_id,     // 当前 lane/source NVL rank。
                                dst_rdma_rank, // 当前 forwarder 目标 RDMA rank。
                                cached_nvl_channel_tail_idx, // 已观察到的源 NVL tail。
                                token_idx,   // 当前正在 combine 的 token。
                                num_tokens_to_combine, // 当前 channel 总 token 数。
                                sub_warp_id, // 当前 large-warp 子 warp。
                                kNumWarpsPerForwarder, // large-warp 总 warp 数。
                                expected_head); // 等待的源 NVL slot。
                            trap();  // 源 NVL sender 未发布 expected_head 时触发，方便定位跨 warp 队列卡点。
                        }
                    }

                    // Combine current token
                    auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;  // token_idx 映射到 RDMA recv/send 环形槽位。
                    void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_token;  // 当前 token 在 RDMA send buffer 中的起始地址。
                    // 中文：下面两个 lambda 把 NVL ring slot 映射为 combine_reduce 可读取的 hidden/weight 地址。
                    auto get_addr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                        // nvl_channel_x 布局为 [src_nvl_rank][slot_idx][hidden_int4_idx]，每个 int4 是 16B 向量化 payload。
                        return reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token) +
                            hidden_int4_idx;
                    };
                    // 中文：权重 lambda 与 hidden lambda 使用同一个 src_nvl_rank/slot_idx，保证数据和权重同源。
                    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
                        // 每个 token 的权重区位于 hidden payload 之后，并跳过 SourceMeta；topk_idx 对应当前专家权重下标。
                        return ld_nc_global(reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token +
                                                                     // 中文：hidden 后先是 SourceMeta，再是 top-k weight 数组。
                                                                     hidden_bytes + sizeof(SourceMeta)) +
                                            topk_idx);
                    };
                    // combine_token 从多个 NVL peer 的 slot 中读取同一 token，按 top-k 权重聚合 hidden payload，
                    // 然后把结果写入 RDMA send buffer：前 hidden_bytes 是 dtype_t payload，后面是 SourceMeta 与 top-k weight。
                    combine_token<LEGACY_NUM_MAX_NVL_PEERS, false, dtype_t, LEGACY_NUM_MAX_NVL_PEERS, true, kNumStages, kNumTMALoadBytes>(
                        expected_head >= 0,  // true 表示当前 lane/NVL peer 对 token 有实际贡献，false 只参与 head 推进。
                        expected_head,       // 源 NVL ring 中待读取的 slot index，负数编码在进入前保留给 combine_token 判断。
                        lane_id,             // lane_id 同时代表 src_nvl_rank，最多覆盖 8 个 NVL peer。
                        hidden_int4,         // hidden 维度按 int4 打包后的长度，例如 fp16 hidden=4096 时为 512。
                        num_topk,            // token 对应的 top-k 专家数，也是 topk weight 区域长度。
                        static_cast<int4*>(shifted),  // 输出 hidden payload 的 RDMA send buffer 地址。
                        reinterpret_cast<float*>(static_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),  // 输出 topk weight 地址。
                        nullptr,                         // 旧版 combine forwarder 这里不需要 bias_0。
                        nullptr,                         // 旧版 combine forwarder 这里不需要 bias_1。
                        num_max_nvl_chunked_recv_tokens_per_rdma,  // 每个 RDMA source 在 NVL ring 中拥有的槽位数。
                        get_addr_fn,                                  // 从 NVL ring 取 hidden payload 的寻址闭包。
                        recv_tw_fn,                                   // 从 NVL ring 取 top-k weight 的寻址闭包。
                        smem_ptr,                                     // TMA scratch 起始地址。
                        tma_phase);                                   // 双缓冲 stage 的 mbarrier phase。

                    // Update head
                    // 当前 token 合并完成后，forwarder_nvl_head 通知 coordinator 对应 NVL source 的可释放进度。
                    if (lane_id < LEGACY_NUM_MAX_NVL_PEERS)
                        // 负数编码只推进到解码出的 slot；正数贡献被消费后推进到 expected_head+1。
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
                }
                sync_large_warp();  // 确保整个 chunk 都已经写入 send_buffer，再由最后一个 sub-warp 发起 RDMA 传输。

                // Issue RDMA send
                // 只让最后一个 sub-warp 发 RDMA put，避免多个 sub-warp 重复发送同一块数据。
                if (sub_warp_id == kNumWarpsPerForwarder - 1) {
                    if (dst_rdma_rank != rdma_rank) {
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;  // chunk 起始 token 对应远端环形槽位。
                        const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_token;  // 本次 put 的连续字节数。
                        const auto dst_ptr =
                            // 远端地址：目标 RDMA rank 的 recv_buffer 中，属于当前源 rdma_rank 的队列槽位。
                            reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        const auto src_ptr =
                            // 本地地址：为目标 dst_rdma_rank 准备好的 send_buffer 槽位，内容刚由 combine_token 写完。
                            reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        // 使用 warp 级 IBGDA put 非阻塞发送；remote PE 通过 dst_rdma_rank/nvl_rank 映射获得。
                        nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr,  // 远端目标 recv_buffer 地址。
                                                          src_ptr,   // 本地 staging/send_buffer 地址。
                                                          num_bytes_per_msg,  // 本 chunk 总字节数。
                                                          translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), // 目标 NVSHMEM PE。
                                                          channel_id, // 当前 channel 对应 IBGDA 队列。
                                                          lane_id,    // warp lane 协作发起 put。
                                                          0);         // wr_id/flag 保留为 0。
                    } else {
                        // 本 RDMA rank 自发自收时无需网络 put，但仍要用 fence 保证本地写入对后续 tail 更新可见。
                        memory_fence();
                    }

                    // Write new RDMA tail
                    __syncwarp();  // 等 warp 内 put/fence 指令都发起后，再发布 tail。
                    if (elect_one_sync()) {
                        // tail 增加 num_chunked_tokens 后，目标 RDMA receiver 就能按 ring 顺序读取这批 combine 结果。
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank), // 远端看到的 tail 计数器。
                                                        num_chunked_tokens, // 本次发布的 token 数。
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), // 目标 PE。
                                                        channel_id, // 当前 channel。
                                                        dst_rdma_rank == rdma_rank); // 本地路径使用 local AMO。
                    }
                }
            }

            // Retired
            // 中文：forwarder 所有 token 处理完后标记 retired，coordinator 可据此释放 head。
            __syncwarp();
            if (elect_one_sync())
                forwarder_retired[warp_id] = true;
            // 中文：forwarder retired 后 coordinator 不再等待该 warp 的 NVL/RDMA head 更新。
        } else if (warp_role == WarpRole::kRDMAReceiver) {
            // Receive from RDMA ranks and write to the output tensor
            // Clean shared memory and sync
            // 中文：receiver warp 每个 lane 追踪一个 RDMA source 的 head，coordinator 用这些 head 释放 RDMA ring。
            EP_DEVICE_ASSERT(kNumRDMARanks <= 32);  // 一个 warp lane 覆盖所有 RDMA source。
            lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;  // 初始化当前 receiver warp 的 RDMA head。
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;  // 初始化 retired 标志。
            sync_rdma_receiver_smem();  // 等 coordinator/receiver 看到初始化状态。

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;  // 当前 channel 负责的 combine 输出 token 范围。
            get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);  // 按 channel 切分 token。

            // Iterate over all tokens and combine
            int cached_channel_tail_idx = 0;  // 当前 lane/RDMA source 的 tail 缓存。
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
                // Read expected head
                EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");  // 编译期保证 RDMA source 可由一个 warp 覆盖。
                int expected_head = -1;  // 当前 lane/RDMA source 对该 token 期望消费的 head。
                if (lane_id < kNumRDMARanks) {
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks + lane_id);  // 读取 cached_notify 修正后的 RDMA head。
                    (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1)
                                        : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);  // 初始化该 lane 的消费 head。
                }

                // Wait lanes to be ready
                auto start_time = clock64();  // 等待 RDMA tail 到达开始时间。
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));  // acquire 读取 RDMA tail。

                    // Timeout check
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        printf(
                            // 中文：RDMA receiver timeout 表示本 receiver 尚未看到源 RDMA tail 推进到 expected。
                            "DeepEP combine RDMA receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, tail: %d, waiting: %ld, "
                            // 中文：expect 是该 source RDMA rank 对应 token 需要等到的 ring tail。
                            "expect: %d\n",
                            // 中文：下面坐标定位本 receiver 等待的源 RDMA rank。
                            channel_id,  // 当前 channel。
                            // 中文：后续参数定位哪个 src RDMA rank 的 tail 没有到达 expected head。
                            rdma_rank,  // 当前 RDMA rank。
                            nvl_rank,  // 当前 NVL rank。
                            lane_id,  // 等待的源 RDMA rank。
                            cached_channel_tail_idx,  // 已观察到的 tail。
                            token_idx,  // 当前 combine token。
                            expected_head);  // 期望至少到达的 head。
                        trap();  // 等 RDMA token 超时。
                    }
                }
                __syncwarp();  // 等所有 lane 可读。

                // Combine current token
                auto get_addr_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                    return reinterpret_cast<int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token) +
                        hidden_int4_idx;  // 返回 RDMA ring 中某 source/slot 的 hidden int4 地址。
                };  // combine_token 读取 payload 的地址函数。
                auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float {
                    return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) +
                                                                       slot_idx * num_bytes_per_token + hidden_bytes + sizeof(SourceMeta)) +
                                        topk_idx);  // 读取 RDMA token 消息中的 topk weight。
                };  // combine_token 读取权重的函数。
                uint32_t dummy_tma_phases[2];  // 非 TMA 路径不使用 phase，仅满足模板参数。
                combine_token<kNumRDMARanks, true, dtype_t, kNumTopkRDMARanks, false, 2>(
                    expected_head >= 0,  // 当前 lane 是否贡献该 token。
                    expected_head,  // ring head。
                    lane_id,  // lane id。
                    hidden_int4,  // hidden int4 长度。
                    num_topk,  // top-k 路数。
                    combined_x + token_idx * hidden_int4,  // 输出行。
                    combined_topk_weights + token_idx * num_topk,  // 输出 topk 权重行。
                    bias_0 == nullptr ? nullptr : bias_0 + token_idx * hidden_int4,  // 可选 bias0。
                    bias_1 == nullptr ? nullptr : bias_1 + token_idx * hidden_int4,  // 可选 bias1。
                    num_max_rdma_chunked_recv_tokens,  // RDMA ring 容量。
                    get_addr_fn,  // payload 地址函数。
                    recv_tw_fn,  // topk 权重读取函数。
                    nullptr,  // 非 TMA 路径无 shared memory。
                    dummy_tma_phases);  // 非 TMA dummy phases。
            }

            // Retired
            __syncwarp();  // 等最后一次 combine 完成。
            if (elect_one_sync())
                rdma_receiver_retired[warp_id] = true;  // 标记 receiver warp 完成。
        } else {
            // Coordinator
            // Sync shared memory status
            is_forwarder_sm ? sync_forwarder_smem() : sync_rdma_receiver_smem();  // 根据 block 类型同步对应 shared 状态。
            const auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;  // 每个 RDMA rank 对应的 forwarder warp 数。

            int last_rdma_head = 0;  // coordinator 已释放的 RDMA head。
            int last_nvl_head[kNumRDMARanks] = {0};  // 每个 RDMA rank 已释放的 NVL head。
            int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;  // 当前 lane 负责的 RDMA rank。
            int dst_nvl_rank = lane_id < LEGACY_NUM_MAX_NVL_PEERS ? lane_id : 0;  // 当前 lane 负责的 NVL rank。
            EP_STATIC_ASSERT(kNumCombineForwarderWarps <= 32, "Invalid number of forwarder warps");  // 一个 warp 能覆盖所有 forwarder。
            while (true) {
                // Retired
                if (not is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id]))
                    break;  // RDMA receiver block 中所有 receiver warp 完成。
                if (is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumForwarders or forwarder_retired[lane_id]))
                    break;  // forwarder block 中所有 forwarder warp 完成。

                // Find minimum head for RDMA ranks
                if (not is_forwarder_sm) {
                    int min_head = std::numeric_limits<int>::max();  // 所有未完成 RDMA receiver 对该 RDMA rank 的最小 head。
                    #pragma unroll
                    for (int i = 0; i < kNumRDMAReceivers; ++i)
                        if (not rdma_receiver_retired[i])
                            min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);  // 统计未 retired receiver。
                    // 中文：只有所有未 retired receiver 都消费超过一整个发送窗口，才安全归还 RDMA ring 空间。
                    if (min_head != std::numeric_limits<int>::max() and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and
                        // 中文：每个 lane 负责一个 dst_rdma_rank 的 head 释放。
                        lane_id < kNumRDMARanks) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                        // 中文：释放量等于本轮新最小 head 与上次释放 head 的差。
                                                        min_head - last_rdma_head,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                        // 中文：combine 使用 channel_id+num_channels，避免和 payload put 的 QP 选择冲突。
                                                        channel_id + num_channels,
                                                        dst_rdma_rank == rdma_rank);  // 原子增加 RDMA head，释放 sender/forwarder 的 ring 槽位。
                        last_rdma_head = min_head;  // 更新已释放 RDMA head。
                    }  // 中文：结束 RDMA head 释放分支。
                } else {
                    // Find minimum head for NVL ranks
                    #pragma unroll
                    // 中文：NVL head 释放要对每个 RDMA 分片分别计算最小值。
                    // 中文：逐 RDMA 分片统计所有 forwarder 对当前 NVL rank 的最小消费 head。
                    for (int i = 0; i < kNumRDMARanks; ++i) {
                        int min_head = std::numeric_limits<int>::max();  // 第 i 个 RDMA 分片下所有 forwarder 对当前 NVL rank 的最小 head。
                        #pragma unroll
                        for (int j = 0; j < num_warps_per_rdma_rank; ++j)
                            if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                                min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);  // 统计未 retired forwarder。
                        if (min_head != std::numeric_limits<int>::max() and min_head > last_nvl_head[i] and lane_id < LEGACY_NUM_MAX_NVL_PEERS)
                            st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head);  // 写 NVL head，释放源 NVL sender 槽位。
                    }
                }

                // Nanosleep and let other warps work
                __nanosleep(LEGACY_NUM_WAIT_NANOSECONDS);  // 避免 coordinator 忙等抢占过多执行资源。
            }  // 中文：coordinator 本轮 head 释放尝试结束。
        }  // 中文：结束 combine coordinator 角色。
    }  // 中文：结束 combine device kernel。
}

void combine(cudaDataType_t type,  // CUDA dtype，目前只支持 BF16。
             void* combined_x,  // 输出 combine 后 token 特征。
             float* combined_topk_weights,  // 输出 combine 后 topk 权重。
             const bool* is_combined_token_in_rank,  // combine 路由表，当前 kernel 参数保留但实际依赖 head。
             const void* x,  // 输入 expert 输出 token 特征。
             const float* topk_weights,  // 输入 topk 权重。
             const void* bias_0,  // 可选 bias0。
             const void* bias_1,  // 可选 bias1。
             const int* combined_rdma_head,  // combine RDMA head。
             const int* combined_nvl_head,  // combine NVL head。
             const void* src_meta,  // dispatch 保存的 SourceMeta。
             const int* rdma_channel_prefix_matrix,  // RDMA channel prefix。
             const int* rdma_rank_prefix_sum,  // RDMA rank prefix。
             const int* gbl_channel_prefix_matrix,  // 全局 channel prefix。
             int num_tokens,  // 输入 token 数。
             int num_combined_tokens,  // 输出 token 数。
             int hidden,  // hidden 维度。
             int num_topk,  // top-k 路数。
             void* rdma_buffer_ptr,  // RDMA symmetric buffer。
             int num_max_rdma_chunked_send_tokens,  // RDMA send chunk 上限。
             int num_max_rdma_chunked_recv_tokens,  // RDMA recv ring 容量。
             void** buffer_ptrs,  // NVL peer buffer 指针表。
             int num_max_nvl_chunked_send_tokens,  // NVL send chunk 上限。
             int num_max_nvl_chunked_recv_tokens,  // NVL recv ring 容量。
             int rank,  // 当前 rank。
             int num_ranks,  // 全局 rank 数。
             cudaStream_t stream,  // CUDA 发射流。
             int num_channels,  // channel 数。
             bool low_latency_mode) {  // 是否 low-latency PE 编号。
    (void)is_combined_token_in_rank;  // 当前实现通过 combined_*_head 表达路由，保留参数匹配 API。
    // internode combine 把 expert 输出按 src_meta 反向送回源 rank。
    // combined_x 形状 [num_combined_tokens, hidden]；如果 combined_topk_weights 非空，
    // 形状为 [num_combined_tokens, num_topk]。bias_0/bias_1 和 x 同 hidden 维，
    // 在发送前或接收规约时加到 BF16 激活上。
    constexpr int kNumCombineForwarderWarps = 24;  // combine forwarder 工作 warp 上限。
    constexpr int kNumTMABytesPerSenderWarp = 16384;  // NVL sender 每 warp TMA buffer 大小。
    constexpr int kNumTMABytesPerForwarderWarp = 9248;  // forwarder 每 warp TMA buffer 大小。
    constexpr int smem_size =
        std::max(kNumTMABytesPerSenderWarp * LEGACY_NUM_MAX_NVL_PEERS, kNumTMABytesPerForwarderWarp * kNumCombineForwarderWarps);  // 两类 block 共享同一动态 shared 配置。

#define COMBINE_LAUNCH_CASE(num_rdma_ranks)                                           \
    {                                                                                 \
        /* combine 的模板实参固定了 RDMA fan-in、数据类型和每类 warp 的 TMA staging 大小。 */ \
        auto combine_func = low_latency_mode ? combine<true,                          /* low-latency 模式。 */ \
                                                       num_rdma_ranks,                /* RDMA rank fan-in。 */ \
                                                       nv_bfloat16,                   /* payload dtype。 */ \
                                                       kNumCombineForwarderWarps,     /* forwarder warp 数。 */ \
                                                       kNumTMABytesPerSenderWarp,     /* sender TMA bytes。 */ \
                                                       kNumTMABytesPerForwarderWarp>  /* forwarder TMA bytes。 */ \
                                             : combine<false,                         /* high-throughput 模式。 */ \
                                                       num_rdma_ranks,                /* RDMA rank fan-in。 */ \
                                                       nv_bfloat16,                   /* payload dtype。 */ \
                                                       kNumCombineForwarderWarps,     /* forwarder warp 数。 */ \
                                                       kNumTMABytesPerSenderWarp,     /* sender TMA bytes。 */ \
                                                       kNumTMABytesPerForwarderWarp>; /* forwarder TMA bytes。 */ \
        /* combine 内部大量使用 TMA，因此 launch 前必须为该模板实例打开足够动态 shared memory。 */ \
        SET_SHARED_MEMORY_FOR_TMA(combine_func);                                      \
        LAUNCH_KERNEL(&cfg,                                                           \
                      combine_func,                                                   \
                      /* combined_x: [num_combined_tokens,hidden]，最终写回给调用方的聚合特征。 */ \
                      reinterpret_cast<int4*>(combined_x),                            \
                      combined_topk_weights,                                          \
                      /* is_combined_token_in_rank: [num_tokens,num_ranks]，记录某 token 是否从对应 rank 收到贡献。 */ \
                      is_combined_token_in_rank,                                      \
                      /* x/topk_weights 是本地 expert 计算后的 partial result，combine 会按 src_meta 回传。 */ \
                      reinterpret_cast<const int4*>(x),                               \
                      topk_weights,                                                   \
                      /* bias_0/bias_1 可选，用 int4 视图和 hidden payload 同粒度读取。 */ \
                      reinterpret_cast<const int4*>(bias_0),                          \
                      reinterpret_cast<const int4*>(bias_1),                          \
                      /* combined_*_head 是 ring head 表，负数编码表示对应槽位已经 ready。 */ \
                      combined_rdma_head,                                             \
                      combined_nvl_head,                                              \
                      /* src_meta 来自 dispatch 输出，保存 token 原始 rank/RDMA/NVL/source 索引。 */ \
                      reinterpret_cast<const SourceMeta*>(src_meta),                  \
                      /* prefix matrix 描述 combine 反向回传时每个 channel/rank 的 token 区间。 */ \
                      rdma_channel_prefix_matrix,                                     \
                      rdma_rank_prefix_sum,                                           \
                      gbl_channel_prefix_matrix,                                      \
                      /* num_tokens 是源 token 总数，num_combined_tokens 是当前 rank 实际需要输出的 token 数。 */ \
                      num_tokens,                                                     \
                      num_combined_tokens,                                            \
                      /* hidden/num_topk 决定 payload 和权重数组的线性 stride。 */      \
                      hidden,                                                         \
                      num_topk,                                                       \
                      /* 后续参数是 RDMA/NVL symmetric buffer 与 ring 容量，和 dispatch 的 staging 协议对称。 */ \
                      rdma_buffer_ptr,                                                \
                      num_max_rdma_chunked_send_tokens,                               \
                      num_max_rdma_chunked_recv_tokens,                               \
                      /* buffer_ptrs 是节点内所有 NVL peer 的 symmetric buffer 基址表。 */ \
                      buffer_ptrs,                                                    \
                      num_max_nvl_chunked_send_tokens,                                \
                      num_max_nvl_chunked_recv_tokens,                                \
                      /* rank/num_ranks 用来拆分 rdma_rank 与 nvl_rank，并选择对应 peer buffer。 */ \
                      rank,                                                           \
                      num_ranks);                                                     \
    }                                                                                 \
    /* 中文：每个 dtype/rank case 启动后退出 switch。 */                              \
    break

    int num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;  // RDMA rank 数。
    auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);  // 每个 RDMA rank 分配的 forwarder warp 数。
    int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;  // 实际 forwarder warp 数。
    EP_HOST_ASSERT(num_rdma_ranks <= kNumCombineForwarderWarps);  // RDMA rank 不能超过 forwarder warp 上限。
    EP_HOST_ASSERT(num_forwarder_warps > LEGACY_NUM_MAX_NVL_PEERS and num_forwarder_warps % num_rdma_ranks == 0);  // forwarder warp 必须覆盖 NVL peers 并能按 RDMA 均分。
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);  // NVL recv ring 按 RDMA source 均分。
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks >
                   std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));  // 每 RDMA 分片容量要大于 RDMA/NVL chunk。
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks - num_warps_per_forwarder >= num_max_nvl_chunked_send_tokens);  // 预留 forwarder pipeline 空间，避免死锁。
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder);  // RDMA chunk 至少容纳每个 forwarder warp 一个 token。
    EP_HOST_ASSERT(type == CUDA_R_16BF);  // legacy internode combine 当前只实例化 BF16。

    SETUP_LAUNCH_CONFIG(num_channels * 2, (num_forwarder_warps + 1) * 32, stream);  // 每 channel 一对 blocks，每 block forwarder_warps + coordinator。
    SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);  // 根据 RDMA rank 数选择模板实例。
#undef COMBINE_LAUNCH_CASE
}

}  // namespace internode  // internode kernel 命名空间结束。

}  // namespace legacy  // legacy 命名空间结束。

}  // namespace deep_ep  // DeepEP 顶层命名空间结束。
