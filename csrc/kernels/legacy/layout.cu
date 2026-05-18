#include <deep_ep/common/exception.cuh>  // 使用 EP_DEVICE_ASSERT/EP_STATIC_ASSERT 等异常与断言工具。

#include "compiled.cuh"  // 引入 legacy 编译期常量与 topk_idx_t 类型。
#include "launch.cuh"  // 引入 SETUP_LAUNCH_CONFIG/LAUNCH_KERNEL/SWITCH_RANKS 发射宏。

namespace deep_ep::legacy {  // legacy MoE 通信实现所在命名空间。

namespace layout {  // layout 阶段负责从 topk_idx 生成 dispatch 所需路由统计。

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(const topk_idx_t* topk_idx,  // 输入 top-k expert id，形状 [num_tokens,num_topk]。
                                    int* num_tokens_per_rank,  // 输出每个全局 rank 接收 token 数，形状 [num_ranks]。
                                    int* num_tokens_per_rdma_rank,  // 可选输出每个 RDMA rank 接收 token 数，形状 [num_rdma_ranks]。
                                    int* num_tokens_per_expert,  // 输出每个全局 expert 接收 token 数，形状 [num_experts]。
                                    bool* is_token_in_rank,  // 输出 token-rank 布尔路由矩阵，形状 [num_tokens,num_ranks]。
                                    int num_tokens,  // token 总数。
                                    int num_topk,  // 每个 token 路由到的 expert 数。
                                    int num_ranks,  // 全局 rank 数。
                                    int num_experts) {  // 全局 expert 数。
    auto sm_id = static_cast<int>(blockIdx.x);  // 当前 block/SM 的逻辑编号。
    auto thread_id = static_cast<int>(threadIdx.x);  // 当前线程在线程块内的编号。

    // 前半部分 SM 统计 expert 粒度的 token 数。num_tokens_per_expert 形状为
    // [num_experts]，用于计算每个本地 expert 接收缓冲区需要多大；每个 SM 负责
    // kNumExpertsPerSM 个 expert，每个线程先私有计数再在 shared memory 中规约。
    __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];  // shared 计数表：[thread,本 SM 负责的 expert 槽]。
    int expert_begin_idx = sm_id * kNumExpertsPerSM, expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);  // 当前 SM 负责的 expert 半开区间。
    if (expert_begin_idx < expert_end_idx) {
        // Per-thread count
        #pragma unroll
        // 中文：每个线程先清空自己负责的 expert 私有计数槽，后续再 shared 规约。
        for (int i = 0; i < kNumExpertsPerSM; ++i)
            num_tokens_per_expert_per_thread[thread_id][i] = 0;  // 每个线程先清零自己的局部 expert 计数。
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;  // 指向第 i 个 token 的 top-k expert 列表。
            #pragma unroll
            for (int j = 0, expert_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);  // 读取第 j 个 top-k expert id。
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                    ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];  // 命中本 SM 负责的 expert 时计数加一。
            }
        }
        __syncthreads();  // 等所有线程完成局部计数后再规约。

        // Sum up
        EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");  // 保证可用前 kNumExpertsPerSM 个线程分别规约一个 expert。
        if (expert_begin_idx + thread_id < expert_end_idx) {
            int sum = 0;  // 当前 expert 的总 token 数累加器。
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_expert_per_thread[i][thread_id];  // 汇总所有线程对该 expert 的局部计数。
            num_tokens_per_expert[expert_begin_idx + thread_id] = sum;  // 写回全局 expert 计数。
        }
        return;  // expert 统计 SM 不参与 rank/RDMA 统计。
    }

    if (num_tokens_per_rdma_rank != nullptr)
        EP_DEVICE_ASSERT(num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0 and num_ranks > LEGACY_NUM_MAX_NVL_PEERS);  // RDMA rank 统计要求 rank 可按节点内 peer 数分组。

    // 后半部分 SM 统计 rank/RDMA-rank 粒度。topk_idx 形状为 [num_tokens, num_topk]；
    // is_token_in_rank 形状为 [num_tokens, num_ranks]，只记录“该 token 是否至少命中
    // 这个 rank 的一个 expert”，避免同 token 多个 top-k lane 在同 rank 内重复发送。
    constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / LEGACY_NUM_MAX_NVL_PEERS;  // 一个 SM 负责的 RDMA rank 数。
    __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];  // shared 计数表：[thread,本 SM 负责的 rank 槽]。
    __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];  // shared 计数表：[thread,本 SM 负责的 RDMA rank 槽]。
    auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;  // rank 统计 SM 的起始 block id。
    int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM, rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);  // 当前 SM 负责的 rank 区间。
    int rdma_rank_begin_idx = rank_begin_idx / LEGACY_NUM_MAX_NVL_PEERS, rdma_rank_end_idx = rank_end_idx / LEGACY_NUM_MAX_NVL_PEERS;  // 对应 RDMA rank 区间。
    if (rank_begin_idx < rank_end_idx) {
        const auto num_expert_per_rank = num_experts / num_ranks;  // 每个 rank 拥有的 expert 数。
        auto expert_begin = rank_begin_idx * num_expert_per_rank;  // 当前 rank 区间对应的 expert 起点。
        auto expert_end = rank_end_idx * num_expert_per_rank;  // 当前 rank 区间对应的 expert 终点。

        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;  // 清零本线程的 rank 计数槽。
        #pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++i)
            num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;  // 清零本线程的 RDMA rank 计数槽。
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;  // 指向第 i 个 token 的 top-k expert 列表。
            int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};  // 当前 token 是否命中对应 rank/RDMA rank。
            #pragma unroll
            for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);  // 读取 top-k expert id。
                if (expert_begin <= expert_idx and expert_idx < expert_end) {
                    // Count single rank
                    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;  // 将 expert id 映射到本 SM rank 槽。
                    is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / LEGACY_NUM_MAX_NVL_PEERS]++;  // 记录 token 命中 rank/RDMA rank。
                }
            }

            // 一个 token 可能有多个 top-k expert 位于同一 rank，但 dispatch 只需要
            // 向该 rank 发送一次完整 hidden 向量，所以这里使用布尔矩阵而非计数矩阵。
            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;  // 第 i 个 token 对应的 rank 布尔行。
            #pragma unroll
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);  // 写 token 是否发往该 rank。
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);  // rank 计数只按布尔命中加一。
            }

            #pragma unroll
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);  // RDMA rank 计数同样按 token 去重。
        }
        __syncthreads();  // 等所有线程完成 rank/RDMA rank 局部计数。

        // Sum up
        EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");  // 保证每个 rank 槽都有线程可负责规约。
        if (rank_begin_idx + thread_id < rank_end_idx) {
            int sum = 0;  // 当前 rank 的 token 总数。
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rank_per_thread[i][thread_id];  // 汇总所有线程的局部 rank 计数。
            num_tokens_per_rank[rank_begin_idx + thread_id] = sum;  // 写回 [num_ranks] 输出。
        }

        if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
            int sum = 0;  // 当前 RDMA rank 的 token 总数。
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];  // 汇总所有线程的局部 RDMA rank 计数。
            num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;  // 写回 [num_rdma_ranks] 输出。
        }
    }
}

void get_dispatch_layout(const topk_idx_t* topk_idx,  // 输入 top-k expert id。
                         int* num_tokens_per_rank,  // 输出 rank 计数。
                         int* num_tokens_per_rdma_rank,  // 可选输出 RDMA rank 计数。
                         int* num_tokens_per_expert,  // 输出 expert 计数。
                         bool* is_token_in_rank,  // 输出 token-rank 布尔矩阵。
                         int num_tokens,  // token 数。
                         int num_topk,  // top-k 路数。
                         int num_ranks,  // rank 数。
                         int num_experts,  // expert 数。
                         cudaStream_t stream) {  // CUDA 发射流。
    constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;  // 每 block 线程数及每 SM 负责的 expert/rank 数。
    // gridDim = expert 统计 SM 数 + rank 统计 SM 数。这样同一个 kernel 同时填充
    // expert/rank/RDMA-rank 三类 layout，避免多次扫描 topk_idx。
    int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) + (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;  // expert 统计 blocks + rank 统计 blocks。
    EP_STATIC_ASSERT(kNumRanksPerSM % LEGACY_NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");  // 每个 rank 统计 block 覆盖完整 RDMA 分组。

    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);  // 构造 CUDA kernel launch config。
    LAUNCH_KERNEL(&cfg,
                  (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),  // 发射模板 kernel。
                  topk_idx,  // 输入 topk_idx。
                  num_tokens_per_rank,  // 输出 rank 计数。
                  num_tokens_per_rdma_rank,  // 可选输出 RDMA rank 计数。
                  num_tokens_per_expert,  // 输出 expert 计数。
                  is_token_in_rank,  // 输出 token-rank 布尔矩阵。
                  num_tokens,  // token 数。
                  num_topk,  // top-k 路数。
                  num_ranks,  // rank 数。
                  num_experts);  // expert 数。
}

}  // namespace layout  // layout kernel 命名空间结束。

}  // namespace deep_ep::legacy  // legacy 命名空间结束。
