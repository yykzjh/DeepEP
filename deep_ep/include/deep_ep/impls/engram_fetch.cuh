#pragma once  // Engram fetch device 实现由 JIT include，防止重复定义。

#include <deep_ep/common/comm.cuh>      // NCCLGin handle 和通信辅助。
#include <deep_ep/common/compiled.cuh>  // nv_bfloat16 等基础类型。
#include <deep_ep/common/layout.cuh>    // 通用 layout helper。
#include <deep_ep/common/ptx.cuh>       // warp/lane 选举等 PTX helper。
// Engram fetch 是一个小型远端 embedding/缓存读取 kernel：按 indices 从各 rank storage 拉取条目。


namespace deep_ep::elastic {
// Elastic/V2 Engram device kernel 命名空间。

template <int kNumQPs,  // Gin/NCCL QP 数；grid.x 通常等于该值，一个 CTA 对应一个 QP。
          int kNumEntriesPerRank,  // 每个 rank 的 storage entry 数，indices 的低位在此范围内取模。
          int kHidden,  // 单个 entry 的 hidden 维度，元素类型固定为 nv_bfloat16。
          int kNumRanks,  // 参与远端 fetch 的 rank 总数。
          int kNumThreads,  // 每个 CTA 的线程数，必须能覆盖 kNumRanks 的 shared 初始化。
          int kNumWarps = kNumThreads / 32,  // 每个 QP/CTA 内用于提交 get/flush 的 warp 数。
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),  // 单条 entry 的通信字节数。
          typename team_t = ncclTeamTagWorld>  // Gin team tag，默认 world team，也可替换成子通信域。
// 模板参数把 QP 数、每 rank entry 数、hidden 维度和 rank 数固化，方便编译器完全展开地址计算。
__global__ void __launch_bounds__(kNumThreads, 1)
engram_fetch_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                  void* storage, void* fetched, int* indices,
                  // 中文：last_gin_requests 记录每个 QP/rank 最后一条 get 请求，wait kernel 会逐项等待。
                  ncclGinRequest_t* last_gin_requests,
                  const int num_tokens) {
    // blockIdx.x 是 QP 维度，每个 QP 里多个 warp 分摊 token。indices[i] 是全局
    // entry id，拆成 src_rank_idx 和 src_entry_idx 后从远端 storage 拉取 hidden 个 bf16。
    // fetched 的逻辑形状为 [num_tokens, kHidden]。
    // storage 在每个 rank 上逻辑形状为 [kNumEntriesPerRank, kHidden]，元素类型 nv_bfloat16。
    const auto qp_idx = static_cast<int>(blockIdx.x);
    // blockIdx.x 对应 QP 编号，一个 block 提交一组 Gin get 请求。
    const auto warp_idx = ptx::get_warp_idx();
    // 当前 block 内 warp 编号。
    const auto global_warp_idx = qp_idx * kNumWarps + warp_idx;
    // 跨所有 QP 的全局 warp 编号，用于按 token 做 grid-stride。
    // 例如 kNumQPs=4、kNumWarps=8 时，全局共有 32 个 warp 分摊 num_tokens。
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // block 内线程编号。

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
    // 每个 QP/block 使用自己的 Gin context，并以 CTA 资源共享方式提交请求。
    // RESOURCE_SHARING_CTA 表示同一个 CTA 内多个 lane/warp 共享该 Gin resource。

    __shared__ bool sent_to_rank[kNumRanks];
    // sent_to_rank[rank] 记录本 QP 是否向该 rank 发出过 get 请求。
    // 这个数组只按 rank 维保存布尔值，不按 token 维保存，目的是减少 shared memory 占用。
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Too many ranks");
    // 初始化 sent_to_rank 时一线程清一个 rank，要求线程数覆盖 rank 数。
    if (thread_idx < kNumRanks)
        // 前 kNumRanks 个线程负责清零 shared 标志。
        sent_to_rank[thread_idx] = false;
        // 每个 rank 一个 bool，初始化成本是 O(kNumRanks)。
    __syncthreads();
    // 确保所有 sent_to_rank 都初始化完成后再提交请求。

    // RDMA get 的源地址是远端 storage[src_entry_idx, :]，目标地址是本地
    // fetched[token_idx, :]。extra_options 可传 AggregateRequests，让多次 get 先聚合
    // 到 doorbell，后面 flush_async 再真正刷出。
    const auto issue_rdma_get = [=](const int& token_idx, const int& src_rank_idx, const int& src_entry_idx,
                                    const int& extra_options = 0) {
        // 根据 token 和全局 entry 拆分结果提交一条 Gin get。
        gin.get<team_t>(math::advance_ptr(storage, static_cast<int64_t>(src_entry_idx) * kNumHiddenBytes),
                        // 远端源地址：storage[src_entry_idx, :]。
                        math::advance_ptr(fetched, static_cast<int64_t>(token_idx) * kNumHiddenBytes),
                        // 本地目标地址：fetched[token_idx, :]。
                        kNumHiddenBytes, src_rank_idx, extra_options);
        // kNumHiddenBytes 是单个 entry 的字节数。
        // Gin get 是异步提交；真正完成由后续 flush_async + wait kernel 保证。
    };

    // 一个 warp 负责一个 token 的请求提交，hidden 维搬运由 GIN 设备侧执行；
    // sent_to_rank 记录本 QP 是否向某个 rank 发过请求，后续只对这些 rank flush。
    // TODO: deal with padded tokens
    if (ptx::elect_one_sync()) {
        // 每个 warp 只用一个 elected lane 提交 Gin get，避免重复发请求。
        // elect_one_sync 通常选出当前 warp 中一个活动 lane，因此下面循环逻辑是 warp 粒度。
        #pragma unroll 4
        for (int i = global_warp_idx; i < num_tokens; i += kNumQPs * kNumWarps) {
            // 所有 QP/warp 按 token 维做 stride 分工。
            const auto global_idx = __ldg(indices + i);
            // 读取全局 entry id。
            // global_idx 的编码约定是 src_rank_idx * kNumEntriesPerRank + src_entry_idx。
            const auto src_rank_idx = global_idx / kNumEntriesPerRank;
            // 全局 entry id 的高位映射到源 rank。
            const auto src_entry_idx = global_idx % kNumEntriesPerRank;
            // 低位映射到该 rank 内 entry 编号。

            // Delay ring DB
            issue_rdma_get(i, src_rank_idx, src_entry_idx, ncclGinOptFlagsAggregateRequests);
            // 聚合 doorbell，减少每个 get 都立即刷 doorbell 的开销。
            sent_to_rank[src_rank_idx] = true;
            // 标记该 rank 后续需要 flush。
            // 即使同一个 rank 有多个 token 请求，也只需要一个 rank 级 flush_async 句柄。
        }
    }
    __syncthreads();
    // 确保所有 warp 对 sent_to_rank 的写入可见。

    // 对每个发过请求的 peer 发 flush_async，并把 request 句柄写入
    // last_gin_requests[qp_idx, rank_idx]。没有请求的 rank 写全 0 句柄，wait kernel
    // 看到全 0 就跳过，避免等待一个不存在的 GIN request。
    if (ptx::elect_one_sync()) {
        // 每个 warp elected lane 负责若干 rank 的 flush/wait request 写入。
        // 这里再次使用 warp elected lane，是因为 flush_async 是 rank 粒度而非 lane 粒度操作。
        for (int i = warp_idx; i < kNumRanks; i += kNumWarps) {
            // 不同 warp 分摊 rank 维。
            const auto request_ptr = last_gin_requests + qp_idx * kNumRanks + i;
            // last_gin_requests 逻辑形状为 [kNumQPs, kNumRanks]。
            if (sent_to_rank[i]) {
                // 本 QP 对该 rank 有请求，必须 flush 并保存 request。
                gin.flush_async<team_t>(i, request_ptr);
                // flush_async 返回的 request 写到 request_ptr，wait kernel 后续等待它。
                // 对同一 rank 的聚合请求在 flush 后形成一个可等待的完成句柄。
            } else {
                // 没有发给该 rank 的请求，写零 request，wait kernel 会跳过。
                EP_STATIC_ASSERT(sizeof(ncclGinRequest_t) == sizeof(int4), "Invalid request size");
                // 用 int4 清零 request，要求 request 大小正好 16B。
                *reinterpret_cast<int4*>(request_ptr) = make_int4(0, 0, 0, 0);
                // 全 0 request 是“无需等待”的哨兵值。
                // wait kernel 只需要检查四个 32-bit 字段是否全 0，就能跳过该 rank。
            }  // 中文：结束当前 rank request 清零。
        }  // 中文：结束本 QP 内所有 rank request 的清理。
    }  // 中文：结束最后请求记录阶段。
}

template <int kNumRanks, int kNumThreads>
// 中文：wait kernel 单独拆出，是为了让 fetch 可以异步返回，调用方稍后再等待远端数据到齐。
// wait kernel 的模板参数只需要 rank 数和线程数，因为它不关心 hidden/entry layout。
__global__ void __launch_bounds__(kNumThreads, 1)
engram_fetch_wait_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                       ncclGinRequest_t* last_gin_requests) {
    // 中文：wait_impl 只消费 last_gin_requests，不访问 storage/fetched payload。
    const auto qp_idx = static_cast<int>(blockIdx.x);
    // blockIdx.x 对应 QP 维。
    // 中文：每个 QP 的 request 数组独立，wait kernel 逐 QP 并行等待。
    const auto thread_idx = static_cast<int>(threadIdx.x);
    // block 内线程编号。

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
    // 使用和 fetch kernel 相同 QP 的 Gin context 等待请求。
    // QP 必须匹配提交请求的 qp_idx，否则等待的队列上下文不对应。

    // wait kernel 通常紧跟在 fetch kernel 之后，由调用方决定是否需要显式等待。
    // 它遍历 last_gin_requests [num_qps, num_ranks]，等待每个非零请求完成。
    for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {
        // 线程按 rank 维分摊 request 等待。
        EP_STATIC_ASSERT(sizeof(ncclGinRequest_t) == sizeof(int4), "Invalid request size");
        // request 以 int4 读取，便于判断是否全 0。
        auto last_gin_req_int4 = __ldg(reinterpret_cast<int4*>(last_gin_requests + qp_idx * kNumRanks + i));
        // 读取 last_gin_requests[qp_idx, i]。
        if (last_gin_req_int4.x != 0 or last_gin_req_int4.y != 0 or
            last_gin_req_int4.z != 0 or last_gin_req_int4.w != 0) {
            // 非零 request 表示 fetch kernel 对该 rank 提交过请求。
            auto last_gin_req = *reinterpret_cast<ncclGinRequest_t*>(&last_gin_req_int4);
            // 把 int4 位模式还原成 ncclGinRequest_t。
            gin.wait(last_gin_req);
            // 等待该 QP 对该 rank 的最后一个请求完成；同 rank 之前请求也按序完成。
            // 因为 Gin 对同一 QP/rank 的请求按提交顺序完成，等待最后一个即可覆盖之前请求。
        }
    }
}

} // namespace deep_ep::elastic
// Elastic Engram device kernel 命名空间结束。
