#pragma once  // combine reduce epilogue device 实现由 JIT include，防止重复定义。

#include <deep_ep/common/compiled.cuh>  // nv_bfloat16、topk_idx_t 等基础类型。
#include <deep_ep/common/ptx.cuh>       // TMA、warp vote/match/gather 等 PTX helper。
#include <deep_ep/common/layout.cuh>    // TokenLayout/BufferLayout 描述通信 buffer 布局。

#include <deep_ep/impls/combine_utils.cuh>  // combine_reduce、slot layout 和向量化 trait。


namespace deep_ep::elastic {  // elastic combine epilogue kernel 所在命名空间。
// Elastic combine reduce epilogue device kernel 命名空间。

template <bool kUseExpandedLayout, bool kAllowMultipleReduction,  // 是否使用 expanded layout，以及是否允许同 rank 多路归约。
          int kNumSMs, int kNumWarps,  // kernel 使用的 SM/block 数和每 block warp 数。
          // TODO: merge these two variables into one (ensure the whole file does not contain "scaleup")
          int kNumScaleoutRanks, int kNumScaleupRanks,  // 二维拓扑维度：跨节点和节点内 rank 数。
          int kHidden,  // hidden 维度，combined_x 的第二维。
          int kNumMaxTokensPerRank,  // recv_buffer 每个 slot 的最大 token 容量。
          int kNumExperts, int kNumTopk,  // 全局 expert 数和 top-k 路数。
          int kNumThreads = kNumWarps * 32,  // 每 block 线程数，由 warp 数推导。
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),  // 单 token hidden payload 字节数。
          int kNumRanks = kNumScaleoutRanks == 1 ? kNumScaleupRanks : kNumScaleoutRanks,  // 非 hybrid 用 scaleup rank，hybrid 用 scaleout rank。
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),  // recv_buffer 第一维是否按 rank。
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>()>  // recv_buffer 第一维槽位数。
__global__ void __launch_bounds__(kNumThreads, 1)
combine_reduce_epilogue_impl(nv_bfloat16* combined_x,  // 输出 [num_combined_tokens, hidden]。
                             float* combined_topk_weights,  // 可选输出 [num_combined_tokens, topk]。
                             topk_idx_t* combined_topk_idx,  // 输入 [num_combined_tokens, topk]，用于确定每路 expert/rank。
                             void* recv_buffer,  // 主 combine kernel 写入的 partial activation buffer。
                             void* bias_0, void* bias_1,  // 可选 bias，均与 combined_x 同形状。
                             const int num_combined_tokens,  // 本 rank 需要输出的 token 数。
                             const int scaleout_rank_idx, const int scaleup_rank_idx) {  // 当前 rank 在二维拓扑中的坐标。
    // reduce epilogue 是 combine 的最后一步：前一个 combine kernel 已经把来自不同
    // rank/top-k slot 的 partial activation 写入 recv_buffer，本 kernel 再对同一原始
    // token 的贡献求和，输出 combined_x [num_combined_tokens, hidden]。如果需要返回
    // combine 后权重，则 combined_topk_weights 为 [num_combined_tokens, num_topk]。
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    // hybrid 下一个 scaleout rank 管辖的 expert 数。
    constexpr int kNumExpertsPerRank = kNumExperts / (kNumScaleupRanks * kNumScaleoutRanks);
    // 每个物理/逻辑 rank 持有的 expert 数。
    EP_STATIC_ASSERT(kNumExperts % (kNumScaleupRanks * kNumScaleoutRanks) == 0, "Invalid number of experts or ranks");
    // expert 必须能被总 rank 数整除。

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);  // 当前 block id，逻辑上对应一个 SM 工作槽。
    // 当前 block/SM 编号。
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();  // block 内 warp id 和 lane id。
    // warp_idx/lane_idx 用于 hidden 维向量化规约。
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;   // NOTES: Here we prioritize distributing tasks to different SMs to ensure that the last wave is evenly concentrated on each SM.
    // 任务分配先按 warp 再按 SM 交错，让尾部 token 更均匀分布到不同 SM。

    // recv_buffer 的第一维可能是 top-k slot，也可能是 rank slot：
    // - allow_multiple_reduction=false 时通常按 top-k 布局
    // - allow_multiple_reduction=true 且 rank 数不超过 topk 时按 rank 布局更省空间
    // kUseRankLayout/kNumTokensInLayout 由 combine_utils.cuh 在编译期决定。
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];  // 每 block 动态 shared，用于 warp staging。
    // shared memory 用作每个 warp 的输出 staging buffer，随后 TMA store 到 combined_x。
    const auto comm_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);  // partial buffer 的单 token layout。
    // recv_buffer 中的 token layout：hidden payload + topk_weights，无 topk_idx。
    const auto comm_buffer = layout::BufferLayout<false>(  // recv_buffer 不带 per-token mbarrier。
        comm_token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, recv_buffer);
    // recv_buffer 逻辑布局：[slot(rank 或 topk), token, token_layout]。

    // Store buffers
    const auto output_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);  // 输出 token 只存 hidden。
    // combined_x 输出只包含 hidden payload。
    const auto output_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, combined_x);  // combined_x 的一维 token buffer。
    // 输出 buffer 逻辑布局：[1, num_combined_tokens, hidden]。
    const auto tma_buffer = layout::BufferLayout<false>(output_token_layout, kNumWarps, 1, smem)  // shared 中按 warp 切 staging。
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    // 每个 warp 在 shared 中独占一个 token staging 槽。

    // bias_0/bias_1 可选，用于两路 bias 融合。它们和 combined_x 同形状
    // [num_combined_tokens, hidden]，combine_reduce 会在 FP32 累加路径里先加 bias。
    const auto bias_0_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_0);  // bias_0 的 token layout。
    // bias_0 与输出同布局；bias_0 为空时后续不会访问。
    const auto bias_1_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_1);  // bias_1 的 token layout。
    // bias_1 与输出同布局；bias_1 为空时后续不会访问。

    // Will block until the main combine kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();  // 等待主 combine kernel 通过 PDL 建立的依赖完成。
    // PDL 同步点：等待主 combine kernel 的依赖数据可见，同时允许 kernel 间重叠启动。

    // Read from buffers and do reduction
    for (int token_idx = global_warp_idx; token_idx < num_combined_tokens; token_idx += kNumWarps * kNumSMs) {  // warp 级网格步进遍历 token。
        // 每个 warp 负责若干 token 的 hidden 规约。
        // Preprocess all indices
        int stored_dst_rank_idx = -1, stored_dst_expert_idx = -1;  // 当前 lane 解析出的目标 rank 和 expert。
        // 当前 lane 对应 top-k expert 的 rank/expert 解析结果。
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        // 一个 warp lane 数最多容纳 32 个 top-k。
        if (lane_idx < kNumTopk) {  // 只有 top-k 范围内的 lane 读取 expert id。
            // lane0..kNumTopk-1 各自读取一个 top-k expert。
            stored_dst_expert_idx = static_cast<int>(combined_topk_idx[token_idx * kNumTopk + lane_idx]);  // 当前 token/lane 的 expert id。
            // combined_topk_idx 逻辑形状 [num_combined_tokens, kNumTopk]。
            stored_dst_rank_idx = stored_dst_expert_idx >= 0 ?  // 无效 expert=-1 时 rank 也保持 -1。
                stored_dst_expert_idx / (kNumScaleoutRanks == 1 ? kNumExpertsPerRank : kNumExpertsPerScaleout) : -1;
            // 单层按每 rank expert 数映射 rank；hybrid 按每 scaleout expert 数映射 scaleout rank。
        }
        __syncwarp();
        // 确保所有 lane 的 top-k rank/expert 已准备好，后续 warp vote/match 使用。

        // 把有效 top-k lane 压缩成 slot 列表，后续 reduce 只遍历有效 slot。deduplicate
        // 是必要的：同一 rank 可能对应多个 expert lane，但通信 buffer 中只需要读取
        // 一份 rank 级 partial，重复读取会导致结果被加多次。
        const auto [should_deduplicate, deduplicate_key] = [&]() -> std::pair<bool, int> {  // 计算是否去重以及去重 key。
            // 根据布局模式决定是否需要对 top-k lane 去重。
            if constexpr (kUseExpandedLayout and not kAllowMultipleReduction) {  // expanded+单路 reduction 时 slot 已经互不重叠。
                // Activations are never reduced before
                return {false, 0};
                // expanded layout 且不允许多路规约时，每个 top-k lane 独立，不需要去重。
            } else if constexpr (kNumScaleoutRanks != 1 and not kUseExpandedLayout and not kAllowMultipleReduction) {  // hybrid 非 expanded 需要按真实 rank 去重。
                // Hybrid mode without expanded layout and multiple reduction. Should deduplicate on a per-rank basis
                return {true, stored_dst_expert_idx >= 0 ? stored_dst_expert_idx / kNumExpertsPerRank : -1};
                // hybrid 非 expanded 路径按真实 rank 去重，避免同 rank 多 expert 重复读取。
            } else {  // 默认按布局 rank/scaleout rank 去重。
                // Should deduplicate on a per-rank (for non-hybrid mode) or a per-scale-rank (for hybrid mode) basis
                return {true, stored_dst_rank_idx};
                // 其它模式按 stored_dst_rank_idx 去重。
            }
        }();
        auto reduce_valid_mask = should_deduplicate ?  // 生成需要读取 partial 的有效 lane bitmask。
            ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0) :
            ptx::gather(stored_dst_rank_idx >= 0);
        // reduce_valid_mask 的 bit 表示哪些 lane 对应需要读取的唯一 slot。
        int topk_slot_idx[kNumTokensInLayout];  // 压缩后的 slot 数组，未使用项填 -1。
        // 压缩后的 slot 列表，长度由 layout 第一维决定。
        compute_topk_slots(
            topk_slot_idx, reduce_valid_mask,  // 输入输出：slot 数组和有效 lane mask。
            [=](const int& idx) {  // 把有效 lane id 映射到通信 buffer slot。
                // idx 是有效 lane id；rank layout 时需要从该 lane 取 rank id，否则直接用 top-k lane。
                return kUseRankLayout ? ptx::exchange(stored_dst_rank_idx, idx) : idx;
            }
        );
        // topk_slot_idx 后续传给 combine_reduce 读取 recv_buffer 对应 slot。

        // 每个 warp 沿 hidden 维分片做向量化规约。kHiddenVec 表示 hidden 维的向量个数，
        // kUnrollFactor 选择能被 warp 整除的展开因子，例如 hidden=7168、vec=int4 时
        // 每个 token 会被拆成 7168*2/16 个 int4 元素。
        using combine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;  // 根据 hidden 对齐选择向量类型。
        // 根据架构/hidden 对齐选择 int4 或 longlong4_t。
        constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);  // hidden 向量元素数。
        // hidden 向量化后的元素个数。
        constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();  // 最大展开因子，最多展开 4。
        // 选择最大可整除展开因子，提升 ILP。
        combine_reduce<kHiddenVec, kUnrollFactor, kNumTokensInLayout>(
            lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),  // 当前 warp lane、slot 列表和 shared 输出。
            /* Get source base */ [=](const int& slot_idx) {  // 根据 slot 返回 partial activation 地址。
                // 根据 slot_idx 返回 recv_buffer 中当前 token 的源 hidden 起点。
                return static_cast<combine_vec_t*>(
                    comm_buffer.get_rank_buffer(slot_idx).get_token_buffer(token_idx).get_base_ptr());
            },
            /* Wait buffer release */ [=]() {  // 在 combine_reduce 首次写 shared 前执行。
                // combine_reduce 首次写 shared staging 前调用，确保此前 TMA store 已释放。
                ptx::tma_store_wait();
                __syncwarp();
            },
            /* Bias 0 */ bias_0 == nullptr ?  // bias_0 为空则传 nullptr，combine_reduce 会跳过。
                nullptr : static_cast<combine_vec_t*>(bias_0_buffer.get_token_buffer(token_idx).get_base_ptr()),
            /* Bias 1 */ bias_1 == nullptr ?  // bias_1 为空则传 nullptr。
                nullptr : static_cast<combine_vec_t*>(bias_1_buffer.get_token_buffer(token_idx).get_base_ptr())
        );
        // combine_reduce 完成后，tma_buffer 中已有该 token 的最终 BF16 hidden。
        ptx::tma_store_fence();  // 发起 TMA store 前对 shared 写入做 fence。
        // 确保 shared staging 写入对 TMA store 可见。
        __syncwarp();
        // warp 内同步后由 elected lane 发起 TMA store。

        // Issue TMA copy
        if (ptx::elect_one_sync()) {  // 每个 warp 只选一个 lane 发起 TMA。
            // 一个 lane 发起 shared->global TMA store。
            ptx::tma_store_1d(output_buffer.get_token_buffer(token_idx).get_base_ptr(),  // 全局输出 token 地址。
                              tma_buffer.get_base_ptr(), kNumHiddenBytes);  // shared staging 地址和拷贝字节数。
            // 把当前 token hidden 写到 combined_x[token_idx, :]。
            ptx::tma_store_commit();  // 提交当前 warp 发起的 TMA store。
            // 提交 TMA store。
        }
        __syncwarp();
        // 确保本 warp 不会提前覆盖 shared staging。

        // top-k weights 不做加和，只从对应 master rank/slot 拷回。若某 lane 无效，
        // 写 0，Python 侧可直接把它当作 padding lane。
        if (combined_topk_weights != nullptr) {  // 只有调用方需要权重输出时才写权重。
            // 调用方需要返回 top-k weights 时才执行。
            const auto master_lane_idx = ptx::get_master_lane_idx(ptx::match(stored_dst_rank_idx));  // 同 rank lane 中选 master。
            // 对同一 rank 的多个 lane 选一个 master lane，避免重复读取 rank layout slot。
            if (lane_idx < kNumTopk) {  // 每个 top-k lane 写一个权重。
                // lane0..topk-1 写对应 top-k 权重。
                float value = 0;  // 默认无效 expert 的输出权重为 0。
                // 无效 lane 默认权重 0。
                if (stored_dst_rank_idx >= 0) {  // 只有有效 expert 才读取通信 buffer。
                    // 有效 top-k lane 从通信 buffer 读取权重。
                    const auto dst_ptr = comm_buffer  // 当前 top-k lane 对应的权重地址。
                        .get_rank_buffer(kUseRankLayout ? stored_dst_rank_idx : master_lane_idx)
                        .get_token_buffer(token_idx).get_topk_weights_ptr() + lane_idx;
                    // rank layout 按 rank 取 slot；top-k layout 用 master lane 对应 slot。
                    value = *dst_ptr;  // 从 partial buffer 取回权重。
                    // 读取该 lane 的 top-k weight。
                }
                combined_topk_weights[token_idx * kNumTopk + lane_idx] = value;  // 写回 [token,lane] 权重。
                // 写回输出权重矩阵。
            }
            __syncwarp();
            // 确保权重写入完成后处理下一个 token。
        }
    }
}

}  // deep_ep::elastic
// Elastic combine reduce epilogue device kernel 命名空间结束。
