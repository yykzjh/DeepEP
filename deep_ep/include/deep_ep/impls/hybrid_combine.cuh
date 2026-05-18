#pragma once  // hybrid combine device 实现会被 JIT 多次 include，使用 pragma once 防止重复定义。

// comm.cuh 提供混合拓扑 barrier、QP 选择和超时轮询。
#include <deep_ep/common/comm.cuh>
// layout.cuh 提供 token/buffer/workspace 的结构化视图。
#include <deep_ep/common/layout.cuh>
// math.cuh 提供 constexpr_ceil_div、pack/unpack、指针偏移等工具。
#include <deep_ep/common/math.cuh>
// ptx.cuh 封装 TMA、mbarrier、warp 操作、system-scope load/store。
#include <deep_ep/common/ptx.cuh>
// combine_utils.cuh 提供 reduce layout 选择、top-k slot 整理、向量化 reduce。
#include <deep_ep/impls/combine_utils.cuh>

// elastic hybrid combine 入口。
namespace deep_ep::elastic {  // elastic hybrid combine kernel 所在命名空间。

// kUseExpandedLayout 表示 dispatch 输出是否按 expert 展开。
template <bool kUseExpandedLayout, bool kAllowMultipleReduction,  // expanded dispatch layout 与多重 reduction 开关。
          // cooperative launch 的 block 数。
          int kNumSMs,
          // scaleup warps 负责节点内收集，forward warps 负责跨 scaleout 回传。
          int kNumScaleupWarps, int kNumForwardWarps,
          // 两层拓扑 rank 数。
          int kNumScaleoutRanks, int kNumScaleupRanks,
          // hidden 是 BF16 元素数。
          int kHidden,
          // 每个最终 rank token 容量。
          int kNumMaxTokensPerRank,
          // MoE expert/top-k 形状。
          int kNumExperts, int kNumTopk,
          // 通信 QP 与超时设置。
          int kNumQPs, int64_t kNumTimeoutCycles,
          // 一个 lane 覆盖若干 scaleup rank。
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),  // 一个 lane 需要覆盖的 scaleup rank 槽数。
          // scaleup tail 周期性更新间隔。
          int kNumScaleupUpdateInterval = 3,  // scaleup tail release 的批量刷新间隔。
          // channel 数与 forward warp 数绑定。
          int kNumChannelsPerSM = kNumForwardWarps,  // 每个 SM 的 forward warp 数等于 channel 数。
          int kNumChannels = kNumChannelsPerSM * kNumSMs,  // 全局 channel 总数。
          // 每个 channel 的最大 token 数。
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),  // 单 channel token 容量上界。
          // 全局 rank 数。
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,  // 二维拓扑展平后的总 rank 数。
          // block 内总 warp/thread 数。
          int kNumWarps = kNumScaleupWarps + kNumForwardWarps,  // 一个 block 中 scaleup+forward warp 总数。
          int kNumThreads = kNumWarps * 32,  // 一个 block 的线程数。
          // BF16 hidden 字节数。
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),  // 单 token hidden payload 字节数。
          // scaleout/scaleup reduce buffer 是否按 rank layout 组织。
          bool kUseScaleoutRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),  // scaleout reduce buffer 是否按 rank slot。
          bool kUseScaleupRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>(),  // scaleup reduce buffer 是否按 rank slot。
          // 两层 reduce layout 的第一维大小。
          int kNumTokensInScaleoutLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),  // scaleout buffer 第一维槽数。
          int kNumTokensInScaleupLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>()>  // scaleup buffer 第一维槽数。
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_combine_impl(nv_bfloat16* x,  // 输入/输出 expert activation，形状依赖 expanded 与 combine 阶段。
                    // topk_weights 非 expanded 路径可写入 reduce token，epilogue 后续使用。
                    float* topk_weights,  // 非 expanded 路径的 top-k 权重输入。
                    // src_metadata 来自 dispatch epilogue，描述原 token 与 expanded 行号。
                    int* src_metadata,  // 每个 reduced token 的源 token 与 top-k/source rank 元数据。
                    // psum_num_recv_tokens_per_scaleup_rank 提供无 CPU sync 时的 token 总数。
                    int* psum_num_recv_tokens_per_scaleup_rank,  // scaleup rank 接收 token 前缀和。
                    // token_metadata_at_forward 来自 hybrid_dispatch forward 分支，用于 replay 跨 scaleout slot。
                    int* token_metadata_at_forward,  // forward 阶段 replay 使用的 token 元数据。
                    // channel_linked_list 由 dispatch_copy_epilogue 写入，scaleup warps 沿它取 token。
                    int* channel_linked_list,  // dispatch epilogue 生成的 per-channel token 链表。
                    // device-side NCCL communicator/window。
                    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,  // device NCCL communicator 和 symmetric window。
                    // buffer/workspace 分别是 payload 区和控制面区。
                    void* buffer, void* workspace,  // payload buffer 和 WorkspaceLayout 控制区。
                    // 当前 GPU 的两层拓扑坐标。
                    const int scaleout_rank_idx, const int scaleup_rank_idx,  // 当前 rank 的二维坐标。
                    // 本地 expert 输出 token 数，可能为哨兵。
                    int num_reduced_tokens) {  // 本 rank reduced token 数，可能是哨兵。
    // hybrid combine 是 hybrid dispatch 的逆过程：
    // 1. scaleup warps 沿 dispatch epilogue 生成的 channel_linked_list，把本节点专家
    //    输出写入 scaleup_buffer；
    // 2. forward warps 把跨 scaleout 的结果经 RDMA 转发；
    // 3. combine_reduce_epilogue 再从返回的 reduce_buffer 写出 combined_x。
    // src_metadata 形状为 [num_reduced_tokens, 2 + kNumTopk]。
    // sm_idx/thread_idx 是 block 与线程坐标。
    const auto sm_idx = static_cast<int>(blockIdx.x);  // 当前 block/SM id。
    const auto thread_idx = static_cast<int>(threadIdx.x);  // 当前 block 内线程 id。
    // warp_idx 决定当前 warp 是 scaleup 角色还是 forward 角色。
    const auto warp_idx = ptx::get_warp_idx();  // 当前线程所属 warp。
    // lane_idx 用于 top-k lane、rank lane 和 warp 内广播。
    const auto lane_idx = ptx::get_lane_idx();  // warp 内 lane id。
    // expanded 且不允许多重 reduction 时需要逐 top-k 槽发送。
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;  // expanded 且单路 reduction 时按 top-k 槽发送。

    // Combine vector type selection
    // combine_vec_t 选择适合 hidden bytes 的向量类型，便于 combine_reduce 向量化累加。
    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;  // hidden 规约使用的向量类型。
    // kHiddenVec 是 hidden 向量化后的元素数量。
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);  // hidden 向量元素数量。

    // Workspaces：包含 channel tail、barrier、scaleout signal 等控制面字段。
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);  // 结构化解释 workspace 控制区。

    // 无 CPU sync 时调用方会传最大哨兵值；真实 token 数从 GPU prefix-sum 最后一项读取。
    // 哨兵值表示调用方没有 CPU 同步真实 reduced token 数。
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)  // 哨兵值表示真实数量需要从 GPU prefix 读取。
        // scale-up rank prefix 的最后一项就是真实 token 数。
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumScaleupRanks - 1);

    // Token layouts：combine token 只包含 hidden 和 top-k weights，不包含 SF。
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);  // combine token layout：hidden + topk weight。

    // TMA buffers：每个 warp 一个 shared token staging buffer。
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];  // 动态 shared，给每个 warp 一个 TMA staging token。
    // tma_buffer 逻辑形状 [kNumWarps, 1]。
    const auto tma_buffer = layout::BufferLayout<true>(  // 每个 token slot 带 mbarrier，便于 TMA producer/consumer 同步。
        token_layout, kNumWarps, 1, smem).get_rank_buffer(warp_idx).get_token_buffer(0);

    // buffer 顺序与 launch_combine 的返回逻辑对应：
    // scaleup_buffer 是本节点内部归并区，scaleout_recv_buffer 是 RDMA 收包区，
    // scaleout_send_buffer 是跨 scaleout 发送区。reduce epilogue 会跳过 scaleup_buffer
    // 后读取最终 reduce buffer。
    // scaleup_buffer 是节点内 scale-up 归并区。
    auto scaleup_buffer = layout::BufferLayout<false>(  // 节点内 scaleup reduce buffer。
        token_layout, kNumTokensInScaleupLayout, kNumScaleoutRanks * kNumMaxTokensPerRank,
        buffer);
    // scaleout_recv_buffer 是跨 scaleout 回传后的接收区。
    auto scaleout_recv_buffer = layout::BufferLayout<false>(  // 跨 scaleout RDMA 回传接收 buffer。
        token_layout, kNumTokensInScaleoutLayout, kNumMaxTokensPerRank,
        scaleup_buffer.get_buffer_end_ptr());
    // scaleout_send_buffer 是跨 scaleout 发送 staging 区；不允许多重 reduction 时按 top-k 拆分。
    auto scaleout_send_buffer = layout::BufferLayout<false>(  // 跨 scaleout RDMA 发送 staging buffer。
        token_layout, kAllowMultipleReduction ? 1 : kNumTopk, kNumChannels * (kNumScaleoutRanks * kNumMaxTokensPerChannel),
        scaleout_recv_buffer.get_buffer_end_ptr());

    // Init TMA for scale-up and forward warps。
    ptx::arrival_phase phase = 0;  // 当前 warp 的 mbarrier arrival phase，TMA 等待时交替使用。
    // mbarrier 在当前 warp 的 tma_buffer 内。
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();  // 当前 warp staging token 的 mbarrier 地址。
    // 每个 warp 只需一个 lane 初始化 mbarrier。
    if (ptx::elect_one_sync())  // 每个 warp 只选一个 lane 初始化 barrier。
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    // 确保初始化完成。
    __syncwarp();

    // NCCL Gin handle
    // Each warp is a channel
    const auto [qp_idx, sharing_mode] =  // 根据 SM/channel 选择 NCCL Gin QP 和共享模式。
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM>(sm_idx, warp_idx % kNumChannelsPerSM);
    // Gin 同时服务 LSA(scale-up) 与 Rail(scale-out) 通信。
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);  // 当前 warp/channel 使用的 Gin 访问句柄。

    // Global parallel barriers for scale-out subteam and scale-up subteam
    // NOTES: this barrier needs a grid sync, as there are channel scale-up tail cleaning before
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,  // hybrid combine 启动前同步 scaleup/scaleout 子团队。
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // Adjust register count at certain cases
    // TODO: support more cases, or try to make channel count more aligned
    // register 分配调优：非 expanded 且 channel 数为 4/8 时给 forward warps 留更多寄存器。
    const bool kAdjustRegisters = (kNumChannelsPerSM == 4 or kNumChannelsPerSM == 8) and not kUseExpandedLayout;  // 是否启用 warpgroup 寄存器再分配。
    // scaleup warps 保留较少寄存器。
    constexpr int kNumRegistersForScaleupWarps = 40;  // scaleup warp 低寄存器预算。
    // forward warps 负责 reduce/通信，分配剩余寄存器。
    constexpr int kNumRegistersForForwardWarps = 256 - kNumRegistersForScaleupWarps;  // forward warp 获得剩余寄存器。

    // 前 kNumScaleupWarps 负责节点内 scale-up 归并，后 kNumForwardWarps 负责
    // scale-out 转发。两类 warp 共用 QP/channel 编号，便于和 dispatch 的链表对应。
    if (warp_idx < kNumScaleupWarps) {  // 前半 warp 进入节点内 scaleup reduce 分支。
        // channel_idx 与 dispatch forward 写链表时的 channel 对齐。
        const auto channel_idx = sm_idx * kNumChannelsPerSM + warp_idx;  // 当前 scaleup warp 对应的全局 channel。

        // Adjust registers
        // scaleup warp 只做链表遍历和局部 reduce，可释放寄存器给 forward warp。
        if constexpr (kAdjustRegisters)  // 需要调寄存器时只影响 scaleup warp 分支。
            ptx::warpgroup_reg_dealloc<kNumRegistersForScaleupWarps>();

        // Shift into the right buffer if using rank layout
        // rank layout 下 scaleup_buffer 第一维已经是本 scaleup rank。
        if constexpr (kUseScaleupRankLayout)  // rank layout 下先切到当前 scaleup rank 槽。
            scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Expanding mode must not be backward
        // expanded layout 中 x 已经展开，不再需要 topk_weights。
        if constexpr (kUseExpandedLayout)  // expanded 反向路径不应再携带 topk_weights。
            EP_DEVICE_ASSERT(topk_weights == nullptr);

        // Tail issuer
        // `st.release.sys` is pretty slow, so do it by an interval
        // update_counter 控制每隔 kNumScaleupUpdateInterval 次才通知一次 tail。
        int update_counter = 0;  // tail 发布节流计数器。
        // stored_num_tokens_sent[j] 是当前 lane/j 负责的 scaleup peer 已发送 token 数。
        int stored_num_tokens_sent[kNumScaleupRanksPerLane] = {};  // 本 lane 覆盖的每个 scaleup peer 已发送数量。
        // stored_old_num_tokens_sent[j] 是上次通知给 peer 的 tail。
        int stored_old_num_tokens_sent[kNumScaleupRanksPerLane] = {};  // 上次发布给 peer 的 tail。
        // tail_ptr 指向当前 channel/本 scaleup rank 的 tail 数组。
        const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);  // 当前 channel 的 scaleup tail 起点。
        // update_tails 将每个目标 scaleup rank 的 tail 发布给 forward warps。
        const auto update_tails = [&](const bool& finish = false) {  // 发布当前 lane 覆盖的 scaleup tail。
            // 每调用一次递增计数，finish 会强制 flush。
            ++ update_counter;  // 累计一次潜在发布机会。
            // 达到间隔或 finish 时发布 tail。
            if (finish or update_counter == kNumScaleupUpdateInterval) {  // 达到间隔或结束时真正写 tail。
                // Wait all TMA stores to finish
                // 发布 tail 前必须保证对应 scaleup_buffer payload 已经写完。
                ptx::tma_store_wait();
                // 所有 lane 等待 store 完成。
                __syncwarp();

                // Issue
                #pragma unroll
                // 每个 lane/j 发布一个 scaleup peer 的 tail。
                for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {  // 遍历该 lane 覆盖的 scaleup rank 槽。
                    // j 是目标 scaleup rank。
                    if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {  // j 是真实 scaleup rank。
                        // NOTES: save some traffic with `stored_old_num_tokens_sent`
                        // Also, we cannot rewrite a finished slot, if the peer is going to clean it
                        // tail 未变化时跳过写，减少 sys-scope release 流量。
                        if (stored_num_tokens_sent[i] != stored_old_num_tokens_sent[i])  // tail 有变化才写系统内存。
                            // 写到目标 scaleup peer 可见的 tail 槽，release 保证 payload 先于 tail 可见。
                            ptx::st_release_sys(gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j), stored_num_tokens_sent[i]);
                        // 记录已发布 tail。
                        stored_old_num_tokens_sent[i] = stored_num_tokens_sent[i];
                    }
                }
                // 重新开始计数。
                update_counter = 0;
            }
            // 保证 warp 内所有 lane 的 tail 发布状态一致。
            __syncwarp();
        };

        // channel_linked_list 形状为
        // [kNumChannels, kNumScaleoutRanks*kNumMaxTokensPerChannel + 1, kNumScaleupRanks]。
        // dispatch_copy_epilogue 在每条链尾写 -1；这里看到所有 rank 都为 -1 就退出。
        // dst_scaleup_rank_idx 用作 round-robin 起点，初值与 channel 绑定以打散访问。
        int dst_scaleup_rank_idx = channel_idx;  // round-robin 当前指针，初始按 channel 打散。
        // stored_ll_idx 是每个 scaleup peer 当前读取到的链表节点位置。
        int stored_ll_idx[kNumScaleupRanksPerLane] = {}, stored_token_idx[kNumScaleupRanksPerLane] = {};  // 每个 lane/j 的链表位置和 token id。
        #pragma unroll
        // 初始化为 -1，避免第一次循环前误认为有有效 token。
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)  // 初始化本 lane 覆盖的所有 rank 槽。
            stored_token_idx[i] = -1;
        // 不断沿 channel_linked_list 消费 token，直到所有 scaleup peer 的链都结束。
        while (true) {  // 持续消费当前 channel 下所有 scaleup peer 的链表。
            // Load token indices in the list
            #pragma unroll
            // 每个 lane/j 读取一个 scaleup peer 当前链表节点中的 token_idx。
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {  // 每个 lane 读取它负责的一组 scaleup rank。
                // j 是目标 scaleup rank。
                const auto j = i * 32 + lane_idx;  // j 是真实 scaleup rank 编号。
                // 越界 rank 直接写 -1；有效 rank 从扁平链表数组读取 token_idx。
                stored_token_idx[i] = i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks ?  // 越界 rank 返回 -1。
                    __ldg(channel_linked_list +
                          channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumScaleupRanks +
                          // 中文：stored_ll_idx 是链表节点编号，乘 kNumScaleupRanks 后定位该节点的 scaleup 列。
                          stored_ll_idx[i] * kNumScaleupRanks + j) : -1;
            }
            // 等待读取完成。
            __syncwarp();

            // Check whether all ranks are finished：所有链当前节点都 <0 表示本 channel 完成。
            bool exited = true;  // 若所有 lane/j 都无 token，则当前 channel 完成。
            #pragma unroll
            // ptx::all 确认 warp 中每个 lane 对应 rank 都完成。
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)  // 检查每个 lane 覆盖的 rank 槽。
                exited &= ptx::all(stored_token_idx[i] < 0);
            // 全部结束则跳出链表消费循环。
            if (exited)  // 所有链表到达 -1 哨兵。
                break;

            // Process tokens for all ranks together using bitmask to skip inactive ranks
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up ranks for 64-bit mask");
            // rank 数 <=32 用 32-bit mask，否则用 64-bit mask。
            using mask_t = std::conditional_t<(kNumScaleupRanks <= 32), uint32_t, uint64_t>;  // 根据 rank 数选择 mask 宽度。
            // wip_mask 标出当前还有 token 可处理的 scaleup peer。
            mask_t wip_mask = 0;  // 当前批次仍需处理的 scaleup rank bitset。
            #pragma unroll
            // 每个 j 组 32 lane 贡献一段 mask。
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)  // 把每组 32 lane 的有效性压入 mask。
                wip_mask |= static_cast<mask_t>(ptx::gather(stored_token_idx[j] >= 0)) << (j * 32);
            // 当前批次中逐个处理有 token 的 scaleup peer。
            while (wip_mask) {  // 当前批次至少还有一个 scaleup rank 有 token。
                // Find next active rank after `dst_scaleup_rank_idx` (round-robin)
                // round-robin 从上次 rank 的下一个开始，避免总是处理低编号 rank。
                const auto start = (dst_scaleup_rank_idx + 1) % kNumScaleupRanks;  // 下一轮公平选择起点。
                // hi_mask 是 start 之后仍有工作的 rank 集合。
                const auto hi_mask = (wip_mask >> start) << start;  // 只保留 start 及之后的待处理 rank。
                // 优先选 start 之后的第一个，否则回绕选第一个。
                dst_scaleup_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);  // 找到下一个 rank bit。
                // 从待处理 mask 中移除这个 rank。
                wip_mask ^= static_cast<mask_t>(1) << dst_scaleup_rank_idx;  // 清掉已选择 rank。

                // Exchange token index from the owning lane using static partition iteration
                // token_idx 从负责 dst_scaleup_rank_idx 的 lane 广播给全 warp。
                int token_idx = -1;  // 即将处理的 reduced token index。
                #pragma unroll
                // 遍历每个 32-rank 分组，找到目标 rank 所在组。
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {  // 找到目标 rank 所属的 32-lane 分组。
                    // src_lane_idx 是目标 rank 在第 j 组中的 lane id。
                    const auto src_lane_idx = dst_scaleup_rank_idx - j * 32;  // 目标 rank 在当前分组内的 lane。
                    // 只有负责 lane 把 stored_token_idx 写入 token_idx。
                    token_idx = src_lane_idx == lane_idx ? stored_token_idx[j] : token_idx;  // 只有 owner lane 写入 token_idx。
                }
                // 从目标 lane 广播 token_idx。
                token_idx = ptx::exchange(token_idx, dst_scaleup_rank_idx % 32);  // owner lane 广播 token_idx。

                // src_metadata 第 0 列是原 token 全局 id，第 1 列包含 slot/topk lane；
                // 根据 kUseScaleupRankLayout 决定写到 rank layout 还是 topk layout。
                // kMetadataStride 与 dispatch_copy_epilogue 的 recv_src_metadata 行宽一致。
                constexpr int kMetadataStride = 2 + kNumTopk;  // src_metadata 每行字段数。
                // src_global_token_idx 编码了原全局 rank 和 token 局部下标。
                const auto src_global_token_idx = __ldg(src_metadata + token_idx * kMetadataStride + 0);  // 原 token 全局编号。
                // src_token_idx 是原 rank 内 token 局部下标。
                const auto src_token_idx = src_global_token_idx % kNumMaxTokensPerRank;  // 原 rank 内 token index。
                // src_scaleout_rank_idx 是原 token 来自哪个 scaleout rank。
                const auto src_scaleout_rank_idx = src_global_token_idx / (kNumMaxTokensPerRank * kNumScaleupRanks);  // 原 token 所在 scaleout rank。
                // token_buffer 是当前 scaleup warp 要写入的目标 scaleup_buffer 位置。
                auto token_buffer = [&]() {  // 根据 layout 模式计算写入 scaleup_buffer 的 token slot。
                    // rank layout：metadata[1]/kNumTopk 是 rank layout 下的 slot。
                    if constexpr (kUseScaleupRankLayout) {
                        const auto src_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) / kNumTopk;  // rank layout 槽号。
                        return scaleup_buffer.get_token_buffer(src_slot_idx);
                    } else {
                        // top-k layout：metadata[1]%kNumTopk 选择第一维，token 维是 scaleout_rank*max+token。
                        const auto master_topk_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) % kNumTopk;  // top-k layout 第一维。
                        return scaleup_buffer
                            .get_rank_buffer(master_topk_idx)
                            // 中文：top-k layout 用 master_topk_idx 作为 rank_buffer 维，token 维合并 scaleout_rank 和 token。
                            .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                    }
                }();
                // 将本地 token_buffer base pointer 转换成目标 scaleup peer 可访问地址。
                token_buffer.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(token_buffer.get_base_ptr(), dst_scaleup_rank_idx));  // 转成目标 scaleup peer 地址。

                // Some checks
                // hidden 必须能按 int4/warp 向量化整除，满足 combine_reduce 和 TMA 向量读写。
                EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");

                // Read source indices for expand mode
                // expanded 模式 metadata[2..] 保存每个 top-k lane 对应的 x 行号。
                int stored_topk_slot_idx = -1;  // expanded 模式下当前 lane 对应的 x 行号。
                if constexpr (kUseExpandedLayout) {  // 只有 expanded layout 才需要读取每路 top-k 的 expanded 行。
                    // 前 kNumTopk lane 读取对应 expanded slot。
                    if (lane_idx < kNumTopk)  // top-k lane 读取对应 metadata。
                        stored_topk_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + (2 + lane_idx));
                    // 后续 gather/exchange 需要所有 lane 完成读取。
                    __syncwarp();
                }

                // 3 cases:
                //  - no-expand, expand + no-reduce
                // 中文：第一类可以直接搬运一个 x 行，不需要把多个 top-k 槽先规约。
                //  - expand + reduce
                //  - expand + send all
                // reduce_valid_mask 标出 expanded 中有效的 x 行。
                auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);  // expanded 中有效 x 行的 lane mask。
                // 普通非 expanded 或 expanded 单有效槽时，不做本地 reduce。
                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);  // 是否可以直接搬运。
                if (no_local_reduce) {  // 无需本地 reduce，直接把 x 行搬到 shared。
                    // 默认读取 x[token_idx]。
                    int token_idx_in_tensor = token_idx;  // 默认 x 行号等于 reduced token id。
                    // expanded 单槽时读取实际 expanded x 行号。
                    if constexpr (kUseExpandedLayout)  // expanded 单槽时使用 metadata 中真实 x 行号。
                        token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

                    // Directly load
                    // 一个 lane 从 x 加载 hidden 到 shared，后面统一 TMA store 到 scaleup_buffer。
                    if (ptx::elect_one_sync()) {  // 一个 lane 发起 TMA load。
                        // load_ptr 指向本地 expert 输出行。
                        const auto load_ptr =  // 本地 x 中要加载的 hidden 行地址。
                            math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                        // 复用 tma_buffer 前等上一轮 store 释放。
                        ptx::tma_store_wait();
                        // 发起 hidden 的 TMA load。
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    }
                    // 等待 TMA load 发射。
                    __syncwarp();
                } else if constexpr (kAllowMultipleReduction) {  // 多个 expanded 槽需要本地 reduce。
                    // Do local reduction
                    // Sort valid top-k indices to front
                    // 中文：把有效 expanded 行号压到数组前面，combine_reduce 只遍历有效槽。
                    // topk_slot_idx 保存需要 reduce 的 x 行号。
                    int topk_slot_idx[kNumTopk];  // 需要读取的 expanded x 行号列表。
                    // 将有效 lane 的 stored_topk_slot_idx 压缩到数组前部。
                    compute_topk_slots(
                        topk_slot_idx, reduce_valid_mask,  // 输出 slot 列表和输入有效 mask。
                        // 从 lane idx 广播 expanded x 行号。
                        [=](const int& idx) {  // 从有效 lane 广播 expanded 行号。
                            return ptx::exchange(stored_topk_slot_idx, idx);
                        }
                    );

                    // Reduce into shared memory
                    // kUnrollFactor 控制 combine_reduce 的循环展开程度。
                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();  // 本地 reduce 最多展开 4。
                    // combine_reduce 把多个 x 行累加到 shared tma_buffer。
                    combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumRanks)>(
                        lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),  // lane、输入槽和 shared 输出。
                        /* Get source base */ [=](const int& slot_idx) {
                            // slot_idx 是 expanded x 行号，返回向量化指针。
                            // x 在 expanded layout 下形状可理解为 [expanded_topk_slots, hidden_bytes]。
                            return math::advance_ptr<combine_vec_t>(
                                x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                        },
                        /* Wait buffer release */ [=]() {  // 首次写 shared 前等待上一轮 store 完成。
                            // 写 shared reduce 结果前等待上一轮 TMA store 完成。
                            // tma_buffer 是复用的 shared staging 区，必须确认上一轮远端 store 已读走。
                            // 中文：hybrid combine 会在本地 reduce、scaleup、scaleout 多路径复用同一 shared staging。
                            ptx::tma_store_wait();
                            __syncwarp();
                        }  // 中文：等待回调结束，shared staging 可以安全写入新 reduce 结果。
                    );
                    // fence 确保普通 shared 写入对后续 TMA store 可见。
                    ptx::tma_store_fence();
                    // 等待 reduce 完成。
                    // 中文：reduce 结果已经在 tma_buffer 中，下一步会按目标 scaleup/scaleout 路径发送。
                    __syncwarp();
                } else {  // 中文：没有本地 reduce 时，直接进入 expanded send。
                    // No local reduction, send all data (expanded send)
                    #pragma unroll
                    // 不允许本地 reduce 时，逐 top-k 槽单独发送。
                    // 中文：这里的 expanded send 与普通 combine 的 expanded 路径语义一致。
                    // 中文：每个 top-k 槽会单独走 scaleup/scaleout 发送流程。
                    // 中文：expanded 模式让每个 top-k 贡献保留独立路由，避免提前累加造成精度差异。
                    for (int k = 0; k < kNumTopk; ++ k) {  // 遍历所有 top-k 槽逐个发送。
                        // 从 lane k 广播 expanded x 行号。
                        // stored_topk_slot_idx 的每个 lane 保存一路 top-k 对应的 expanded 行号。
                        int topk_slot_idx = ptx::exchange(stored_topk_slot_idx, k);  // lane k 的 expanded 行号。
                        // 无效槽跳过。
                        if (topk_slot_idx < 0)  // 该 top-k 槽无有效 expert。
                            continue;

                        if (ptx::elect_one_sync()) {  // 一个 lane 串行搬运当前 top-k 槽。
                            // Load
                            // expanded send 读取 topk_slot_idx；非 expanded 读取 token_idx。
                            // load_ptr 指向本 rank expert 计算出的 BF16 hidden 行，字节长度固定为 kNumHiddenBytes。
                            const auto load_ptr = math::advance_ptr(x, static_cast<int64_t>(kDoExpandedSend ? topk_slot_idx : token_idx) * kNumHiddenBytes);  // 当前要读取的 x 行。
                            // 等待上一次 store 完成后复用 tma_buffer。
                            ptx::tma_store_wait();
                            // 加载当前槽 hidden。
                            ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                            // 等待 TMA load 到达。
                            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                            // NOTES: We don't need to care about `topk_weights` since we are in expand mode

                            // Store
                            // 目标 buffer 第一维为 top-k 槽 k，token 维为源 scaleout/token。
                            const auto dst_token_buffer = scaleup_buffer  // 当前 top-k 槽写入的 scaleup 目标 token。
                                .get_rank_buffer(k)
                                .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                            // 直接写目标 scale-up peer 的 LSA symmetric pointer。
                            // 这里的 rank_buffer(k) 按 top-k 槽分片，接收侧再根据 metadata 取回对应槽。
                            ptx::tma_store_1d(
                                gin.get_sym_ptr<ncclTeamTagLsa>(dst_token_buffer.get_base_ptr(), dst_scaleup_rank_idx),
                                tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                            // 提交 TMA store。
                            ptx::tma_store_commit();
                        }
                        // 等待当前 top-k 槽发送请求发出。
                        __syncwarp();
                    }
                }

                // Write top-k weights
                // 非 expanded 路径需要把 top-k weight 随 token 写给 scaleup_buffer。
                if (not kUseExpandedLayout and topk_weights != nullptr and lane_idx < kNumTopk) {  // 非 expanded 才需要复制权重。
                    // 读取本地 expert 输出行对应的 top-k weight。
                    // topk_weights 形状为 [num_tokens,num_topk]，lane_idx 直接对应 top-k 列。
                    const float value = __ldg(topk_weights + (token_idx * kNumTopk + lane_idx));  // 当前 token/top-k 权重。
                    // 写入 shared token buffer 的 weight 区。
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = value;
                    // fence 让 leader lane 的 TMA store 能看到这些 weight。
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Issue TMA stores into remote scale-up buffer
                // NOTES: `kDoExpandedSend` mode has already issued
                // 非 expanded-send 路径只需把当前 token/reduce 结果写一次到 token_buffer。
                if (not kDoExpandedSend and ptx::elect_one_sync()) {  // 普通路径由一个 lane 发起完整 token store。
                    // Wait TMA arrival (only for non-reduced cases)
                    // no_local_reduce 分支前面只发起了 TMA load，这里等待它到达。
                    if (no_local_reduce) {  // 直接搬运路径需要等待 TMA load 到达。
                        // reduce 分支已经在 combine_reduce 中完成 shared 写入，所以只在 no_local_reduce 等 TMA load。
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    }

                    // Issue stores
                    // token_buffer 已经被映射到目标 scale-up peer 的 symmetric pointer。
                    // 非 expanded-send 一次写完整 token layout，包含 hidden 和可选 top-k weights。
                    ptx::tma_store_1d(
                        token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                        token_layout.get_num_bytes<false>());
                    // 提交 TMA store。
                    ptx::tma_store_commit();
                }
                #pragma unroll
                // 成功处理一个目标 scaleup rank 后，推进对应 tail 计数。
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)  // 推进被本轮处理的目标 rank tail。
                    stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                // 等待 store/计数更新完成。
                __syncwarp();
            }

            // Update the tails together
            // NOTES: TMA wait is inside
            // 周期性发布 scale-up tail，forward warps 会等待这些 tail 再读取 scaleup_buffer。
            update_tails();

            // Move linked list：当前 rank 消费了一个 token，则链表节点向后移动。
            #pragma unroll
            // 只有 stored_token_idx 有效的链推进。
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)  // 各 rank 链表指针按是否消费推进。
                stored_ll_idx[i] += (stored_token_idx[i] >= 0);
        }

        // Update for the unissued ones：退出前强制 flush tail，通知 forward 端没有更多 token。
        update_tails(true);
    } else {
        // forward_warp_idx 是跨 scaleout 回传角色内的 warp id。
        const auto forward_warp_idx = warp_idx - kNumScaleupWarps;  // forward 子团队内 warp id。
        // channel_idx 与 dispatch forward metadata/tail 一致。
        const auto channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;  // 当前 forward warp 对应 channel。

        // Adjust registers
        // forward warp 做 replay/reduce/RDMA，调回更多寄存器。
        if constexpr (kAdjustRegisters)  // forward warp 获取更多寄存器。
            ptx::warpgroup_reg_alloc<kNumRegistersForForwardWarps>();

        // Shift into the right buffer
        // 当前 channel 的 scaleout send buffer，token 维覆盖所有 scaleout rank 的 channel slot。
        scaleout_send_buffer = scaleout_send_buffer.get_channel_buffer<kNumScaleoutRanks * kNumMaxTokensPerChannel>(channel_idx);  // 切到当前 channel send staging。

        // Shape of `token_metadata_at_forward`: `[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        // forward metadata 行布局与 hybrid_dispatch forward 记录保持一致。
        // 末尾额外的 +1 行用于写 src_token_global_idx=-1 的终止哨兵，forward warp 看到它就停止 replay。
        // 每行前两列是 (src_token_global_idx, is_token_last_in_chunk)，后面两段 top-k 长度数组分别记录
        // src_scaleup_rank_idx 与 src_slot_idx。combine 必须按 dispatch 记录的顺序重放，才能和 tail 递增顺序一致。
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;  // forward metadata 每行字段数。
        // 偏移到当前 channel metadata 区。
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);  // 偏移到当前 channel metadata。

        // Overlap TMA stores and reduction
        // last_* 保存上一轮已经写到 send_buffer/local recv_buffer 的 token，下一轮 reduce 前再发 RDMA。
        int last_src_scaleout_rank_idx = -1;  // 上一轮 token 的目标 scaleout rank，-1 表示无待 flush 项。
        int last_is_token_last_in_chunk = 0;  // 上一轮 token 是否为 chunk 尾，用于 Gin flush 策略。
        void* last_recv_token_buffer_ptr = nullptr;  // 远端最终 recv buffer 地址。
        void* last_send_token_buffer_ptr = nullptr;  // 本地 send staging 地址。
        // flush_last_tma_and_issue_rdma 用于把上一轮 TMA store 与本轮 reduce 重叠。
        const auto flush_last_tma_and_issue_rdma = [&]() {  // 等上一轮 TMA 完成并按需发起 RDMA put。
            // last_src_scaleout_rank_idx<0 表示还没有可 flush 的上一轮。
            if (last_src_scaleout_rank_idx >= 0 and ptx::elect_one_sync()) {  // 有待 flush 项时由一个 lane 执行。
                // 等待上一轮写 send/local recv buffer 的 TMA store 完成。
                ptx::tma_store_wait();

                // Issue only if not local rank
                // 源 scaleout 就是本 rank 时已经写到 recv buffer，无需 Rail put。
                if (last_src_scaleout_rank_idx != scaleout_rank_idx) {  // 非本地 scaleout 才需要 Rail put。
                    // 跨 scaleout 时把 send buffer 写回源 scaleout 的 recv buffer。
                    gin.put<ncclTeamTagRail>(
                        last_recv_token_buffer_ptr,
                        last_send_token_buffer_ptr,
                        // 中文：token_layout 字节数覆盖 hidden payload 及当前 layout 所需的对齐 padding。
                        token_layout.get_num_bytes<false>(),
                        last_src_scaleout_rank_idx,
                        // chunk 最后一个 token 不 aggregate，让远端尽快看到完成边界。
                        // 中文：这个 flush 边界与远端 tail/signal 等待逻辑联动，不能全部聚合。
                        last_is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests
                    );
                }  // 中文：最后一个聚合请求已 flush 到 Gin，远端后续才能观察到完整 chunk。
            }  // 中文：结束 scaleout RDMA 请求批处理。
            // 等待 flush/RDMA 请求发起完成。
            __syncwarp();
        };

        // Replay the dispatch：按 hybrid_dispatch forward 记录的 metadata 重放 token 顺序。
        int stored_num_tokens_recv[kNumScaleupRanksPerLane] = {}, stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};  // 已消费计数和 tail 缓存。
        // stored_num_tokens_recv[j] 是本 forward warp 已从 scale-up rank(j*32+lane) 消费的 token 数。
        // stored_cached_scaleup_tail[j] 是对应 scale-up rank 已发布的 tail 缓存。
        // 这里“消费计数”按 scale-up rank 独立递增，而不是按 top-k lane 递增；
        // 因为同一个 scale-up rank 可能在一个 token 的多个 top-k lane 中重复出现，deduplicate 后只应消费一个 slot。
        // i 是当前 channel metadata 行号，直到读到 src_token_global_idx=-1。
        for (int i = 0; ; ++ i) {  // 顺序重放当前 channel 的 forward metadata 行。
            // 第 0 列是源 token 全局 id，-1 表示 metadata 结束。
            const auto src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);  // 源 token 全局 id 或 -1 哨兵。
            // 第 1 列标记当前 token 是否是 dispatch forward chunk 的最后一个。
            const auto is_token_last_in_chunk = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 1);  // 当前 token 是否 chunk 尾。
            // 源全局 rank。
            // src_token_global_idx 编码为 src_rank_idx*Tmax + src_token_idx。
            const auto src_rank_idx = src_token_global_idx / kNumMaxTokensPerRank;  // 源 token 所属全局 rank。
            // 源 scaleout rank。
            const auto src_scaleout_rank_idx = src_rank_idx / kNumScaleupRanks;  // 源 token 所属 scaleout rank。
            // 源 rank 内 token 局部下标。
            const auto src_token_idx = src_token_global_idx % kNumMaxTokensPerRank;  // 源 rank 内 token index。
            // metadata 第 2.. 保存各 top-k lane 对应的源 scale-up rank。
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?  // 当前 top-k lane 依赖的 scaleup rank。
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + lane_idx) : -1;
            // metadata 后半段保存各 top-k lane 对应的 scale-up slot。
            // slot 是 scaleup_buffer rank_buffer 中的 token 槽位，用来找到 expert partial result。
            auto stored_src_slot_idx = lane_idx < kNumTopk ?  // 当前 top-k lane 在 scaleup_buffer 中的 slot。
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + kNumTopk + lane_idx) : -1;
            // 结束标记：没有更多 metadata 行。
            if (src_token_global_idx < 0)  // metadata 结束哨兵。
                break;

            // Scaleup rank mask
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up peers");
            // rank 数 <=32 用 32-bit mask，否则用 64-bit mask。
            using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;  // scaleup rank bitmask 类型。
            // scaleup_mask 标出当前 metadata 行依赖哪些 scale-up rank 的 payload。
            const auto scaleup_mask = ptx::reduce_or(  // 汇总当前 token 依赖的所有 scaleup rank。
                stored_src_scaleup_rank_idx >= 0 ?
                (mask_t(1) << stored_src_scaleup_rank_idx) : mask_t(0));
            // stored_is_scaleup_rank_needed[j] 是当前 lane/j 对应 rank 是否需要等待。
            // 这里把 top-k lane 的多个 source rank 合并成 bit mask，避免逐 top-k 反复轮询 tail。
            bool stored_is_scaleup_rank_needed[kNumScaleupRanksPerLane];  // 当前 lane/j 对应 rank 是否需要等待。
            #pragma unroll
            // 从 scaleup_mask 中拆出每个 lane/j 的 rank 位。
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)  // 解码 scaleup_mask 到本 lane 覆盖的 rank。
                stored_is_scaleup_rank_needed[j] = (scaleup_mask >> (j * 32 + lane_idx)) & 1;

            // Wait all tails to arrive：只有 scaleup warp 发布 tail 后，对应 scaleup_buffer 数据才可读。
            // forward metadata 告诉我们当前 token 依赖哪些 scale-up rank；只有这些 rank 的
            // stored_cached_scaleup_tail 大于已消费计数时，才能安全读取 scaleup_buffer 中的对应 slot。
            // 这相当于一个 per-channel/per-rank 的生产者消费者队列：scaleup warp 生产，forward warp 消费。
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {  // 等待所需 scaleup tail 到达。
                // arrived 表示当前 token 依赖的所有 scale-up rank 都已有未消费 payload。
                bool arrived = true;  // 当前 token 依赖的所有 rank 是否已有可消费 payload。
                #pragma unroll
                // 不需要的 rank 直接视为 arrived；需要的 rank 必须 recv_count < cached_tail。
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)  // 检查每个需要的 scaleup rank。
                    arrived &= not stored_is_scaleup_rank_needed[j] or stored_num_tokens_recv[j] < stored_cached_scaleup_tail[j];
                // warp 内所有 lane 都确认 arrived 后才继续。
                if (ptx::all(arrived))  // warp 内所有 lane 都确认到达。
                    return true;

                // Reload cached
                #pragma unroll
                // 重新读取每个 scale-up rank 的 tail。
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {  // 重新读取每个 scaleup rank 的 tail。
                    // k 是目标 scale-up rank。
                    const auto k = j * 32 + lane_idx;  // 当前 lane 对应真实 scaleup rank。
                    // acquire_sys 与 scaleup warp 的 st_release_sys 配对。
                    // tail 表示 scaleup warp 已写入 scaleup_buffer 的 token 数。
                    stored_cached_scaleup_tail[j] = j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks ?  // 越界 rank tail 为 -1。
                        ptx::ld_acquire_sys(workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k)) : -1;
                }

                // Timeout
                // 超时时打印每个 lane/j 的 recv/tail/wait 状态。
                if (is_last_check) {
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                        // 中文：每个 lane 负责一个 j*32+lane_idx 的 scaleup rank，因此逐项打印。
                        printf("DeepEP combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, lane: %d, recv: %d, tail: %d (wait=%d)\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,  // 当前 rank/拓扑。
                               channel_idx, j * 32 + lane_idx,  // channel 和被检查的 scaleup lane。
                               stored_num_tokens_recv[j],  // 已消费 token 数。
                               stored_cached_scaleup_tail[j],  // 已观察到的 tail。
                               stored_is_scaleup_rank_needed[j]);  // 当前 metadata 是否需要该 rank。
                    }  // 中文：结束当前 scaleup lane 的 timeout 打印。
                }  // 中文：所有需要的 scaleup rank 尚未到齐，继续等待。
                return false;
            });

            // Increase received count：当前 metadata 行消费了需要的 scale-up rank 各一个 token。
            // 中文：stored_num_tokens_recv 是每个 scaleup peer 的链表游标，推进后读取下一节点。
            #pragma unroll
            // 每个 lane/j 的 recv counter 在对应 rank 被需要时加 1。
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)  // 消费当前 token 依赖的每个 scaleup rank。
                stored_num_tokens_recv[j] += static_cast<int>(stored_is_scaleup_rank_needed[j]);

            // kAllowMultipleReduction=false 时，不能在 forward warp 合并多个 scale-up payload。
            if constexpr (not kAllowMultipleReduction) {  // 禁止 forward 阶段合并，逐 slot 转发。
                // Cases where multiple reduction is disabled. We need to forward all data from scaleup peers to scaleout peers
                // TODO: Let scale-up warps directly put data into `send_buffer`?
                // scaleup_buffer 的 token 维编码为 src_scaleout_rank*max_tokens+src_token_idx。
                // 禁止多路 reduce 时，forward warp 只做转发，不在本地把多个 scale-up payload 合成一个。
                // 这个分支的输出仍保持 top-k layout，后续 reduce epilogue 会按 top-k 槽逐项处理，
                // 因而每个有效 top-k 槽都要搬一次，而不是合成单个 reduced token。
                const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;  // scaleup_buffer token 维索引。
                // expanded layout 逐 top-k lane 有效；非 expanded 需要对 scaleup rank 去重。
                auto topk_valid_mask = kUseExpandedLayout ?  // 需要转发的 top-k lane mask。
                    ptx::gather(stored_src_scaleup_rank_idx >= 0) :
                    ptx::gather(ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);  // Deduplicate w.r.t. scaleup rank index if expanded mode is disabled
                // 一个 leader lane 顺序搬每个有效 top-k 槽。
                if (ptx::elect_one_sync()) {  // 一个 lane 串行转发所有有效 top-k 槽。
                    #pragma unroll
                    // k 遍历 top-k layout 的第一维。
                    for (int k = 0; k < kNumTopk; ++ k) {  // 遍历 top-k 第一维。
                        // 当前 k 不在有效 mask 中则跳过。
                        if ((topk_valid_mask & (1u << k)) == 0u)  // 当前 top-k 槽无有效 payload。
                            continue;

                        // Issue TMA load, and wait
                        // 从 scaleup_buffer[k, src_slot_idx] 加载 scale-up peer 写好的 token。
                        ptx::tma_load_1d(
                            tma_buffer.get_base_ptr(), scaleup_buffer.get_rank_buffer(k).get_token_buffer(src_slot_idx).get_base_ptr(),
                            mbarrier_ptr, token_layout.get_num_bytes<false>());
                        // 等待 TMA load 完成。
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                        // Issue TMA store, and wait
                        // recv_buffer_ptr 是源 scaleout rank 最终要读取的 reduce buffer 地址。
                        const auto recv_buffer_ptr = scaleout_recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx).get_base_ptr();  // 源 scaleout 最终接收地址。
                        // 本地 scaleout 直接写 recv buffer；远端 scaleout 先写 send buffer。
                        const auto send_buffer_ptr = src_scaleout_rank_idx == scaleout_rank_idx ?  // 本地则直接写 recv，远端先写 send staging。
                            recv_buffer_ptr : scaleout_send_buffer.get_rank_buffer(k).get_token_buffer(i).get_base_ptr();
                        // 写入本地 send/recv buffer。
                        ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                        ptx::tma_store_commit();
                        // RDMA 源必须等待 TMA store 完成。
                        ptx::tma_store_wait();

                        // Issue IBGDA
                        // 从 mask 中清掉当前 k，用于判断是否最后一个 put。
                        topk_valid_mask ^= 1u << k;  // 清掉当前已转发 top-k 槽。
                        // 跨 scaleout 时通过 Rail put 写回源 scaleout。
                        if (src_scaleout_rank_idx != scaleout_rank_idx) {  // 远端源节点需要 Rail put。
                            gin.put<ncclTeamTagRail>(
                                recv_buffer_ptr,  // 远端源 scaleout rank 的最终 recv buffer 地址。
                                send_buffer_ptr,  // 本地 send buffer 地址；本地 bypass 时不会走到 put。
                                token_layout.get_num_bytes<false>(),  // 一个 combine token slot 的字节数。
                                src_scaleout_rank_idx,  // 目标 rail/scaleout rank，即原 token 所在节点。
                                // Gin 聚合请求用于减少 doorbell；当前 chunk 最后一次 put 必须不聚合，
                                // 让底层把前面积累的 WR 刷出去，远端 reduce epilogue 才能及时看到完整 chunk。
                                // 中文：topk_valid_mask==0 表示当前 token 所有有效 top-k 槽已经发送完。
                                topk_valid_mask == 0 and is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests  // 非最后一次 put 聚合请求，最后一次负责 flush。
                            );
                        }  // 中文：结束当前 top-k 槽的发送判断。
                    }  // 中文：结束当前 token 的 top-k 遍历。
                }  // 中文：结束顺序发送分支。
                // 等待顺序搬运结束。
                __syncwarp();
            } else {  // 允许 forward 阶段把多个 scaleup payload 规约成一个 token。
                // NOTES: we must do deduplicate and only add once from one rank
                // 允许多重 reduction 时，同一 scale-up rank 的重复 lane 只取一次。
                auto reduce_valid_mask = ptx::gather(  // 去重后的有效 scaleup rank lane mask。
                    ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);

                // Calculate the source buffer index：把 rank/slot 或 top-k/slot 映射为 scaleup_buffer 线性 token index。
                int stored_src_buffer_idx = 0;  // 当前 lane 对应的 scaleup_buffer 线性 token index。
                // rank layout 中，buffer index = scaleup_rank * tokens_per_rank + slot。
                if constexpr (kUseScaleupRankLayout) {  // rank layout：第一维是 scaleup rank。
                    stored_src_buffer_idx =
                        stored_src_scaleup_rank_idx * scaleup_buffer.num_max_tokens_per_rank + stored_src_slot_idx;
                } else {
                    // top-k layout 中，slot 维是源 scaleout/token，第一维由 lane/top-k 表示。
                    const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;  // top-k layout 的 token 维。
                    stored_src_buffer_idx = stored_src_slot_idx == -1 ? -1 :
                        lane_idx * scaleup_buffer.num_max_tokens_per_rank + src_slot_idx;
                }

                // Preprocess top-k indices
                // topk_slot_idx 保存需要 reduce 的 scaleup_buffer 线性 token index。
                int topk_slot_idx[kNumTokensInScaleupLayout];  // combine_reduce 输入 slot 列表。
                // 把有效 lane 的 source buffer index 压缩到数组前部。
                compute_topk_slots(
                    topk_slot_idx, reduce_valid_mask,  // 输出压缩 slot 与输入有效 mask。
                    // 从 lane idx 广播它保存的 source buffer index。
                    [=](const int& idx) {  // 从有效 lane 广播 scaleup_buffer 线性 index。
                        return ptx::exchange(stored_src_buffer_idx, idx);
                    }
                );

                // Do reduce
                // forward warp 可能寄存器更多，因此 kAdjustRegisters 时允许更大展开。
                constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, kAdjustRegisters ? 8 : 4>();  // 根据寄存器策略选择展开因子。
                // combine_reduce 从多个 scaleup_buffer token 累加到 shared tma_buffer。
                combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumScaleoutRanks)>(
                    lane_idx,  // 当前 warp lane，按 hidden vector 和 top-k 槽协作。
                    topk_slot_idx,  // 待规约的 scaleup_buffer 线性 slot 列表。
                    static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),  // shared memory 中的 reduce 输出缓冲。
                    /* Get source base */ [=](const int& slot_idx) {
                        // slot_idx 是 scaleup_buffer 的线性 token index。
                        return static_cast<combine_vec_t*>(scaleup_buffer.get_token_buffer(slot_idx, true).get_base_ptr());
                    },
                    /* Wait buffer release */ [=]() {
                        // reduce 开始前先 flush 上一轮 TMA/RDMA，实现通信计算重叠。
                        flush_last_tma_and_issue_rdma();
                    }
                );

                // Merge topk weights
                // NOTES: the slot indices must follow the master lane
                // 对同一 scale-up rank 的重复 lane，权重读取也跟随 master lane 的 source buffer index。
                // 这里先用 match 找到同一 source scale-up rank 的 lane group，再取 master lane 的 slot；
                // 否则重复 top-k lane 可能从不同 slot 读取 weight，导致 combine epilogue 权重和 hidden 不匹配。
                stored_src_buffer_idx = ptx::exchange(  // 将重复 lane 的 slot 对齐到 master lane。
                    stored_src_buffer_idx, ptx::get_master_lane_idx(ptx::match(stored_src_scaleup_rank_idx)));
                // 非 expanded 路径需要把每个 top-k lane 的 weight 一并写入 reduce token。
                if (not kUseExpandedLayout and stored_src_scaleup_rank_idx >= 0) {  // 非 expanded 需要复制 weight。
                    tma_buffer.get_topk_weights_ptr()[lane_idx] =
                        scaleup_buffer.get_token_buffer(stored_src_buffer_idx, true)
                                    // 中文：scaleup_buffer 中保存的是 forward 阶段带回的 topk weight 副本。
                                    .get_topk_weights_ptr()[lane_idx];
                }
                // 确保 weight 写入对后续 TMA store 可见。
                ptx::tma_store_fence();
                // Necessary to let the leader lane see the writes
                __syncwarp(); // Necessary to let the leader lane see the writes

                // Assign send and receive buffers
                // NOTES: as we only have 1 destination, we will use "send" as "recv" for local transfer
                // scaleout_recv_buffer_rank_idx 是最终 reduce buffer 第一维，依赖 rank/top-k layout。
                int scaleout_recv_buffer_rank_idx;  // scaleout_recv_buffer 第一维槽位。
                // rank layout 下第一维使用当前 scaleout rank。
                if constexpr (kUseScaleoutRankLayout) {  // rank layout 用当前 scaleout rank 作为槽。
                    scaleout_recv_buffer_rank_idx = scaleout_rank_idx;
                } else {
                    // top-k layout 下第一维使用有效 lane 中的 master top-k。
                    const int src_topk_idx = ptx::get_master_lane_idx(ptx::gather(stored_src_scaleup_rank_idx >= 0));  // top-k layout 的 master 槽。
                    scaleout_recv_buffer_rank_idx = src_topk_idx;
                }
                // recv_token_buffer 是源 scaleout 最终应该接收的 token 位置。
                const auto recv_token_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_recv_buffer_rank_idx).get_token_buffer(src_token_idx);  // 源 scaleout 最终接收 token。
                // 本地 scaleout 直接写 recv buffer；远端 scaleout 先写当前 channel send buffer。
                const auto send_token_buffer = src_scaleout_rank_idx == scaleout_rank_idx ?  // 本地 bypass 或远端 send staging。
                    recv_token_buffer :
                    scaleout_send_buffer.get_token_buffer(i);

                // Write into scale-out send buffer or local rank recv buffer bypass
                // leader lane 把 reduce 结果从 shared 写到 send/recv token buffer。
                if (ptx::elect_one_sync()) {  // 一个 lane 发起 reduce 结果 TMA store。
                    ptx::tma_store_1d(send_token_buffer.get_base_ptr(),  // 目标 send/recv token buffer。
                                      tma_buffer.get_base_ptr(),         // shared reduce 结果。
                                      token_layout.get_num_bytes<false>()); // 写完整 token slot。
                    // 提交 TMA store，下一轮 flush_last_tma_and_issue_rdma 会等待并 put。
                    ptx::tma_store_commit();
                }
                // 等待 store 请求发出。
                __syncwarp();

                // Record RDMA info to issue later：下一次 reduce 前 flush，实现 store/RDMA 与计算重叠。
                // last_* 四元组描述“刚刚 TMA store 到本地 send/recv buffer 的 token”。
                // 下一轮进入 combine_reduce 前调用 flush_last_tma_and_issue_rdma，把这次 store 等完并发起 Gin put。
                // 这样可以把本轮 reduce 计算和上一轮网络发送重叠起来。
                last_src_scaleout_rank_idx = src_scaleout_rank_idx;  // 记录下一次 flush 时要 put 到哪个 scaleout rank。
                last_is_token_last_in_chunk = is_token_last_in_chunk;  // 是否当前 chunk 最后 token，决定 Gin 聚合请求是否收尾。
                last_recv_token_buffer_ptr = recv_token_buffer.get_base_ptr();  // 远端最终接收地址。
                last_send_token_buffer_ptr = send_token_buffer.get_base_ptr();  // 本地 staging 地址。
            }
        }

        // Issue the last RDMA：循环结束后还有最后一轮 TMA/RDMA 未 flush。
        if constexpr (kAllowMultipleReduction)
            // flush_last_tma_and_issue_rdma 会先等待最后一次 TMA store，再提交最后一批 Gin put。
            flush_last_tma_and_issue_rdma();  // flush 最后一轮延迟发送的 RDMA。

        // Clean scaleup tails：forward 已消费完当前 channel 的所有 scale-up tail。
        #pragma unroll
        // 每个 lane/j 清理一个 scale-up rank 的 tail。
        for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {  // 清理当前 channel 的所有 scaleup tail。
            // k 是目标 scale-up rank。
            const auto k = j * 32 + lane_idx;  // 当前 lane 负责的 scaleup rank。
            // 越界 rank 跳过。
            // 清零 tail 是下一轮 channel 复用 workspace 的前置条件。
            if (j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks)  // 越界 rank 不写。
                *workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k) = 0;
        }
        // 等待 tail 清理完成。
        __syncwarp();

        // Update, wait and clean：通知所有 scaleout peer 本 channel combine 已完成，并等待它们也完成。
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid ranks");
        // 前 kNumScaleoutRanks 个 lane 分别负责一个 scaleout peer。
        if (lane_idx < kNumScaleoutRanks) {  // 每个有效 lane 负责一个 scaleout peer 完成信号。
            // Update remote tails
            // expected_signal=(finish=1, tail=0)，表示当前 channel 的 combine 完成。
            // pack2 的高/低字段与 dispatch/combine workspace signal 的解析方式保持一致。
            // 这里不再传 tail 进度，而是只用 finish=1 做“所有 scaleout peer 已完成本 channel”的收尾 barrier。
            // signal 清零后 workspace 可被下一轮 combine 复用。
            const auto expected_signal = math::pack2<int, int64_t>(1, 0);  // finish=1、tail=0 的打包信号。
            // 向每个 scaleout peer 的 signal 槽累加完成信号。
            // red_add_rel 使用 release 语义，确保本 channel 的 RDMA/TMA 写入先于完成信号可见。
            gin.red_add_rel<ncclTeamTagRail>(
                workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx),
                expected_signal, lane_idx);

            // Wait tail arrival
            // wait_ptr 是本 rank 等待 peer lane_idx 写入的完成信号。
            const auto wait_ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx);  // 本 rank 等待 peer lane_idx 的信号槽。
            // 等所有 peer 都回写 expected_signal。
            comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {  // 等待所有 scaleout peer 完成当前 channel。
                // acquire_sys 与 peer 的 release/atomic 写配对。
                const auto signal = ptx::ld_acquire_sys<int64_t>(wait_ptr);  // acquire 读取 peer release/atomic 信号。
                // 收到期望信号后清理槽位。
                if (signal == expected_signal) {  // 收到目标完成信号。
                    // Clean for next usages
                    // 当前 lane 只清理自己等待的 peer signal，避免多个 lane 写同一槽。
                    *wait_ptr = 0;
                    return true;
                }

                // 超时输出实际 signal 与期望 signal。
                if (is_last_check) {
                    printf("DeepEP combine (scale-out wait all) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, lane: %d, signal: %lld, expected: %lld\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,  // 当前 rank/拓扑。
                               // 中文：signal/expected_signal 可定位是哪一个 peer 的完成信号没有到达。
                               channel_idx, lane_idx,  // channel 和等待的 peer lane。
                           signal, expected_signal);  // 实际信号与期望信号。
                }  // 中文：当前 lane 的 signal 未达到 expected_signal，继续等待或超时。
                return false;
            });
        }
        // 等待所有 signal 清理完成。
        __syncwarp();
    }

    // No barrier at epilogue：后续 reduce epilogue 由 launch 顺序/外层同步保证读取时机。
    // 这里不再做昂贵的全局 barrier，避免每个 channel 结束时额外串行化。
}

}  // namespace deep_ep::elastic
