#pragma once  // 单层 elastic combine device 实现由 JIT include，避免重复定义。

// device-side NCCL/Gin 类型，用于在 CUDA kernel 内发起 put 与解析 symmetric pointer。
#include <nccl_device.h>

// comm.cuh 提供 combine 前后的 GPU barrier 与 QP 分配。
#include <deep_ep/common/comm.cuh>
// layout.cuh 提供 TokenLayout/BufferLayout/WorkspaceLayout 等结构化 buffer 视图。
#include <deep_ep/common/layout.cuh>
// math.cuh 提供 ceil_div、advance_ptr 等整数与指针工具。
#include <deep_ep/common/math.cuh>
// ptx.cuh 提供 TMA、mbarrier、warp mask/reduction 等 PTX 封装。
#include <deep_ep/common/ptx.cuh>

// combine_utils.cuh 提供 reduce buffer 布局选择、top-k 槽整理、向量化 reduce 实现。
#include <deep_ep/impls/combine_utils.cuh>


// elastic combine 主 kernel 位于 deep_ep::elastic 命名空间。
namespace deep_ep::elastic {  // elastic 单层 combine kernel 命名空间。

// kIsScaleupNVLink 表示目标 rank 是否可通过 NVLink/LSA 直接写远端 buffer。
template <bool kIsScaleupNVLink,  // scaleup rank 间是否可用 NVLink/LSA symmetric pointer。
          // kUseExpandedLayout 表示 dispatch 输出已按 expert 展开；combine 需要按 metadata 找回原 token。
          bool kUseExpandedLayout, bool kAllowMultipleReduction,  // expanded layout 和多重 reduction 开关。
          // kNumSMs/kNumWarps 决定 cooperative grid 与每个 block 内并行 warp 数。
          int kNumSMs, int kNumWarps,  // cooperative grid SM 数和每 block warp 数。
          // kNumRanks 是 scale-up rank 数。
          int kNumRanks,  // scaleup rank 数。
          // kHidden 是 BF16 hidden 元素个数，kNumHiddenBytes 由它推导。
          int kHidden,  // hidden BF16 元素数。
          // 每个 rank 的 reduce/recv buffer token 上限。
          int kNumMaxTokensPerRank,  // 每 rank 最大 token 容量。
          // kNumExperts/kNumTopk 描述 MoE 路由形状。
          int kNumExperts, int kNumTopk,  // expert 总数和 top-k 路数。
          // QP 数与超时周期由通信配置传入。
          int kNumQPs, int64_t kNumTimeoutCycles,  // Gin QP 数和超时周期。
          // block 线程数为 kNumWarps*32。
          int kNumThreads = kNumWarps * 32,  // block 线程数。
          // combine kernel 当前只处理 BF16 x，因此 hidden bytes 固定为 kHidden*sizeof(nv_bfloat16)。
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),  // 单 token hidden 字节数。
          // 根据是否允许多重 reduction 决定 reduce buffer 的第一维是否按 rank 组织。
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),  // reduce buffer 第一维是否按 rank。
          // reduce buffer 第一维 token/rank layout 的大小，具体规则见 combine_utils.cuh。
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),  // reduce buffer 第一维槽数。
          // NVLink 模式使用 LSA team；跨节点/RDMA 使用 world team。
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>  // 根据互联选择 Gin team。
__global__ void __launch_bounds__(kNumThreads, 1)
combine_impl(nv_bfloat16* x,  // 本地 expert 输出，形状 [num_reduced_tokens, hidden] 或 expanded 行。
             // topk_weights 非 expanded 路径可选写回到 token buffer，epilogue 做加权聚合。
             float* topk_weights,  // 非 expanded 路径的 top-k 权重。
             // src_metadata 来自 dispatch epilogue，逻辑形状 [num_reduced_tokens, 2+kNumTopk]。
             int* src_metadata, int* psum_num_recv_tokens_per_scaleup_rank,  // dispatch epilogue 元数据和 rank prefix。
             // NCCL device communicator/window 交给 Gin 做 device-side 通信。
             const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,  // device communicator/window。
             // buffer 存放 dispatch 接收区和 combine send/reduce 区；workspace 存 barrier 控制面。
             void* buffer, void* workspace,  // payload buffer 和控制面 workspace。
             // 当前 scale-up rank。
             const int rank_idx,  // 当前 scaleup rank。
             // 本地 expert 已计算完成、需要 combine 回去的 token 行数。
             int num_reduced_tokens) {  // 本地 reduced token 数，可能为容量哨兵。
    // combine 是 dispatch 的反向通信：
    // - 输入 x: [num_reduced_tokens, hidden]，是本 rank 本地 expert 计算后的输出；
    // - src_metadata: [num_reduced_tokens, 2 + kNumTopk]，来自 dispatch 阶段，告诉 kernel
    //   每个接收 token 对应原始 token index、源 rank、top-k slot；
    // - 输出不会在本 kernel 中直接写到 combined_x，而是先写入远端 reduce buffer，
    //   之后由 combine_reduce_epilogue 再完成权重归约和 bias 处理。
    // Utils：当前 block/线程坐标。
    const auto sm_idx = static_cast<int>(blockIdx.x);  // 当前 block/SM id。
    // thread_idx 是 block 内线性线程编号，用于 barrier 与可能的清理循环。
    const auto thread_idx = static_cast<int>(threadIdx.x);  // 当前线程 id。
    // warp_idx 加 rank_idx 做轻微轮转，让不同 rank 的 warp 访问模式错开一点。
    const auto warp_idx = (ptx::get_warp_idx() + rank_idx) % kNumWarps;  // rank 轮转后的 warp id。
    // lane_idx 是 warp 内线程编号。
    const auto lane_idx = ptx::get_lane_idx();  // warp 内 lane id。
    // global_warp_idx 用于把所有 warp 拉平成 token 分片编号。
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;  // 全局 warp id。
    // expanded 且不允许多重 reduction 时，每个 top-k 槽都要单独发送，不能本地合并。
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;  // expanded 且禁止本地 reduce 时逐槽发送。

    // We should assign the real number of received tokens if without CPU sync
    // 若调用方传入最大容量哨兵，说明没有 CPU 同步真实 token 数，需要从 prefix-sum 末项读取。
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)  // 容量哨兵表示需要从 GPU prefix 读取真实数量。
        // rank 维 inclusive prefix 的最后一项就是本 rank 本轮实际需要 combine 的 token 数。
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumRanks - 1);

    // Buffer layouts
    // recv_buffer 是 dispatch 阶段在 buffer 开头放置的接收区；send_buffer 紧跟其后，用于
    // combine 时暂存需要通过 RDMA 送回远端的 token。kUseRankLayout/kNumTokensInLayout 来自
    // combine_utils.cuh，会根据 allow_multiple_reduction、rank/topk 数决定 reduce buffer 的组织方式。
    // shared memory 作为每个 warp 的 TMA staging buffer。
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];  // 每 warp TMA staging shared memory。
    // combine token 只需要 hidden 与 top-k weights，不包含 SF，也不需要源 token 字段。
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);  // combine token layout。
    // tma_buffer 逻辑形状 [kNumWarps, 1]，每个 warp 独占一个 token 暂存区。
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumWarps, 1, smem)  // shared staging，每 warp 一个 token。
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    // recv_buffer 是远端最终接收 reduce payload 的区域，第一维由 rank/topk layout 策略决定。
    const auto recv_buffer = layout::BufferLayout<false>(  // 源 rank 最终读取的 reduce buffer。
        token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, buffer);
    // send_buffer 紧跟 recv_buffer，非 NVLink 路径先写本地 send_buffer 再 RDMA put。
    const auto send_buffer = layout::BufferLayout<false>(  // 非直达路径的本地 RDMA send staging。
        token_layout, kNumRanks,
        kNumMaxTokensPerRank * (kDoExpandedSend ? kNumTopk : 1),
        recv_buffer.get_buffer_end_ptr());

    // Init TMA：每个 warp 初始化自己的 mbarrier。
    ptx::arrival_phase phase = 0;  // 当前 TMA mbarrier phase。
    // mbarrier 存在 shared token buffer 内，用于等待 TMA load 完成。
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();  // 当前 warp staging 的 mbarrier。
    // 只由一个 lane 初始化 barrier，expected arrivals 为 1。
    if (ptx::elect_one_sync())  // leader lane 初始化 mbarrier。
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    // 确保 mbarrier 初始化对整个 warp 可见。
    __syncwarp();

    // Expanding mode must not be backward：expanded layout 的 x 已经按 expert 行展开，不再使用 topk_weights。
    if constexpr (kUseExpandedLayout)  // expanded combine 不应携带 topk_weights。
        EP_DEVICE_ASSERT(topk_weights == nullptr);

    // Gin handle
    // We treat each warp as a "channel"
    // QP 分配按 sm/warp 选择，使多个 warp 可以并发使用不同通信队列。
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumWarps>(sm_idx, warp_idx);  // 当前 warp 的 QP。
    // gin 负责 NVLink symmetric pointer 解析与 RDMA/NVLink put。
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);  // Gin 通信句柄。

    // Full barrier to ensure the remote buffer is available：等待上一阶段清理/使用远端 buffer 完成。
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);  // 控制面 workspace。
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kCombineTag0, false, false, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // Do TMA writes into the remote buffers
    // 每个 warp 处理一段 token。TMA staging buffer 位于共享内存中：
    // 1. 从本地 x 读取 expert 输出到 tma_buffer；
    // 2. 根据 src_metadata 找到原 rank 的目标 token slot；
    // 3. 若目标 rank NVLink 可达，直接写它的 recv_buffer；否则先写本地 send_buffer，后续走 RDMA。
    // 每个 warp 负责连续的一段 reduced token，便于按 i 顺序读取 src_metadata。
    int num_tokens_per_warp = math::ceil_div(num_reduced_tokens, kNumSMs * kNumWarps);  // 每 warp token 数上界。
    // 当前 warp 的起始 token 行。
    const int token_start_idx = num_tokens_per_warp * global_warp_idx;  // 当前 warp 起始 token。
    // 当前 warp 的结束 token 行，最后一段可能短于 num_tokens_per_warp。
    const int token_end_idx = min(token_start_idx + num_tokens_per_warp, num_reduced_tokens);  // 当前 warp 结束 token。
    // 顺序扫描当前 warp 的 reduced token 行。
    for (int i = token_start_idx; i < token_end_idx; ++ i) {  // 遍历当前 warp 负责的 reduced token。
        // The master slot index during dispatch
        // metadata 每行长度固定为 2+kNumTopk：
        // [0]=src_token_global_idx，[1]=src_rank_topk_idx，[2..]=expanded 源行号。
        constexpr int kMetadataStride = 2 + kNumTopk;  // src_metadata 每行字段数。
        // src_token_idx 取模后得到源 rank 内的 token 局部下标。
        const int src_token_idx = __ldg(src_metadata + i * kMetadataStride) % kNumMaxTokensPerRank;  // 源 token 局部 index。
        // src_rank_topk_idx 把“源 rank 或 slot layout 维度”和 top-k lane 打包在一个整数里。
        const int src_rank_topk_idx = __ldg(src_metadata + i * kMetadataStride + 1);  // 源 rank/top-k 打包字段。
        // 高位除以 kNumTopk 得到源 rank 或 layout rank。
        const int src_rank_idx = src_rank_topk_idx / kNumTopk;  // 源 rank 或 layout rank。
        // 低位取模得到原 token 的 top-k lane。
        const int src_topk_idx = src_rank_topk_idx % kNumTopk;  // 源 top-k lane。

        // Directly to the remote or via RDMA
        // nvlink_bypass 是一个重要优化：当目标 rank 与当前 rank 在同一个 NVLink/LSA 域时，
        // gin.get_sym_ptr 可以得到目标 rank 对应 token buffer 的可直接访问指针，省掉 send_buffer
        // 与 RDMA 转发；跨节点时则必须先进入 send_buffer，等待后续通信阶段处理。
        // true 表示能直接把结果写回源 rank 的 reduce buffer。
        const bool nvlink_bypass = gin.is_nvlink_accessible<team_t>(src_rank_idx);  // 是否可直接写源 rank。
        // master_token_buffer 是本 token 最终要写入的目标 token buffer 视图。
        layout::TokenLayout master_token_buffer = [=]() {  // 计算当前 token 的目标 buffer。
            // NVLink bypass
            // 直达路径：先构造本地同形状地址，再转换成 src_rank_idx 的 symmetric pointer。
            if (nvlink_bypass) {
                // rank layout 时第一维用当前 rank_idx；top-k layout 时第一维用 src_topk_idx。
                auto token_buffer = recv_buffer.get_rank_buffer(kUseRankLayout ? rank_idx : src_topk_idx).get_token_buffer(src_token_idx);  // 目标 recv token。
                // 把 token_buffer 的 base pointer 从本地窗口地址转换成远端 rank 的可访问地址。
                token_buffer.set_base_ptr(gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx));
                return token_buffer;
            }

            // Use RDMA
            // RDMA 路径：先写本地 send_buffer[src_rank_idx, src_token_idx]，之后显式 gin.put。
            return send_buffer.get_rank_buffer(src_rank_idx).get_token_buffer(src_token_idx);
        }();

        // Hidden requirements：combine_reduce 使用 int4 级向量化，hidden 必须能按 warp 向量整除。
        EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
        // combine_vec_t 是在不超过 128-bit 的条件下选择的向量类型，例如 int4/int2。
        using combine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;  // combine reduce 向量类型。
        // kHiddenVec 是 hidden 向量化后的元素个数。
        constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);  // hidden 向量元素数。

        // Read source indices for expand mode：expanded metadata 第 2.. 列记录每个 top-k 槽对应的 x 行号。
        int stored_topk_slot_idx = -1;  // expanded 模式当前 lane 的 x 行号。
        if constexpr (kUseExpandedLayout) {  // expanded layout 需要读取 top-k 源行。
            // 前 kNumTopk 个 lane 分别读取一个 expanded 槽位；无效槽为 -1。
            if (lane_idx < kNumTopk)
                stored_topk_slot_idx = __ldg(src_metadata + i * kMetadataStride + (2 + lane_idx));
            // 读取完成后，后续 gather/exchange 需要 warp 内一致。
            __syncwarp();
        }

        // 3 cases:
        //  - no expand + no reduce, or expand + no reduce
        // 中文：第一类只需要从一个 x 行搬到目标 token buffer。
        //  - expand + reduce
        //  - expand + send all
        // expand layout 下，一个原 token 可能为多个 top-k 槽保留独立位置。
        // 若多个有效槽最终落到同一 rank 且允许多重 reduction，可以先在本 warp 内做局部 reduce，
        // 减少远端写入次数；若不允许，则必须把每个槽都单独发送，交给 epilogue 做最终聚合。
        // reduce_valid_mask 标出 expanded metadata 中有效的 top-k 源行。
        auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);  // 有效 expanded 行 mask。
        // no_local_reduce 覆盖普通非 expanded 情况，或 expanded 但本 token 只有一个有效槽。
        auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);  // 是否可直接搬运。
        if (no_local_reduce) {  // 直接搬运一行 x。
            // 默认输入 x 的行号就是当前 reduced token 行 i。
            int token_idx_in_tensor = i;  // 默认 x 行号。
            // expanded 单槽情况下，要从有效 lane 广播真实 x 行号。
            if constexpr (kUseExpandedLayout)  // expanded 单槽时广播真实行号。
                token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

            // No reduce
            // 一个 lane 发起完整 hidden 的 TMA load/store，其他 lane 只负责同步。
            if (ptx::elect_one_sync()) {  // leader lane 发起 TMA load/store。
                // load_ptr 指向本地 expert 输出 x[token_idx_in_tensor]。
                const auto load_ptr =  // 当前输入 x 行地址。
                    math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                // 复用 shared buffer 前先等待上一轮 store 完成。
                ptx::tma_store_wait();
                // 把 BF16 hidden 从 x 搬到 shared tma_buffer。
                ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                // 登记本次 TMA load 的字节数。
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                // 等待 TMA load 到达并翻转 phase。
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                // 把 shared hidden 写到目标 token buffer；目标可能是远端 NVLink 指针或本地 send_buffer。
                ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                // 提交 TMA store 请求。
                ptx::tma_store_commit();
            }
            // 等待单槽发送路径完成，再处理 weights/RDMA。
            __syncwarp();
        } else if constexpr (kAllowMultipleReduction) {  // expanded 多槽且允许本地 reduce。
            // Do local reduction
            // Sort valid top-k indices to front
            // topk_slot_idx 保存需要本地 reduce 的 x 行号，末尾以 -1 填充。
            int topk_slot_idx[kNumTopk];  // 待 reduce 的 expanded x 行号列表。
            // compute_topk_slots 会把 reduce_valid_mask 中的有效 lane 压到数组前部。
            compute_topk_slots(
                topk_slot_idx, reduce_valid_mask,  // 输出 slot 列表和有效 mask。
                // 通过 exchange 从 lane idx 广播该 lane 保存的 expanded slot 行号。
                [=](const int& idx) {  // 从指定 lane 广播 slot 行号。
                    return ptx::exchange(stored_topk_slot_idx, idx);
                }
            );

            // Reduce into shared memory：把多个 expanded 行在本 warp 内累加到 tma_buffer。
            // kUnrollFactor 控制每个 lane 连续处理多少向量，兼顾寄存器压力与访存吞吐。
            constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();  // 最大展开因子。
            // combine_reduce 将 topk_slot_idx 中的若干 x 行 BF16 累加，结果写入 shared tma_buffer。
            combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumRanks)>(
                lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),  // lane、输入 slot、输出 shared。
                /* Get source base */ [=](const int& slot_idx) {
                    // slot_idx 是 expanded x 的行号，返回该行向量化后的起始地址。
                    // 中文：combine_vec_t 让 reduce 按向量宽度读写 hidden，减少指令数量。
                    return math::advance_ptr<combine_vec_t>(
                        x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                },
                /* Wait buffer release */ [=]() {  // 写 shared 前等待上一轮 store。
                    // combine_reduce 写 shared 前等待上一轮 TMA store 释放。
                    // 中文：同一个 warp 复用 tma_buffer，若不等待可能覆盖尚未被 TMA 读完的数据。
                    ptx::tma_store_wait();
                    __syncwarp();
                }  // 中文：等待回调结束，combine_reduce 可以安全写入 shared staging。
            );
            // fence 确保普通 shared 写入对随后的 TMA store 可见。
            ptx::tma_store_fence();
            // 等待所有 lane 完成本地 reduce。
            __syncwarp();

            // Issue TMA stores：本地 reduce 后只需要发送一份累加后的 hidden。
            if (ptx::elect_one_sync()) {  // leader lane 发起 reduce 结果 TMA store。
                // 把 tma_buffer 中的 reduce 结果写到目标 token buffer。
                ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                // 提交 TMA store。
                ptx::tma_store_commit();
            }
            // 等待发送请求发出。
            __syncwarp();
        } else {  // 中文：进入 expanded send，保留每个 top-k 槽的独立贡献。
            // No local reduction, send all data (expanded send)
            #pragma unroll
            // 不允许本地 reduce 时，逐 top-k 槽发送每个 expanded 行。
            // 中文：expanded send 会保留每个 top-k 槽的独立贡献，后续由目标端做最终规约。
            for (int k = 0; k < kNumTopk; ++ k) {  // 逐 top-k 槽发送 expanded 行。
                // 中文：此路径保留每个 top-k 槽独立 payload，避免本地多重规约带来的精度/语义变化。
                // 从第 k 个 lane 广播它保存的 expanded x 行号。
                const auto slot_idx = ptx::exchange(stored_topk_slot_idx, k);  // 第 k lane 的 expanded 行号。
                // slot_idx<0 表示该 top-k 槽无效，跳过发送。
                if (slot_idx >= 0) {  // 有效槽才发送。
                    // src_token_ptr 指向 expanded x[slot_idx]，使用 int4 指针满足 TMA 对齐/向量化。
                    const auto src_token_ptr = math::advance_ptr<int4>(x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));  // expanded x 行地址。
                    // token_buffer 是远端 reduce buffer 中 top-k layout 的第 k 行。
                    const auto token_buffer = recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx);  // top-k layout 目标 token。
                    if (ptx::elect_one_sync()) {  // leader lane 串行搬运当前槽。
                        // Load
                        // 每次发送前先等上一轮 store 完成，避免复用 tma_buffer 冲突。
                        ptx::tma_store_wait();
                        // 加载当前 expanded 行到 shared。
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), src_token_ptr, mbarrier_ptr, kNumHiddenBytes);
                        // 登记并等待 TMA load 到达。
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                        // NVLink 可达则直接写远端 token_buffer 的相同位置。
                        if (nvlink_bypass) {
                            // Write into the same position
                            // 中文：bypass 模式无需 RDMA staging，直接用 LSA 对称地址写回源 rank。
                            // 中文：目标地址通过 gin.get_sym_ptr 映射到源 rank 的 symmetric token_buffer。
                            ptx::tma_store_1d(gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx),
                                              // 中文：bypass 和 RDMA staging 都从同一个 shared tma_buffer 取数据。
                                              tma_buffer.get_base_ptr(), kNumHiddenBytes);
                            ptx::tma_store_commit();
                        } else {  // 中文：非 NVLink bypass 时进入 RDMA staging 路径。
                            // Write to the RDMA send buffer
                            // 中文：非 bypass 模式先写本地 send buffer，再由后续 RDMA put 发往源 rank。
                            // expanded send 的 send_buffer token 维度扩展为 src_token_idx*kNumTopk+k。
                            // 中文：这样同一个源 token 的不同 top-k 槽不会互相覆盖 staging 槽。
                            const auto send_token_buffer =  // expanded send 本地 staging token。
                                send_buffer.get_rank_buffer(src_rank_idx).get_token_buffer(src_token_idx * kNumTopk + k);
                            // 先把 shared 写入本地 send buffer。
                            ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                            ptx::tma_store_commit();
                            // 等待本地 send buffer 可作为 RDMA 源。
                            ptx::tma_store_wait();

                            // Issue RDMA
                            // 再通过 gin.put 写到源 rank 的目标 reduce buffer。
                            // 中文：team_t 可能是 world/rail/lsa，模板决定具体通信域。
                            gin.put<team_t>(token_buffer.get_base_ptr(), send_token_buffer.get_base_ptr(),
                                            kNumHiddenBytes, src_rank_idx);
                        }  // 中文：结束 bypass/RDMA staging 二选一路径。
                    }  // 中文：结束当前 token/top-k 槽的发送。
                    // 等待当前 top-k 槽发送路径完成，再复用 staging buffer。
                    __syncwarp();
                }  // 中文：结束当前 k 槽的所有 token 发送。
            }  // 中文：结束 expanded send 的 top-k 循环。
        }

        // Write topk weights：非 expanded 路径需要把原 top-k 权重随 token 一起带回源 rank。
        if (not kUseExpandedLayout and topk_weights != nullptr and lane_idx < kNumTopk) {  // 非 expanded 复制权重。
            // 当前 lane 读取 x[i] 对应的第 lane_idx 个 top-k 权重。
            const float value = __ldg(topk_weights + (i * kNumTopk + lane_idx));  // 当前 top-k 权重。
            // 写入目标 token buffer 的权重区域，后续 reduce epilogue 会读取并应用。
            master_token_buffer.get_topk_weights_ptr()[lane_idx] = value;
        }
        // 等待所有权重 lane 写入完成，再处理可能的 RDMA put。
        __syncwarp();

        // Wait send buffer's TMA store and issue RDMA send
        // NOTES: `kDoExpandedSend` mode has already issued
        // 非 expanded send 且不能 NVLink bypass 时，前面只是把 token 写到本地 send_buffer，
        // 这里才真正把完整 token buffer 通过 RDMA put 到源 rank 的 recv_buffer。
        if (not kDoExpandedSend and not nvlink_bypass and ptx::elect_one_sync()) {  // 普通 RDMA 路径发起 put。
            // 等待 master_token_buffer 对应的本地 send_buffer TMA store 完成。
            ptx::tma_store_wait();
            // 目标 reduce buffer 的第一维依赖布局：rank layout 用当前 rank，top-k layout 用 src_topk_idx。
            const auto dst_ptr = recv_buffer.get_rank_buffer(kUseRankLayout ? rank_idx : src_topk_idx)  // 远端目标 reduce token。
                .get_token_buffer(src_token_idx).get_base_ptr();
            // 把完整 token（hidden + 可选 topk weights）写回源 rank。
            // 中文：master_token_buffer 同时包含 hidden 和非 expanded 路径的 topk weights。
            gin.put<team_t>(dst_ptr, master_token_buffer.get_base_ptr(),
                            // 中文：master_token_buffer 的字节数由 TokenLayout 决定，确保 weights 与 hidden 同步抵达。
                            master_token_buffer.get_num_bytes<false>(), src_rank_idx);
        }
    }

    // Final barrier to ensure data arrival：保证所有 combine payload 到达后，reduce epilogue 才开始读。
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,  // combine payload 完成后的同步。
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kCombineTag1, true, true, false>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);
}

}  // deep_ep::elastic
