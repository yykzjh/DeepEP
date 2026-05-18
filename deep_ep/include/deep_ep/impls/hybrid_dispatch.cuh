#pragma once  // hybrid dispatch device 实现会被 JIT include，避免重复定义。

// comm.cuh 提供跨 scale-out/scale-up 子团队的 barrier 与 QP 选择。
#include <deep_ep/common/comm.cuh>
// compiled.cuh 提供 topk_idx_t、sf_pack_t 等编译期类型。
#include <deep_ep/common/compiled.cuh>
// exception.cuh 提供 EP_STATIC_ASSERT/EP_DEVICE_ASSERT。
#include <deep_ep/common/exception.cuh>
// layout.cuh 提供 workspace、token、buffer 的结构化视图。
#include <deep_ep/common/layout.cuh>
// math.cuh 提供 constexpr 对齐、前缀编码、指针偏移等工具。
#include <deep_ep/common/math.cuh>
// ptx.cuh 封装 TMA、mbarrier、warp shuffle、system-scope load/store。
#include <deep_ep/common/ptx.cuh>


// elastic hybrid dispatch 入口。
namespace deep_ep::elastic {  // elastic hybrid dispatch kernel 所在命名空间。

// kDoCPUSync 控制是否把计数写到 host mapped workspace。
template <bool kDoCPUSync,  // 是否把最终计数同步到 mapped host workspace。
          // kReuseSlotIndices 表示 slot/链表位置由上一次 deterministic 路径复用。
          bool kReuseSlotIndices,  // 是否复用已有 slot/链表位置，服务 deterministic/cached 路径。
          // kNumSMs 是 cooperative launch block 数。
          int kNumSMs,  // cooperative kernel 的 block/SM 数。
          // notify/scaleout/forward 三类 warp 在同一个 block 内并发执行。
          int kNumNotifyWarps, int kNumScaleoutWarps, int kNumForwardWarps,  // 三类 warp 数。
          // scaleout 是跨节点/rail 维度，scaleup 是节点内 NVLink 维度。
          int kNumScaleoutRanks, int kNumScaleupRanks,  // 跨节点和节点内 rank 维度。
          // hidden/SF 描述单 token payload 字节布局。
          int kNumHiddenBytes, int kNumSFPacks,  // hidden payload 字节数和 FP8 scale pack 数。
          // 每个最终 rank 最大可接收 token 数。
          int kNumMaxTokensPerRank,  // 每 rank 最大 token 容量。
          // expert/top-k 路由形状与本地 expert 对齐粒度。
          int kNumExperts, int kNumTopk, int kExpertAlignment,  // expert/top-k 形状和 expert 对齐粒度。
          // QP 数与超时周期控制通信并发和死锁诊断。
          int kNumQPs, int64_t kNumTimeoutCycles,  // Gin QP 数和超时周期。
          // 每个 lane 负责若干 scale-up rank，便于 kNumScaleupRanks>32 时覆盖全部 rank。
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),  // 每个 lane 覆盖的 scaleup rank 分组数。
          // 每个 SM 的 channel 数等于 scaleout warps 数，channel 与 warp 一一对应。
          int kNumChannelsPerSM = kNumScaleoutWarps,  // 每个 SM 内 channel 数。
          // 全局 channel 数。
          int kNumChannels = kNumScaleoutWarps * kNumSMs,  // 全局 channel 数。
          // 每个 channel 分到的 token 容量上限。
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),  // 每 channel token 容量上界。
          // scaleout tail 每累计若干 slot 通知一次 peer，减少 system-scope release 次数。
          int kScaleoutUpdateInterval = 3,  // scaleout tail 批量发布间隔。
          // forward 每次从某个 scaleout rank 消费的 slot chunk 大小。
          int kNumSlotsPerForwardChunk = kScaleoutUpdateInterval,  // forward 每次消费的 slot chunk 大小。
          // 全局 rank 数等于两层拓扑乘积。
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,  // 总 rank 数。
          // 三类 warp 对应的线程数。
          int kNumNotifyThreads = kNumNotifyWarps * 32,  // notify 子团队线程数。
          int kNumScaleoutSendThreads = kNumScaleoutWarps * 32,  // scaleout send 子团队线程数。
          int kNumForwardThreads = kNumForwardWarps * 32,  // forward 子团队线程数。
          // block 总线程数。
          int kNumThreads = kNumNotifyThreads + kNumScaleoutSendThreads + kNumForwardThreads>  // block 总线程数。
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_dispatch_impl(
    // x/sf/topk_idx/topk_weights 是本 rank 输入 token 与路由信息。
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,  // 输入 token、scale factor、top-k id 和权重。
    // copied_topk_idx 是可选输出副本，不参与通信协议。
    topk_idx_t* copied_topk_idx,  // 可选 topk_idx 副本输出。
    // cumulative_local_expert_recv_stats 记录本 rank 本地 expert 的累计接收量。
    int* cumulative_local_expert_recv_stats,  // 本地 expert 累计接收统计。
    // rank/expert 前缀和由 notify 分支生成，copy epilogue 使用。
    int* psum_num_recv_tokens_per_scaleup_rank,  // 输出 scaleup rank 接收 token 前缀和。
    int* psum_num_recv_tokens_per_expert,  // 输出本地 expert 接收 token 前缀和。
    // dst_buffer_slot_idx 保存 forward 分配的 scale-up slot，deterministic/复用路径会读取。
    int* dst_buffer_slot_idx,  // 每个 token/top-k 的目标 buffer slot。
    // token_metadata_at_forward 记录 forward 阶段的源 token、slot 和 top-k 信息，用于链表/复用。
    int* token_metadata_at_forward,  // forward replay 元数据。
    // num_tokens 是当前 rank 输入 token 数。
    const int num_tokens,  // 当前 rank 输入 token 数。
    // FP8 scale factor tensor 的 token/hidden stride。
    const int sf_token_stride, const int sf_hidden_stride,  // FP8 scale factor tensor stride。
    // TODO(NCCL): so many params, plans to optimize?
    // device-side NCCL communicator/window，由 Gin 用于 Rail/LSA 通信。
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,  // device NCCL 通信器与 symmetric window。
    // buffer 是三段通信 payload 区，workspace 是控制面同步区。
    void* buffer,  // payload 通信 buffer。
    void* workspace, void* mapped_host_workspace,  // GPU workspace 和可选 host mapped workspace。
    // 当前 GPU 在 scaleout/scaleup 两层拓扑中的坐标。
    const int scaleout_rank_idx, const int scaleup_rank_idx) {  // 当前 rank 的二维拓扑坐标。
    // hybrid dispatch 同时处理两层拓扑：
    // - notify warps 统计 rank/expert token 数并跨 scale-out rank 交换；
    // - scaleout warps 把本节点要发往其他 scale-out 的 token 写入 RDMA buffer；
    // - forward warps 把收到的 scale-out token 转发到本节点 scale-up peer。
    // recv 侧 epilogue 会基于这里写出的 scaleup_buffer 生成 recv_x/metadata。
    // 每个最终 rank 管辖的 expert 数。
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;  // 每个全局 rank 的 expert 数。
    // 每个 scaleout group 覆盖的 expert 数，用于先判断 token 是否跨 scaleout。
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;  // 每个 scaleout group 的 expert 数。
    // expert 必须能按 scale-up rank 均匀切分。
    EP_STATIC_ASSERT(kNumExperts % kNumScaleupRanks == 0, "Invalid number of experts or ranks");
    // notify warp 数按 4 的倍数组织，匹配 named barrier/warpgroup 使用假设。
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");
    // scaleout send 和 forward 需要一一配对 channel。
    EP_STATIC_ASSERT(kNumScaleoutWarps == kNumForwardWarps, "Invalid warp size");

    // Utils
    // NOTES: a warp is a channel (different channels may share QPs)
    // sm_idx/thread_idx 是当前 block 和 block 内线程编号。
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);  // block/线程坐标。
    // warp_idx/lane_idx 用于划分三类 warp 角色和 top-k lane。
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();  // warp/lane 坐标。
    // 二维 rank 坐标压平成全局 rank id。
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;  // 二维 rank 展平。

    // workspace 保存跨 kernel 的控制面：notify reduction、rank/expert count、
    // channel tail、barrier flag 等；mapped_host_workspace 是 CPU 可读映射，用于
    // kDoCPUSync 路径把计数直接带回 Python。
    // GPU workspace 用于所有 rank 间控制消息。
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);  // GPU 控制面布局。
    // host workspace 只有 kDoCPUSync 路径会写，CPU 读取它拿到真实输出形状。
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);  // host mapped 控制面布局。

    // The kernel uses a fixed space of dynamic shared memory (no static shared memory)
    // smem 前段给 notify 计数，后段给 scaleout/forward warps 做 TMA staging。
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];  // 动态 shared，前段 notify 后段 TMA staging。
    // notify 区容纳全局 rank_count[kNumRanks]+expert_count[kNumExperts]。
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?  // notify shared 区字节数。
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    // 后续 tma_buffer 从 notify 区之后开始，因此 notify 区必须 TMA 对齐。
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");

    // Named barrier indices：notify 专用 barrier，不阻塞其他 warp 角色。
    constexpr int kNotifyBarrierIndex = 1;  // notify 子团队使用的 named barrier id。

    // NCCL Gin handle
    // Each warp is a channel
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, (kNumNotifyWarps > 0)>(  // 选择当前 warp 的 QP 模式。
        sm_idx, (warp_idx - kNumNotifyWarps) % kNumChannelsPerSM, warp_idx < kNumNotifyWarps);
    // Gin 同时支持 Rail(scale-out/RDMA) 和 LSA(scale-up/NVLink) team。
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);  // 当前 warp 使用的 Gin 句柄。

    // Global parallel barriers for scale-out subteam and scale-up subteam
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,  // hybrid dispatch 开始前的团队同步。
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag0, false, false, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // token_layout 是整个 hybrid dispatch 的统一 token 包格式：
    // hidden bytes + 可选 FP8 scale packs + topk_idx/topk_weights + 源 token metadata。
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);  // 通信 token 包布局。
    // notify warps 不使用 TMA staging，因此索引用 warp_idx-kNumNotifyWarps。
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumScaleoutWarps + kNumForwardWarps, 1,  // 非 notify warp 的 shared staging。
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(warp_idx - kNumNotifyWarps).get_token_buffer(0);

    // buffer 被顺序切成三段：
    // scaleup_buffer [scaleup_rank, scaleout_rank*max_tokens]：最终给本节点专家读取；
    // scaleout_send_buffer [1, max_tokens]：本 rank 发 RDMA 前的 staging；
    // scaleout_recv_buffer [scaleout_rank, channel*max_tokens_per_channel]：RDMA 收包区。
    auto scaleup_buffer = layout::BufferLayout<false>(  // 最终写给节点内 scaleup peer 的 buffer。
        token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);
    auto scaleout_send_buffer = layout::BufferLayout<false>(  // 跨 scaleout 发送 staging buffer。
        token_layout, 1, kNumMaxTokensPerRank, scaleup_buffer.get_buffer_end_ptr());
    auto scaleout_recv_buffer = layout::BufferLayout<false>(  // 跨 scaleout 接收 buffer。
        token_layout, kNumScaleoutRanks, kNumChannels * kNumMaxTokensPerChannel, scaleout_send_buffer.get_buffer_end_ptr());

    // Init TMA for scale-out and forward warps
    // phase 记录 TMA mbarrier phase；scaleout/forward warp 每轮 load 后翻转。
    ptx::arrival_phase phase = 0;  // 当前 warp TMA mbarrier phase。
    // mbarrier 放在对应 warp 的 token staging buffer 内。
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();  // 当前 warp staging token 的 mbarrier。
    // notify warp 没有 tma_buffer，因此只有非 notify warp 初始化 mbarrier。
    if (warp_idx >= kNumNotifyWarps and ptx::elect_one_sync())  // 非 notify warp 由 leader lane 初始化。
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    // 确保 mbarrier 初始化完成。
    __syncwarp();

    // warp role 按 warp_idx 划分，三类 warp 在同一个 cooperative kernel 中并发，
    // 通过 named barrier、workspace tail 和 GIN signal 交接数据。
    if (warp_idx < kNumNotifyWarps) {  // notify 子团队：统计并交换 rank/expert 计数。
        // Assign shared memory：notify 只使用 smem 开头的 rank/expert 计数区。
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);  // shared notify 区 int 元素数。
        // rank_expert_count 是 rank_count 和 expert_count 的连续拼接。
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);  // notify shared 计数区起点。

        // Clean initial counts
        // NOTES: if you want to change the order of different warp roles, please take care of the `thread_idx`
        // rank_count 长度 kNumRanks；expert_count 长度 kNumExperts。
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;  // rank/expert 计数数组。
        #pragma unroll
        // notify 线程分摊清零对齐后的 shared 计数区。
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)  // 当前 notify 线程清理多个 int。
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        // 确保计数从 0 开始。
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // notify warp 对 topk_idx [num_tokens, kNumTopk] 做一次全局扫描：
        // expert_count 按 expert 计数，rank_count 按目标全局 rank 去重计数。
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        // global_warp_idx 交错分配 token，避免某个 SM 单独处理连续大块输入。
        const auto global_warp_idx = sm_idx * kNumNotifyWarps + warp_idx;  // 全局 notify warp id。
        // 每个 notify warp 以全局 notify warp 数为 stride 扫描。
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {  // 按 warp 网格步进扫描 token。
            // Expert choice can not be redundant
            // NOTES: no assertions here as they are expensive
            // 前 kNumTopk 个 lane 分别读取一个 expert id，其余 lane 为 -1。
            const auto dst_expert_idx = lane_idx < kNumTopk ?  // 当前 lane 对应的目标 expert。
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            // expert_count 统计全局 expert 级别的选择次数。
            if (dst_expert_idx >= 0)  // 无效 expert=-1 不计数。
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // Rank choice should do deduplication here：同一 token 发往同一全局 rank 只发送一次。
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;  // expert 映射到全局 rank。
            // deduplicate 保留同 rank 的一个 lane，避免 rank_count 重复。
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)  // 同一 token 同 rank 只计一次。
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        // 等待所有 notify warp 完成 shared 计数。
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // 每个 SM 把局部计数以 (arrival_count<<32)|count 的方式 red_add 到
        // workspace。SM0 等 arrival_count==kNumSMs 后得到全 grid 计数。
        #pragma unroll
        // 每个 notify 线程负责若干 rank/expert 字段。
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            // 高 32 位代表当前 SM 到达一次，低 32 位携带该 SM 局部计数。
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            // red_add 聚合所有 SM 的同一字段。
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // Do the remaining work by SM 0：等待全 grid 计数并跨 scale-out/scale-up 发布。
        if (sm_idx == 0) {  // 只有 SM0 负责把全 grid 聚合结果发布出去。
            // Reduce all SM's count
            // Wait all SMs' arrival
            #pragma unroll
            // SM0 的 notify 线程分摊等待所有字段的 arrival_count。
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {  // 分摊等待每个计数字段。
                // timeout_while 防止某个 SM 没到时无限等待。
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {  // 等待所有 SM 的 red_add 到达。
                    // status 高 32 位是到达 SM 数，低 32 位是累计 count。
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);  // 高位 arrival，低位 count。
                    // 所有 SM 到达后，将结果编码到 scaleout send workspace。
                    if ((status >> 32) == kNumSMs) {
                        // Encode and write into the send buffer
                        // get_scaleout_rank_expert_count_ptr<true>() 是本地发送缓冲区。
                        workspace_layout.get_scaleout_rank_expert_count_ptr<true>()[i] =
                            math::encode_decode_positive<int>(status & 0xffffffffll);

                        // Clean for the next usage：清空 reduction workspace，供下一轮使用。
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    // 最后一次检查输出详细拓扑坐标，便于定位卡在哪个 rank。
                    if (is_last_check) {
                        // 中文：status 高 32 位是到达计数，低 32 位是 reduction 后的 expert/rank count。
                        printf("DeepEP hybrid notify (GPU reduction) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                               // 中文：打印当前 logical rank 坐标和 status，便于区分 scaleout/scaleup 哪一层卡住。
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }
                    return false;
                    // 中文：timeout_while 收到 false 会继续轮询，直到超过最终检查后触发外层处理。
                });
            }
            // 确保所有字段都已写入 scaleout send workspace。
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // 把本 scaleout rank 的计数写到所有 scaleout peer 的 workspace 中。
            // rank_count 写 kNumScaleupRanks 个 int，expert_count 只写本 scaleout
            // 负责的 kNumExpertsPerScaleout 个 expert。
            EP_STATIC_ASSERT(kReuseSlotIndices or kNumScaleoutRanks <= kNumNotifyThreads,
                             "kNumScaleoutRanks must be less than kNumNotifyThreads");
            // 前 kNumScaleoutRanks 个线程分别负责一个 scaleout peer。
            if (thread_idx < kNumScaleoutRanks) {
                // 目标 scaleout peer id。
                const auto dst_scaleout_rank_idx = thread_idx;
                // rank_count 段按 [scaleout_rank, scaleup_rank] 组织，发送当前 scaleout_rank_idx 的一行。
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx),  // 远端写入本 scaleout 行的 recv rank-count 区。
                    workspace_layout.get_scaleout_rank_count_ptr<true>(dst_scaleout_rank_idx),  // 本地 send workspace 中目标 peer 对应的 rank-count 行。
                    kNumScaleupRanks * sizeof(int), dst_scaleout_rank_idx,  // 拷贝一整行 scaleup rank count 到目标 scaleout peer。
                    ncclGinOptFlagsAggregateRequests);  // rank-count put 先聚合，后续 expert-count put 负责推进队列。
                // expert_count 段只发送当前 scaleout rank 覆盖的 expert 切片。
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_rank_idx),  // 远端 recv expert-count 中属于本 scaleout 的切片。
                    workspace_layout.get_scaleout_expert_count_ptr<true>(dst_scaleout_rank_idx),  // 本地 send expert-count 切片。
                    kNumExpertsPerScaleout * sizeof(int), dst_scaleout_rank_idx);  // 发送本 scaleout 管辖的 expert 计数。
            }
            // 等待 Rail put 请求发出。
            __syncwarp();

            // Util functions to get metadata from scale-out peers
            // NOTES: this is correct as RDMA operations has a minimum write granularity of 1024 bytes (a whole integer write is atomic)
            // recv_and_reduce 从所有 scaleout peer 拉取同一个字段并求和。
            // 逻辑维度可以理解为：
            //   rank_count[scaleout_peer, dst_scaleup_rank] 或
            //   expert_count[scaleout_peer, expert_in_current_scaleout_slice]。
            // 每个字段都由远端 RDMA 写入 ready 编码，当前 notify 线程负责等待、解码、累加并清零。
            const auto recv_and_reduce = [=](const auto& get_ptr_func, const bool& is_expert_reduction = false) -> int {  // 等待并汇总所有 scaleout peer 的字段。
                // count 累加 kNumScaleoutRanks 个 peer 的 decoded 值。
                int count = 0;  // 当前字段跨 scaleout peer 的累计值。
                #pragma unroll
                // j 遍历每个 scaleout peer。
                for (int j = 0; j < kNumScaleoutRanks; ++ j) {  // 遍历每个 scaleout peer。
                    // get_ptr_func(j) 返回 peer j 写入本 rank workspace 的字段地址。
                    const auto ptr = get_ptr_func(j);  // peer j 写入的 workspace 字段。
                    // decoded 在 timeout loop 内更新，ready 后用于累加。
                    int decoded;  // 解码后的正数 count。
                    // 等待该 peer 的 ready 编码出现。
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check){  // 等待该 peer 字段 ready。
                        // acquire_sys 保证看到 peer RDMA 写入的数据。
                        decoded = math::encode_decode_positive(ptx::ld_acquire_sys<int>(ptr));  // acquire 读取并解码正数。
                        // ready 表示 peer 已经把正数编码写入该字段。
                        if (math::is_decoded_positive_ready(decoded))  // 该字段已经由 peer 写入。
                            return true;

                        // 超时打印是 rank count 还是 expert count reduction。
                        if (is_last_check) {
                            // 中文：decoded 仍未 ready 时，说明对应 scaleout peer 还没写入该字段。
                            printf("DeepEP hybrid notify (scale-out %s reduction) timeout, "
                                   "scale-out: %d, scale-up: %d, "
                                   "thread: %d, wait scale-out: %d, decoded: %d\n",
                                   is_expert_reduction ? "expert" : "rank",  // 当前 reduction 类型。
                                   scaleout_rank_idx, scaleup_rank_idx, thread_idx, j,  // 本 rank 坐标、线程和等待的 scaleout peer。
                                   decoded);  // 当前解码值。
                        }
                        return false;
                    });

                    // Add and clean for next usages
                    // 累加该 peer 的贡献，并清空字段供下一轮复用。
                    count += decoded, *ptr = 0;
                }
                // 返回所有 scaleout peer 的合计。
                return count;
            };

            // Write into all scale-up peers' rank-level counters
            #pragma unroll
            // i 是目标 scale-up rank，本 rank 需要把“所有 scaleout 汇总后的 rank count”写给它。
            for (int i = thread_idx; i < kNumScaleupRanks; i += kNumNotifyThreads) {
                // Wait scale-out arrival and reduce
                // 对固定 scale-up rank i，汇总所有 scaleout peer 发给它的 token 数。
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {  // 汇总目标 scaleup rank i 的 rank count。
                    // scaleout_rank_count 的二维逻辑为 [scaleout_peer, dst_scaleup_rank]。
                    return workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_peer_idx, i);
                });

                // Write into the remote scale-up peer
                // 高 32 位写 kNumScaleupRanks，表示目标端需要等待这么多 scale-up peer 的贡献。
                const int64_t counter = (static_cast<int64_t>(kNumScaleupRanks) << 32ll) | count;  // 到达数一次性设置为 scaleup peer 总数。
                // 低 32 位是真实 token 数；高 32 位是“贡献到达个数”，目标端用它判断是否 ready。
                // 写到目标 scale-up peer 的 rank_count[scaleup_rank_idx] 槽。
                gin.put_value<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx,
                    // 中文：目标是同一 scaleout 内的第 i 个 scaleup peer。
                    counter, i);
            }
            // 等待 rank-level counter 通知发出。
            __syncwarp();

            // Atomic add into all scale-up peers' expert-level counters
            #pragma unroll
            // i 遍历当前 scaleout expert 切片内的 expert。
            for (int i = thread_idx; i < kNumExpertsPerScaleout; i += kNumNotifyThreads) {
                // Wait scale-out arrival and reduce
                // 对固定 expert i，汇总所有 scaleout peer 的选择次数。
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {  // 汇总 expert i 的 expert count。
                    // expert count 以 scaleout 内 expert id 编号，后面再映射到目标 scale-up rank 的本地 expert。
                    return workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_peer_idx, i);
                }, true);

                // Write into the remote scale-up peer
                // expert-level 使用 red_add_rel，因为多个源 scale-up peer 会累加到同一目标 expert。
                const int64_t counter = (1ll << 32ll) | count;  // expert red_add 每次贡献一个 arrival。
                // expert counter 的高 32 位每次只加 1，目标端等到 kNumScaleupRanks 次贡献。
                // 当前 scaleout expert 切片内的 i 映射到目标 scale-up rank。
                const auto dst_scaleup_rank_idx = i / kNumExpertsPerRank;  // expert i 所属 scaleup rank。
                // expert_idx_in_dst_rank 是目标 rank 内的本地 expert 下标。
                const auto expert_idx_in_dst_rank = i % kNumExpertsPerRank;  // 目标 rank 内 expert 下标。
                // release-scope red_add 同时累加到达数和 count。
                gin.red_add_rel<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_expert_count_ptr<false>() + expert_idx_in_dst_rank,
                    // 中文：counter 高位是到达次数、低位是 token count，目标端等待所有 scaleup peer。
                    counter, dst_scaleup_rank_idx);
            }
            // There are shared memory reads above, a barrier is necessary
            // 中文：rank_expert_count 仍在 shared 中被后续阶段复用，不能让 notify warp 提前覆盖。
            // 等待所有 scale-up peer 通知请求发出，再复用 shared 计数区。
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // NOTES: from now on, the `rank` and `expert`s size change into the local size
            // 从这里开始只关心本 scale-up rank 看到的 rank_count[kNumScaleupRanks] 和本地 expert_count。
            // 中文：前面是 scaleout 维聚合，后面转为 scaleup rank 内的局部 prefix 计算。
            expert_count = rank_expert_count + kNumScaleupRanks;

            // Wait local counters to be ready
            // NOTES: here we only care the prefix sum by scale-up peers (used for later epilogue), not all ranks
            // 中文：这里只等待 scale-up peer 写给本 rank 的局部计数，全局 scaleout 发送不在此处阻塞。
            EP_STATIC_ASSERT(kNumNotifyWarps == 0 or kNumScaleupRanks + kNumExpertsPerRank <= kNumNotifyWarps * 32,
                             "Insufficient notify threads");
            // 等待本 scale-up rank 的 rank/expert counter 全部由 peer 写入完成。
            comm::timeout_while<kNumTimeoutCycles>(thread_idx < kNumScaleupRanks + kNumExpertsPerRank,
                [&](const bool& is_last_check) {  // 等待当前 thread_idx 对应字段 ready。
                // 每个 thread_idx 对应一个本地 rank/expert 字段。
                const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + thread_idx);  // 本地 scaleup 聚合状态。
                // 高 32 位达到 kNumScaleupRanks，表示所有 scale-up peer 的贡献到齐。
                if ((status >> 32ull) == kNumScaleupRanks) {  // arrival_count 到齐。
                    // Clean GPU workspace and write into host workspace
                    // 低 32 位是真实计数。
                    const auto count = static_cast<int>(status & 0xffffffffll);  // 真实 token/expert count。
                    // status 是 (arrival_count << 32) | token_count，两个字段共用一个 64-bit 原子槽。
                    // rank count 不需要对齐，expert count 需要按 kExpertAlignment 对齐。
                    const auto aligned_count = math::align<int>(  // expert count 按 tile 对齐。
                        count, thread_idx < kNumScaleupRanks ? 1 : kExpertAlignment);

                    // 清理 workspace 槽。
                    workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] = 0;
                    // CPU sync 路径把 ready 编码写到 mapped host workspace。
                    if constexpr (kDoCPUSync) {  // 需要 CPU 同步输出形状时写 host mapped 区。
                        // host 侧轮询正数编码；encode_decode_positive 用负/正翻转避免读取半更新状态。
                        host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] =
                            math::encode_decode_positive(aligned_count);
                    }

                    // Update statistics counters
                    // 统计使用未对齐 count；thread_idx>=kNumScaleupRanks 表示本地 expert 字段。
                    if (cumulative_local_expert_recv_stats != nullptr and thread_idx >= kNumScaleupRanks)  // 本地 expert 统计字段。
                        atomicAdd(cumulative_local_expert_recv_stats + (thread_idx - kNumScaleupRanks), count);

                    // Save for later prefix sum calculation
                    // 保存对齐后的值，prefix sum 用它生成实际输出布局。
                    // rank_expert_count 前 kNumScaleupRanks 个是 rank 维计数，后面是本地 expert 维计数。
                    rank_expert_count[thread_idx] = aligned_count;
                    return true;
                }

                // 超时输出当前字段的到达计数和值。
                if (is_last_check) {
                    // 中文：scale-up reduction 超时通常说明同一 scaleout 组内某个 scaleup rank 没有写入本字段。
                    printf("DeepEP hybrid notify (scale-up reduction) timeout,"
                           "scale-out: %d/%d, scale-up: %d/%d, "
                           "thread: %d, status: %d | %d, expected: %d\n",
                           // 中文：status 高/低 32 位与 GPU reduction 路径一致，expected 是 scaleup rank 数。
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                           // 中文：打印 status 的高低半字，便于判断是到达数不足还是 count 值异常。
                           static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumScaleupRanks);
                }  // 中文：超时打印只在最后一次检查触发，避免自旋期间刷屏。
                return false;
            });
            // 中文：wait 循环结束后，rank_count/expert_count 在 shared 中已经是本 rank 的局部视图。
            // 等待所有本地字段写入 shared。
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Do prefix sum by the warps of the first SM
            // NOTES: we may have fast implementation with `cub::BlockScan`, but it is too heavy to use
            // 中文：数组长度只有 rank/expert 级别，用轻量 warp scan 更适合 JIT 模板展开。
            // do_psum 与单层 dispatch 类似：小数组用一个 warp 做 prefix，避免引入 CUB。
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {  // warp 内小数组前缀和。
                // psum 保存前一批 32 lane 的累计和。
                int psum = 0;  // 已处理批次的累计和。
                #pragma unroll
                // exclusive 模式多输出一个前缀起点，因此循环长度是 n+1。
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {  // 每轮处理最多 32 个元素。
                    // idx 是输出位置。
                    const auto idx = i * 32 + lane_idx;  // 当前 lane 对应输出位置。
                    // exclusive 模式输入索引比输出索引小 1。
                    const auto mem_idx = idx - is_exclusive;  // exclusive 模式输入左移一位。
                    // 越界 lane 贡献 0。
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;  // 越界贡献 0。
                    // warp 内 inclusive prefix 加上前面批次 psum。
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);  // 当前批次 prefix 加历史累计。

                    // Store into global memory
                    // rank 前缀输出 n 项，expert exclusive 前缀输出 n+1 项。
                    if (idx < n + is_exclusive)  // 有效输出位置才写。
                        out[idx] = sum;

                    // Update `psum` by using the last lane's value
                    // lane31 是当前 32 项总和，广播给下一批。
                    psum = ptx::exchange(sum, 31);
                }
            };
            // warp0 生成 scale-up rank 维 inclusive 前缀。
            if (warp_idx == 0) {  // notify warp0 计算 rank 维 prefix。
                // Inclusive prefix sum
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumScaleupRanks, 0);
            // warp1 生成本地 expert 维 exclusive 前缀，expanded epilogue 用。
            } else if (warp_idx == 1) {  // notify warp1 计算 expert 维 prefix。
                // Exclusive prefix sum for later expanding
                // 中文：exclusive expert prefix 给 expanded copy epilogue 提供每个 expert 的起始位置。
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }  // 中文：notify warp 的 prefix 计算结束。
        }  // 中文：结束 hybrid notify 子团队逻辑。
    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
        // scaleout_warp_idx 是 scaleout send 角色内的 warp 编号。
        const int scaleout_warp_idx = warp_idx - kNumNotifyWarps;  // scaleout send 子团队内 warp id。
        // channel_idx 全局唯一，后续 scaleout_recv_buffer 与 tail 都按 channel 切片。
        const int channel_idx = sm_idx * kNumChannelsPerSM + scaleout_warp_idx;  // 当前 sender channel。
        // 当前 scaleout rank 只写自己 rank 行下的 recv buffer。
        scaleout_recv_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_rank_idx);
        // 再切到当前 channel 的 token 段，容量 kNumMaxTokensPerChannel。
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);

        // Channel metadata maintenance
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid number of scale-out ranks");
        // stored_scaleout_tail 是本 channel 对各 scaleout 目标发出的 token 数，由 lane 按 rank 持有。
        int stored_scaleout_tail = 0, stored_old_scaleout_tail = 0;  // 当前 tail 和上次已发布 tail。
        // update_scaleout_tail 把 tail 增量通知给每个 scaleout peer 的 forward warp。
        const auto update_scaleout_tail = [&](const bool& finish_flag = false) {  // 发布当前 channel 的 scaleout tail。
            // 只有 rank lane 负责通知；达到 interval 或 finish 时才写，降低 release/atomic 频率。
            if (lane_idx < kNumScaleoutRanks and
                (stored_scaleout_tail >= stored_old_scaleout_tail + kScaleoutUpdateInterval or finish_flag)) {
                // signaled_tail 同时打包 finish 标志和 tail 值。
                const auto signaled_tail = math::pack2<int, int64_t>(finish_flag, stored_scaleout_tail);  // 打包 finish/tail。
                // ptr 指向“目标 scaleout peer 看到的本 channel/源 scaleout tail”槽。
                const auto ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx);  // 本 channel/source scaleout 信号槽。
                // old_signaled_tail 用于计算增量，red_add_rel 避免覆盖并发更新。
                const auto old_signaled_tail = math::pack2<int, int64_t>(0, stored_old_scaleout_tail);  // 上次已发布状态。

                // NOTES: the "release" scope will be `sys` for the local rank (we may involve NVLink so not `gpu`)
                // For RDMA requests, "release" is ensured by "atomic"
                // lane_idx 同时是目标 scaleout rank；Rail team 负责跨 scaleout 通知。
                gin.red_add_rel<ncclTeamTagRail>(ptr, signaled_tail - old_signaled_tail, lane_idx);
                // 记录已经通知过的 tail，下一次只发送增量。
                stored_old_scaleout_tail = stored_scaleout_tail;
            }
            // 确保同 warp 对 tail 的视图一致。
            __syncwarp();
        };

        // Preload next token：提前把下一 token 的 hidden/SF 异步搬到 shared，和 RDMA 发起重叠。
        const auto preload_next_token = [&](const int& token_idx) {  // 预取下一 token 到 shared staging。
            // token_idx 越界表示没有下一 token。
            if (token_idx >= num_tokens)  // 没有下一 token 时跳过。
                return;

            // Issue TMA load
            // 转成 int64 做字节偏移。
            const auto token_i64_idx = static_cast<int64_t>(token_idx);  // 字节偏移使用 64-bit。
            // 一个 lane 发起 hidden 的 TMA load。
            if (ptx::elect_one_sync()) {  // leader lane 发起 hidden TMA load。
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            // 等待 TMA 发射。
            __syncwarp();

            // Issue SF `cp.async`
            if constexpr (kNumSFPacks > 0) {
                // SF pack 要满足 cp.async 的基础对齐。
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                // gmem_src_ptr 是当前 token 的 SF 起始地址。
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                // smem_dst_ptr 是 shared token buffer 的 SF 区。
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                // 每个完整轮由 32 lane 搬 32 个 SF pack。
                constexpr auto kNumFullIters = kNumSFPacks / 32;  // 完整 32-lane SF 搬运轮数。
                #pragma unroll
                // k 轮覆盖第 k*32 到 k*32+31 个 SF pack。
                for (int k = 0; k < kNumFullIters; ++ k) {  // 每轮所有 lane 各搬一个 SF pack。
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                // 尾部不足 32 个 pack 时，只让有效 lane 搬运。
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {  // 尾部不足 32 个 pack。
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                // 把 cp.async 到达也关联到同一 mbarrier。
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                // 等待所有 lane 发完 SF 异步拷贝。
                __syncwarp();
            }
        };

        // Iterate all tokens：channel_idx 决定该 warp 处理的第一枚 token。
        preload_next_token(channel_idx);
        // 每个 channel 以 kNumChannels 为 stride 扫描输入 token。
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += kNumChannels) {  // 当前 channel 按 stride 扫描 token。
            // Load top-k indices and weights
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            // 当前 lane 对应的目标 scaleout rank。
            int stored_dst_scaleout_rank_idx = -1;  // 当前 top-k lane 的目标 scaleout rank。
            // 前 kNumTopk lane 读取 top-k expert。
            if (lane_idx < kNumTopk) {  // top-k lane 读取路由信息。
                // 保留原始 topk 类型用于 copied_topk_idx。
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);  // 原始 topk_idx 元素。
                // 转 int 后用于计算 scaleout rank。
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);  // 转为 int 方便计算 rank。
                // 先按每个 scaleout 覆盖的 expert 区间判断目标 scaleout。
                stored_dst_scaleout_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerScaleout : -1;
                // 把全局 expert id 写入 token metadata，forward 阶段再映射 scale-up rank。
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                // topk weight 可选写入 token buffer。
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                // 可选复制原 topk_idx 给上层。
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            // 等待 top-k metadata 写入 shared。
            __syncwarp();

            // Add source metadata (rank index and token index)
            // 源 token 全局 id 编码为全局 rank*kNumMaxTokensPerRank+token_idx。
            if (ptx::elect_one_sync())  // leader lane 写源 token 全局 id。
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            // fence 确保 shared metadata 对后续 TMA store 可见。
            ptx::tma_store_fence();
            // staging metadata 完成。
            __syncwarp();

            // Deduplicate ranks and assign slots：按目标 scaleout rank 分配 channel 内 slot。
            int stored_dst_slot_idx = -1;  // 当前 top-k lane 分配到的 scaleout recv slot。
            // 通过目标 rank 对应 lane 广播当前 tail，作为该 token 的旧 slot。
            const auto stored_old_slot_idx = ptx::exchange(  // 从目标 scaleout rank lane 读取当前 tail。
                stored_scaleout_tail, stored_dst_scaleout_rank_idx >= 0 ? stored_dst_scaleout_rank_idx : 0);
            // 同 token 多个 top-k 可能落到同一 scaleout，只保留一个 lane 消耗 slot。
            if (ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx) and stored_dst_scaleout_rank_idx >= 0)  // 同 scaleout 去重后分配 slot。
                stored_dst_slot_idx = stored_old_slot_idx;

            // Update scale-out tail：统计本 token 涉及哪些 scaleout rank，并推进对应 tail。
            const auto scaleout_rank_mask = ptx::reduce_or(stored_dst_scaleout_rank_idx >= 0 ? (1u << stored_dst_scaleout_rank_idx) : 0u);  // 当前 token 命中的 scaleout 集合。
            // 每个 rank lane 持有自己的 tail，mask 中该位为 1 则 tail+1。
            stored_scaleout_tail += (scaleout_rank_mask >> lane_idx) & 1;

            // Wait TMA arrival and issue the TMA store into send buffer。
            if (ptx::elect_one_sync()) {  // leader lane 等待预取并写 send buffer。
                // 等待 preload_next_token 发出的 hidden/SF 到达 shared。
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                // So if no ranks will go by RDMA, we skip the send buffer stores
                // 如果所有目标都只是本地 scaleout_rank_idx，则不需要写 scaleout_send_buffer。
                if (scaleout_rank_mask ^ (1 << scaleout_rank_idx)) {  // 存在非本地 scaleout 目标。
                    // 需要跨 scaleout 时，先把完整 token 写入本地 send buffer[token_idx]。
                    // 中文：如果只有本地目标，后续直接写本地 recv buffer，可跳过 send staging。
                    ptx::tma_store_1d(scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                                      tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                }  // 中文：结束跨 scaleout staging 写入判断。
            }  // 中文：结束当前 token 的本地 shared/staging 准备。
            // 等待 shared 数据到达和可能的 send buffer store 发起。
            __syncwarp();

            // Local rank can be bypassed：目标 scaleout 就是本 rank 时直接写本地 recv channel。
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx == scaleout_rank_idx) {  // 本地 scaleout 目标直接 bypass。
                ptx::tma_store_1d(scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                  tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            }
            // 提交本轮 TMA store。
            ptx::tma_store_commit();
            // 等待本地 send/recv buffer store 完成，RDMA 源必须已经可读。
            ptx::tma_store_wait();
            // 同步后预取下一 token。
            __syncwarp();

            // Preload the next token (overlapping with the IBGDA issues)：预取与下面的 put 重叠。
            preload_next_token(token_idx + kNumChannels);

            // Issue IBGDA requests：跨 scaleout 目标通过 Rail team put 到目标 recv channel。
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx != scaleout_rank_idx) {  // 远端 scaleout 目标走 RDMA put。
                gin.put<ncclTeamTagRail>(
                        // 中文：目标地址是远端 scaleout recv slot，源地址是本地 staging send slot。
                        scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                        tma_buffer.get_num_bytes<false>(),
                        // 中文：stored_dst_scaleout_rank_idx 是 rail team 里的目标 scaleout rank。
                        stored_dst_scaleout_rank_idx,
                        ncclGinOptFlagsAggregateRequests);
            }
            // 等待本 token 的 RDMA 请求发起完成。
            __syncwarp();

            // Issue scale-out tail update：周期性通知 forward warp 有新 slot 可消费。
            update_scaleout_tail();
        }

        // Flush unflushed tails：循环结束后把未满 interval 的 tail 和 finish 标志发出去。
        update_scaleout_tail(true);
    } else {
        // forward_warp_idx 是 forward 角色内的 warp 编号。
        const int forward_warp_idx = warp_idx - (kNumNotifyWarps + kNumScaleoutWarps);  // forward 子团队内 warp id。
        // forward 与 scaleout send 使用同一个 channel 编号，才能消费对应 recv channel。
        const int channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;  // 当前 forward channel。
        // 切到当前 channel 的 scaleout_recv_buffer，后续第一维再按来源 scaleout rank 选择。
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);
        // scaleup_buffer 当前 rank 行是本 scale-up 组内可映射到 peer 的基址。
        scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Shape of `token_metadata_at_forward`: `[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        // forward metadata 每行保存源 token、chunk 结束标志、top-k 目标 scaleup rank 和 slot。
        // 这个布局必须和 hybrid_combine.cuh 中的 replay 逻辑完全一致：
        // combine 会逐行读取 source token、chunk 尾标记、scale-up rank/slot，并按这里记录的顺序反向回传。
        // 额外的 +1 行用于写 -1 结束哨兵，避免 combine 需要额外传入每个 channel 的行数。
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;  // forward metadata 每行字段数。
        // 指针偏移到当前 channel 的 metadata 区。
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);  // 偏移到当前 channel。

        // Shape of `dst_buffer_slot_idx`: `[kNumChannels, kNumScaleoutRanks, kNumMaxTokensPerChannel, kNumTopk]`
        // 当前 channel 的 dst slot 表用于复用路径，记录每个 RDMA slot/top-k lane 的 scale-up slot。
        dst_buffer_slot_idx += channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel * kNumTopk);  // 偏移到当前 channel slot 表。

        // Transform linked list index
        // 链表全局下标把 [channel, logical_list_idx, scaleup_rank] 压平成一维。
        // copy epilogue 会按这个一维下标把 token 串成 per-scaleup-rank 链表；
        // 因此 transform_linked_list_idx 的编码要同时包含 channel 和 scaleup_rank，防止不同 channel 的节点互相覆盖。
        const auto transform_linked_list_idx = [=](const int& idx) {  // 把局部链表序号转为全局一维下标。
            // 每个 channel 有 kNumScaleoutRanks*kNumMaxTokensPerChannel 个真实节点，加 1 个尾节点。
            constexpr int kNumTokensInLinkedList = kNumMaxTokensPerChannel * kNumScaleoutRanks + 1;  // 每 channel 链表节点数含尾节点。
            // scaleup_rank_idx 作为最后一维，便于同一 logical idx 同时记录多个 scale-up rank 的链。
            return channel_idx * (kNumTokensInLinkedList * kNumScaleupRanks) +
                idx * kNumScaleupRanks + scaleup_rank_idx;
        };

        // Forward tokens from scale-out ranks
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Too many scale-out ranks");
        // num_tokens_processed 是当前 channel 已写 metadata 的行数。
        int num_tokens_processed = 0;  // 当前 channel 已写入 forward metadata 的 token 行数。
        // stored_scaleout_old_tail_idx 是当前 lane 对应 scaleout rank 已消费到的 tail。
        int stored_scaleout_old_tail_idx = 0;  // 当前 lane 对应来源 scaleout 的已消费 tail。
        // 每个 lane/j 持有一个目标 scale-up rank 的发送计数，用作 linked list idx/slot。
        int stored_scaleup_send_counters[kNumScaleupRanksPerLane] = {};  // 每个目标 scaleup rank 已发送 slot 计数。
        // 对 lane>=scaleout ranks 的无效 lane，finish 直接为 true。
        int stored_finish_flag = lane_idx >= kNumScaleoutRanks;  // 越界 lane 默认完成。
        // 当前 lane 对应 scaleout rank 已通知的 tail。
        int stored_scaleout_tail_idx = 0;  // 当前 lane 观察到的来源 scaleout tail。
        // round-robin 从 channel_idx 映射的 rank 开始，避免所有 forward warp 同时偏向 rank0。
        int recv_scaleout_rank_idx = channel_idx % kNumScaleoutRanks;  // round-robin 当前来源 rank。
        // wip_mask 标出哪些 scaleout rank 有新 token 或尚未 finish。
        uint32_t wip_mask;  // 仍有工作或未 finish 的 scaleout rank mask。
        // 只要还有任一 scaleout rank 未完成或有新 tail，就继续 forward。
        while ((wip_mask = ptx::gather(stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag == 0))) {  // 直到所有来源 rank finish 且无积压 slot。
            // Pick next rank in round-robin
            // 从上次处理 rank 的下一个 rank 开始找，保证公平。
            const auto offset = (recv_scaleout_rank_idx + 1) % kNumScaleoutRanks;  // 下一次选择起点。
            // hi_mask 是 offset 之后仍有工作的 rank 集合。
            const auto hi_mask = (wip_mask >> offset) << offset;  // offset 之后仍有工作的 rank。
            // 优先选 offset 后的第一个 rank，否则回绕到 wip_mask 的第一个 rank。
            recv_scaleout_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);  // 选出要消费的来源 rank。

            // Wait for this rank to have data (or finish)
            // 等待被选中的 scaleout rank tail 增长，或它已经 finish。
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {  // 等待选中来源 rank 有新 slot 或 finish。
                // 当前 lane 自己的 arrived/finished 状态。
                const uint32_t arrived_or_finished =  // 当前 lane 对应 rank 是否可处理。
                    stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag > 0;
                // 从 recv_scaleout_rank_idx 对应 lane 广播该 rank 是否可处理。
                if (ptx::exchange(arrived_or_finished, recv_scaleout_rank_idx))  // 从目标 rank lane 广播状态。
                    return true;

                // Timeout
                // 超时只让有效 scaleout lane 打印当前 tail 状态。
                // 中文：forwarding timeout 通常意味着某个 scaleout sender 未推进 tail。
                if (is_last_check) {
                    if (lane_idx < kNumScaleoutRanks) {
                        // 中文：最后一次检查仍未看到目标 tail 推进时，打印 forwarding 诊断。
                        printf("DeepEP hybrid dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, "
                               "channel: %d, lane: %d, old scale-out tail: %d, scale-out tail: (%d, %d)\n",
                               // 中文：前两个坐标定位当前 rank，channel/lane 定位被等待的转发通道。
                               scaleout_rank_idx, scaleup_rank_idx,  // 当前 rank 二维坐标。
                               // 中文：old tail 是本 forwarder 已消费位置，tail 二元组展示远端发布进度。
                               // 中文：lane 对应被检查的 scaleout rank。
                               channel_idx, lane_idx, stored_scaleout_old_tail_idx,  // channel、lane 和已消费 tail。
                               stored_finish_flag, stored_scaleout_tail_idx);  // finish 标志与最新 tail。
                    }
                    return false;
                }

                // Read new signaled tails
                // 有效 scaleout lane 读取自己对应来源 rank 的 signaled tail。
                if (lane_idx < kNumScaleoutRanks) {  // 有效 scaleout lane 读取自己的 signal。
                    // acquire_sys 与 scaleout sender 的 release/atomic 配对，保证看到数据后再消费。
                    const auto signaled_tail = ptx::ld_acquire_sys<int64_t>(  // 读取 finish/tail 打包信号。
                        workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx));
                    // signaled_tail 打包了 finish_flag 和 tail_idx。
                    math::unpack2<int, int64_t>(signaled_tail, stored_finish_flag, stored_scaleout_tail_idx);
                }
                // 更新 tail 后同步，再下一轮检查。
                __syncwarp();
                return false;
            });

            // Process one chunk from the current rank
            // start_slot_idx 是该来源 rank 本次要消费的起点。
            const auto start_slot_idx = ptx::exchange(stored_scaleout_old_tail_idx, recv_scaleout_rank_idx);  // 当前 chunk 起点。
            // end_slot_idx 不超过已通知 tail，且每次最多消费 kNumSlotsPerForwardChunk 个。
            // 这样 forward warp 可以在多个 scaleout 来源之间轮询，避免单个来源长队列饿死其它来源。
            // chunk 大小与 scaleout sender 的 tail 更新间隔一致，通常一个通知就对应最多 kNumSlotsPerForwardChunk 个新 slot。
            const auto end_slot_idx = std::min(  // 当前 chunk 终点。
                ptx::exchange(stored_scaleout_tail_idx, recv_scaleout_rank_idx),
                start_slot_idx + kNumSlotsPerForwardChunk
            );
            // 只有来源 rank 对应 lane 更新 old_tail。
            if (lane_idx == recv_scaleout_rank_idx)  // 只有来源 rank lane 更新 old tail。
                stored_scaleout_old_tail_idx = end_slot_idx;

            // recv_buffer 是当前来源 scaleout rank 在本 channel 的 RDMA 收包区。
            const auto recv_buffer = scaleout_recv_buffer.get_rank_buffer(recv_scaleout_rank_idx);  // 当前来源 scaleout 的 recv buffer。
            // 顺序消费 [start_slot_idx, end_slot_idx) 的 RDMA slot。
            for (int slot_idx = start_slot_idx; slot_idx < end_slot_idx; ++ slot_idx) {  // 顺序消费本 chunk 内 slot。
                // token_buffer 是 RDMA sender 已写入的完整 token 包。
                const auto token_buffer = recv_buffer.get_token_buffer(slot_idx);  // RDMA 收到的 token 包。

                // Wait TMA arrival
                // 复用 tma_buffer 前等待上一轮 store 完成。
                ptx::tma_store_wait();
                // 全 warp 同步后加载新 token。
                __syncwarp();

                // TMA load into shared memory
                // 一个 lane 把 RDMA recv token 从 global buffer 加载到 shared staging。
                if (ptx::elect_one_sync()) {  // leader lane 发起 TMA load。
                    ptx::tma_load_1d(tma_buffer.get_base_ptr(), token_buffer.get_base_ptr(),
                                     // 中文：mbarrier 记录本次 TMA load 的事务字节数，wait 会阻塞到 shared 数据可读。
                                     mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                }  // 中文：leader lane 完成当前 token 的 TMA load 等待。
                // shared token 完整到达。
                __syncwarp();

                // Read top-k indices
                EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
                // 当前 lane 对应目标 scale-up rank，默认 -1 表示不属于本 scaleout rank。
                int stored_dst_scaleup_rank_idx = -1;  // 当前 lane 解析出的目标 scaleup rank。
                // 从 token metadata 读取全局 expert id。
                auto dst_expert_idx = lane_idx < kNumTopk ? tma_buffer.get_topk_idx_ptr()[lane_idx] : -1;  // 当前 top-k expert。
                // 减去当前 scaleout 的 expert 起点，得到 scaleout 内部 expert id。
                // expert 全局 id 的区间按 scaleout 切分：[scaleout_rank*kNumExpertsPerScaleout, ...)。
                dst_expert_idx -= scaleout_rank_idx * kNumExpertsPerScaleout;
                // 只有落在当前 scaleout expert 区间内的 lane 才需要转发到某个 scale-up rank。
                stored_dst_scaleup_rank_idx = 0 <= dst_expert_idx and dst_expert_idx < kNumExpertsPerScaleout ?  // 本 scaleout 内目标才映射。
                    dst_expert_idx / kNumExpertsPerRank : -1;

                // Write the per-scaleup channel index for this token：为每个目标 scale-up rank 分配链表节点。
                int linked_list_idx = -1;  // 当前 token 对目标 scaleup rank 的链表位置。
                #pragma unroll
                // kNumScaleupRanksPerLane 覆盖所有 scale-up rank。
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {  // 遍历 lane 覆盖的 scaleup rank 组。
                    // src_lane_idx 表示目标 rank 在第 j 组 32 lane 中对应的 lane。
                    const auto src_lane_idx = stored_dst_scaleup_rank_idx - j * 32;  // 目标 rank 在组内 lane。
                    // valid 表示当前 lane/j 组合负责该目标 rank。
                    const bool valid = 0 <= src_lane_idx and src_lane_idx < 32;  // 当前 j 是否覆盖目标 rank。
                    // 从负责 lane 广播该 rank 当前发送计数，作为 linked_list_idx。
                    const auto exchanged = ptx::exchange(  // 从 owner lane 取该 rank 当前发送计数。
                        stored_scaleup_send_counters[j], valid ? src_lane_idx : 0);
                    // 只有 valid 组合更新 linked_list_idx。
                    // linked_list_idx 表示该 token 在目标 scale-up rank 当前 channel 队列中的逻辑位置。
                    linked_list_idx = valid ? exchanged : linked_list_idx;
                }
                // 非复用路径把链表节点写入 token，copy epilogue 会把该节点指向 recv token i。
                if (not kReuseSlotIndices and lane_idx < kNumTopk) {  // 新分配路径写 linked-list metadata。
                    // transform_linked_list_idx 会把无效/局部索引编码成 epilogue 能识别的形式。
                    tma_buffer.get_linked_list_idx_ptr()[lane_idx] = transform_linked_list_idx(linked_list_idx);
                    // fence 确保 linked_list_idx 写入对随后的 TMA store 可见。
                    ptx::tma_store_fence();
                }
                // 等待链表 metadata 写完。
                __syncwarp();

                // Deduplicate for scale-up ranks：同一个 token 对同一目标 scale-up rank 只写一次 payload。
                int stored_dst_slot_idx = -1;  // 当前 lane 对应的目标 scaleup slot。
                // dst_slot_idx_ptr 指向当前 channel/来源 scaleout/slot 的 top-k slot 记录。
                // 维度是 [channel, source_scaleout, slot, topk_lane]；cached/reuse 路径会用同一 top-k lane
                // 直接取回上一次分配的 scale-up slot，避免重新做 atomicAdd 和 linked-list 构造。
                const auto dst_slot_idx_ptr = dst_buffer_slot_idx +  // 当前 RDMA slot/top-k 的 slot 记录地址。
                    recv_scaleout_rank_idx * (kNumMaxTokensPerChannel * kNumTopk) + slot_idx * kNumTopk;
                // 复用路径直接读取上次 forward 写出的 slot。
                if constexpr (kReuseSlotIndices) {
                    // cached forward 保持 topk 到 slot 的映射不变，因此直接复用 dst_buffer_slot_idx。
                    if (lane_idx < kNumTopk)  // top-k lane 读取复用 slot。
                        stored_dst_slot_idx = __ldg(dst_slot_idx_ptr + lane_idx);
                } else {
                    // Deduplicate for NVLink ranks
                    // 非复用路径由去重后的目标 scale-up rank lane 原子分配 slot。
                    // deduplicate 保证同一 token 多个 top-k 若落到同一 scale-up rank，只分配一个 payload slot。
                    if (ptx::deduplicate(stored_dst_scaleup_rank_idx, lane_idx) and stored_dst_scaleup_rank_idx >= 0)  // 对目标 scaleup rank 去重。
                        stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_scaleup_rank_idx, 1);
                }
                // 等待 slot 分配完成。
                __syncwarp();

                // Issue TMAs：把 shared token 写入目标 scale-up rank 的 scaleup_buffer[slot]。
                if (stored_dst_slot_idx >= 0) {  // 当前 lane 负责写一个目标 scaleup slot。
                    // LSA symmetric pointer 把本地 scaleup_buffer 地址转换成目标 scale-up rank 地址。
                    const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(  // 目标 scaleup peer 的 symmetric 地址。
                        scaleup_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        stored_dst_scaleup_rank_idx);
                    // TMA store 完整 token payload+metadata。
                    // 这里写的是节点内/scaleup peer 的 buffer，后续 copy epilogue 从该 slot 拷到最终 recv_x。
                    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                    // 提交 TMA store。
                    ptx::tma_store_commit();
                }
                // 等待 store 发起完成。
                __syncwarp();

                // Add per-scale-up counter：更新每个目标 scale-up rank 已发送 token 数。
                EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Invalid number of scale-up peers");
                // rank 数 <=32 用 unsigned，否则用 64-bit mask。
                using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;  // scaleup rank mask 类型。
                // scaleup_send_mask 标出当前 token 命中的所有目标 scale-up rank。
                const auto scaleup_send_mask = ptx::reduce_or(  // 当前 token 命中的所有 scaleup rank。
                    stored_dst_scaleup_rank_idx >= 0 ?
                    (mask_t(1) << stored_dst_scaleup_rank_idx) : mask_t(0));
                #pragma unroll
                // 每个 lane/j 维护一个 rank 的 counter；mask 对应位为 1 则加一。
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)  // 更新各目标 scaleup rank 的发送计数。
                    stored_scaleup_send_counters[j] += (scaleup_send_mask >> (j * 32 + lane_idx)) & 1;

                // Record metadata at forward：非复用路径记录 slot 信息，下一轮可直接复用。
                if constexpr (not kReuseSlotIndices) {
                    EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of selections");
                    // 当前 processed 行的 metadata 起始地址。
                    const auto metadata_ptr = token_metadata_at_forward +  // 当前 metadata 行起点。
                        num_tokens_processed * kNumForwardMetadataDims;

                    // Source token index and last token index flag
                    // 第 0 列保存源 token 全局 id，第 1 列保存当前 chunk 是否最后一个 slot。
                    // chunk 尾标记会被 hybrid_combine 用来决定最后一次 Gin put 是否关闭 aggregate，
                    // 从而及时 flush 本 chunk 的网络写。
                    if (ptx::elect_one_sync()) {  // leader lane 写源 token 和 chunk 尾标记。
                        metadata_ptr[0] = tma_buffer.get_src_token_global_idx_ptr()[0];
                        metadata_ptr[1] = slot_idx == (end_slot_idx - 1);
                    }

                    // Second, original top-k indices and destination slots
                    // 第 2..2+kNumTopk-1 保存目标 scale-up rank；
                    // 第 2+kNumTopk.. 保存对应分配的 scale-up slot。
                    if (lane_idx < kNumTopk) {  // top-k lane 写 rank/slot 两段 metadata。
                        metadata_ptr[2 + lane_idx] = stored_dst_scaleup_rank_idx;
                        metadata_ptr[2 + kNumTopk + lane_idx] = stored_dst_slot_idx;
                        // 同时把 slot 写入 dst_buffer_slot_idx，供 kReuseSlotIndices 路径读取。
                        dst_slot_idx_ptr[lane_idx] = stored_dst_slot_idx;
                    }
                }
                // 当前 channel forward metadata 行数加一。
                num_tokens_processed += 1;
                // 等待 metadata 和计数更新完成。
                __syncwarp();
            }
        }

        // Assign the source token index part of the metadata into `-1` as an ending mark。
        // hybrid_combine 读取 token_metadata_at_forward 时以 -1 判断 channel 结束。
        if (not kReuseSlotIndices and ptx::elect_one_sync())
            token_metadata_at_forward[num_tokens_processed * kNumForwardMetadataDims] = -1;
        // 注意这里只写 metadata 第 0 维 source token idx；combine forward 读到 -1 后停止遍历本 channel。
        // 等待结束标记写入。
        __syncwarp();

        // Update linked list's ending position：写每个目标 scale-up rank 的链表尾节点位置。
        if constexpr (not kReuseSlotIndices) {
            // tail_ptr 是当前 channel/本 scale-up rank 记录尾节点的 workspace 指针。
            // 这里写的不是 token 数，而是“链表尾节点的全局 index”。copy epilogue 拿到它后会把尾节点 next 置为 -1，
            // 这样每个 scale-up rank 都能沿链表遍历属于自己的接收 token。
            const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
            #pragma unroll
            // 每个 lane/j 对应一个目标 scale-up rank 的尾 counter。
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                // j 是目标 scale-up rank。
                if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {  // j 是有效目标 scaleup rank。
                    // 写入尾节点的全局 linked-list index；epilogue 会把该节点置 -1。
                    ptx::st_relaxed_sys(
                        gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j),  // 目标 scaleup rank 的 channel tail 指针。
                        transform_linked_list_idx(stored_scaleup_send_counters[i]));  // 将本地 counter 转成链表全局 index。
                }
            }
        }
        // 等待 tail 写入完成。
        __syncwarp();

        // Clean tails for next usages：当前 forward 已消费所有 scaleout tail，清零 workspace。
        if (lane_idx < kNumScaleoutRanks)  // 每个 scaleout lane 清理一个 signal 槽。
            *workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx) = 0;
        // 等待清理完成。
        __syncwarp();
    }

    // Scale-up barrier to ensure data arrival
    // As scale-out tokens have already been consumed by forwarders, no need to do scale-out barrier again
    // 这里只需要 scale-up 子团队 barrier，保证 forward 写入 scaleup_buffer 后 epilogue 再读。
    // scaleout sender/forward 之间的同步已经通过 signaled_tail 完成；此处若再做 scaleout barrier 会多一次跨节点等待。
    // scaleup barrier 负责确保所有 LSA/TMA store 对同节点 peer 可见，并给 programmatic epilogue 一个安全起点。
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,  // dispatch epilogue 前的 scaleup 同步。
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag1, true, true, false>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, /* do not scale-out */ false, true);  // 只做 scaleup barrier。

    // Trigger the copy epilogue kernel：通知 dispatch_copy_epilogue 可以整理 scaleup_buffer。
    cudaTriggerProgrammaticLaunchCompletion();  // 触发 programmatic dependent launch 的 copy epilogue。

    // Clean scale-up counters
    // All scale-out counters should be cleaned before
    // forward 分支使用的 scaleup_atomic_sender_counter 在这里清零。
    EP_STATIC_ASSERT(kNumScaleupRanks <= kNumThreads, "Insufficient threads");
    // 只让 SM0 的前 kNumScaleupRanks 个线程写，避免重复清理。
    if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumScaleupRanks)  // 只在新分配路径清零 atomic sender counter。
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace deep_ep::elastic
