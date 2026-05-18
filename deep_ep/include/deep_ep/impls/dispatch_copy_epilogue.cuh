#pragma once  // dispatch copy epilogue 是 JIT include 的 device 实现头，防止重复定义模板。

// compiled.cuh 提供 topk_idx_t、sf_pack_t 等编译期类型别名。
#include <deep_ep/common/compiled.cuh>
// layout.cuh 负责把一段线性 buffer 解释成 hidden/SF/top-k/metadata 的结构化视图。
#include <deep_ep/common/layout.cuh>
// math.cuh 提供 constexpr_ceil_div、advance_ptr 等指针/整数工具。
#include <deep_ep/common/math.cuh>
// ptx.cuh 封装 TMA、mbarrier、warp shuffle、lane 选举等底层 PTX 操作。
#include <deep_ep/common/ptx.cuh>


// elastic 版本的 dispatch/combine kernel 都放在 deep_ep::elastic 命名空间下。
namespace deep_ep::elastic {  // Elastic/V2 device kernel 命名空间。

// kDoExpand 控制输出是否按 expert 展开；false 时一个输入 token 对应一行输出。
template <bool kDoExpand, bool kCachedMode,
          // kDoExpand=true 时把 token 按有效 expert lane 拆成多行；kCachedMode=true 时复用既有布局。
          // NOTES: this channel concept only applies for scale-out ranks
          // kNumSMs/kNumChannels/kNumWarps 描述 grid 与 scale-out channel 的并行形状。
          int kNumSMs, int kNumChannels, int kNumWarps,
          // kNumScaleoutRanks 是跨节点/跨 NVSHMEM 域 rank 数，kNumScaleupRanks 是节点内 rank 数。
          int kNumScaleoutRanks, int kNumScaleupRanks,
          // kNumHiddenBytes 是单 token hidden 向量字节数；kNumSFPacks 是 FP8 scale pack 个数。
          int kNumHiddenBytes, int kNumSFPacks,
          // 每个 rank 最多接收的 token 数，用于推导通信 buffer 的二维布局上限。
          int kNumMaxTokensPerRank,
          // kNumExperts 是全局 expert 数；kNumTopk 是每个 token 选择的 expert 数。
          int kNumExperts, int kNumTopk,
          // 全局 rank 数等于 scale-out 与 scale-up 两层 rank 的笛卡尔积。
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          // 一个 CTA/block 内线程数固定为 warp 数乘 32。
          int kNumThreads = kNumWarps * 32,
          // 每个 channel 的 token 上限用于 hybrid 链表和 channel 内 slot 编址。
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          // 只有多 scale-out 且非 cached 模式需要为 combine 建立 channel 链表。
          bool kDoCreateLinkedList = (kNumScaleoutRanks > 1 and not kCachedMode)>
__global__ void __launch_bounds__(kNumThreads, 1)  // 每个 block 使用 kNumThreads，且期望每 SM 至少驻留 1 个 block。
// buffer 是主 dispatch kernel 写好的 scale-up 接收缓冲；workspace 保存跨 kernel 状态。
dispatch_copy_epilogue_impl(void* buffer, void* workspace,
                            // psum_num_recv_tokens_per_scaleup_rank: [kNumScaleupRanks]，每个元素是 rank 维度前缀和。
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            // expanded 模式下，每个本地 expert 的输出行数通过这里原子递增。
                            int* psum_num_recv_tokens_per_expert,
                            // recv_x 是最终 hidden 输出，逻辑形状为 [num_recv_tokens 或 expanded_rows, hidden]。
                            void* recv_x, sf_pack_t* recv_sf,
                            // recv_topk_idx/weights 保存路由元信息，combine 反向聚合会再次读取。
                            topk_idx_t* recv_topk_idx, float* recv_topk_weights,
                            // recv_src_metadata 保存源 token 与 buffer slot 的反向路由信息。
                            int* recv_src_metadata,
                            // hybrid 非 cached 模式的 channel 链表，combine 沿它恢复 slot 顺序。
                            int* channel_linked_list,
                            // num_recv_tokens 可为真实值，也可为哨兵值触发设备端读取真实值。
                            int num_recv_tokens,
                            // recv_sf 的两个 stride 来自 PyTorch tensor，单位分别是 token 与 hidden pack 维度。
                            const int recv_sf_token_stride, const int recv_sf_hidden_stride,
                            // 当前 GPU 在两层拓扑中的坐标，用于把全局 expert 映射成本地 expert。
                            const int scaleout_rank_idx, const int scaleup_rank_idx) {
    // 本 epilogue 在主 dispatch kernel 之后运行：主 kernel 已把 token 写入通信 buffer，
    // epilogue 负责把分散在 rank/channel 布局里的数据整理成 PyTorch 输出张量。
    // 输出形状：
    //   recv_x: [num_recv_tokens, hidden]
    //   recv_sf: FP8 时为 [num_recv_tokens, hidden / 128] 的打包尺度，stride 由 Python 传入
    //   recv_topk_idx / recv_topk_weights: 非 expanded 为 [num_recv_tokens, num_topk]
    //   recv_src_metadata: [num_recv_tokens, 2 + num_topk]，供 combine 反向路由使用。
    // sm_idx 是当前 CTA 编号；thread_idx 是 CTA 内线性线程编号。
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);  // block/线程线性编号。
    // warp_idx/lane_idx 把 CTA 内线程拆成 warp 与 lane，后续大多按 warp 协作处理一个 token。
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();  // CTA 内 warp id 与 lane id。
    // global_warp_idx 把所有 CTA 的同名 warp 拉平成全局 warp id，用作 token 循环起点。
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;

    // 每个 rank 拥有连续的 expert 区间。收到的全局 expert id 会被转换成本 rank 内
    // 局部 id；例如 num_experts=64、num_ranks=8 时 rank3 管 [24,32)，expert 27
    // 会被写成局部 expert 3。非本 rank 的 top-k lane 会写 -1。
    // 每个 rank 平均拥有 kNumExperts/kNumRanks 个 expert，本文件假设 expert 均匀切分。
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;  // 每 rank 均分到的 expert 数。
    // rank_idx 把二维拓扑坐标压成全局 rank id：scaleout 为高位，scaleup 为低位。
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;  // 当前 rank 的全局线性 id。
    // 当前 rank 负责的 expert 区间为 [expert_start_idx, expert_end_idx)。
    const auto expert_start_idx = kNumExpertsPerRank * rank_idx, expert_end_idx = kNumExpertsPerRank * (rank_idx + 1);  // 当前 rank 管辖的 expert 半开区间。

    // shared memory 中每个 warp 准备一个 token 临时区，TMA 从 scaleup_buffer 加载
    // 完整 token，再写回连续输出。scaleup_buffer 的逻辑维度是
    // [kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank]。
    // smem 按 TMA 对齐要求声明，避免 tma_load/tma_store 对齐不满足导致运行时错误。
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];  // warp 私有 TMA staging shared memory。
    // TokenLayout 描述单个 token 的内存布局：hidden bytes、SF bytes、top-k 个数，并带 mbarrier/链表字段。
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);  // 单 token 布局。
    // tma_buffer 给每个 warp 分配一个 shared-memory token 暂存区，逻辑形状 [kNumWarps, 1]。
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    // scaleup_buffer 是 global-memory 输入，逻辑形状 [scaleup_rank, scaleout_rank*kNumMaxTokensPerRank]。
    const auto scaleup_buffer = layout::BufferLayout<false>(token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);  // dispatch 主 kernel 写入的输入 layout。

    // Init TMA
    // phase 记录 mbarrier 当前 phase，每次等待后翻转，匹配 PTX mbarrier 协议。
    ptx::arrival_phase phase = 0;  // 当前 mbarrier parity phase。
    // mbarrier 放在当前 warp 的 shared token 区，用来等待 TMA load 完成。
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();  // 当前 warp staging buffer 内的 mbarrier。
    // 一个 warp 只需要一个 lane 初始化 mbarrier，elect_one_sync 会稳定选出一个参与 lane。
    if (ptx::elect_one_sync())
        // expected arrivals=1，因为只有一次 TMA transaction 到达该 barrier。
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    // 确保同 warp 其他 lane 在使用 mbarrier 前看到初始化结果。
    __syncwarp();

    // Will block until the main dispatch kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    // PDL 依赖同步：等待主 dispatch kernel 完成，使 global buffer 的写入对 epilogue 可见。
    cudaGridDependencySynchronize();

    // 如果调用方没有先把 num_recv_tokens 从 GPU 同步到 CPU，会传入最大哨兵值；
    // 此时 epilogue 直接读 prefix-sum 的最后一项得到真实 token 数，避免一次 CPU 同步。
    // 约定最大容量值作为“未知真实 token 数”的哨兵。
    if (num_recv_tokens == kNumMaxTokensPerRank * kNumRanks)
        // prefix-sum 最后一项就是所有 scale-up rank 的总接收 token 数。
        num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[kNumScaleupRanks - 1];

    // psum_num_recv_tokens_per_scaleup_rank 是按 scale-up rank 的前缀和，例如
    // [5, 9, 12] 表示 rank0 有 5 个、rank1 有 4 个、rank2 有 3 个 token。
    // current_rank_start/end 用这个前缀和把全局接收序号 i 映射回来源 scale-up rank。
    // current_rank_idx 表示当前全局输出序号 i 落在哪个来源 scale-up rank。
    int current_rank_idx = -1, stored_psum_num_recv_tokens;  // 当前来源 rank 与临时 prefix 值。
    // current_rank_start/end 表示该 rank 在全局输出序号空间中的半开区间。
    int current_rank_start = 0, current_rank_end = 0;  // 当前来源 rank 的 token 序号区间。
    #pragma unroll
    // 每个全局 warp 以 stride=kNumWarps*kNumSMs 处理一串 recv token。
    for (int i = global_warp_idx; i < num_recv_tokens; i += kNumWarps * kNumSMs) {
        // i 超出当前 rank 区间时，推进到下一个 rank 的 prefix-sum 终点。
        while (i >= current_rank_end) {
            // 先移动 rank 指针，再读取这个 rank 的累计结束位置。
            current_rank_idx += 1;
            // 防御性检查：prefix-sum 区间数量不应超过 scale-up rank 数。
            EP_DEVICE_ASSERT(current_rank_idx < kNumScaleupRanks);
            // 每个 rank 的 prefix 值由该 rank_idx 对应 lane 读取，再通过 warp exchange 广播。
            const auto stored_lane_idx = current_rank_idx % 32;  // 保存当前 prefix 的 lane id。
            // lane0/32/64... 批量读取一组 rank 的 prefix，避免所有 lane 重复读同一地址。
            if (stored_lane_idx == 0 and current_rank_idx + lane_idx < kNumScaleupRanks)
                stored_psum_num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[current_rank_idx + lane_idx];
            // 上一个 end 成为当前 rank 的 start。
            current_rank_start = current_rank_end;
            // 从 stored_lane_idx 对应 lane 取出当前 rank 的 prefix end。
            current_rank_end = ptx::exchange(stored_psum_num_recv_tokens, stored_lane_idx);
        }
        // 在当前来源 rank 的 buffer 中，局部 token 下标是 i-current_rank_start。
        const auto buffer_token = scaleup_buffer.get_rank_buffer(current_rank_idx).get_token_buffer(i - current_rank_start);  // 输入 buffer 中的源 token。

        // 等待之前可能发出的 TMA store 全部释放，防止复用 shared token 暂存区时覆盖未提交数据。
        ptx::tma_store_wait();
        // 保证整个 warp 都完成 store wait 后再发起新的 TMA load。
        __syncwarp();

        // 发起 TMA load：一次把 hidden、SF、top-k metadata 等单 token 结构搬到 shared memory。
        if (ptx::elect_one_sync()) {
            // 只有一个 lane 提交 TMA 指令；目标是 warp 私有的 tma_buffer。
            ptx::tma_load_1d(tma_buffer.get_base_ptr(), buffer_token.get_base_ptr(),
                             mbarrier_ptr, tma_buffer.get_num_bytes<false>());
            // 告诉 mbarrier 本次 transaction 的字节数，后面 wait 会等它完成。
            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, tma_buffer.get_num_bytes<false>());
        }
        // 其他 lane 等提交 lane 完成 TMA 发射，随后可以并行做元数据预读。
        __syncwarp();

        // 这里直接从 global buffer 读取 top-k expert id，而不是等 TMA 完成后从 shared 读；
        // 这样可以用 metadata 处理掩盖一部分 TMA load 延迟。
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        // 默认 -1 表示当前 lane 没有有效 top-k 目标。
        int dst_expert_idx = -1;  // 当前 lane 对应的目标 local expert，-1 表示无效。
        // 前 kNumTopk 个 lane 分别读取一个 top-k expert id。
        if (lane_idx < kNumTopk)
            dst_expert_idx = buffer_token.get_topk_idx_ptr()[lane_idx];
        // 确保所有 lane 已经得到自己负责的 expert id，后面 gather/match 需要 warp 内一致。
        __syncwarp();

        // 判断当前 top-k expert 是否属于本 rank 管辖的 expert 区间。
        const auto in_range = expert_start_idx <= dst_expert_idx and dst_expert_idx < expert_end_idx;  // 当前 top-k 是否属于本 rank。
        // master_src_topk_idx 是本 rank 命中的第一个 top-k lane，用于 metadata 和链表索引。
        const auto master_src_topk_idx = ptx::get_master_lane_idx(ptx::gather(in_range));  // 命中本 rank 的代表 top-k lane。
        // 把全局 expert id 转成本 rank 内局部 expert id；非本 rank 的 lane 统一置 -1。
        dst_expert_idx = in_range ? dst_expert_idx - expert_start_idx : -1;
        // 同一个 token 不允许在本 rank 内命中重复 expert，否则 expanded/combine 语义会歧义。
        EP_DEVICE_ASSERT(ptx::deduplicate(dst_expert_idx, lane_idx) or dst_expert_idx == -1);
        // 非 expanded 模式保留 [token, topk] 形状，逐 lane 写回本地 expert id 或 -1。
        if (not kDoExpand and lane_idx < kNumTopk)
            recv_topk_idx[i * kNumTopk + lane_idx] = static_cast<topk_idx_t>(dst_expert_idx);
        // 等待 top-k id 与断言相关的 warp 操作完成，再进入输出行号分配。
        __syncwarp();

        // expanded 模式会把一个 token 的多个有效 top-k lane 拆成多行输出，所以需要
        // atomicAdd 到对应 expert 的 prefix 计数；非 expanded 模式保持一收一出，dst=i。
        // dst_tensor_idx 是最终 recv_x/recv_sf/metadata 中的输出行号。
        int dst_tensor_idx = -1;  // 最终输出张量行号，expanded 模式按 expert lane 分配。
        // 非 expanded 模式一进一出，由选举 lane 把输出行设为全局接收序号 i。
        if (not kDoExpand and ptx::elect_one_sync()) {
            dst_tensor_idx = i;
        // expanded 模式中每个有效 expert lane 都要得到一行独立输出。
        } else if (kDoExpand and dst_expert_idx >= 0) {
            // psum_num_recv_tokens_per_expert 初始为 expert 的 prefix 起点，atomicAdd 返回本 lane 的唯一行号。
            dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);
        }
        // 确保每个 lane 的 dst_tensor_idx 分配完成，再等待 TMA 数据到达。
        __syncwarp();

        // 等待当前 token 的 TMA load 到达 shared memory。
        if (ptx::elect_one_sync())
            // wait 后 phase 翻转，下一轮可以复用同一个 mbarrier。
            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
        // 所有 lane 等待 shared token 完整可读。
        __syncwarp();

        // hybrid 非 cached 模式下，dispatch 阶段为每个 channel/scaleup rank 建链；
        // combine 之后沿链表反向找到“第几个 slot 对应哪个 recv token”。这里把链表
        // 中间节点写成当前输出 i，尾节点在函数末尾统一写 -1。
        if constexpr (kDoCreateLinkedList) {
            // linked_list_idx 由 dispatch 写入 shared token，指向 channel_linked_list 中待填的节点。
            if (ptx::elect_one_sync())
                channel_linked_list[tma_buffer.get_linked_list_idx_ptr()[master_src_topk_idx]] = i;
            // 写链表后同步，避免后续复用 shared token 影响 master_src_topk_idx 关联的字段。
            __syncwarp();
        }

        // 把 hidden 向量从 shared memory 连续写入最终输出 recv_x。
        if (kDoExpand ? (dst_tensor_idx >= 0) : ptx::elect_one_sync()) {
            // expanded 模式每个有效 lane 负责自己的行；非 expanded 只由一个 lane 代表整个 token 发 store。
            ptx::tma_store_1d(math::advance_ptr(recv_x, static_cast<int64_t>(dst_tensor_idx) * kNumHiddenBytes),
                              tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
            // commit 把本 warp 已提交的 TMA store 推入异步队列。
            ptx::tma_store_commit();
        }
        // 确保所有 lane 在 SF 与 metadata 写入前都已经完成 hidden store 发射。
        __syncwarp();

        // Store SF：FP8 场景下，scale factor pack 与 hidden 分开存放，需要普通 lane 写回。
        if constexpr (kNumSFPacks > 0) {
            // 每轮 32 个 lane 共搬 32 个 sf_pack，完整轮数为 kNumSFPacks/32。
            constexpr auto kNumFullIters = kNumSFPacks / 32;  // 每 warp 完整写 32 个 SF pack 的轮数。
            // 余数轮只允许 lane_idx 落在剩余 pack 范围内的 lane 访问。
            const bool do_last_iter = (kNumSFPacks % 32 != 0) and (kNumFullIters * 32 + lane_idx < kNumSFPacks);  // 是否处理尾部不足 32 的 SF pack。
            // sf_pack_t 必须能用至少 32-bit 对齐写回，避免未对齐全局访存。
            EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");

            // 先把 shared SF 读到寄存器，后面 expanded 模式可能要把同一份 SF 写到多个输出行。
            const auto smem_src_ptr = tma_buffer.get_sf_ptr();  // shared token 中 SF pack 起点。
            // +1 为可能存在的余数轮保留寄存器槽位。
            sf_pack_t reg_src[kNumFullIters + 1];  // 当前 lane 缓存的 SF pack，+1 容纳尾部轮。
            #pragma unroll
            // 完整轮：lane l 读取第 k*32+l 个 SF pack。
            for (int k = 0; k < kNumFullIters; ++ k)
                reg_src[k] = smem_src_ptr[k * 32 + lane_idx];
            // 余数轮：只读真实存在的尾部 SF pack。
            if (do_last_iter)
                reg_src[kNumFullIters] = smem_src_ptr[kNumFullIters * 32 + lane_idx];

            // stride 从 Python tensor 传入，转 int64 防止大 tensor 偏移溢出 32-bit。
            const auto recv_sf_token_stride_i64 = static_cast<int64_t>(recv_sf_token_stride);
            // hidden stride 表示相邻 SF pack 在目标 tensor hidden 维上的间隔。
            const auto recv_sf_hidden_stride_i64 = static_cast<int64_t>(recv_sf_hidden_stride);

            // expanded 模式可能多个 lane 有有效 dst_tensor_idx；非 expanded 只需要 lane0 对应的一行。
            auto mask = kDoExpand ? ptx::gather(dst_tensor_idx >= 0) : 1;  // 需要写 SF 的 expanded lane 集合。
            // 逐个有效 lane 广播其 dst_tensor_idx，并由全 warp 合作写那一行 SF。
            while (mask) {
                // __ffs 返回最低置位 bit 的 1-based 下标，因此减 1 得到 lane id。
                const int valid_lane_idx = __ffs(mask) - 1;  // 当前要广播输出行号的有效 lane。
                // 从 valid_lane_idx 广播目标行号，定位到该 token 的 SF 起始地址。
                const auto gmem_dst = math::advance_ptr<sf_pack_t>(recv_sf,
                    ptx::exchange(dst_tensor_idx, valid_lane_idx) * (recv_sf_token_stride_i64 * sizeof(sf_pack_t)));
                #pragma unroll
                // 每个 lane 写自己负责的 SF pack 到带 stride 的 PyTorch tensor。
                for (int k = 0; k < kNumFullIters; ++ k)
                    gmem_dst[(k * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[k];
                // 尾部 pack 只由有效 lane 写，避免越界访问 recv_sf。
                if (do_last_iter)
                    gmem_dst[(kNumFullIters * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[kNumFullIters];
                // 清除已经处理的 lane，继续处理下一个 expanded 输出行。
                mask ^= 1 << valid_lane_idx;
            }
        }

        // 写 top-k weight：forward 可选，backward/combine 会用它做加权回传。
        if (kDoExpand and recv_topk_weights != nullptr and dst_tensor_idx >= 0) {
            // expanded 模式每个有效 lane 对应一行，因此权重是一维 [expanded_rows]。
            recv_topk_weights[dst_tensor_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        } else if (not kDoExpand and recv_topk_weights != nullptr and lane_idx < kNumTopk) {
            // For backward, weights are optional
            // 非 expanded 模式保持二维展平布局 [num_recv_tokens, kNumTopk]。
            recv_topk_weights[i * kNumTopk + lane_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        }
        // 等待所有 weight 写入结束，再写 metadata，保持输出张量视图一致。
        __syncwarp();

        // recv_src_metadata 第 0 列保存原始 token 全局 index；第 1 列保存 combine
        // 需要的路由位置信息：
        //   - 单层模式：source_scaleup_rank * num_topk + master_topk_lane
        //   - hybrid 模式：slot_idx * num_topk + master_topk_lane
        // 第 2.. 列在 expanded 模式下记录每个 top-k lane 拆分后的 dst_tensor_idx。
        // 每个 token 的 metadata 行长度固定为 2+kNumTopk，expanded 情况后 kNumTopk 列存拆分行号。
        constexpr int kMetadataStride = 2 + kNumTopk;  // recv_src_metadata 每个 token 的列数。
        // 前两列只需要一个 lane 写，避免重复写同一地址。
        if (ptx::elect_one_sync()) {
            // 第 0 列：原始输入 token 的全局下标，combine 回填原始顺序时使用。
            recv_src_metadata[i * kMetadataStride + 0] = *tma_buffer.get_src_token_global_idx_ptr();
            // 单层 scale-up 下，slot 由来源 rank 与 top-k lane 唯一确定。
            if constexpr (kNumScaleoutRanks == 1) {
                recv_src_metadata[i * kMetadataStride + 1] = current_rank_idx * kNumTopk + master_src_topk_idx;
            } else {
                // 多 scale-out 下，slot 是当前 rank 内 token 局部序号乘 top-k，再加命中 lane。
                recv_src_metadata[i * kMetadataStride + 1] = (i - current_rank_start) * kNumTopk + master_src_topk_idx;
            }
        }
        // 等待前两列写完，再由各 lane 写 expanded 的 reduction source index。
        __syncwarp();

        // expanded 模式下，第 2+lane 列记录该 top-k lane 被展开后的输出行号；无效 lane 为 -1。
        if (kDoExpand and lane_idx < kNumTopk)
            recv_src_metadata[i * kMetadataStride + 2 + lane_idx] = dst_tensor_idx;
        // 当前 token 的全部输出写入完成，下一轮可以复用 shared token 区。
        __syncwarp();
    }

    // 链表尾部使用 -1 作为终止标记；同时把 workspace 中的 tail 指针清零，给下一次
    // dispatch/combine 复用。这个清理和 hybrid_combine.cuh 的链表读取逻辑强关联。
    if constexpr (kDoCreateLinkedList) {
        // 一个 lane 可能负责多个 scale-up rank，ceil_div 保证覆盖全部 rank。
        constexpr int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32);  // 每个 lane 负责的 scaleup rank 批数。
        // workspace_layout 暴露 channel_scaleup_tail_ptr，尾指针由 dispatch 阶段维护。
        const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);  // 解析 dispatch workspace 控制区。
        // 所有 warp/SM 分摊 kNumChannels 个链表尾节点的收尾工作。
        for (int i = global_warp_idx; i < kNumChannels; i += kNumSMs * kNumWarps) {
            #pragma unroll
            // 每个 lane 处理若干 scale-up rank 的 tail。
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                // k 是当前 lane/j 对应的 scale-up rank；最后一轮要防止越界。
                if (const auto k = j * 32 + lane_idx; j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks) {
                    // 把链表最后一个节点写成 -1，combine 读到 -1 即停止遍历。
                    channel_linked_list[
                        *workspace_layout.get_channel_scaleup_tail_ptr(i, k)
                    ] = -1;

                    // Clean for combine usages
                    // tail 指针清零，避免下一次 dispatch 看到上一次遗留的尾节点地址。
                    *workspace_layout.get_channel_scaleup_tail_ptr(i, k) = 0;
                }
            }
            // 同一 channel 的所有 rank tail 处理完再进入下一个 channel。
            __syncwarp();
        }  // 中文：结束当前 channel 的 linked-list tail 初始化。
    }  // 中文：结束所有 channel 的 epilogue metadata 初始化。
}

}  // namespace deep_ep::elastic
