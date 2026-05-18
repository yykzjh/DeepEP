#pragma once  // deterministic prologue device 实现由 JIT include，防止重复定义模板。

// cooperative_groups 用于跨整个 cooperative grid 做同步，保证所有 SM 的计数都可见。
#include <cooperative_groups.h>

// compiled.cuh 提供 topk_idx_t 等编译期类型别名。
#include <deep_ep/common/compiled.cuh>
// math.cuh 提供 ceil_div 等整数工具。
#include <deep_ep/common/math.cuh>
// ptx.cuh 封装 warp match/shuffle/lane id 等底层操作。
#include <deep_ep/common/ptx.cuh>


// elastic dispatch 的确定性 prologue 放在 elastic 命名空间。
namespace deep_ep::elastic {  // Elastic/V2 deterministic dispatch prologue 命名空间。

// TODO: support scale-out。当前实现只覆盖 scale-up 域，也就是单 scale-out rank 内的稳定 slot 分配。
template <int kNumSMs, int kNumWarps,
          // kNumSMs/kNumWarps 决定全局 warp 数，也决定 token 连续分片的粒度。
          // kNumScaleupRanks 是节点内目标 rank 数，slot 的高维前缀按这个 rank 维度统计。
          int kNumScaleupRanks,
          // 每个 rank 的 buffer 最大 token 容量；最终 slot 会加上 scaleup_rank_idx*kNumMaxTokensPerRank。
          int kNumMaxTokensPerRank,
          // kNumExperts/kNumScaleupRanks 推导每个 rank 管辖的 expert 数。
          int kNumExperts, int kNumTopk,
          // 每个 block 线程数固定为 warp 数乘 32。
          int kNumThreads = kNumWarps * 32>
__global__ void __launch_bounds__(kNumThreads, 1)  // 每个 CTA 使用固定线程数，至少驻留 1 个 block。
dispatch_deterministic_prologue_impl(
    // topk_idx 逻辑形状 [num_tokens, kNumTopk]，每个元素是全局 expert id。
    topk_idx_t* topk_idx,
    // rank_count_buffer 逻辑形状 [kNumSMs, kNumScaleupRanks]，存每个 SM 发往各 rank 的计数。
    int* rank_count_buffer,
    // dst_buffer_slot_idx 逻辑形状 [num_tokens, kNumTopk]，输出每个 top-k lane 的目标 buffer slot。
    int* dst_buffer_slot_idx,
    // num_tokens 是当前 dispatch 输入 token 数。
    const int num_tokens,
    // scaleup_rank_idx 是本 GPU 的 rank id，用于把本地 slot 转成全局 buffer slot。
    const int scaleup_rank_idx
) {
    // deterministic prologue 只支持 scale-up 域，用于在真正 dispatch 前预先计算
    // 每个 token/top-k lane 应写入的目标 slot。dst_buffer_slot_idx 形状为
    // [num_tokens, kNumTopk]，元素为 scaleup_rank_idx*kNumMaxTokensPerRank+slot，
    // 无效或重复 rank lane 写 -1。
    // 每个 scale-up rank 管辖相同数量 expert，例如 64 experts / 8 ranks = 每 rank 8 个。
    constexpr int kNumExpertsPerRank = kNumExperts / kNumScaleupRanks;
    // expert 必须能均匀切分到 rank，否则 expert->rank 的整数除法映射会出错。
    EP_STATIC_ASSERT(kNumExperts % kNumScaleupRanks == 0, "Invalid number of experts or ranks");

    // Utils：把 CUDA 线程坐标拆成 block、warp、lane 三层。
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    // warp_idx 是 block 内 warp 编号；lane_idx 是 warp 内 0..31 的 lane 编号。
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    // global_warp_idx 用于给每个 warp 分配一段连续 token。
    const auto global_warp_idx = sm_idx * kNumWarps + warp_idx;

    // 每个 warp 负责一段连续 token，先统计 rank 计数，再二次扫描写 slot；
    // 连续分片保证同一输入在不同运行中得到相同 slot 顺序。
    // 每个 warp 获得 ceil(num_tokens / 全局warp数) 个 token，保证覆盖所有 token。
    const auto num_tokens_per_warp = math::ceil_div(num_tokens, kNumSMs * kNumWarps);
    // 当前 warp 的起始 token，下标按输入顺序连续。
    const auto start_token_idx = global_warp_idx * num_tokens_per_warp;
    // 结尾 token 不超过 num_tokens，最后几个 warp 可能为空。
    const auto end_token_idx = min(start_token_idx + num_tokens_per_warp, num_tokens);

    // 一个 warp 内把 32 个 lane 切成若干 token group。若 topk=8，则
    // kNumTokensPerGroup=4，每 8 个 lane 处理一个 token 的 top-k。
    // 一个 warp 同时处理 kNumTokensPerGroup 个 token；每个 token 占 kNumTopk 个 lane。
    constexpr int kNumTokensPerGroup = 32 / kNumTopk;
    // token_idx_offset 表示当前 lane 属于本 warp group 中的第几个 token。
    const auto token_idx_offset = lane_idx / kNumTopk;
    // token_mask 只覆盖同一 token 的 top-k lane，用来在 ptx::match 结果中屏蔽其他 token。
    const unsigned token_mask = ((1u << kNumTopk) - 1) << (token_idx_offset * kNumTopk);
    // 本实现假设一个 token 的 top-k 可以放在一个 warp 内。
    EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k");

    // shared memory 分三段：
    // rank_count_global_psum [kNumScaleupRanks]
    // rank_count_warp_sum [kNumWarps, kNumScaleupRanks]
    // rank_count_warp_psum [kNumScaleupRanks]
    // 第一段汇总前序 SM 的计数，第二段保存本 block 每个 warp 的局部计数。
    // smem 是 int 数组视图，下面三个逻辑区域首尾相接。
    extern __shared__ int8_t smem[];
    // rank_count_global_psum[rank] 保存当前 SM 之前所有 SM 对该 rank 的累计 token 数。
    const auto rank_count_global_psum = math::advance_ptr<int>(smem, 0);
    // rank_count_warp_sum 指向“当前 warp”的局部计数行，布局总共是 [kNumWarps, kNumScaleupRanks]。
    const auto rank_count_warp_sum = math::advance_ptr<int>(rank_count_global_psum, (kNumScaleupRanks + warp_idx * kNumScaleupRanks) * sizeof(int));
    // rank_count_warp_psum[rank] 保存当前 warp 写 slot 时的动态前缀计数。
    const auto rank_count_warp_psum = math::advance_ptr<int>(rank_count_warp_sum, kNumWarps * kNumScaleupRanks * sizeof(int));

    // 初始化 shared memory 的三个区域，元素总数为 kNumScaleupRanks*(1+2*kNumWarps)。
    for (int i = thread_idx; i < kNumScaleupRanks * (1 + 2 * kNumWarps); i += kNumThreads)
        // smem 按 int 使用；这里每个线程 stride 初始化，避免旧 block 残留数据影响计数。
        reinterpret_cast<int*>(smem)[i] = 0;
    // block 内所有 shared 计数清零后才能开始 rank 统计。
    __syncthreads();

    // Util functions：把 expert id 映射到 scale-up rank；负 expert 表示无效。
    const auto map_expert_to_rank_idx = [&](const int& expert_idx) {
        // expert 连续均分给 rank，因此整数除法即可得到目标 rank。
        return expert_idx >= 0 ? expert_idx / kNumExpertsPerRank : -1;
    };
    // 判断当前 lane 是否是同一 token 内同一 rank 的第一个 lane。
    const auto is_unique = [&](const int& rank_idx) {
        // ptx::match(rank_idx) 返回 warp 内 rank_idx 相同的 lane mask，再用 token_mask 限定到当前 token。
        return ((ptx::match(rank_idx) & token_mask) >> lane_idx) == 1;
    };
    // 统计 mask 中 bit_idx 之前有多少个 1，用于计算同一 rank 内的 lane 局部偏移。
    const auto count_ones_before = [&](const unsigned& mask, const int& bit_idx) {
        // 例如 mask=0b10110, bit_idx=4，则统计低 4 位 0b0110 中的 1 个数为 2。
        return __popc(mask & ((1u << bit_idx) - 1));
    };
    // 取得另一个 warp 的 rank_count_warp_sum 行，用于跨 warp 前缀求和。
    const auto get_other_rank_count_warp_sum = [&](const int& other_warp_idx) {
        // NOTES: pass negative num_bytes to advance pointer
        // rank_count_warp_sum 当前指向本 warp 行，因此用相对偏移 other_warp_idx-warp_idx。
        return math::advance_ptr<int>(rank_count_warp_sum, (other_warp_idx - warp_idx) * kNumScaleupRanks * sizeof(int));
    };

    // 第一遍扫描：只统计“每个 warp 会向每个 rank 写多少个 token”。
    for (int i = start_token_idx; i < end_token_idx; i += kNumTokensPerGroup) {
        // token_idx 是当前 lane 所属 token 的真实输入下标。
        const auto token_idx = i + token_idx_offset;
        // active lane 必须位于本组 kNumTopk*kNumTokensPerGroup 范围内，且 token 未越过 end。
        const auto is_active_thread = lane_idx < kNumTopk * kNumTokensPerGroup and token_idx < end_token_idx;
        // 只有 active lane 读取 topk；无效 lane 置 -1，后续 rank 映射自然失效。
        const int expert_idx = is_active_thread ? static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
        // 将 expert id 转成目标 rank id；无效 expert 得到 -1。
        const auto rank_idx = map_expert_to_rank_idx(expert_idx);

        // 一个 token 的多个 expert 可能落在同一 rank，只需要发送一次 token。
        // is_unique 保留该 rank 的第一个 lane，其他 lane deduped_rank_idx=-1。
        // 例如 topk expert 对应 rank [2,2,5,7]，只有第一个 rank2 lane 会保留 2。
        const auto deduped_rank_idx = is_unique(rank_idx) ? rank_idx : -1;
        // 找出 warp 内所有 deduped_rank_idx 相同的 lane；-1 lane 也会匹配，但后面会过滤。
        const auto rank_idx_mask = ptx::match(deduped_rank_idx);

        // 只让这一组相同 rank 的第一个 lane 更新计数；表达式右移后当前 lane 位为 1 表示它是最低位。
        if ((rank_idx_mask >> lane_idx) == 1 and deduped_rank_idx >= 0)
            // __popc(rank_idx_mask) 是这一批 token-group 中要发往该 rank 的去重 token 数。
            rank_count_warp_sum[deduped_rank_idx] += __popc(rank_idx_mask);
    }
    // 等待所有 warp 的局部计数完成，后面 block 汇总会读取每个 warp 的行。
    __syncthreads();

    // 先把本 block 每个 rank 的计数写到全局 rank_count_buffer
    // [kNumSMs, kNumScaleupRanks]，再 cooperative grid sync，让所有 SM 计数可见。
    for (int rank_idx = thread_idx; rank_idx < kNumScaleupRanks; rank_idx += kNumThreads) {
        // rank_count_block_sum 汇总当前 SM/block 内所有 warp 发往 rank_idx 的 token 数。
        int rank_count_block_sum = 0;
        // 遍历 block 内每个 warp 的局部计数行。
        for (int i = 0; i < kNumWarps; i++)
            // get_other_rank_count_warp_sum(i)[rank_idx] 取第 i 个 warp 对该 rank 的计数。
            rank_count_block_sum += get_other_rank_count_warp_sum(i)[rank_idx];
        // 写入全局计数矩阵，后续所有 SM 都会读取它来计算全局前缀。
        rank_count_buffer[sm_idx * kNumScaleupRanks + rank_idx] = rank_count_block_sum;
    }
    // cooperative grid sync 保证所有 SM 都已经写完 rank_count_buffer。
    cooperative_groups::this_grid().sync();

    // 计算当前 SM 之前所有 SM 对每个 rank 的累计计数，得到本 SM 的全局起始 slot。
    for (int rank_idx = lane_idx; rank_idx < kNumScaleupRanks; rank_idx += 32) {
        // rank_count 是 rank_idx 在所有前序 SM 中累计占用的 slot 数。
        int rank_count = 0;
        // warp_idx 维度分摊前序 SM 的求和，避免单线程串行扫描全部 block。
        for (int i = warp_idx; i < sm_idx; i += kNumWarps)
            // rank_count_buffer 行维度是 SM，列维度是 rank。
            rank_count += rank_count_buffer[i * kNumScaleupRanks + rank_idx];
        // 多个 warp 可能同时贡献同一个 rank_idx，因此用 block 级 atomicAdd 合并到 shared psum。
        atomicAdd_block(rank_count_global_psum + rank_idx, rank_count);
    }
    // 等待全局前缀 rank_count_global_psum 全部写完。
    __syncthreads();

    // 计算当前 warp 的初始 prefix：前序 SM + 当前 SM 内前序 warp。
    for (int rank_idx = lane_idx; rank_idx < kNumScaleupRanks; rank_idx += 32) {
        // 从前序 SM 的全局前缀开始。
        int rank_count = rank_count_global_psum[rank_idx];
        // 加上当前 block 内所有早于本 warp 的局部计数。
        for (int i = 0; i < warp_idx; i++)
            // 这样每个 warp 对每个 rank 都有一个稳定且互不重叠的起始 slot。
            rank_count += get_other_rank_count_warp_sum(i)[rank_idx];
        // 写到 shared，第二遍扫描时会边写 slot 边递增。
        rank_count_warp_psum[rank_idx] = rank_count;
    }
    // warp 内同步足够，因为每个 rank_idx 由某个 lane 写，随后同 warp 读取。
    __syncwarp();

    // 第二遍扫描：根据 prefix 给每个有效 token/top-k lane 写确定性的目标 slot。
    for (int i = start_token_idx; i < end_token_idx; i += kNumTokensPerGroup) {
        // 与第一遍保持完全相同的 token/lane 映射，保证计数和写 slot 对齐。
        const auto token_idx = i + token_idx_offset;
        // active 条件也必须与第一遍一致，否则 prefix 与实际写入数量会不匹配。
        const auto is_active_thread = lane_idx < kNumTopk * kNumTokensPerGroup and token_idx < end_token_idx;
        // 读取当前 top-k lane 的 expert id；无效 lane 仍置 -1。
        const auto expert_idx = is_active_thread ? static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
        // 映射到目标 rank。
        const auto rank_idx = map_expert_to_rank_idx(expert_idx);

        // 避免同一 token 对同一 rank 重复发送：只保留该 token 内第一个命中的 top-k lane。
        const auto deduped_rank_idx = is_unique(rank_idx) ? rank_idx : -1;
        // rank_idx_mask 标出本 warp group 中所有相同 deduped_rank_idx 的 lane。
        const auto rank_idx_mask = ptx::match(deduped_rank_idx);

        // rank_count_warp_psum[rank] 是当前 warp 对该 rank 的起始 slot；
        // count_ones_before(rank_idx_mask,lane) 给出同一 group 内当前 lane 前面已有几个同 rank token。
        // 因此 stored_dst_slot_idx = rank 前缀 + 当前 lane 在这批同 rank lane 中的局部序号。
        const auto stored_dst_slot_idx = deduped_rank_idx >= 0 ?
                                         rank_count_warp_psum[deduped_rank_idx] + count_ones_before(rank_idx_mask, lane_idx) : -1;
        // value 把 rank 内 slot 转成通信 buffer 的全局 slot；-1 表示这个 top-k lane 不写入。
        const auto value = stored_dst_slot_idx >= 0 ?
                           scaleup_rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
        // 对 active lane 写出结果；非 active lane 不碰输出，避免越界到 token 尾部之外。
        if (is_active_thread)
            dst_buffer_slot_idx[i * kNumTopk + lane_idx] = value;

        // 当前 group 写完后，将该 rank 的 prefix 向后推进本 group 消耗的 slot 数。
        if ((rank_idx_mask >> lane_idx) == 1 and deduped_rank_idx >= 0)
            // 与第一遍同样只由最低 lane 更新一次，推进量为相同 rank lane 的个数。
            rank_count_warp_psum[deduped_rank_idx] += __popc(rank_idx_mask);
        // 每一批 token group 后同步，保证 rank_count_warp_psum 的递增对本 warp 下一批可见。
        __syncwarp();
    }
}

}  // namespace deep_ep::elastic
