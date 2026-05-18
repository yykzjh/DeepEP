#pragma once  // combine 公共工具会被多个 JIT device 实现 include，防止重复定义。

#include <stdexcept>  // constexpr helper 找不到展开因子时抛 logic_error。


namespace deep_ep::elastic {  // elastic device 端 combine 工具所在命名空间。
// Elastic combine device 工具命名空间。

template <bool kAllowMultipleReduction, int kNumRanks, int kNumTopk>  // 模板参数描述是否允许同 rank 多路归约、rank 数和 top-k 数。
constexpr bool use_rank_layout() {  // 编译期决定 combine 临时 buffer 第一维采用 rank 还是 top-k。
    // 多路 reduction 时，如果 rank 数 <= topk，用 rank 作为第一维可以避免不同
    // top-k lane 映射到同一 rank 后发生覆盖；否则使用 topk 维，节省 buffer 空间。
    if constexpr (not kAllowMultipleReduction)  // 不允许同 rank 多 expert 合并时，直接使用 top-k layout。
        // 不允许多路 reduction 时，每个 top-k slot 独立保存即可，不需要按 rank 合并。
        return false;
    return kNumRanks <= kNumTopk;  // rank 数不多于 top-k 时，用 rank layout 能避免覆盖且不会增加空间。
    // rank 数小于等于 top-k 时使用 rank layout，避免同 rank 多 expert 覆盖并保持空间可控。
}

template <bool kAllowMultipleReduction, int kNumRanks, int kNumTopk>  // 与 use_rank_layout 使用同一组编译期形状。
constexpr int get_num_tokens_in_layout() {  // 返回临时 layout 的槽位数量。
    // 返回通信 buffer 第一维槽位数：rank layout 用 rank 数，否则用 top-k 数。
    // 这个值会传给 BufferLayout 的 num_ranks 字段，虽然语义可能是 rank 或 top-k slot。
    return use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>() ? kNumRanks : kNumTopk;
}

template <int kLength, int kMaxUnrollFactor, int kWarpSize = 32>  // kLength 通常是 hidden_vec，warp 默认 32 lane。
constexpr int get_max_unroll_factor() {  // 为 hidden 维循环选择最大安全展开因子。
    // 选择最大的展开因子，使 hidden 向量长度能被 warp_size * unroll 整除。
    // 这样每个 lane 的循环次数完全一致，不需要处理最后一个不完整 tile。
    for (int i = kMaxUnrollFactor; i >= 1; -- i)  // 从大到小试探，优先暴露更多 ILP。
        // 从最大展开因子往下试，优先选择更大的 ILP。
        if (kLength % (kWarpSize * i) == 0)
            // hidden 向量数必须能被 warp_size*unroll 整除。
            return i;
    throw std::logic_error("Invalid length, cannot find unrolling factor");
    // 理论上 host wrapper 已限制 hidden，本分支表示模板参数不受支持。
}

// Determine the vector type for combine loads/stores based on arch and hidden size alignment
template <int kHiddenBytes>  // hidden 字节数用于判断全 hidden 向量是否满足更宽向量类型对齐。
struct CombineVecTraits {  // combine load/store 向量类型选择器。
    // CombineVecTraits 根据架构和 hidden 对齐选择向量化 load/store 类型。
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // 只有 device SM100+ 编译路径才尝试 256-bit 向量。
    // SM100+ 支持 256-bit LD/ST。hidden 对齐时用 longlong4_t 一次搬 32B，否则回退
    // int4 的 16B，避免未对齐访问拖慢或触发非法指令。
    static constexpr bool kUseLonglong4 = (kHiddenBytes % sizeof(longlong4_t) == 0) and
                                          ((kHiddenBytes / sizeof(longlong4_t)) % 32 == 0);
    using vec_t = std::conditional_t<kUseLonglong4, longlong4_t, int4>;  // 根据对齐结果选择 32B 或 16B 向量。
    // hidden 对齐到 32B 且 warp 分片也对齐时使用 longlong4_t，否则使用 int4。
#else  // SM90 及 host 解析路径没有 longlong4_t 优化分支。
    using vec_t = int4;  // 默认使用 16B int4，兼容所有当前 CUDA 架构。
    // SM90 及以下保守使用 16B int4 向量。
#endif
};

template <int kNumValidTopk, typename fetch_func_t>  // kNumValidTopk 是压缩后最多保留的有效输入路数。
__device__ __forceinline__
void compute_topk_slots(int (&topk_slot_idx)[kNumValidTopk], uint32_t mask,  // 输出 slot 数组和输入 lane mask。
                        const fetch_func_t& fetch_func) {  // fetch_func 把 lane id 映射到实际 slot。
    // mask 中的 1 表示有效 lane。这里按从低到高的 lane 顺序压缩成 slot 数组；
    // fetch_func 可把 lane id 转换成 rank slot 或 top-k slot。即使 lowest_idx=-1
    // 也执行 fetch，是为了减少 warp 内分支发散。
    #pragma unroll
    for (int k = 0; k < kNumValidTopk; ++ k) {
        // 每次取 mask 中最低有效 lane，构造压缩后的第 k 个 slot。
        const int lowest_idx = __ffs(mask) - 1;
        // __ffs 返回 1-based bit index；mask=0 时 lowest_idx=-1。
        // Here we perform the exchange unconditionally to avoid `BRA.DIV`
        const auto fetched = fetch_func(lowest_idx);  // 将最低有效 lane 转为 rank/top-k slot。
        // 即使 lowest_idx=-1 也调用 fetch_func，减少 warp 内分支。
        mask &= mask - 1;
        // 清掉最低的 1 bit。
        topk_slot_idx[k] = lowest_idx >= 0 ? fetched : -1;
        // 没有更多有效 lane 时填 -1，后续 ldg_with_gez_pred 会跳过。
    }
}

template <int kHiddenVec, int kUnrollFactor, int kNumExpectedTopk, int kNumValidTopk,  // hidden 向量数、展开因子和输入路数。
          typename vec_t, typename get_src_buffer_ptr_func_t, typename wait_buffer_func_t>  // 向量类型、源地址 getter 和等待回调。
__device__ __forceinline__
void combine_reduce(const int& lane_idx, int (&topk_slot_idx)[kNumValidTopk],  // lane id 与已压缩输入 slot 列表。
                    vec_t* dst_buffer_ptr,  // 输出 shared staging buffer。
                    const get_src_buffer_ptr_func_t& get_src_buffer_ptr_func,  // 按 slot 返回源 buffer 指针。
                    const wait_buffer_func_t& wait_buffer_func,  // 首次写前等待 staging buffer 可用。
                    vec_t* bias_0 = nullptr, vec_t* bias_1 = nullptr) {  // 可选 bias，布局与输出 token hidden 一致。
    // combine_reduce 在一个 warp 内沿 hidden 维对多个 top-k/rank slot 做加和，
    // 输出写入 dst_buffer_ptr 指向的 shared staging buffer。
    constexpr int kNumElemsPerVec = sizeof(vec_t) / sizeof(nv_bfloat16);
    // 一个向量类型包含多少个 BF16 元素，例如 int4=8、longlong4_t=16。
    EP_STATIC_ASSERT(kNumElemsPerVec % 2 == 0, "Invalid number of elements");
    // 后续用 nv_bfloat162/float2 pair 规约，元素数必须为偶数。
    EP_STATIC_ASSERT(kHiddenVec % (kUnrollFactor * 32) == 0, "Invalid unrolling");
    // 每个 warp tile 必须整除 hidden 向量数。

    // 当只有一到两路输入且无 bias 时，直接使用 BF16 向量加法，避免 BF16->FP32->BF16
    // 的转换成本；多路或带 bias 时才进入 FP32 accumulator 路径保证数值正确。
    const bool enable_hadd_bypass =
        (bias_0 == nullptr and bias_1 == nullptr) and
        (kNumValidTopk <= 2 or topk_slot_idx[2] < 0);
    // 只有最多两路输入且无 bias 时可用 BF16 hadd bypass。
    EP_STATIC_ASSERT(kNumValidTopk > 0, "Invalid top-k");
    // 至少要有一个可能的输入 slot。

    if (enable_hadd_bypass) {  // 进入最多两路输入的 BF16 直接相加快路径。
        // 快路径：最多两路 BF16 直接相加，避免 FP32 accumulator。
        // 注意该路径牺牲一点多路累加精度换取速度，但仅限最多两路输入且无 bias。
        #pragma unroll 1
        for (int i = 0; i < kHiddenVec / (kUnrollFactor * 32); ++ i) {  // 每轮覆盖 kUnrollFactor*32 个 vec_t。
            // i 遍历 hidden 的 warp tile。
            // Read values 0
            const auto slot_0 = topk_slot_idx[0];
            // 第一条输入 slot，可能为 -1。
            const auto src_base_ptr_0 = get_src_buffer_ptr_func(slot_0);
            // 获取 slot_0 对应的源 buffer 起点。
            vec_t values_0[kUnrollFactor] = {};
            // 当前 lane 读取的第一路向量数组。
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                // 每个 lane 展开读取多个向量。
                values_0[j] = ptx::ldg_with_gez_pred(
                    src_base_ptr_0 + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), slot_0);
                // slot_0<0 时 predicate 阻止非法读取并返回零值。
            }

            // Read values 1
            vec_t values_1[kUnrollFactor] = {};
            // 第二路输入，若不存在保持零。
            const auto slot_1 = kNumValidTopk == 1 ? -1 : topk_slot_idx[1];
            // top-k 只有 1 时强制第二路无效。
            const auto src_base_ptr_1 = get_src_buffer_ptr_func(slot_1);
            // 获取第二路源 buffer 起点。
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                // 读取第二路 hidden 分片。
                values_1[j] = ptx::ldg_with_gez_pred(
                    src_base_ptr_1 + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), slot_1);
                // slot_1<0 时返回零。
            }

            // Wait buffer releases for the first write
            if (i == 0)  // 只在第一次写当前 token 的 staging buffer 前等待一次。
                // 第一次写 dst 前等待上游 buffer release，避免覆盖仍在被 TMA 读取的数据。
                wait_buffer_func();

            // Reduce into shared memory
            const auto bf162_view_0 = reinterpret_cast<nv_bfloat162*>(values_0);
            // 把第一路向量解释成 BF16 pair 视图。
            const auto bf162_view_1 = reinterpret_cast<nv_bfloat162*>(values_1);
            // 把第二路向量解释成 BF16 pair 视图。
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                // 对每个展开向量做 pairwise BF16 加法。
                #pragma unroll
                for (int l = 0; l < kNumElemsPerVec / 2; ++ l)  // 遍历 vec_t 内所有 BF16 pair。
                // 每个 vec_t 包含 kNumElemsPerVec/2 个 BF16 pair。
                bf162_view_0[j * (kNumElemsPerVec / 2) + l] += bf162_view_1[j * (kNumElemsPerVec / 2) + l];
                // BF16 pair 加法直接在 values_0 上原地完成。
                dst_buffer_ptr[i * (kUnrollFactor * 32) + j * 32 + lane_idx] = values_0[j];
                // 写入 shared staging buffer。
            }
        }
    } else {  // 进入通用 FP32 累加路径，支持多路输入和 bias。
        // 通用路径：多路输入或带 bias 时使用 FP32 accumulator，最后转回 BF16。
        #pragma unroll 1
        for (int i = 0; i < kHiddenVec / (kUnrollFactor * 32); ++ i) {  // 每次处理一个 warp tile。
            // 遍历 hidden warp tile。
            // Add bias
            float2 reduced[kUnrollFactor * kNumElemsPerVec / 2] = {};
            // FP32 pair accumulator，覆盖当前 lane 的所有展开元素。
            const auto add_bias = [&](const vec_t* base_ptr) {  // device lambda：读取一份 bias 并加到 accumulator。
                // 把 bias 向量读入并累加到 reduced。
                // base_ptr 指向 bias_0 或 bias_1 的 hidden 向量起点，布局与 x 完全一致。
                // Read
                vec_t values[kUnrollFactor];
                // 暂存 bias 向量。
                #pragma unroll
                for (int j = 0; j < kUnrollFactor; ++ j)  // 读取当前 lane 展开的多个 bias 向量。
                    // 读取当前 tile 的第 j 个向量。
                    values[j] = ptx::ldg(base_ptr + i * (kUnrollFactor * 32) + j * 32 + lane_idx);

                // Reduce
                const auto bf162_view = reinterpret_cast<nv_bfloat162*>(values);
                // bias 向量按 BF16 pair 解释。
                #pragma unroll
                for (int j = 0; j < kUnrollFactor * kNumElemsPerVec / 2; ++ j)
                    // 累加所有 BF16 pair 到 FP32 accumulator。
                    ptx::accumulate(reduced[j], bf162_view[j]);
            };
            bias_0 != nullptr ? add_bias(bias_0) : void();
            // bias_0 存在时先加 bias_0。
            bias_1 != nullptr ? add_bias(bias_1) : void();
            // bias_1 存在时再加 bias_1。
            // 这里用三元表达式是为了保持 device lambda 内联且避免额外 if block 干扰展开。

            #pragma unroll
            for (int k = 0; k < kNumValidTopk; ++ k) {  // 遍历压缩后的每一路输入贡献。
                // 依次累加每个有效 top-k/rank slot。
                // We have a limitation on `k` to reduce the branch instruction count
                if (k >= kNumExpectedTopk and topk_slot_idx[k] < 0)  // 期望路数之外遇到无效 slot 就提前停止。
                    // 超过期望 top-k 且当前 slot 无效时提前退出，减少无意义循环。
                    break;

                // Read values
                const auto src_base_ptr = get_src_buffer_ptr_func(topk_slot_idx[k]);
                // 获取第 k 路输入 slot 的源 buffer。
                // topk_slot_idx[k] 可能为 -1，后续 predicate load 会把该路当作全零输入。
                vec_t values[kUnrollFactor] = {};
                // 暂存第 k 路输入向量。
                #pragma unroll
                for (int j = 0; j < kUnrollFactor; ++ j) {
                    // 每个 lane 按 hidden stride 读取多个向量。
                    values[j] = ptx::ldg_with_gez_pred(
                        src_base_ptr + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), topk_slot_idx[k]);
                    // slot 无效时 predicate 返回零，避免非法读。
                }

                // Reduce
                const auto bf162_view = reinterpret_cast<nv_bfloat162*>(values);
                // 输入向量解释为 BF16 pair。
                #pragma unroll
                for (int j = 0; j < kUnrollFactor * kNumElemsPerVec / 2; ++ j)
                    // 把 BF16 pair 累加到 FP32 accumulator。
                    ptx::accumulate(reduced[j], bf162_view[j]);
            }

            // Wait buffer releases for the first write
            if (i == 0)  // 首次写 staging buffer 前等待消费者释放。
                // 首次写前等待目标 staging buffer 可写。
                wait_buffer_func();

            // Cast into shared memory
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {  // 把当前 lane 的每个展开向量写回 shared。
                // 把 FP32 accumulator 转回 BF16 向量。
                vec_t casted_value;
                // 当前输出向量。
                auto bf162_view = reinterpret_cast<nv_bfloat162*>(&casted_value);
                // 输出向量的 BF16 pair 视图。
                // casted_value 未初始化没关系，下面会覆盖它的所有 BF16 pair。
                #pragma unroll
                for (int l = 0; l < kNumElemsPerVec / 2; ++ l)  // 遍历 vec_t 内全部 BF16 pair。
                    // 每个 pair 从 float2 round-to-nearest 转 BF16。
                    bf162_view[l] = __float22bfloat162_rn(reduced[j * (kNumElemsPerVec / 2) + l]);
                dst_buffer_ptr[i * (kUnrollFactor * 32) + j * 32 + lane_idx] = casted_value;
                // 写回 shared staging buffer，后续通常由 TMA store 发到全局内存。
            }
        }  // 中文：结束当前 unroll 段的 BF16 pair 写回。
    }  // 中文：结束当前 lane 负责的 reduce 写回循环。
}

}  // namespace deep_ep::elastic
// Elastic combine 工具命名空间结束。
