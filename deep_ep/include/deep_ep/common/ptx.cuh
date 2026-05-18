#pragma once  // PTX helper 被所有 Elastic device kernel include，防止重复定义。

#include <cuda_bf16.h>  // nv_bfloat16/nv_bfloat162 类型和转换 intrinsic。

#include <deep_ep/common/compiled.cuh>   // longlong4_t、全局编译配置。
#include <deep_ep/common/exception.cuh>  // EP_STATIC_ASSERT 等检查宏。

namespace deep_ep::elastic::ptx {
// Elastic device PTX/intrinsic helper 命名空间。
// 这个头文件集中封装“CUDA C++ 不容易直接表达”的底层指令：
// - TMA/mbarrier：用于 dispatch/combine 中 shared memory 与 global memory 的异步搬运；
// - acquire/release/system scope LD/ST：用于跨 GPU 发布 ring tail、flag、prefix 等控制字；
// - predicate load/store 和 warp intrinsic：用于 top-k 无效 lane、rank 去重、prefix sum 等细粒度控制。
// 调用点通常只依赖这些 helper 的语义，不直接写 PTX，从而把复杂的内存顺序和指令约束集中在这里。

// host 侧占位结构和 device mbarrier 保持相同大小/对齐，layout 代码在 host/JIT
// 生成阶段也会计算 sizeof(mbarrier)，因此不能只在 __CUDA_ARCH__ 下定义。
struct alignas(8) mbarrier { uint64_t __placeholder; };
// host/device 共用的 mbarrier 占位类型；真实操作通过 PTX 指令完成。
using arrival_phase = uint32_t;
// mbarrier parity phase 类型，通常在 0/1 间翻转。

// TMA 最小 16B，但 SM100 的 longlong4 256-bit 访存要求 32B 对齐；统一用 32B，
// 可以让 TMA buffer 和向量化 combine buffer 共用同一套 layout。
static constexpr int kNumTMAAlignBytes = 32;
// DeepEP 统一要求 TMA/shared staging 至少 32B 对齐。

#ifdef __CUDACC__  // 只有 CUDA 编译器路径才真正暴露 device intrinsic 和内联 PTX。

/// Exceptions
__forceinline__ __device__ void trap() {
    // 触发设备端异常，常用于超时后终止 kernel。
    asm volatile("trap;");  // 直接发出 PTX trap 指令，GPU 会中止当前 kernel 并向 host 报错。
}

/// Thread layout
__forceinline__ __device__ int get_warp_idx() {
    // 返回当前线程所在 warp id，并通过 shfl 保证同 warp 所有 lane 使用 lane0 计算结果。
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);  // threadIdx.x/32 是 warp id，lane0 广播可避免每 lane 重复依赖表达式。
}

__forceinline__ __device__ int get_lane_idx() {
    // 读取硬件 lane id，范围 [0, 31]。
    int lane_idx;  // 保存 %laneid 特殊寄存器读取结果。
    // PTX 输出寄存器。
    asm volatile("mov.s32 %0, %laneid;" : "=r"(lane_idx));  // "=r" 表示输出到 32-bit 通用寄存器。
    // 从特殊寄存器 %laneid 读取 lane 编号。
    return lane_idx;
    // 返回 lane id。
}

/// Election
__forceinline__ __device__ int elect_one_sync() {
#ifndef DISABLE_SM90_FEATURES  // 默认启用 SM90+ elect.sync；兼容模式下走 lane0 回退。
    // elect.sync 会在一个 warp 中选出一个活动 lane，比固定 lane0 更适合有谓词
    // 分支的路径；禁用 SM90 特性时回退到 lane0。
    int pred = 0;  // 先置 0；只有被 elect 的 lane 会在 PTX 中写成 1。
    // pred=1 表示当前 lane 被 elect.sync 选中。
    // 下面这段内联 PTX 的结构是：声明临时通用寄存器 rx 和谓词 px，
    // elect.sync 在给定 mask 内选出一个活跃 lane，并只在被选中的 lane 上把 pred 写成 1。
    asm volatile(
        "{\n"
        // 中文：rx/px 是内联 PTX 的临时寄存器声明，使用 %% 转义给编译器。
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        // 中文：内联 PTX 中声明的寄存器只在花括号作用域内有效，避免与外部 asm 冲突。
        // 中文：elect.sync 写 rx/px；这里只关心 px 是否选中了当前 lane。
        // 中文：rx 是被选 lane 的 lane id，px 是“当前 lane 是否被选中”的谓词。
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        // 中文：pred 既是输入也是输出，+r 约束让编译器分配可读写寄存器。
        : "+r"(pred)
        : "r"(0xffffffff));  // 使用全 warp mask；不活跃 lane 不会参与 elect。
    // 在完整 warp mask 下选择一个活跃 lane。
    return pred;
    // 只有一个活跃 lane 返回 1。
#else  // 禁用 SM90 特性时没有 elect.sync，使用 lane0 作为唯一 leader。
    return get_lane_idx() == 0;  // 返回 bool 会隐式转成 int，lane0 为 1，其它 lane 为 0。
    // 旧架构/禁用 SM90 特性时固定 lane0。
#endif  // DISABLE_SM90_FEATURES 分支结束。
}

/// TMA and `cp.async`
__forceinline__ __device__ void mbarrier_init_with_fence(mbarrier* ptr, const int& arrive_count = 1) {
    // 初始化 shared memory mbarrier，并用 fence 确保初始化对 cluster/CTA 可见。
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::
                 "r"(arrive_count), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));  // ptr 必须是 shared address，arrive_count 是参与者数量。
    // arrive_count 是等待该 barrier 的 producer/arrival 数。
    asm volatile("fence.mbarrier_init.release.cluster;" ::);  // release.cluster 保证 cluster 内其它 CTA 能看到初始化完成。
    // release fence 保证 mbarrier 初始化完成。
}

__forceinline__ __device__ void mbarrier_invalidate(mbarrier* ptr) {
    // 使 mbarrier 失效，通常在 shared memory 区域复用前调用。
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));  // invalidate 后这块 shared memory 可安全复用。
}

__forceinline__ __device__ void mbarrier_arrive(mbarrier* ptr) {
    // 对 mbarrier 做一次普通 arrive，不设置 TMA expected bytes。
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));  // "_" 表示不接收 arrive token。
}

__forceinline__ __device__ void mbarrier_arrive_and_set_tx(mbarrier* ptr, const int& num_bytes) {
    // expect_tx 告诉 mbarrier 本次 TMA 将写入多少字节；等待方只有在数据到达并且
    // arrive 计数满足时才会继续。
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::
                 "r"(num_bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));  // expect_tx 把 TMA 字节数绑定到 barrier。
    // num_bytes 必须和对应 TMA load/store 的事务字节数一致。
}

__forceinline__ __device__ void mbarrier_wait_and_flip_phase(mbarrier* ptr, arrival_phase& phase) {
    // 等待 mbarrier 达到指定 parity phase，然后翻转 phase，供下次复用。
    // mbarrier.try_wait.parity 的第三个立即数不是 DeepEP 语义层面的超时；
    // 它只是 PTX 指令的等待提示，真正的通信超时由外层 comm::timeout_while 或 clock64 逻辑控制。
    asm volatile(
        "{\n\t"  // 汇编作用域开始。
        ".reg .pred       P1; \n\t"  // 声明等待谓词寄存器。
        "LAB_WAIT: \n\t"  // 自旋等待标签。
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"  // 按 phase 尝试等待 mbarrier。
        "@P1 bra DONE; \n\t"  // 满足条件则跳出。
        "bra     LAB_WAIT; \n\t"  // 未满足则继续自旋。
        "DONE: \n\t"  // 等待完成标签。
        "}" ::  // 汇编作用域结束。
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "r"(phase), "r"(0x989680));
    // 0x989680 是 try_wait 的 spin hint/timeout immediate，实际超时由外层控制。
    phase ^= 1;
    // 当前 stage 下一次复用时使用相反 phase。
}

__forceinline__ __device__ void tma_store_fence() {
    // shared 写入后、TMA store 读取前使用 proxy fence，解决 async proxy 可见性。
    asm volatile("fence.proxy.async.shared::cta;");  // proxy fence 处理普通 shared 写与 async/TMA proxy 之间的可见性。
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait() {
    // 等待 TMA store group，模板参数表示允许剩余多少未完成 group。
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(kNumRemainingWaits) : "memory");  // "n" 要求模板常量立即数，memory clobber 防止编译器重排。
}

enum TMACacheHint: int64_t {  // TMA/ldg 使用的 L2 cache hint literal 枚举。
    // TMA cache hint 常量，直接传给 PTX 指令的 L2::cache_hint operand。
    kEvictFirst = 0x12f0000000000000ll,  // PTX cache hint literal：evict-first。
    // 数据短暂使用，倾向更早淘汰。
    kEvictNormal = 0x1000000000000000ll  // PTX cache hint literal：normal eviction。
    // 正常缓存策略。
};

__forceinline__ __device__ void tma_load_1d(
    const void* dst_ptr, const void* src_ptr, mbarrier* ptr, const int& num_bytes,
    const TMACacheHint& hint = TMACacheHint::kEvictFirst) {
    // global -> shared 的 bulk copy。dispatch/combine 通常只短暂使用载入内容，
    // 默认 evict-first，减少对普通 global load 的 L2 污染。
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr))),  // shared 目标地址。
        "l"(src_ptr),  // global 源地址。
        "r"(num_bytes),  // 传输字节数。
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),  // 完成信号 mbarrier。
        "l"(hint)  // L2 cache hint literal。
        : "memory");
    // 指令完成由 ptr 指向的 mbarrier 标记，不会立即阻塞当前线程。
}

__forceinline__ __device__ void tma_store_1d(
    const void* dst_ptr, const void* src_ptr, const int& num_bytes,
    const TMACacheHint& hint = TMACacheHint::kEvictNormal) {
    // shared -> global 的 bulk copy。写出的通信 buffer 往往马上被后续 epilogue
    // 或对端读取，默认 evict-normal。
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::
                 "l"(dst_ptr),  // global 目标地址。
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),  // shared 源地址。
                 "r"(num_bytes),  // 传输字节数。
                 "l"(hint)  // L2 cache hint literal。
                 : "memory");
    // store 被加入 bulk group，后续需 tma_store_commit/tma_store_wait 保证完成。
}

__forceinline__ __device__ void tma_store_commit() {
    // 提交当前 TMA bulk store group。
    asm volatile("cp.async.bulk.commit_group;");  // commit 后该 group 才能被 wait_group 观察。
}

template <class dtype_t>
__forceinline__ __device__ void cp_async_ca(const dtype_t* gmem_src, const dtype_t* smem_dst) {
    // cp.async.ca 从 global 到 shared 复制 4/8/16B，缓存策略为 cache-all。
    EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8 or sizeof(dtype_t) == 16, "Invalid dtype bytes");
    // cp.async 支持的单次复制粒度。
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], %2;\n" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst))),  // shared 目标地址。
                 "l"(gmem_src),  // global 源地址。
                 "n"(sizeof(dtype_t)));  // 复制字节数必须是编译期立即数。
    // smem_dst 需要转换成 shared address。
}

__forceinline__ __device__ void cp_async_mbarrier_arrive(mbarrier* ptr) {
    // cp.async 完成后对 mbarrier arrive。
    asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));  // 让 cp.async pipeline 对 mbarrier arrive。
}

/// Barriers
template <int kNumThreads>
__forceinline__ __device__ void named_barrier(const int& idx) {
    // 等价于 barrier.sync.aligned：同一 named barrier 上的所有线程必须以相同控制流到达，
    // 因此只在 warp/CTA 分工非常明确的 kernel 中使用。
    asm volatile("bar.sync %0, %1;" ::"r"(idx), "r"(kNumThreads));  // idx 是 named barrier id，kNumThreads 是参与线程数。
    // idx 是 named barrier 编号，kNumThreads 是参与线程数。
}

/// LD/ST instructions
__forceinline__ __device__ int4 ldg_with_gez_pred(const int4* ptr, const int& value, const TMACacheHint& cache_hint = TMACacheHint::kEvictFirst) {
    // 带谓词的只读 global load：value >= 0 时加载，否则返回 0。combine_reduce 用它
    // 避免无效 top-k slot 分支，同时保证无效 slot 不参与求和。
    // 例子：topk_idx == -1 表示该路 expert 无效，传入 value=-1 后不会访问 ptr。
    int4 ret = make_int4(0, 0, 0, 0);  // predicate false 时返回零向量，避免无效 lane 影响归约。
    // 这里使用 inline PTX 是因为 CUDA intrinsic 没有直接提供“带谓词 + L2 hint + nc”的组合。
    asm volatile(
        "{\n\t"
        // p 是 PTX 谓词寄存器，后续 @p 让 load 指令在硬件层面被屏蔽。
        "  .reg .pred p;\n\t"
        // setp.ge.s32 将 value >= 0 的判断写入 p，对应函数名中的 gez。
        "  setp.ge.s32 p, %5, 0;\n\t"
        // v4.s32 一次读取 16B，正好对应 int4；L1 no_allocate 避免流式数据污染 L1。
        "  @p ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s32 {%0, %1, %2, %3}, [%4], %6;\n\t"
        "}"
        : "+r"(ret.x), "+r"(ret.y), "+r"(ret.z), "+r"(ret.w)
        : "l"(ptr), "r"(value), "l"(cache_hint)  // ptr 是 global 地址，value 用来生成谓词，cache_hint 传给 L2 hint。
        : "memory");
    return ret;  // 返回加载到的 int4，或 predicate false 时的零向量。
}

__forceinline__ __device__ int4 ldg_with_gtz_pred(const int4* ptr, const int& value, const TMACacheHint& cache_hint = TMACacheHint::kEvictFirst) {
    // value > 0 时执行 non-coherent global load，否则返回零 int4。
    // gtz 常用于“计数大于 0 才读取 payload”的路径，比 gez 多排除 0 这个空状态。
    int4 ret = make_int4(0, 0, 0, 0);  // predicate false 时默认返回零。
    // 默认零值用于 predicate false。
    asm volatile(
        "{\n\t"
        // 用 predicate 生成条件读，避免在 CUDA C++ 层产生额外分支和地址访问。
        "  .reg .pred p;\n\t"
        // setp.gt.s32 对应 value > 0，适合正数长度/正数计数判断。
        "  setp.gt.s32 p, %5, 0;\n\t"
        // nc load 表示 non-coherent 路径，和 DeepEP 对共享 staging buffer 的流式读取模式匹配。
        "  @p ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s32 {%0, %1, %2, %3}, [%4], %6;\n\t"
        "}"
        : "+r"(ret.x), "+r"(ret.y), "+r"(ret.z), "+r"(ret.w)
        : "l"(ptr), "r"(value), "l"(cache_hint)  // value > 0 时才触发 load。
        : "memory");
    return ret;
    // 返回加载结果或零。
}

__forceinline__ __device__ int4 ld_with_gez_pred(const int4* ptr, const int& value, const TMACacheHint& cache_hint = TMACacheHint::kEvictFirst) {
    // value >= 0 时执行普通 global load，否则返回零 int4。
    // 和 ldg_with_gez_pred 的区别是这里没有 .nc，适合需要普通 global 一致性语义的读。
    int4 ret = make_int4(0, 0, 0, 0);  // predicate false 时默认返回零。
    // 默认零值。
    asm volatile(
        "{\n\t"
        // p 为 per-thread predicate，每个 lane 可以独立决定是否真的发起 load。
        "  .reg .pred p;\n\t"
        // value 常来自 top-k 或 head 编码，-1 这类哨兵值会使读被跳过。
        "  setp.ge.s32 p, %5, 0;\n\t"
        // 使用 cache_hint 让调用点能选择 evict-first/normal 等 L2 策略。
        "  @p ld.L1::no_allocate.L2::cache_hint.global.v4.s32 {%0, %1, %2, %3}, [%4], %6;\n\t"
        "}"
        : "+r"(ret.x), "+r"(ret.y), "+r"(ret.z), "+r"(ret.w)
        : "l"(ptr), "r"(value), "l"(cache_hint)  // 普通 global load 带 cache_hint。
        : "memory");
    return ret;
    // 返回加载结果或零。
}

#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // SM100+ 支持更宽的 longlong4_t 256-bit 路径。
__forceinline__ __device__ longlong4_t ldg_with_gez_pred(const longlong4_t* ptr, const int& value, const TMACacheHint& cache_hint = TMACacheHint::kEvictFirst) {
    // SM100+ 版本的 256-bit predicate load，value >= 0 时读取 longlong4_t。
    // longlong4_t 是 4 个 64-bit lane，整体 32B；用于新架构更宽的数据搬运。
    longlong4_t ret = make_longlong4_t(0, 0, 0, 0);  // predicate false 时返回 256-bit 零。
    // 默认零值。
    asm volatile(
        "{\n\t"
        // 谓词寄存器仍然只控制当前线程，因此不同 lane 可以读取不同有效性的数据。
        "  .reg .pred p;\n\t"
        // value >= 0 与 int4 版本保持一致，方便模板调用点只替换向量宽度。
        "  setp.ge.s32 p, %5, 0;\n\t"
        // v4.s64 对应 4 个 64-bit 输出寄存器，PTX operand 需要用 l 约束。
        "  @p ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s64 {%0, %1, %2, %3}, [%4], %6;\n\t"
        "}"
        : "+l"(ret.x), "+l"(ret.y), "+l"(ret.z), "+l"(ret.w)
        : "l"(ptr), "r"(value), "l"(cache_hint)  // SM100+ 256-bit load 参数。
        : "memory");
    return ret;
    // 返回加载结果或零。
}

__forceinline__ __device__ longlong4_t ldg(const longlong4_t* ptr) {
    // SM100+ 256-bit non-coherent global load。
    // 该 helper 不带 predicate，调用者必须保证 ptr 有效且满足 longlong4_t 对齐。
    longlong4_t ret;  // 保存 256-bit load 的四个 64-bit lane。
    // 输出寄存器。
    asm volatile(
        // no_allocate 说明读取后不会长期复用，避免占用 L1 cache。
        "ld.L1::no_allocate.global.nc.v4.s64 {%0, %1, %2, %3}, [%4];\n\t"
        : "=l"(ret.x), "=l"(ret.y), "=l"(ret.z), "=l"(ret.w)
        : "l"(ptr)  // global 源地址。
        : "memory");
    return ret;
    // 返回加载结果。
}
#endif  // SM100+ longlong4_t helper 结束。

__forceinline__ __device__ int4 ldg(const int4* ptr) {
    // int4 只读加载，使用 CUDA __ldg intrinsic。
    // 这里保留简单 intrinsic，是因为 int4 路径不需要谓词或特殊 L2 hint。
    return __ldg(ptr);  // 使用只读缓存路径读取 int4。
}

template <typename dtype_t>
__forceinline__ __device__ void st_with_gez_pred(dtype_t* ptr, dtype_t value, const int& condition) {
    // condition >= 0 时写 32-bit value，否则跳过写。
    // 典型用途：只有有效 top-k lane 写回结果，避免无效 lane 覆盖目标缓冲。
    EP_STATIC_ASSERT(sizeof(dtype_t) == 4, "Invalid data type");
    // 当前 helper 只支持 4B 标量。
    auto view = *reinterpret_cast<int*>(&value);  // 保留 dtype_t 的原始 32-bit 位模式。
    // 按 int 位模式传给 PTX。
    asm volatile(
        "{\n\t"
        // 用谓词控制 store，和 predicate load 一样避免生成额外控制流。
        "  .reg .pred p;\n\t"
        // condition 可以是 expert id、rank id 或 slot id，负数通常代表无效。
        "  setp.ge.s32 p, %2, 0;\n\t"
        // st.global.s32 只写 4B，dtype_t 通过位模式透传，不做数值转换。
        "  @p st.global.s32 [%0], %1;\n\t"
        "}"
        :: "l"(ptr), "r"(view), "r"(condition)  // condition >= 0 时写 view。
        : "memory");
}

template <typename dtype_t>
__forceinline__ __device__ dtype_t ld_volatile(const void* ptr) {
    // volatile global load，常用于轮询控制字，避免编译器缓存。
    // 例子：receiver 轮询 tail 指针时必须每次真正读取 global memory，不能复用旧寄存器值。
    if constexpr (sizeof(dtype_t) == 4) {
        // 32-bit volatile load。
        uint32_t value;  // 保存 32-bit 原始位模式。
        // 原始位模式。
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr));  // volatile 防止轮询 load 被缓存。
        return reinterpret_cast<const dtype_t&>(value);
        // 按目标类型返回。
    } else if constexpr (sizeof(dtype_t) == 8) {
        // 64-bit volatile load。
        // 64-bit 分支常用于时间戳、指针或 packed 控制字段。
        uint64_t value;  // 保存 64-bit 原始位模式。
        // 原始位模式。
        asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(value) : "l"(ptr));  // volatile 64-bit load。
        return reinterpret_cast<const dtype_t&>(value);
        // 按目标类型返回。
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
        // 仅支持 4/8B 控制字。
    }
}

__forceinline__ __device__ void red_add(const int64_t* ptr, const int64_t& value) {
    // TODO(NVCC): why don't NVCC support `s64`?
    // GPU-scope 64-bit add，不带 release 语义。
    // 只适合单 GPU 内部计数；跨 GPU/CPU 可见的计数器应使用 release.sys 版本。
    asm volatile("red.gpu.global.add.u64 [%0], %1;" :: "l"(ptr), "l"(value));  // red 是 fire-and-forget reduction，不返回旧值。
}

__forceinline__ __device__ void red_add_rel_sys(const int* ptr, const int& value) {
    // system-scope release 32-bit atomic add，用于跨 GPU 可见的计数器。
    // release.sys 是 DeepEP ring 协议的关键：先写 payload，再发布 tail/counter。
    // 典型配对关系：producer 先写通信 buffer，再调用 red_add_rel_sys；consumer 用
    // ld_acquire_sys 读取计数后，才能安全读取对应 payload。若这里误用 relaxed/gpu scope，
    // 跨 NVLink/RDMA peer 可能看到 counter 已更新但 payload 仍不可见。
    asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));  // release.sys 确保此前 payload 写入先于计数发布。
}

__forceinline__ __device__ void red_add_rel_sys(const int64_t* ptr, const int64_t& value) {
    // system-scope release 64-bit atomic add。
    // 64-bit 版本通常用于更宽的累计字节数、token 计数或 packed 状态。
    asm volatile("red.release.sys.global.add.u64 [%0], %1;" :: "l"(ptr), "l"(value));  // 64-bit system-scope release reduction。
}

template <typename dtype_t>
__forceinline__ __device__ dtype_t ld_acquire_sys(const dtype_t* ptr) {
    // system-scope acquire load，和 release store/add 配对。
    // 如果 producer 用 st_release_sys 发布 tail，consumer 必须用这里读取，才能看到 tail 前的 payload 写入。
    if constexpr (sizeof(dtype_t) == 4) {
        // 32-bit acquire load。
        uint32_t value;  // acquire load 原始位模式。
        asm volatile("ld.acquire.sys.L1::no_allocate.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr));  // system acquire 32-bit load。
        return reinterpret_cast<const dtype_t&>(value);  // 按调用方 dtype_t 返回。
    } else if constexpr (sizeof(dtype_t) == 8) {
        // 64-bit acquire load。
        // acquire 不只读取数值，还建立“读到发布值之后可见此前写入”的内存顺序。
        uint64_t value;  // acquire load 原始位模式。
        asm volatile("ld.acquire.sys.L1::no_allocate.global.u64 %0, [%1];" : "=l"(value) : "l"(ptr));  // system acquire 64-bit load。
        return reinterpret_cast<const dtype_t&>(value);  // 按调用方 dtype_t 返回。
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
        // 只支持 4/8B 控制字。
    }
}

template <typename dtype_t>
__forceinline__ __device__ void st_relaxed_sys(void* ptr, dtype_t value) {
    // system-scope relaxed store，不提供 release 排序。
    // 适合初始化或单独状态写入；不能用来发布“payload 已写完”这类同步事件。
    if constexpr (sizeof(dtype_t) == 4) {
        // 32-bit relaxed store。
        uint32_t int_value = reinterpret_cast<const uint32_t&>(value);  // 保留原始 32-bit 位模式。
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(int_value));  // system relaxed 32-bit store。
    } else if constexpr (sizeof(dtype_t) == 8) {
        // 64-bit relaxed store。
        // 仍然是 system scope，可被其它 GPU/CPU 观察，但没有先后顺序保证。
        uint64_t int_value = reinterpret_cast<const uint64_t&>(value);  // 保留原始 64-bit 位模式。
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(ptr), "l"(int_value));  // system relaxed 64-bit store。
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
        // 只支持 4/8B。
    }
}

template <typename dtype_t>
__forceinline__ __device__ void st_release_sys(void* ptr, dtype_t value) {
    // system-scope release store，保证此前写入对 acquire load 可见。
    // 例子：先写 ring buffer 中的 token，再用 st_release_sys 写 tail，receiver acquire 读 tail 后即可安全读 token。
    if constexpr (sizeof(dtype_t) == 4) {
        // 32-bit release store。
        uint32_t int_value = reinterpret_cast<const uint32_t&>(value);  // 保留原始 32-bit 位模式。
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(int_value));  // system release 32-bit store。
    } else if constexpr (sizeof(dtype_t) == 8) {
        // 64-bit release store。
        // 用于 64-bit tail/head 或 packed 指针状态时，同样承担发布语义。
        uint64_t int_value = reinterpret_cast<const uint64_t&>(value);  // 保留原始 64-bit 位模式。
        asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(ptr), "l"(int_value));  // system release 64-bit store。
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
        // 只支持 4/8B。
    }
}

// Adjust registers
// Hopper/Blackwell warpgroup 指令允许局部调高/调低寄存器预算，配合 producer/consumer warp 分工。
template <int kNumRegs>
__device__ __forceinline__ void warpgroup_reg_alloc(){
    // 增加 warpgroup 可用寄存器上限，常用于生产者/消费者分工优化。
    // 该指令要求同一 warpgroup 中的线程以一致控制流执行；否则寄存器预算状态会不一致。
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));  // kNumRegs 必须是编译期立即数。
}

template <int kNumRegs>
__device__ __forceinline__ void warpgroup_reg_dealloc(){
    // 释放此前分配的寄存器额度，让同 CTA 其它 warpgroup 可使用。
    // 通常和 warpgroup_reg_alloc 成对出现，放在 producer/consumer 阶段切换边界。
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));  // 归还寄存器额度。
}

/// General fences
__device__ __forceinline__ void fence_acq_rel_sys() {
    // system-scope acquire/release fence，用于跨 GPU 内存顺序。
    asm volatile("fence.acq_rel.sys;" ::: "memory");  // memory clobber 防止 host compiler 重排内联 asm 前后访存。
}

/// Intrinsics
template <typename dtype_t>
__device__ __forceinline__ dtype_t exchange(dtype_t ptr, const int& src_lane_idx) {
    // 从 src_lane_idx 广播任意 4B 倍数大小的数据到整个 warp。
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
    // 通过 int 片段 shfl，因此类型大小必须是 4B 倍数。
    const auto send_int_values = reinterpret_cast<int*>(&ptr);  // 将待广播对象按 int 分片。
    // 待发送值的 int 视图。
    dtype_t recv_dtype;  // 接收端重组后的对象。
    // 接收值。
    auto recv_int_values = reinterpret_cast<int*>(&recv_dtype);  // 接收对象的 int 分片视图。
    // 接收值的 int 视图。
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
        // 逐个 32-bit 片段广播。
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], src_lane_idx);  // 从 src lane 广播第 i 个 int 分片。
    return recv_dtype;
    // 返回从 src lane 拿到的完整值。
}

__device__ __forceinline__ unsigned gather(const bool& value) {
    // warp ballot，把每个 lane 的 bool 收集成 32-bit mask。
    return __ballot_sync(0xffffffff, value);  // 第 i bit 表示 lane i 的 value 是否为 true。
}

__device__ __forceinline__ bool all(const bool& value) {
    // warp all，所有 lane value 都为 true 时返回 true。
    return __all_sync(0xffffffff, value);  // 所有 active lane 为 true 才返回 true。
}

__device__ __forceinline__ bool any(const bool& value) {
    // warp any，任意 lane value 为 true 时返回 true。
    return __any_sync(0xffffffff, value);  // 任意 active lane 为 true 即返回 true。
}

__device__ __forceinline__ unsigned reduce_or(const unsigned& value) {
    // warp 内 32-bit OR 规约。
    return __reduce_or_sync(0xffffffff, value);  // 硬件 warp OR reduction。
}

__device__ __forceinline__ unsigned long long reduce_or(const unsigned long long& value) {
    // warp 内 64-bit OR 规约，通过高低 32-bit 分开规约实现。
    const auto low = __reduce_or_sync(0xffffffff, static_cast<unsigned>(value));
    // 低 32 bit OR。
    const auto high = __reduce_or_sync(0xffffffff, static_cast<unsigned>(value >> 32));  // 高 32 bit OR。
    // 高 32 bit OR。
    return (static_cast<unsigned long long>(high) << 32) | low;
    // 合并成 64-bit 结果。
}

__device__ __forceinline__ int reduce_add(const int& value) {
    // warp 内 int 加法规约。
    return __reduce_add_sync(0xffffffff, value);  // 硬件 warp int 加法规约。
}

__device__ __forceinline__ unsigned match(const int& value) {
    // 返回 warp 中与当前 lane value 相同的 lane mask。
    return __match_any_sync(0xffffffff, value);  // 返回同值 lane 集合 mask。
}

__device__ __forceinline__ int fns(const unsigned& value, const int& offset) {
    // find-nth-set：返回 value 中第 offset 个置位 bit 的 index。
    return __fns(value, 0, offset);  // 从 bit 0 开始找第 offset 个 set bit。
}

template <typename dtype_t>
__device__ __forceinline__ auto ffs(const dtype_t& value) {
    // 返回最低置位 bit 的 0-based index；没有置位时返回 -1。
    if constexpr (sizeof(dtype_t) == 4) {
        // 32-bit 路径。
        return __ffs(static_cast<int>(value)) - 1;  // CUDA __ffs 返回 1-based，因此减 1。
    } else {
        // 64-bit 路径。
        EP_STATIC_ASSERT(sizeof(dtype_t) == 8, "Invalid data type");
        return __ffsll(static_cast<long long>(value)) - 1;  // 64-bit 版本同样转为 0-based。
    }
}

__device__ __forceinline__ int get_master_lane_idx(const unsigned& mask) {
    // Equivalent to `31 - __clz(mask)`
    int highest_idx;  // 保存最高置位 bit index。
    // bfind 返回最高置位 bit。
    asm volatile("bfind.u32 %0, %1;" : "=r"(highest_idx) : "r"(mask));  // bfind.u32 返回最高 set bit。
    return highest_idx;
    // 作为一组相同 key lane 的 master lane。
}

__device__ __forceinline__ bool deduplicate(const int& value, const int& lane_idx) {
    // 同 value 的 lane 中只保留最高 lane，常用于 top-k/rank 去重。
    // 选择最高 lane 而不是最低 lane 是本项目的一致约定：get_master_lane_idx(match(value))
    // 同时被 dispatch metadata、combine source lane 选择和部分 prefix 逻辑使用，避免同一 token
    // 在不同 helper 中出现“代表 lane”不一致。
    return get_master_lane_idx(match(value)) == lane_idx;  // 只有同值 lane 集合中的最高 lane 返回 true。
}

__device__ __forceinline__ int warp_inclusive_sum(int value, const int& lane_idx) {
    // warp 内 inclusive prefix sum。
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        // 依次从 1、2、4、8、16 lane 距离累加。
        const auto synced = __shfl_up_sync(0xffffffff, value, offset);  // 从当前 lane 前 offset 个 lane 读取值。
        // 读取前 offset 个 lane 的部分和。
        if (lane_idx >= offset)
            // 只有 lane_idx 足够大时才累加。
            value += synced;  // 累加前缀部分和。
    }
    return value;  // inclusive sum 包含当前 lane 自身。
    // 返回包含当前 lane 的前缀和。
}

__device__ __forceinline__ float2 fadd2(const float2& a, const float2& b) {
    // 两个 float2 相加；SM100+ 可用向量化 fadd2 指令。
    #if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // SM100+ 有原生 float2 向量加法 intrinsic。
    return __fadd2_rn(a, b);  // SM100+ 使用向量化 float2 加法。
    #else  // 旧架构没有 __fadd2_rn，退化为两个标量 FP32 加法。
    return {a.x + b.x, a.y + b.y};  // 旧架构手动逐分量相加。
    #endif  // fadd2 架构分支结束。
}

__device__ __forceinline__ void accumulate(float2& a, nv_bfloat162 b) {
    // 将 BF16 pair 累加到 FP32 pair accumulator。
    #if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)  // SM100+ 可直接用 BF16->FP32 add PTX。
    // Use `add.rn.f32.bf16` instruction to perform fused (cast + add) operation on SM100
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));
    // SM100 直接用 BF16->FP32 add 指令累加 x。
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));
    // 累加 y。
    #else  // 旧架构先把 bfloat162 转成 float2，再逐分量累加。
    const auto [x, y] = __bfloat1622float2(b);  // BF16 pair 转成两个 FP32。
    // 旧架构先转 float2。
    a.x += x, a.y += y;  // 累加到 FP32 accumulator。
    // 手动累加。
    #endif  // accumulate 架构分支结束。
}

#endif  // __CUDACC__ device helper 分支结束；host 侧只保留 mbarrier 类型和常量。

} // namespace deep_ep::elastic::ptx
// Elastic PTX helper 命名空间结束。
