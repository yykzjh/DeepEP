#pragma once  // legacy CUDA 低层工具头：PTX load/store、fence、向量化 copy 等都在这里。

#include <deep_ep/common/compiled.cuh>  // 引入 topk_idx_t、架构宏、公共编译配置。
#include <deep_ep/common/exception.cuh>  // 引入 EP_STATIC_ASSERT/EP_DEVICE_ASSERT。

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                     \
    {                                                                                                                                 \
        /* warp 级连续拷贝宏：每个 lane 处理间隔 32 的元素，UNROLL_FACTOR 控制一次循环 */                                             \
        /* 读写多少个向量寄存器。legacy dispatch/combine 大量用它搬 hidden_int4 向量。 */                                             \
        constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                                                                             /* 一个 warp 一轮覆盖 32*UNROLL_FACTOR 个元素。 */ \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                          /* 每 lane 暂存 UNROLL_FACTOR 个加载值。 */ \
        auto __src = (SRC);                                                                                                           /* 固定源指针表达式，避免宏参数重复求值。 */ \
        auto __dst = (DST);                                                                                                           /* 固定目标指针表达式，避免宏参数重复求值。 */ \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                      /* 主循环只处理完整 unroll 块。 */ \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); /* 先批量 load，提高指令级并行。 */ \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);  /* 再批量 store，保持每 lane 间隔 32。 */ \
        }                                                                                                                             /* 主循环结束。 */ \
        {                                                                                                                             /* tail 块处理 N 不是 kLoopStride 倍数的剩余元素。 */ \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                                                                  /* tail 起点加当前 lane id。 */ \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       /* tail load 仍保持 unroll。 */ \
                if (__i + __j * 32 < (N)) {                                                                                           /* 越界保护。 */ \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);                                                           /* 加载有效 tail 元素。 */ \
                }                                                                                                                     /* 结束越界保护。 */ \
            }                                                                                                                         /* tail load 结束。 */ \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       /* tail store 仍保持 unroll。 */ \
                if (__i + __j * 32 < (N)) {                                                                                           /* 只写有效 tail 元素。 */ \
                    ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);                                                            /* 写回 tail 元素。 */ \
                }                                                                                                                     /* 结束越界保护。 */ \
            }                                                                                                                         /* tail store 结束。 */ \
        }                                                                                                                             \
    }

namespace deep_ep::legacy {  // legacy kernel helper 命名空间。

template <int kBytes>
struct VecInt {};  // 默认模板不定义 vec_t，只有支持的字节宽度会特化。
// VecInt 把字节宽度映射到同宽整数/向量类型，方便通用模板把任意 dtype_t
// reinterpret 成 PTX inline asm 支持的寄存器类型。
template <>
struct VecInt<1> {
    using vec_t = int8_t;  // 1 字节类型映射到 int8_t。
};
template <>
struct VecInt<2> {
    using vec_t = int16_t;  // 2 字节类型映射到 int16_t。
};
template <>
struct VecInt<4> {
    using vec_t = int;  // 4 字节类型映射到 int。
};
template <>
struct VecInt<8> {
    using vec_t = int64_t;  // 8 字节类型映射到 int64_t。
};
template <>
struct VecInt<16> {
    using vec_t = int4;  // 16 字节类型映射到 int4，匹配向量化 LD/ST。
};

template <typename FuncT>
struct PatternVisitor {
    FuncT func;  // 保存可调用对象，通常是 device lambda。

    // PatternVisitor 让 lambda 可以用 visitor[i] 的形式被模板代码访问，兼容 host/device
    // 两侧编译，常用于把布局计算公式作为“虚拟数组”传入通用拷贝逻辑。
    __device__ __host__ explicit PatternVisitor(FuncT&& func) : func(std::forward<FuncT>(func)) {}  // 转发保存 lambda。

    __device__ __host__ auto operator[](const uint32_t& i) { return func(i); }  // 用数组下标语法调用 lambda。
};

__device__ __forceinline__ void trap() {
    asm("trap;");  // 触发 device trap，常用于不可恢复的 device assert。
}

__device__ __forceinline__ void memory_fence() {
    // legacy kernel 手写了不少跨 GPU/跨 CTA 的 flag 协议，系统域 fence 保证数据写入
    // 先于 release flag 对远端可见。
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void memory_fence_gpu() {
    // GPU scope fence 只保证本 GPU 内 CTA/SM 间可见，适合不跨设备的队列状态。
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void memory_fence_cta() {
    // CTA scope fence 用于同 block 内共享/全局状态的小范围同步，开销低于 sys/gpu scope。
    asm volatile("fence.acq_rel.cta;" ::: "memory");
}

__device__ __forceinline__ void st_relaxed_sys_global(const int* ptr, int val) {
    // relaxed store 不携带顺序保证，通常用于写非同步关键路径的普通计数或初始化值。
    asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_release_sys_global(const int* ptr, int val) {
    // release store 用于发布队列 tail/signal：在它之前的 payload 写入必须先对远端可见。
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_release_cta(const int* ptr, int val) {
    // 只在 CTA 内发布状态，常用于同 block 内 coordinator/worker 的轻量握手。
    asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");  // CTA scope release store。
}

__device__ __forceinline__ int ld_acquire_sys_global(const int* ptr) {
    // acquire load 与远端 release store 配对，读取到新 tail/signal 后才能安全读取 payload。
    int ret;  // 32-bit acquire load 结果。
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));  // system scope acquire load，读取远端发布的 flag/counter。
    return ret;  // 返回读取值。
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t* ptr) {
    // 64-bit system acquire 常用于读取跨 GPU 发布的 packed 状态或 64-bit 计数器。
    uint64_t ret;  // 读取到的 64-bit 值。
    // acquire.sys 与远端 release.sys 配对，读到新值后才能安全访问其保护的数据。
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));  // system scope acquire 64-bit load。
    return ret;  // 返回读取结果。
}

__device__ __forceinline__ int ld_acquire_global(const int* ptr) {
    // GPU scope acquire 只保证同一 GPU 内可见，适合本地 SM 间同步，不适合 NVSHMEM 远端发布。
    int ret;  // 读取到的 32-bit 值。
    // 这里不使用 sys scope，可以减少跨系统域同步成本。
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));  // GPU scope acquire load。
    return ret;  // 返回读取结果。
}

__device__ __forceinline__ int atomic_add_release_sys_global(const int* ptr, int value) {
    // release atomic add 用于“增加计数并发布之前写入”。返回旧值，可作为当前线程获得的 slot。
    int ret;  // atomic add 返回旧值。
    asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));  // system scope release atomic add。
    return ret;  // 返回旧值，可作为当前线程分配到的 slot。
}

__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
    // GPU scope 版本用于本 GPU 内 ring slot 分配，不需要跨 NVSHMEM PE 的 system scope。
    int ret;  // atomic add 返回旧值，可作为 slot id。
    // 返回旧值这一点很关键：多个线程同时 add 时，每个线程拿到不同旧值作为独占写槽。
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));  // GPU scope release atomic add。
    return ret;  // 返回旧值。
}

__device__ __forceinline__ int ld_acquire_cta(const int* ptr) {
    // CTA scope acquire 与 st_release_cta 配对，适合 shared/global 中的 block 内控制变量。
    int ret;  // CTA scope load 结果。
    asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));  // CTA scope acquire load。
    return ret;  // 返回读取结果。
}

__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t* ptr) {
    // no-allocate relaxed load：读通信 buffer 中一次性数据，不希望污染 L1 cache。
    uint16_t ret;  // 8-bit asm 输出借用 16-bit half 寄存器约束。
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));  // no-allocate relaxed 8-bit load。
    return static_cast<uint8_t>(ret);  // 截断低 8 bit 返回。
}

__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t* ptr) {
    // 16-bit no-allocate 读常用于 half/bfloat16 或短控制字段的原始位模式读取。
    uint16_t ret;  // 16-bit load 结果。
    // relaxed 表示不提供 acquire/release 顺序，只做普通数据读取。
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));  // no-allocate relaxed 16-bit load。
    return ret;  // 返回读取结果。
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t* ptr) {
    // 32-bit no-allocate 读覆盖 int/float 位模式，是通信 payload 和 metadata 的常用宽度。
    uint32_t ret;  // 32-bit load 结果。
    // L1::no_allocate 避免一次性流式读取占据 L1 cache。
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));  // no-allocate relaxed 32-bit load。
    return ret;  // 返回读取结果。
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t* ptr) {
    // 64-bit no-allocate 读用于 packed int2、指针值或长计数字段。
    uint64_t ret;  // 64-bit load 结果。
    // relaxed 读不会和 release 写建立同步；如果读的是控制 flag 应使用 acquire helper。
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));  // no-allocate relaxed 64-bit load。
    return ret;  // 返回读取结果。
}

__device__ __forceinline__ int ld_volatile_global(const int* ptr) {
    // volatile load 用于轮询 flag/counter，避免编译器把循环内 load 提出或缓存到寄存器。
    int ret;  // volatile 32-bit load 结果。
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));  // 轮询 flag/counter 时强制每次真正读取 global memory。
    return ret;  // 返回读取值。
}

__device__ __forceinline__ float ld_volatile_global(const float* ptr) {
    // float volatile 主要用于轮询或调试读取 FP32 状态，普通 payload 不建议用 volatile 路径。
    float ret;  // float load 结果。
    // volatile 只保证编译器每次发起 load，不等价于 acquire 内存顺序。
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));  // volatile float load。
    return ret;  // 返回读取结果。
}

__device__ __forceinline__ int64_t ld_volatile_global(const int64_t* ptr) {
    // signed 64-bit volatile 读取保留符号解释，适合读取可能为负的哨兵编码。
    int64_t ret;  // 64-bit signed load 结果。
    // 例如 head 使用负数表示 ready 时，signed 类型能直接参与 <0 判断。
    asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));  // volatile signed 64-bit load。
    return ret;  // 返回读取结果。
}

__device__ __forceinline__ int64_t ld_volatile_global(const uint64_t* ptr) {
    // 这个 overload 保留旧调用方返回 int64_t 的习惯，但底层按 u64 读取位模式。
    int64_t ret;  // 64-bit unsigned load 结果以 int64_t 返回，保持调用方旧类型。
    // 如果高位被置位，返回的 int64_t 可能表现为负数，调用点需要理解这是位模式复用。
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));  // volatile unsigned 64-bit load。
    return ret;  // 返回读取结果。
}

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"  // 激进路径：non-coherent/no-allocate/L2 256B sector hint。
#else
#define LD_NC_FUNC "ld.volatile.global"  // 兼容路径：使用 volatile global load。
#endif

// no-allocate load 避免通信 buffer 扫描污染 L1；在 SASS 中会变成 LDG.E.NA.*。
// 如果调试或架构不支持 aggressive PTX，可以通过宏退回 volatile load。
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t* ptr) {
    auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));  // 先按同宽整数/向量类型加载。
    return *reinterpret_cast<dtype_t*>(&ret);  // 再 reinterpret 回原 dtype，避免为每种 dtype 重写 asm。
}

template <>
__device__ __forceinline__ uint8_t ld_nc_global(const uint8_t* ptr) {
    uint16_t ret;  // inline asm 8-bit 输出借用 16-bit 约束保存。
    // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit constraint letter (`h` below means unsigned 16-bit)
    // 这条特化服务于 bool/mask 之类的字节字段；C++ 层看是 uint8_t，PTX 约束层实际用 half 寄存器承接。
    // PTX 读出的是低 8 bit；高位只是为了满足 inline asm 寄存器约束。
    asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);  // 截取低 8 位作为结果。
}

template <>
__device__ __forceinline__ int ld_nc_global(const int* ptr) {
    int ret;  // 32-bit load 结果。
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));  // non-coherent/no-allocate 32-bit load。
    return ret;  // 返回读取结果。
}

template <>  // int64_t 特化用于 64-bit prefix、指针偏移或计时值。
__device__ __forceinline__ int64_t ld_nc_global(const int64_t* ptr) {
    int64_t ret;  // 64-bit load 结果。
    asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));  // non-coherent/no-allocate 64-bit load。
    return ret;  // 返回读取值。
}

template <>  // float 特化用于 top-k weight、scale 等 FP32 元数据。
__device__ __forceinline__ float ld_nc_global(const float* ptr) {
    float ret;  // float load 结果。
    asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));  // non-coherent/no-allocate float load。
    return ret;  // 返回读取值。
}

template <>  // int2 向量 load 一次搬 8B，适合小型 packed metadata。
__device__ __forceinline__ int2 ld_nc_global(const int2* ptr) {
    int2 ret;  // 两个 32-bit lane 的向量结果。
    asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));  // non-coherent/no-allocate int2 load。
    return ret;  // 返回 int2。
}

template <>  // int4 向量 load 一次搬 16B，是 legacy hidden payload 的主要搬运粒度。
__device__ __forceinline__ int4 ld_nc_global(const int4* ptr) {
    int4 ret;  // 四个 32-bit lane 的向量结果。
    asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));  // non-coherent/no-allocate int4 load。
    return ret;  // 返回 int4。
}

__device__ __forceinline__ void st_na_relaxed(const uint8_t* ptr, uint8_t val) {
    // 8-bit store 用于 mask/flag 等字节数据；PTX 约束用 16-bit 寄存器承载立即值。
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(static_cast<uint16_t>(val)));  // no-allocate relaxed 8-bit store。
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t* ptr, uint16_t val) {
    // 16-bit store 可用于 half/bfloat16 位模式，但这里按 uint16_t 处理避免数值转换。
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));  // no-allocate relaxed 16-bit store。
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t* ptr, uint32_t val) {
    // 32-bit relaxed store 用于普通计数或 FP32 位模式，不提供发布语义。
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));  // no-allocate relaxed 32-bit store。
}

__device__ __forceinline__ void st_na_relaxed(const int* ptr, int val) {
    // signed int 版本和 uint32_t 使用同宽 PTX，只是 C++ 调用侧类型不同。
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));  // no-allocate relaxed 32-bit signed store。
}

__device__ __forceinline__ void st_na_relaxed(const int4* ptr, int4 val) {
    // int4 store 是 16B 向量写，常用于 BF16 hidden 的打包写回。
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"  // no-allocate relaxed int4 向量 store。
                 :
                 : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__device__ __forceinline__ void st_na_release(const int* ptr, int val) {
    // no-allocate + release 常用于写 tail/head，既发布顺序又避免控制变量污染 L1。
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));  // no-allocate release 32-bit store，用于发布 GPU scope flag。
}

__device__ __forceinline__ void st_na_release(const uint32_t* ptr, uint32_t val) {
    // uint32_t 版本服务于位模式为 unsigned 的控制字。
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));  // uint32 版本 release store。
}

__device__ __forceinline__ void st_na_release(const uint64_t* ptr, uint64_t val) {
    // 64-bit release store 用于 packed head/tail 或双字段状态。
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));  // uint64 版本 release store。
}

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
// 注意这里的 no_allocate 是性能提示而不是同步语义；需要发布顺序时应使用 st_na_release 或 fence+普通 store。
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"  // 激进路径：store 不分配 L1。
#else
#define ST_NA_FUNC "st.global"  // 兼容路径：普通 global store。
#endif

template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t* ptr, const dtype_t& value) {
    // 泛型 no-allocate store 会先把 dtype_t 映射成同宽 VecInt，再调用特化版本生成 PTX。
    // 这样调用点可以传 SourceMeta、int2/int4 或其它 POD 类型，仍然生成固定宽度的全局写；
    // 前提是 sizeof(dtype_t) 必须命中 VecInt 特化，否则编译期会暴露不支持的宽度。
    st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),  // 把目标指针转成同宽整数/向量指针。
                 *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));  // 把 value 的位模式转成同宽整数/向量。
}

template <>
__device__ __forceinline__ void st_na_global(const int* ptr, const int& value) {
    // int32 特化常用于写 head/tail、slot index、prefix 等控制面整数。
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));  // int32 no-allocate/global store。
}

template <>
__device__ __forceinline__ void st_na_global(const int64_t* ptr, const int64_t& value) {
    // int64 特化用于较大的 prefix/计数，或 host/device 共享的 64-bit 状态。
    asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));  // int64 no-allocate/global store。
}

template <>
__device__ __forceinline__ void st_na_global(const float* ptr, const float& value) {
    // float 特化主要写 top-k weight、scale 和累计统计，不做类型 punning。
    asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));  // float no-allocate/global store。
}

template <>
__device__ __forceinline__ void st_na_global(const int4* ptr, const int4& value) {
    // int4 特化是旧版 payload 的高频路径，一次写 16B，可覆盖 8 个 BF16 或 16 个 FP8。
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));  // int4 向量 store。
}

__device__ __forceinline__ float log2f_approx(const float& x) {
    // 低延迟 FP8/LogFMT 路径只需要近似 log2/exp2，用 PTX approximate 指令减少 latency。
    float ret;  // 近似 log2 输出。
    asm volatile("lg2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));  // PTX 近似 log2 指令。
    return ret;  // 返回近似值。
}

__device__ __forceinline__ float exp2f_approx(const float& x) {
    // 和 log2f_approx 配对，常用于把量化 scale 从 log 域还原到线性域。
    float ret;  // 近似 exp2 输出。
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));  // PTX 近似 2^x 指令。
    return ret;  // 返回近似值。
}

__forceinline__ __device__ int get_lane_id() {
    // 比 threadIdx.x % 32 更直接，且在一些编译路径中能避免额外算术指令。
    int lane_id;  // 当前线程在 warp 内的 lane id，范围 [0,31]。
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));  // 直接读取特殊寄存器 %laneid。
    return lane_id;  // 返回 lane id。
}

__device__ __forceinline__ uint32_t elect_one_sync() {
#ifndef DISABLE_SM90_FEATURES
    // elect.sync 在当前 active mask 中选一个 lane 作为 leader；相比 lane0，在分支后
    // lane0 不活跃时仍能正常选择 leader。
    // legacy kernel 中大量 “只发起一次 TMA/put/store” 的逻辑都依赖它；
    // 如果直接写 lane_id==0，分支收敛后的子 warp 可能没有 lane0，从而漏发一次性操作。
    uint32_t pred = 0;  // elect.sync 会把被选中 lane 的 predicate 置 true，这里用整数承接。
    // 下面的 busy-wait 汇编使用 parity bit 等待对应 stage 的 mbarrier 满足；
    // 与 common/ptx.cuh 一样，0x989680 只是 PTX try_wait 的等待提示，不是业务层超时。
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"    // %%rx 接收被选中 lane id，但当前实现只关心 predicate。
        ".reg .pred %%px;\n"   // %%px 表示当前 lane 是否被 elect 选中。
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n" // 只有被选中 lane 写 1，其它 lane 保持 0。
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));  // active mask 固定全 warp；分支内 inactive lane 会由硬件 mask 处理。
    return pred;  // 返回非零表示当前 lane 是 leader，可安全执行一次性操作。
#else
    return get_lane_id() == 0;  // 非 SM90 回退：固定 lane0 作为 leader。
#endif
}

// TMA PTX instructions
#ifndef DISABLE_SM90_FEATURES

__device__ __forceinline__ void fence_barrier_init() {
    // mbarrier 初始化属于 async proxy/cluster 可见性范畴，release fence 避免后续 TMA 看到未初始化 barrier。
    // 调用顺序通常是 mbarrier_init -> fence_barrier_init -> __syncwarp；缺少 fence 时，
    // 后续 cp.async.bulk 可能观察到旧 barrier 状态。
    asm volatile("fence.mbarrier_init.release.cluster; \n" ::);  // 初始化 mbarrier 前的 cluster release fence。
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
    // arrive_count 表示 barrier 期待的参与者数量；TMA 场景常设为 1 或 32。
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));  // mbarrier 地址必须转成 shared address。
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count), "r"(mbar_int_ptr));  // 初始化 arrive count。
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar_ptr) {
    // 失效化用于 shared memory 被复用前清理 barrier 状态，防止旧 phase 干扰新 pipeline。
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));  // 转 shared address。
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mbar_int_ptr));  // 失效化 mbarrier。
}

template <bool kWithMultiStages = false>
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase, int stage_idx = 0) {
    // kWithMultiStages=true 时 phase 的每一 bit 对应一个 stage；单 stage 时直接翻转 0/1。
    // 双缓冲 TMA 中 stage0/stage1 共享一个 phase 变量的不同 bit，避免为每个 stage 单独保存状态。
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));  // mbarrier shared address。
    const auto wait = kWithMultiStages ? (phase >> stage_idx) & 1 : phase;  // 当前 stage 需要等待的 parity。
    // try_wait.parity 用 parity 而不是绝对计数，因此每次 wait 成功后必须翻转 phase。
    asm volatile(
        "{\n\t"  // 汇编作用域开始。
        ".reg .pred       P1; \n\t"  // P1 保存 try_wait 是否成功。
        "LAB_WAIT: \n\t"  // 自旋等待标签。
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"  // 按 parity 等待 mbarrier。
        "@P1 bra DONE; \n\t"  // 成功则跳出。
        "bra     LAB_WAIT; \n\t"  // 未成功则继续等待。
        "DONE: \n\t"  // 等待完成标签。
        "}" ::"r"(mbar_int_ptr),  // 汇编作用域结束并传入 mbarrier shared address。
        "r"(wait),
        "r"(0x989680));  // 0x989680 是 try_wait 的 immediate spin hint。
    phase ^= kWithMultiStages ? (1 << stage_idx) : 1;  // 等待完成后翻转对应 parity bit。
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
    // expect_tx 告诉 barrier 后续异步传输的字节数；TMA 完成这些字节后 barrier 才会满足。
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));  // mbarrier shared address。
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::"r"(num_bytes), "r"(mbar_int_ptr));  // arrive 并声明即将传输 num_bytes。
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_ptr) {
    // 普通 arrive 不绑定传输字节，适合非 TMA 的线程间同步。
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));  // mbarrier shared address。
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::"r"(mbar_int_ptr));  // 普通 arrive。
}

__device__ __forceinline__ void tma_store_fence() {
    // 普通 shared store 和 TMA async proxy 属于不同内存代理；没有这个 fence，
    // leader lane 发起 TMA store 时可能看不到其它 lane 刚写入 shared 的数据。
    asm volatile("fence.proxy.async.shared::cta;");  // TMA store 前后用于 async proxy 的 shared fence。
}

constexpr uint64_t kEvictFirst = 0x12f0000000000000;  // L2 cache hint：优先驱逐，适合一次性 staging 数据。
constexpr uint64_t kEvictNormal = 0x1000000000000000;  // L2 cache hint：普通缓存策略。

__device__ __forceinline__ void tma_load_1d(
    const void* smem_ptr, const void* gmem_ptr, uint64_t* mbar_ptr, int num_bytes, bool evict_first = true) {
    // 1D TMA global->shared。mbarrier 负责通知数据到达，evict_first 用于短生命周期 staging 数据。
    // 调用方必须随后执行 mbarrier_arrive_and_expect_tx/等待 parity；否则 shared 中的数据可能尚未到达。
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));  // mbarrier shared address。
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));  // shared memory 目标地址。
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;  // 根据生命周期选择 L2 hint。
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(smem_int_ptr),
        "l"(gmem_ptr),  // global 源地址。
        "r"(num_bytes),  // 传输字节数。
        "r"(mbar_int_ptr),  // 完成通知 mbarrier。
        "l"(cache_hint)  // L2 cache hint。
        : "memory");
}

__device__ __forceinline__ void tma_store_1d(const void* smem_ptr, const void* gmem_ptr, int num_bytes, bool evict_first = true) {
    // 1D TMA shared->global，并立即 commit 到 bulk group；调用方随后用 tma_store_wait 等待完成。
    // 参数顺序和 tma_load_1d 相反：这里 smem_ptr 是源，gmem_ptr 是目标。
    // store 前若 shared 数据由普通 store 产生，必须先调用 tma_store_fence，让 async proxy 看见这些写入。
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));  // shared memory 源地址。
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;  // L2 cache hint。
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::"l"(gmem_ptr),
                 "r"(smem_int_ptr),  // shared 源地址。
                 "r"(num_bytes),  // 传输字节数。
                 "l"(cache_hint)  // L2 cache hint。
                 : "memory");
    asm volatile("cp.async.bulk.commit_group;");  // 提交当前 bulk async store group。
}

template <int N>  // N 是 wait_group immediate，表示允许保留的未完成 async bulk group 数。
__device__ __forceinline__ void tma_store_wait() {
    // 例如 tma_store_wait<0>() 等待所有 bulk store 完成；tma_store_wait<1>() 允许保留 1 组在途以做流水。
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");  // 等待至多 N 个 bulk group 未完成。
}

#endif

template <typename dtype_t>  // 与 Python/C++ host 侧容量计算保持同一公式。
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;  // 向上整除。
}

template <typename dtype_t>  // 向上对齐常用于 ring 容量、expert token 容量和 shared memory offset。
__host__ __device__ constexpr dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;  // 向上对齐到 b 的倍数。
}

template <typename dtype_t>  // 向下对齐用于截断到合法 tile 边界。
__host__ __device__ constexpr dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;  // 向下对齐到 b 的倍数。
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int& token_start_idx, int& token_end_idx) {
    // 把 token 维均匀切给 channel/SM。最后一个 channel 可能少于 num_tokens_per_sm。
    // 例：num_tokens=100、num_sms=6 时，每 SM 17 个 token，最后一个 SM 只处理 [85,100)。
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);  // 每个 SM/channel 理论处理 token 数。
    token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);  // 当前 SM 起始 token index。
    token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);  // 当前 SM 结束 token index，右开。
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    // 把两个小整数打包成一个大整数，常用于 layout_range 中保存 (count, offset)。
    // 这是位级打包而不是数值转换，例如两个 int16 会按内存顺序拼成一个 int32。
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");  // 大类型大小必须正好容纳两个小类型。
    dtype_b_t packed;  // 打包后的大类型变量。
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);  // 把 packed 视作两个小类型槽。
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;  // 低位放 x，高位放 y。
    return packed;  // 返回打包值。
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");  // 解包类型大小必须匹配 pack2 的布局。
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);  // 把 packed 视作两个小类型槽。
    x = unpacked_ptr[0], y = unpacked_ptr[1];  // 解出 x/y。
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t& ptr, int src_lane_idx) {
    // 对任意 sizeof(dtype_t) 为 int 倍数的类型做 warp broadcast，例如 int2/int4
    // 元数据结构。逐 int shuffle 后再 reinterpret 回原类型。
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");  // 只能广播 4B 整数倍大小的结构。
    auto send_int_values = reinterpret_cast<int*>(&ptr);  // 把待广播对象拆成若干 int。
    int recv_int_values[sizeof(dtype_t) / sizeof(int)];  // 保存从 src lane 收到的 int 分片。
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)  // 逐 int 分片做 warp shuffle。
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], src_lane_idx);  // 从 src_lane_idx 广播第 i 个分片。
    return *reinterpret_cast<dtype_t*>(recv_int_values);  // 把分片数组重新解释成 dtype_t。
}

constexpr float kFP8Margin = 1e-4;  // FP8 scale 计算中的小 margin，避免边界溢出。
constexpr float kFinfoAmaxE4M3 = 448.0f;  // E4M3 FP8 最大有限值。
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;  // E4M3 最大值倒数，减少运行时除法。

__forceinline__ __device__ float fast_pow2(int x) {
    // 通过直接构造 IEEE754 exponent bits 计算 2^x；调用方保证 -126<=x<=127，
    // 所以不处理 subnormal/overflow。
    uint32_t bits_x = (x + 127) << 23;  // FP32 exponent bias=127，mantissa/sign 置 0。
    return *reinterpret_cast<float*>(&bits_x);  // 直接把 bit pattern 解释为 float。
}

__forceinline__ __device__ int fast_log2_ceil(float x) {
    // 读取 FP32 exponent/mantissa 快速得到 ceil(log2(x))，用于把 FP8 scale 取整到
    // 2 的幂。mantissa 非 0 时说明需要向上取一档。
    auto bits_x = *reinterpret_cast<uint32_t*>(&x);  // 读取 FP32 原始 bit。
    auto exp_x = (bits_x >> 23) & 0xff;  // 提取 exponent 字段。
    auto man_bits = bits_x & ((1 << 23) - 1);  // 提取 mantissa 字段。
    return exp_x - 127 + (man_bits != 0);  // mantissa 非 0 时 ceil 需要加 1。
}

__forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale, float& scale_inv, bool round_scale) {
    // round_scale=True 使用 2 的幂 scale，便于 UE8M0 存储；False 使用精确比例
    // E4M3_MAX/amax，数值更准但 scale 需要 FP32 存储。
    // amax 是当前 token/hidden block 的绝对值最大值；scale 用于量化，scale_inv 用于反量化。
    if (round_scale) {  // 使用 2 的幂 scale，便于压缩存储。
        auto exp_scale_inv = fast_log2_ceil(amax * kFinfoAmaxInvE4M3);  // 计算 scale_inv 的 exponent。
        scale = fast_pow2(-exp_scale_inv);  // scale = 2^-exp。
        scale_inv = fast_pow2(exp_scale_inv);  // scale_inv = 2^exp。
    } else {
        scale_inv = amax * kFinfoAmaxInvE4M3;  // 精确反 scale，amax 映射到 448。
        scale = kFinfoAmaxE4M3 / amax;  // 精确 scale。
    }
}

template <bool kIsUE8M0, typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
__forceinline__ __device__ out_dtype_t extract_required_scale_format(float value) {
    // UE8M0 只保留 FP32 exponent 字段作为无符号 8-bit scale；否则直接返回 FP32 scale。
    // UE8M0 路径的输出通常写入 packed scale buffer，非 UE8M0 路径则写入 FP32 scale tensor。
    if constexpr (kIsUE8M0) {
        return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);  // 取 FP32 exponent 8 bit。
    } else {
        return value;  // 非 UE8M0 路径保留 FP32 scale。
    }
}

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int** barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);  // block 内线程 id。

    // kSyncOnly=false 时，barrier 不仅同步控制流，还发布本 block 之前的全局内存写入；
    // legacy notify/dispatch 使用它确保 peer 看到计数后也能看到对应 payload。
    if constexpr (not kSyncOnly) {
        memory_fence();  // 发布本 block 之前的系统域写入。
        __syncthreads();  // block 内线程同步，确保所有线程都完成 fence。
    }

    // Add self-ranks, sub other ranks
    if (thread_id < kNumRanks) {  // 前 kNumRanks 个线程分别负责一个 peer。
        atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, LEGACY_FINISHED_SUM_TAG);  // 在本 rank 的 peer 槽加完成标记。
        atomicSub_system(barrier_signal_ptrs[thread_id] + rank, LEGACY_FINISHED_SUM_TAG);  // 在对端观察本 rank 的槽减完成标记。
    }
    EP_DEVICE_ASSERT(kNumRanks <= blockDim.x);  // 确保有足够线程覆盖所有 rank。

    // Check timeout
    auto start_time = clock64();  // 记录开始等待的 GPU 时钟。
    while (true) {  // 轮询所有 peer 是否到达 barrier。
        auto value = thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;  // 每个线程读一个 peer 槽。
        if (__all_sync(0xffffffff, value <= 0))  // warp 内所有被观察槽都归零/非正时 barrier 完成。
            break;  // 退出轮询。

        if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {  // 超时保护，避免死等。
            printf("DeepEP timeout check failed: rank = %d, thread = %d, value = %d)\n", rank, thread_id, value);  // 打印 rank/thread/残留值。
            trap();  // 触发 device trap，暴露同步错误。
        }
    }
    __syncthreads();  // barrier 完成后 block 内再次同步。
}

__forceinline__ __device__ int atomic_cas_cta_acquire(int* addr, int x, int y) {
    // CTA acquire CAS 用于 shared-memory mutex；拿锁成功后可见前一个持锁者 release 的写入。
    int ret;  // CAS 返回旧值。
    asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;" : "=r"(ret) : "l"(addr), "r"(x), "r"(y) : "memory");  // CTA scope acquire CAS。
    return ret;  // 返回旧值。
}

__forceinline__ __device__ int atomic_exch_cta_release(int* addr, int x) {
    // CTA release exchange 用于释放 shared-memory mutex，并发布临界区内写入。
    int ret;  // exchange 返回旧值。
    asm volatile("atom.release.cta.shared::cta.exch.b32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(x) : "memory");  // CTA scope release exchange。
    return ret;  // 返回旧值。
}

__forceinline__ __device__ void acquire_lock(int* mutex) {
    // To make later memory operations valid, we must use `acquire` for memory semantics
    // 这个锁只用于 CTA 内 shared-memory 临界区，例如多个 sender warp 合并推进 shared tail/window。
    // acquire 保证拿到锁之后能看到前一个 release_lock 持锁者写出的窗口状态。
    while (atomic_cas_cta_acquire(mutex, 0, 1) != 0)  // CAS 0->1 成功才拿到锁。
        ;  // 自旋等待锁释放。
}

__forceinline__ __device__ void release_lock(int* mutex) {
    // To make previous memory operations visible to other threads, we must use `release` for memory semantics
    // release 把临界区内对 shared tail/window 的更新发布出去，下一位 acquire_lock 的线程才能按新状态继续。
    atomic_exch_cta_release(mutex, 0);  // release 写 0，释放锁并发布临界区写入。
}

// Operation functors
template <typename T>
struct ReduceSum {
    // functor 形式便于 warp_reduce 以模板参数内联不同归约操作。
    __device__ T operator()(T a, T b) const { return a + b; }  // 求和归约。
};
template <typename T>
struct ReduceMax {
    // 最大值归约常和 head/tail 进度搭配：一组 lane 中只要某个 lane 走得最远，就用 max 广播该进度。
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }  // 最大值归约。
};
template <typename T>
struct ReduceMin {
    // 最小值归约可表示“所有 lane 都至少完成到哪里”，因为最慢 lane 决定安全边界。
    __device__ T operator()(T a, T b) const { return a < b ? a : b; }  // 最小值归约。
};
template <typename T>
struct ReduceAnd {
    // 按位与常用于全员 ready 检测：只有所有 lane 的同一 bit 为 1，结果 bit 才为 1。
    __device__ T operator()(T a, T b) const { return a & b; }  // 按位与归约。
};
template <typename T>
struct ReduceOr {
    // 按位或常用于合并路由 mask：任一 lane 命中某 rank/expert，结果 mask 就保留该 bit。
    __device__ T operator()(T a, T b) const { return a | b; }  // 按位或归约。
};

// Unified reduction function
template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
    // kIntergroupReduce=false：在每个 lane group 内部归约；
    // kIntergroupReduce=true：把多个 group 的同位置 lane 进一步跨组归约。
    // 例如 kNumLanesPerGroup=8 时，一个 warp 分成 4 组；false 得到每组内部结果，
    // true 则会把 4 个 group 的对应 lane 合并。
    // XOR shuffle 的好处是每一步都在固定距离交换，不需要显式计算 group 起点；
    // 只要 kNumLanesPerGroup 是 2 的幂，就能通过编译期 if 裁剪出最少的 shuffle 指令。
    EP_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or kNumLanesPerGroup == 4 or
                         kNumLanesPerGroup == 2 or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");  // 只支持 2 的幂 lane group，便于 XOR shuffle 归约。
    constexpr uint32_t mask = 0xffffffff;  // 使用全 warp active mask。
    if constexpr (kIntergroupReduce) {  // intergroup 模式把多个小组的结果继续合并。
        // intergroup 方向是“跨 group 合并同一相对 lane”，所以 group 越小，需要跨越的距离越多。
        if constexpr (kNumLanesPerGroup <= 1)  // 1-lane 小组先跨 1 合并。
            value = op(value, __shfl_xor_sync(mask, value, 1));  // XOR shuffle 交换相邻 lane。
        if constexpr (kNumLanesPerGroup <= 2)  // 2-lane 或更小继续跨 2 合并。
            value = op(value, __shfl_xor_sync(mask, value, 2));  // 合并距离 2 的 lane。
        if constexpr (kNumLanesPerGroup <= 4)  // 4-lane 或更小继续跨 4 合并。
            value = op(value, __shfl_xor_sync(mask, value, 4));  // 合并距离 4 的 lane。
        if constexpr (kNumLanesPerGroup <= 8)  // 8-lane 或更小继续跨 8 合并。
            value = op(value, __shfl_xor_sync(mask, value, 8));  // 合并距离 8 的 lane。
        if constexpr (kNumLanesPerGroup <= 16)  // 16-lane 或更小继续跨 16 合并。
            value = op(value, __shfl_xor_sync(mask, value, 16));  // 合并半 warp。
    } else {
        // intragroup 方向是“组内合并”，所以 group 越大，需要从半组距离开始逐步缩小。
        if constexpr (kNumLanesPerGroup >= 32)  // 32-lane 组先合并半 warp。
            value = op(value, __shfl_xor_sync(mask, value, 16));  // 合并距离 16 的 lane。
        if constexpr (kNumLanesPerGroup >= 16)  // 16-lane 组继续合并距离 8。
            value = op(value, __shfl_xor_sync(mask, value, 8));  // 合并距离 8 的 lane。
        if constexpr (kNumLanesPerGroup >= 8)  // 8-lane 组继续合并距离 4。
            value = op(value, __shfl_xor_sync(mask, value, 4));  // 合并距离 4 的 lane。
        if constexpr (kNumLanesPerGroup >= 4)  // 4-lane 组继续合并距离 2。
            value = op(value, __shfl_xor_sync(mask, value, 2));  // 合并距离 2 的 lane。
        if constexpr (kNumLanesPerGroup >= 2)  // 2-lane 组最终合并相邻 lane。
            value = op(value, __shfl_xor_sync(mask, value, 1));  // 合并距离 1 的 lane。
    }
    return value;  // 返回当前 lane 的归约结果；是否所有 lane 都有效取决于调用约定。
}

// Convenience aliases
// 下面这些 alias 只是固定 Op 类型，调用点可以只写 warp_reduce_sum/max 等更可读的名字。
// kNumLanesPerGroup 默认 32 表示整 warp 归约；设置为 8/4 等时常用于一个 warp 内多个 rank/专家小组。
template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    // 求和常用于统计 token 数、累计 offset 或合并局部计数。
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});  // warp/group 求和。
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
    // 最大值常用于聚合 tail/head 或等待状态中的最远进度。
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});  // warp/group 最大值。
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
    // 最小值可用于判断一组 lane 是否都到达某个阈值。
    // 例：每 lane 持有一个 peer 的剩余计数，min<=0 可帮助定位最慢/已完成 peer。
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMin<T>{});  // warp/group 最小值。
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_and(T value) {
    // 按位与常用于 all-true/all-ready 状态压缩。
    // 对 bool/int mask 调用时，调用方通常只读取每个 group 的代表 lane 结果。
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceAnd<T>{});  // warp/group 按位与。
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_or(T value) {
    // 按位或常用于把多个 lane 的 rank mask/top-k mask 合成一个 bitset。
    // 在 top-k 路由里，一个 warp 可用 or 把 32 个 lane 的专家命中信息压成一个 rank mask。
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceOr<T>{});  // warp/group 按位或。
}

}  // namespace deep_ep::legacy
