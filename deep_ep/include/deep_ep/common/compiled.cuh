#pragma once  // 公共编译期类型/宏头，避免重复包含。

// 让 CLion/IDE 在没有真实 nvcc 编译上下文时也能解析 CUDA 条件编译分支。
// 这些宏只影响 IDE 索引，不会进入正常编译路径。
// __CLION_IDE__ 由 JetBrains CUDA 语法解析器定义，真实 nvcc 编译不会走这里。
#ifdef __CLION_IDE__  // 仅 IDE 索引路径生效。
// 伪造 SM90，保证 IDE 能看到 TMA/FP8 等 Hopper 分支声明。
#define __CUDA_ARCH__ 900
// 伪造 relocatable device code 标志，避免 device-side NCCL/Gin 声明被 IDE 跳过。
#define __CUDACC_RDC__
// 伪造 CUDACC，启用 CUDA 相关 include 和 __device__ 语法路径。
#define __CUDACC__
#endif

// PyTorch 扩展默认会禁用部分 half/bfloat16 运算符；DeepEP kernel 内部需要直接做
// BF16/FP8 向量计算，因此在本公共头里解除这些限制。
// 如果上游已经定义禁用宏，这里统一 undef，让 CUDA intrinsic/operator 可用。
#ifdef __CUDA_NO_HALF_CONVERSIONS__  // 如果 PyTorch 禁用了 half 转换则解除。
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
// 解除 half 标量运算符禁用。
#ifdef __CUDA_NO_HALF_OPERATORS__  // 如果 PyTorch 禁用了 half operator 则解除。
#undef __CUDA_NO_HALF_OPERATORS__
#endif
// 解除 half2 向量运算符禁用。
#ifdef __CUDA_NO_HALF2_OPERATORS__  // 如果 PyTorch 禁用了 half2 operator 则解除。
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
// 解除 bfloat16 转换禁用。
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__  // 如果 PyTorch 禁用了 bfloat16 转换则解除。
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
// 解除 bfloat162 向量运算符禁用。
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__  // 如果 PyTorch 禁用了 bfloat162 operator 则解除。
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

// cstdint 提供 int8_t/int16_t/int32_t/int64_t 等固定宽度整型。
#include <cstdint>
// cuda_bf16.h 提供 nv_bfloat16/nv_bfloat162 类型和 intrinsic。
#include <cuda_bf16.h>
// cuda_runtime.h 提供 CUDA runtime 类型、CUDART_VERSION 和错误码。
#include <cuda_runtime.h>

// DISABLE_SM90_FEATURES 用于在不支持 FP8/TMA 的平台上关闭 SM90 特性。
#ifndef DISABLE_SM90_FEATURES  // 默认启用 SM90 FP8/TMA 相关头。
// Hopper/SM90+ 路径使用 CUDA 官方 FP8 类型。
#include <cuda_fp8.h>
#else
// Ampere does not support FP8 features
// 旧架构没有 cuda_fp8.h 中的枚举和类型，这里给出占位定义让源码可编译。
#define __NV_E4M3 0
#define __NV_E5M2 1
// 中文：这些 typedef 只用于类型占位，禁用 SM90 特性时不会实际执行 FP8 Tensor Core 路径。
typedef int __nv_fp8_interpretation_t;
// 中文：用 int/uint8_t 占位可让同一套模板代码在 Ampere 上通过编译。
typedef int __nv_fp8x4_e4m3;
typedef uint8_t __nv_fp8_storage_t;
#endif

// CUDA 13 开始提供 longlong4_32a 作为 256-bit 对齐 LD/ST 类型；旧 CUDA 版本手工
// 定义相同大小/对齐的结构，供 SM100+ combine 规约路径一次搬 32B 使用。
// CUDART_VERSION 只有包含 cuda_runtime.h 后才可靠。
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000  // CUDA 13+ 已提供官方 32B 对齐类型。
// CUDA 13+ 直接复用 runtime 提供的 32B 对齐类型。
using longlong4_t = longlong4_32a;
// 构造函数名也做同名别名，隐藏 CUDA 版本差异。
#define make_longlong4_t make_longlong4_32a
#else
// 旧 CUDA 手动定义 32B 对齐的四 long long 结构。
struct alignas(32) longlong4_t { long long x, y, z, w; };  // 旧 CUDA 下手工定义 256-bit 向量。
// 设备端 inline 构造函数，行为与 CUDA 13 的 make_longlong4_32a 对齐。
__device__ __forceinline__ longlong4_t make_longlong4_t(
    const long long& x, const long long& y, const long long& z, const long long& w) {  // 四个 64-bit 分量。
    // 聚合初始化返回四个 64-bit lane。
    return {x, y, z, w};
}
#endif

// EP_NUM_TOPK_IDX_BITS 可由编译参数覆盖，用于选择 topk_idx_t 位宽。
#ifndef EP_NUM_TOPK_IDX_BITS  // 未通过编译参数指定时使用默认位宽。
// 默认使用 64-bit，兼容大 expert id 或上层期望 long tensor 的情况。
#define EP_NUM_TOPK_IDX_BITS 64
#endif

// 公共类型位于 deep_ep 命名空间，elastic 子命名空间会直接引用这些别名。
namespace deep_ep {  // DeepEP 公共类型命名空间。

// kEnableSM90Features 暴露给模板/host 侧代码，用于选择是否启用 FP8/TMA 特性。
#ifndef DISABLE_SM90_FEATURES
constexpr bool kEnableSM90Features = true;  // 编译期标记：SM90 特性可用。
#else
constexpr bool kEnableSM90Features = false;  // 编译期标记：SM90 特性禁用。
#endif

// int_with_bits 把 bit 数映射到对应固定宽度整型。
template <int kNumBits> struct int_with_bits;  // bit 数到整型类型的映射模板。
// 8-bit top-k index 类型。
template <> struct int_with_bits<8>  { using type = int8_t;  };  // 8-bit 特化。
// 16-bit top-k index 类型。
template <> struct int_with_bits<16> { using type = int16_t; };  // 16-bit 特化。
// 32-bit top-k index 类型。
template <> struct int_with_bits<32> { using type = int32_t; };  // 32-bit 特化。
// 64-bit top-k index 类型。
template <> struct int_with_bits<64> { using type = int64_t; };  // 64-bit 特化。

// topk_idx_t 是 kernel 内统一的 top-k expert id 类型。
using topk_idx_t = int_with_bits<EP_NUM_TOPK_IDX_BITS>::type;  // top-k expert id 类型别名。

// sf_pack_t 同时承载 FP32 scale 和 UE8M0x4 打包 scale。FP8 dispatch 中，
// Python 侧看到的 scale 逻辑形状通常是 [num_tokens, hidden / 128]，
// kernel 侧会按 sf_pack_t 向量化搬运以满足 16B 对齐。
union sf_pack_t {  // FP8 scale factor 的统一存储单元。
    // FP32 scale 路径直接以 float 解释。
    float fp32;  // FP32 scale 表示。
    // UE8M0x4 路径把 4 个 8-bit scale 打包到一个 int。
    int ue8m0x4;  // 4 个 UE8M0 scale 打包表示。
};

// TMA/通信 buffer 的基础对齐字节数。
constexpr int kNumTMAAlignedBytes = 16;  // TMA 最小对齐字节数。
// 一个 16B 对齐块内可容纳多少个 sf_pack_t。
constexpr int kNumAlignedSFPacks = 16 / sizeof(sf_pack_t);  // 一个 16B 块里的 scale pack 数。

// 通信 channel 的上界用于 workspace 中静态控制数组；实际 channel 数由 dispatch
// 配置和 rank 拓扑决定，不能超过这个常量。
// 这个常量也帮助 host 侧提前分配足够大的 channel metadata 区。
constexpr int kNumMaxChannels = 1024;  // workspace 静态预留最大 channel 数。

} // namespace deep_ep
