#pragma once  // JIT CUDA handle 抽象会被 KernelRuntime/LaunchRuntime 包含，防止重复定义。

#include <cuda.h>                      // CUDA runtime/driver handle 类型都来自这里。
#include <filesystem>                  // load_kernel 接收 cubin 文件路径。
#include <deep_ep/common/exception.cuh> // CUDA_RUNTIME_CHECK/CUDA_DRIVER_CHECK/EP_HOST_ASSERT。

#include "../utils/lazy_driver.hpp"  // driver API 分支通过 lazy_cu* 懒加载符号。

namespace deep_ep::jit {  // DeepEP JIT 子系统命名空间。

#if CUDART_VERSION >= 12080 and defined(EP_JIT_USE_RUNTIME_API)

// Use CUDA runtime API
// CUDA 12.8+ 的 runtime API 可以直接加载 cubin 并拿到 cudaKernel_t；否则走 driver API。
// 两套 API 的类型不同，但下面统一 typedef 成 LibraryHandle/KernelHandle/LaunchConfigHandle，
// 上层 LaunchRuntime 就不需要关心具体后端。
using LibraryHandle = cudaLibrary_t;        // runtime API 中已加载 cubin library 的句柄。
using KernelHandle = cudaKernel_t;          // runtime API 中可 launch kernel 的句柄。
using LaunchConfigHandle = cudaLaunchConfig_t; // runtime API 的 launch config 结构。
using LaunchAttrHandle = cudaLaunchAttribute;  // runtime API 的 launch attribute 结构。

#define EP_CUDA_UNIFIED_CHECK CUDA_RUNTIME_CHECK  // 上层统一错误检查宏，在 runtime 分支映射到 CUDA runtime 检查。

static KernelHandle load_kernel(const std::filesystem::path& cubin_path, const std::string& func_name,
                                LibraryHandle *library_opt = nullptr) {
    LibraryHandle library;  // 保存 cudaLibraryLoadFromFile 返回的 library 句柄。
    KernelHandle kernel{};  // 保存 cudaLibraryGetKernel 返回的 kernel 句柄。
    CUDA_RUNTIME_CHECK(cudaLibraryLoadFromFile(&library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));  // 从 cubin 文件加载 CUDA library。
    nvshmemx_culibrary_init(library);  // 让 NVSHMEM 识别这个 CUDA library 中的 device 符号。
    CUDA_RUNTIME_CHECK(cudaLibraryGetKernel(&kernel, library, func_name.c_str()));  // 按符号名取出真正的 kernel。

    if (library_opt != nullptr)  // 调用方想保存 library 以便析构时 unload。
        *library_opt = library;  // 把 library handle 写回输出指针。
    return kernel;  // 返回可 launch 的 kernel handle。
}

static void unload_library(const LibraryHandle& library) {
    nvshmemx_culibrary_finalize(library);  // 通知 NVSHMEM 这个 CUDA library 即将卸载。
    const auto error = cudaLibraryUnload(library);  // 卸载 runtime API library。
    EP_HOST_ASSERT(error == cudaSuccess or error == cudaErrorCudartUnloading);  // 允许进程退出时 cudart 正在卸载的特殊返回。
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel,
                                                  const cudaStream_t& stream, const int& smem_size,
                                                  const dim3& grid_dim, const dim3& block_dim, const int& cluster_dim) {
    if (smem_size > 0)  // 只有动态 shared memory 大于 0 时才需要 opt-in 设置。
        CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));  // 允许 kernel 使用更大动态共享内存。

    LaunchConfigHandle config;              // runtime API launch config。
    config.gridDim = grid_dim;              // 设置 grid 维度。
    config.blockDim = block_dim;            // 设置 block 维度。
    config.dynamicSmemBytes = smem_size;    // 设置动态共享内存字节数。
    config.stream = stream;                 // 设置 launch 所在 CUDA stream。
    config.numAttrs = 0;                    // 默认没有 launch attribute。
    config.attrs = nullptr;                 // 默认 attribute 指针为空。

    // TODO: support cooperative and dependent kernel launch
    // runtime API 分支目前只设置 cluster attribute；driver API 分支支持 cooperative 和 PDL。
    // 因此如果需要完整 SM90 launch 属性，默认仍建议走 driver API 分支。
    // NOTES: must use `static` or the `attr` will be deconstructed
    static LaunchAttrHandle attr;  // 必须静态存储，避免 config 返回后 attrs 指向栈内存。
    if (cluster_dim > 1) {  // cluster_dim=1 表示不启用 thread block cluster。
        attr.id = cudaLaunchAttributeClusterDimension;  // 指定 attribute 类型为 cluster 维度。
        attr.val.clusterDim = {static_cast<unsigned>(cluster_dim), 1, 1};  // 只使用 x 维 cluster。
        config.attrs = &attr;  // 把 attribute 地址挂到 config。
        config.numAttrs = 1;   // runtime API 分支目前只设置一个 attribute。
    }
    return config;  // 返回完整 launch config。
}

template<typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };  // CUDA launch API 要求传入参数地址数组，每个元素指向一个 kernel 参数。
    return cudaLaunchKernelExC(&config, kernel, ptr_args);  // 使用 runtime API 启动 kernel。
}

#else

// Use CUDA driver API
// driver API 分支是当前 JIT launch 的主路径。它能设置 cooperative launch、cluster dim、
// programmatic dependent launch(PDL)，这些属性对 DeepEP 的跨 block barrier/TMA overlap 很重要。
using LibraryHandle = CUmodule;          // driver API 中已加载 cubin module 的句柄。
using KernelHandle = CUfunction;         // driver API 中可 launch kernel function 的句柄。
using LaunchConfigHandle = CUlaunchConfig; // driver API launch config 结构。
using LaunchAttrHandle = CUlaunchAttribute; // driver API launch attribute 结构。

#define EP_CUDA_UNIFIED_CHECK CUDA_DRIVER_CHECK  // 上层统一错误检查宏，在 driver 分支映射到 CUDA driver 检查。

static KernelHandle load_kernel(const std::filesystem::path& cubin_path, const std::string& func_name,
                               LibraryHandle *library_opt = nullptr) {
    LibraryHandle library;  // 保存 cuModuleLoad 返回的 module。
    KernelHandle kernel;    // 保存 cuModuleGetFunction 返回的 kernel function。
    CUDA_DRIVER_CHECK(lazy_cuModuleLoad(&library, cubin_path.c_str()));  // 从 cubin 文件加载 CUDA module。
    CUDA_DRIVER_CHECK(lazy_cuModuleGetFunction(&kernel, library, func_name.c_str()));  // 按符号名取出 kernel function。

    if (library_opt != nullptr)  // 调用方需要保存 module，析构时卸载。
        *library_opt = library;  // 把 module handle 写回输出参数。
    return kernel;  // 返回可 launch 的 CUfunction。
}

static void unload_library(const LibraryHandle& library) {
    const auto error = lazy_cuModuleUnload(library);  // 卸载 CUDA driver module。
    EP_HOST_ASSERT(error == CUDA_SUCCESS or error == CUDA_ERROR_DEINITIALIZED);  // 允许 CUDA driver 已经反初始化时的退出路径。
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel,
                                                 const cudaStream_t& stream, const int& smem_size,
                                                 const dim3& grid_dim, const dim3& block_dim, const int& cluster_dim,
                                                 // 中文：cooperative/PDL 会影响底层 launch API 和 kernel 调度语义。
                                                 const bool& cooperative, const bool& enable_pdl) {
    if (smem_size > 0)  // 只有动态 shared memory 大于 0 时才需要设置函数属性。
        CUDA_DRIVER_CHECK(lazy_cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size));  // opt-in 更大动态共享内存。

    LaunchConfigHandle config;              // driver API launch config。
    config.gridDimX = grid_dim.x;           // grid x 维。
    config.gridDimY = grid_dim.y;           // grid y 维。
    config.gridDimZ = grid_dim.z;           // grid z 维，DeepEP 通常为 1。
    config.blockDimX = block_dim.x;         // block x 维线程数。
    config.blockDimY = block_dim.y;         // block y 维，DeepEP 通常为 1。
    config.blockDimZ = block_dim.z;         // block z 维，DeepEP 通常为 1。
    config.sharedMemBytes = smem_size;      // 动态共享内存大小。
    config.hStream = stream;                // launch 所在 CUDA stream。

    // Create attributes
    static LaunchAttrHandle attrs[3];  // 最多 cooperative、cluster、PDL 三个属性；静态存储避免返回后悬挂。
    config.attrs = attrs;              // 把属性数组挂到 config。
    config.numAttrs = 0;               // 后续按需追加属性。

    // Cooperative launch
    // dispatch/combine 中的 device barrier 可能使用 cooperative_groups::this_grid().sync()，
    // 因此 host launch 必须显式设置 cooperative attribute。
    if (cooperative) {
        auto& attr = attrs[config.numAttrs ++];  // 取下一个 attribute 槽位并递增数量。
        attr.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;  // 设置 cooperative launch 属性。
        attr.value.cooperative = 1;  // 启用 cooperative 模式。
    }

    // Cluster size
    // NOTES: must use `static` or the `attr` will be deconstructed
    if (cluster_dim > 1) {
        auto& attr = attrs[config.numAttrs ++];  // 取下一个 attribute 槽位。
        attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;  // 设置 thread block cluster 维度属性。
        attr.value.clusterDim.x = cluster_dim;  // cluster x 维大小。
        attr.value.clusterDim.y = 1;            // cluster y 维固定 1。
        attr.value.clusterDim.z = 1;            // cluster z 维固定 1。
    }

    // Dependent kernel launch
    // PDL 允许后续 kernel 在前序 kernel 尚未完全结束时按 CUDA 的程序化依赖启动，
    // DeepEP 在 copy/reduce epilogue 场景中可借此减少 kernel 间空泡。
    if (enable_pdl) {
        auto& attr = attrs[config.numAttrs ++];  // 取下一个 attribute 槽位。
        attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;  // 设置 PDL 相关属性。
        attr.value.programmaticStreamSerializationAllowed = 1;  // 允许程序化 stream serialization。
    }
    return config;  // 返回完整 driver launch config。
}

template<typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };  // driver launch 也要求参数地址数组，每个元素指向一个 kernel 参数。
    return lazy_cuLaunchKernelEx(&config, kernel, ptr_args, nullptr);  // 使用 cuLaunchKernelEx 启动 kernel。
}

#endif

} // namespace deep_ep::jit  // 结束 DeepEP JIT 子系统命名空间。
