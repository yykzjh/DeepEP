#pragma once  // CUDA driver 懒加载 wrapper 会被多个后端文件包含，防止重复定义。

#include <cuda.h>   // 声明 cuModuleLoad/cuLaunchKernelEx/cuMemMap 等 CUDA driver API。
#include <dlfcn.h>  // 使用 dlopen/dlsym 在运行时解析 libcuda.so.1 中的符号。

#include <deep_ep/common/exception.cuh>  // EP_HOST_ASSERT 用于检查 driver library 和符号是否成功加载。

namespace deep_ep {  // DeepEP C++ 公共工具命名空间。

// Lazy loading all driver symbols
// 通过 dlopen/dlsym 懒解析 CUDA driver API，避免链接期强依赖所有 driver 符号。
// JIT driver launch、CUDA Fabric memory、batched memory op 等路径才会真正触发这些函数解析。
static void* get_driver_handle() {
    static void* handle = nullptr;  // 进程内缓存 libcuda.so.1 的动态库句柄，避免每次 wrapper 调用都 dlopen。
    if (handle == nullptr) {  // 首次使用任意 driver API 时才真正加载 libcuda。
        handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);  // RTLD_LAZY 延迟解析符号，RTLD_LOCAL 避免污染全局符号表。
        EP_HOST_ASSERT(handle != nullptr and "Failed to load CUDA driver `libcuda.so.1`");  // 没有 NVIDIA driver 时直接给出明确错误。
    }
    return handle;  // 返回缓存句柄，供 dlsym 查找具体 cu* 函数。
}

// Macro to define wrapper functions named `lazy_cu{API name}`
#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name) \
/* 中文：宏生成的 lazy wrapper 在首次调用时解析 driver symbol，之后复用函数指针。 */ \
template <typename... Args> \
static auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&name); /* 保存原始 cu* 函数指针类型，确保 wrapper 返回值和参数完全匹配。 */ \
    static FuncType func = nullptr; /* 每个 API 单独缓存 dlsym 结果，避免重复符号查找。 */ \
    if (func == nullptr) { /* 首次调用 lazy_cuXXX 时解析真实 driver 符号。 */ \
        func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), #name)); /* 从 libcuda.so.1 中按函数名查找符号。 */ \
        EP_HOST_ASSERT(func != nullptr and "Failed to load CUDA driver API"); /* 当前 driver 缺少该 API 时立即报错。 */ \
    } \
    return func(std::forward<decltype(args)>(args)...); /* 完美转发参数到真实 CUDA driver API。 */ \
}

DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);                  // 懒加载 cuGetErrorName，用于把 CUresult 转成错误名。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorString);                // 懒加载 cuGetErrorString，用于把 CUresult 转成可读错误说明。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuFuncSetAttribute);              // 懒加载 cuFuncSetAttribute，JIT kernel 设置 opt-in shared memory 时使用。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleLoad);                    // 懒加载 cuModuleLoad，KernelRuntime 加载编译出的 cubin 时使用。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleUnload);                  // 懒加载 cuModuleUnload，KernelRuntime 析构/释放 cubin module 时使用。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleGetFunction);             // 懒加载 cuModuleGetFunction，从 cubin module 中取 kernel symbol。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLaunchKernelEx);                // 懒加载 cuLaunchKernelEx，LaunchRuntime 用它支持 launch attribute。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemSetAccess);                  // 懒加载 cuMemSetAccess，CUDA Fabric 映射后设置多 GPU 访问权限。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemRetainAllocationHandle);     // 懒加载 cuMemRetainAllocationHandle，从指针反查 driver allocation handle。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemGetAddressRange_v2);         // 懒加载 cuMemGetAddressRange_v2，查询映射指针对应的地址范围和大小。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemAddressReserve);             // 懒加载 cuMemAddressReserve，为 Fabric allocation 预留虚拟地址。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemAddressFree);                // 懒加载 cuMemAddressFree，释放预留的虚拟地址范围。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemMap);                        // 懒加载 cuMemMap，把 driver allocation 映射到虚拟地址。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemUnmap);                      // 懒加载 cuMemUnmap，解除虚拟地址和 allocation 的映射。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemCreate);                     // 懒加载 cuMemCreate，创建可导出 Fabric handle 的 allocation。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemRelease);                    // 懒加载 cuMemRelease，释放 driver allocation handle 引用。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuCtxGetDevice);                  // 懒加载 cuCtxGetDevice，查询当前 CUDA context 绑定的 device。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemImportFromShareableHandle);  // 懒加载 cuMemImportFromShareableHandle，peer 进程导入 Fabric handle。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemExportToShareableHandle);    // 懒加载 cuMemExportToShareableHandle，导出 Fabric shareable handle。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemGetAllocationGranularity);   // 懒加载 cuMemGetAllocationGranularity，计算 Fabric allocation 对齐大小。
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuStreamBatchMemOp);              // 懒加载 cuStreamBatchMemOp，批量写/等同步 flag。

}  // namespace deep_ep  // 结束 DeepEP C++ 公共工具命名空间。
