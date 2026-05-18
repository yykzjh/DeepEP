#pragma once  // 共享内存 helper 被 legacy buffer 和测试路径多处包含，防止重复定义。

#include <cuda.h>                        // CUDA driver API：CUmemFabricHandle、CUmemMap 等。
#include <deep_ep/common/exception.cuh>  // CUDA_RUNTIME_CHECK/CUDA_DRIVER_CHECK/EP_HOST_ASSERT。

#include "lazy_driver.hpp"  // 延迟加载 CUDA driver 符号，兼容不同 CUDA 版本/运行环境。

namespace deep_ep::shared_memory {  // legacy 共享 GPU buffer 的工具命名空间。

// MemHandle 同时支持 CUDA IPC 和 CUDA Fabric 两类共享句柄。legacy Buffer 会把本 rank 的
// MemHandle 通过 Python all_gather 发送给同节点 peer，peer 打开后得到可被 GPU kernel
// 直接访问的远端 buffer 指针。
union MemHandleInner {
    cudaIpcMemHandle_t cuda_ipc_mem_handle;     // 常规 CUDA IPC 句柄，适合单机多进程共享 cudaMalloc 内存。
    CUmemFabricHandle cu_mem_fabric_handle;     // CUDA Fabric 句柄，适合支持 Fabric 的跨设备/跨进程映射。
};

struct MemHandle {
    MemHandleInner inner;  // 实际共享句柄；由 use_fabric 决定解释为 IPC 还是 Fabric。
    size_t size;           // Fabric 打开映射时需要知道映射长度；IPC 分支也记录下来方便统一传输结构。
};

static void cu_mem_set_access_all(void* ptr, size_t size) {
    int device_count;                               // 当前进程可见 GPU 数量。
    CUDA_RUNTIME_CHECK(cudaGetDeviceCount(&device_count));  // Fabric 映射需要给每张 GPU 设置访问权限。

    constexpr int kMaxDeviceCount = 8;              // 本实现用栈数组保存访问描述，假定单节点最多 8 卡。
    EP_HOST_ASSERT(0 < device_count and device_count <= kMaxDeviceCount);  // 超出假设时直接报错，避免数组越界。

    CUmemAccessDesc access_desc[kMaxDeviceCount];   // 每个可见 GPU 一条访问规则。
    for (int i = 0; i < device_count; ++ i) {       // 给所有本地 GPU 授权，peer kernel 才能读写映射地址。
        access_desc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;  // 访问主体是 GPU device。
        access_desc[i].location.id = i;                              // 第 i 张 GPU。
        access_desc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;   // 允许读写，用于 token payload 和同步 flag。
    }
    CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(reinterpret_cast<CUdeviceptr>(ptr), size, access_desc, device_count));  // 应用访问权限。
}

static void cu_mem_free(void* ptr) {
    CUmemGenericAllocationHandle handle;  // driver allocation 的底层 handle，需要 release。
    CUDA_DRIVER_CHECK(lazy_cuMemRetainAllocationHandle(&handle, ptr));  // 从虚拟地址反查 allocation handle。

    size_t size = 0;  // 查询映射长度，后续 unmap/address free 都要用相同 size。
    CUDA_DRIVER_CHECK(lazy_cuMemGetAddressRange_v2(nullptr, &size, reinterpret_cast<CUdeviceptr>(ptr)));  // 查询地址范围。

    CUDA_DRIVER_CHECK(lazy_cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), size));       // 解除虚拟地址到物理 allocation 的映射。
    CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size)); // 释放保留的虚拟地址范围。
    CUDA_DRIVER_CHECK(lazy_cuMemRelease(handle));                                       // 释放底层 allocation handle。
}

class SharedMemoryAllocator {
public:
    explicit SharedMemoryAllocator(const bool& use_fabric) : use_fabric(use_fabric) {}  // use_fabric 决定走 CUDA Fabric 还是 CUDA IPC。

    void malloc(void** ptr, size_t size) const {
        // Fabric 分支需要按 driver API 的 allocation granularity 对齐，并显式设置所有 device
        // 可读写权限；普通分支直接 cudaMalloc，之后通过 cudaIpcGetMemHandle 暴露给 peer。
        if (use_fabric) {
            CUdevice device;  // Fabric allocation 绑定到当前 CUDA context 所在 device。
            CUDA_DRIVER_CHECK(lazy_cuCtxGetDevice(&device));  // 查询当前 device id。

            CUmemAllocationProp prop = {};                          // 描述将要创建的 driver allocation。
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;              // GPU pinned allocation，允许显式 map/unmap。
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;        // allocation 位于某个 CUDA device。
            prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;   // 要求导出 Fabric shareable handle。
            prop.location.id = device;                               // 绑定当前 device。

            size_t alignment = 0;  // driver API 要求 size 按 allocation granularity 对齐。
            EP_HOST_ASSERT(size > 0);  // 共享 buffer 长度必须为正。
            CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&alignment, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));  // 获取最小对齐。
            size = ((size + alignment - 1) / alignment) * alignment;  // 向上对齐，例如 1.5MB 对齐到 2MB。

            CUmemGenericAllocationHandle handle;  // 底层物理 allocation handle。
            CUDA_DRIVER_CHECK(lazy_cuMemCreate(&handle, size, &prop, 0));  // 创建可导出 Fabric handle 的 allocation。
            // Fabric 分支显式拆分“物理 allocation”和“虚拟地址”，方便跨进程导入同一 allocation。
            CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(ptr), size, alignment, 0, 0));  // 预留虚拟地址。
            CUDA_DRIVER_CHECK(lazy_cuMemMap(reinterpret_cast<CUdeviceptr>(*ptr), size, 0, handle, 0));  // 把 allocation map 到虚拟地址。
            cu_mem_set_access_all(*ptr, size);  // 授权所有本地 device 读写这段地址。
        } else {
            CUDA_RUNTIME_CHECK(cudaMalloc(ptr, size));  // 普通 IPC 分支用 cudaMalloc，后续通过 cudaIpcGetMemHandle 导出。
        }
    }

    void free(void* ptr) const {
        if (use_fabric) {
            cu_mem_free(ptr);  // Fabric 分支必须按 driver API 的 unmap/addressFree/release 顺序释放。
        } else {
            CUDA_RUNTIME_CHECK(cudaFree(ptr));  // IPC 分支按普通 cudaMalloc 释放。
        }
    }

    void get_mem_handle(MemHandle* mem_handle, void* ptr) const {
        size_t size = 0;  // 记录导出 allocation 的实际映射大小。
        CUDA_DRIVER_CHECK(lazy_cuMemGetAddressRange_v2(nullptr, &size, reinterpret_cast<CUdeviceptr>(ptr)));  // 查询 ptr 所属地址范围。
        mem_handle->size = size;  // Python all_gather 传递 MemHandle 时连同 size 一起传给 peer。
        // size 必须随 handle 一起传递，因为导入方 reserve/map 需要完全一致的映射长度。

        if (use_fabric) {
            CUmemGenericAllocationHandle handle;  // 从虚拟地址拿到底层 allocation handle。
            CUDA_DRIVER_CHECK(lazy_cuMemRetainAllocationHandle(&handle, ptr));  // 增加引用以便导出。
            // retain 后的 handle 用于 export；导入方再通过 ImportFromShareableHandle 拿到同一 allocation。
            CUDA_DRIVER_CHECK(lazy_cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));  // 导出 Fabric handle。
        } else {
            CUDA_RUNTIME_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));  // 导出 CUDA IPC handle。
        }
    }

    void open_mem_handle(void** ptr, MemHandle* mem_handle) const {
        // 打开 peer 传来的 handle 后，本进程得到一个远端映射指针。
        // csrc/legacy/buffer.hpp 会把这些指针复制到 buffer_ptrs_gpu，legacy intranode kernel
        // 通过该表完成跨 rank NVLink 写入。
        if (use_fabric) {
            size_t size = mem_handle->size;  // peer 侧按导出方记录的大小 reserve/map。

            CUmemGenericAllocationHandle handle;  // 导入后的本进程 allocation handle。
            CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));  // 导入 Fabric handle。

            CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(ptr), size, 0, 0, 0));  // 在本进程预留虚拟地址。
            CUDA_DRIVER_CHECK(lazy_cuMemMap(reinterpret_cast<CUdeviceptr>(*ptr), size, 0, handle, 0));        // 将 peer allocation 映射进本进程。
            cu_mem_set_access_all(*ptr, size);  // 授权本地所有 GPU 访问这个 peer 映射。
            // 打开后的 *ptr 是本进程虚拟地址，但底层物理页属于导出方 allocation。
        } else {
            CUDA_RUNTIME_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));  // 懒开启 peer access，降低初始化开销。
        }
    }

    void close_mem_handle(void* ptr) const {
        if (use_fabric) {
            // Fabric peer 映射和本地 allocation 都用 cu_mem_free 收尾，但语义分别是 close map / free owner map。
            cu_mem_free(ptr);  // 关闭 Fabric 映射，释放本进程虚拟地址和 handle 引用。
        } else {
            CUDA_RUNTIME_CHECK(cudaIpcCloseMemHandle(ptr));  // 关闭 CUDA IPC 映射；不释放导出方的原始 cudaMalloc。
        }
    }

private:
    bool use_fabric;  // true 表示使用 CUDA Fabric handle；false 表示使用传统 CUDA IPC handle。
};

}  // namespace deep_ep::shared_memory
