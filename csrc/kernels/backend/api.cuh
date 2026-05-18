#pragma once  // backend 对外声明会被 legacy/elastic runtime 多处包含，使用 pragma once 防止重复声明。

#include <memory>    // 预留给后端上下文智能指针；当前头里主要由实现文件使用。
#include <vector>    // unique id 字节数组、NCCL window 指针列表都用 std::vector 表达。
#include <optional>  // NVSHMEM barrier 可选接收 cudaStream_t，表示同步可以绑定到某个 stream。

#include <nccl.h>         // host 侧 ncclComm_t/ncclUniqueId/ncclWindow_t 等类型。
#include <nccl_device.h>  // device 侧 ncclDevComm_t/Gin 相关类型，Elastic kernel 需要通过 handle 间接使用。

// TODO: make a unified API
namespace deep_ep::nvshmem {  // legacy/V1 跨节点后端封装；主要服务 csrc/legacy/buffer.hpp。

std::vector<uint8_t> get_unique_id();  // rank0 生成 NVSHMEM 初始化所需的唯一 ID，再由用户进程广播给其他 rank。

int init(const std::vector<uint8_t>& root_unique_id_val,
         const int& rank,
         const int& num_ranks,
         const int& team_split_stride);  // 初始化 NVSHMEM team；team_split_stride 用于按节点/设备拓扑切分通信域。

void* alloc(const size_t& size, const size_t& alignment);  // 分配 NVSHMEM 对称内存，各 rank 返回同语义地址空间的 buffer。

void free(void* ptr);  // 释放 alloc 得到的 NVSHMEM 对称内存。

void barrier(const bool& with_cpu_sync, const std::optional<cudaStream_t>& stream_opt = std::nullopt);  // rank 间 barrier，可选择 CPU 同步或 stream 上排队同步。

void finalize();  // 关闭 NVSHMEM runtime；对应 init 的生命周期收尾。

}  // deep_ep::nvshmem

namespace deep_ep::nccl {  // Elastic/V2 主要后端封装；通过 NCCL window/Gin 暴露 GPU 间通信能力。

pybind11::bytearray get_local_unique_id();  // 生成 ncclUniqueId 并作为 Python bytearray 返回，方便 torch.distributed 广播。

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx);  // 基于广播后的 unique id 创建 ncclComm_t，并用 int64_t 句柄传回 Python。

void destroy_nccl_comm(const int64_t& nccl_comm);  // 销毁 create_nccl_comm 返回的 communicator 句柄。

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm);  // 返回物理拓扑维度，通常是 (num_rdma_ranks, num_nvl_ranks)。

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode);  // 返回逻辑 scaleout/scaleup 维度。

// TODO: make it header only?
// 中文：该 context 封装 NCCL symmetric memory window 与 Gin 通信所需的 host/device 句柄。
struct NCCLSymmetricMemoryContext {
private:
    // Can not use this unmapped pointer from outside
    void* raw_window_ptr;  // NCCL window 原始分配指针；外部使用 mapped_window_ptr，避免误用未映射地址。

public:
    // NCCLSymmetricMemoryContext 是 Elastic/V2 后端的核心对象：
    // - 基于已有 ncclComm_t 创建 NCCL window，使所有 rank 的一块 GPU 内存可被 Gin 访问；
    // - 推导 physical domain(RDMA/NVLink) 与 logical domain(scaleout/scaleup)；
    // - 保存 host/device communicator、window、mapped_window_ptr 等 kernel launch 需要的句柄。
    // csrc/elastic/buffer.hpp 会把 mapped_window_ptr 切成 workspace 与 payload buffer。
    // Global
    int rank_idx;   // 当前 rank 在 global communicator 中的编号，范围 [0, num_ranks)。
    int num_ranks;  // communicator 总 rank 数；Elastic dispatch/combine 会按它拆分 scaleout/scaleup。

    // Logical
    int num_scaleout_ranks, num_scaleup_ranks;      // 逻辑二维拓扑：跨节点 scaleout 数、节点内/高速域 scaleup 数。
    int scaleout_rank_idx, scaleup_rank_idx;        // 当前 rank 在逻辑二维拓扑中的坐标。

    // Physical
    int num_rdma_ranks, num_nvl_ranks;              // 物理二维拓扑：RDMA 域数量和 NVLink/本地域大小。
    int rdma_rank_idx, nvl_rank_idx;                // 当前 rank 的物理坐标。
    bool is_scaleup_nvlink;                         // logical scaleup 是否正好落在 NVLink 域内；hybrid kernel 会据此选通信策略。

    // NCCL handles
    ncclComm_t comm;                    // host 侧 communicator；来自 create_nccl_comm 的句柄。
    ncclDevComm_t dev_comm;             // device 侧 communicator；kernel 通过它发起 Gin/NCCL 设备端操作。
    ncclWindow_t window;                // NCCL symmetric memory window；把各 rank GPU buffer 注册成可远端访问区域。
    void* mapped_window_ptr;            // 当前 rank 可直接传给 CUDA kernel 的 window 映射地址。
    std::vector<void*> nvl_window_ptrs;  // NVLink 域内各 peer rank 的 window 指针，设备端可用它做快速 peer 访问。

    // Configs
    int num_allocated_qps;  // 已申请 queue pair 数；Elastic kernel 的 dispatch/combine 会按 SM/QP 分摊通信通道。

    NCCLSymmetricMemoryContext(const int64_t& nccl_comm,  // Python 传入的 host ncclComm_t 句柄。
                               const int& num_ranks, const int& rank_idx,  // world size 与当前 rank。
                               const size_t& size, const size_t& alignment,  // symmetric window 总字节数与地址对齐要求。
                               const bool& allow_hybrid_mode,  // 是否按 RDMA scaleout + NVLink scaleup 推导逻辑拓扑。
                               const int& sl_idx, const int& num_allocated_qps);  // RDMA service level 与预申请 Gin/QP 数。
    // 构造函数会分配 NCCL symmetric memory、注册 window、创建 ncclDevComm_t，并填充
    // mapped_window_ptr/nvl_window_ptrs。调用方必须保证所有 rank 以相同 size/alignment 进入，
    // 否则对称地址 offset 映射会失效。

    // TODO: finish this with `explicit_destroy`
    // ~NCCLSymmetricMemoryContext();

    void* get_sym_ptr(void* ptr, const int& dst_rank_idx) const;  // 把本地 window 内指针转换为 dst rank 上对应的对称地址。

    void finalize() const;  // 释放 NCCL window/device communicator 等资源；由 ElasticBuffer 析构/显式释放路径调用。
};

}  // deep_ep::nccl

namespace deep_ep::cuda_driver {  // host 侧 CUDA driver 批量 flag 操作，常用于通信 kernel 之间做轻量同步。

void batched_write(CUstream stream, const std::vector<void*>& ptrs, const int& value);  // 在 stream 上批量把多个 flag 地址写成同一个 value。

void batched_wait(CUstream stream, const std::vector<void*>& ptrs, const int& value);  // 在 stream 上等待多个 flag 地址都变为 value。

void batched_write_and_wait(CUstream stream, const std::vector<void*>& write_ptrs, const std::vector<void*>& wait_ptrs, const int& value);  // 先写本地/远端 flag，再等待对应 flag，避免 host 多次 launch。

}  // namespace deep_ep::cuda_driver
