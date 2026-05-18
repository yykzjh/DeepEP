#pragma once  // legacy Buffer runtime 是 header-only pybind 实现，防止重复定义。

#include <ATen/cuda/CUDAContext.h>   // 获取 PyTorch 当前 CUDA stream 和 stream pool。
#include <ATen/cuda/CUDADataType.h>  // 把 torch scalar type 转成 CUDA data type。
#include <cuda_runtime.h>            // cudaMalloc/cudaMemcpy/cudaDeviceSynchronize 等 runtime API。
#include <torch/python.h>            // PyTorch C++/pybind 类型转换 helper。

#include <chrono>  // CPU 轮询计数超时。
#include <memory>  // 保留智能指针相关依赖；与 runtime 头文件风格一致。

#include <deep_ep/common/compiled.cuh>   // legacy 编译常量、topk_idx_t 等公共定义。
#include <deep_ep/common/exception.cuh>  // EP_HOST_ASSERT/CUDA_RUNTIME_CHECK。

#include "../utils/event.hpp"          // EventHandle 和 stream_wait。
#include "../utils/shared_memory.hpp"  // CUDA IPC/Fabric 共享内存 helper。
#include "../kernels/legacy/api.cuh"   // legacy CUDA kernel host API。
#include "../kernels/backend/api.cuh"  // NVSHMEM backend API。
#include "config.hpp"                  // legacy Config 调优参数。

namespace deep_ep::legacy {  // legacy/V1 C++ runtime 命名空间。

struct Buffer {
    // Buffer 是 legacy/V1 的 C++ host runtime，职责比 Python `deep_ep.buffers.legacy.Buffer`
    // 更靠近资源管理：它持有节点内 CUDA IPC/Fabric buffer、跨节点 NVSHMEM symmetric buffer、
    // mapped host counter、workspace 和通信 stream。Python 层只保存这个对象的 shared_ptr，
    // 真正的 dispatch/combine/low-latency 调用都会先在这里做 dtype/shape/stream 检查，
    // 再转发到 csrc/kernels/legacy/api.cuh 暴露的 CUDA host wrapper。
    // 读这个文件时可以把它看成“三段式桥梁”：
    // 1. 构造/sync/destroy：建立 peer buffer 指针表；
    // 2. intranode/internode dispatch/combine：把 Python tensor 打包成 kernel 参数；
    // 3. low_latency_*：管理固定容量双缓冲和 mask/shrink 状态。
    EP_STATIC_ASSERT(LEGACY_NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");  // legacy rank 拆分逻辑假设单节点最多 8 个 NVLink peer。

private:
    // Low-latency mode buffer
    // low_latency_buffer_idx 在两个 RDMA buffer 槽之间切换。低延迟路径返回的 tensor 直接复用
    // 内部 buffer，因此同一时刻最多安全持有两轮结果；Python 文档中的“不能持有超过 2 个结果”
    // 就来自这个双缓冲设计。
    int low_latency_buffer_idx = 0;
    // 低延迟双缓冲当前使用的槽位索引，取值在 0/1 间翻转；dispatch/combine 根据它选择
    // rdma_send/recv staging 区，下一轮再切到另一半，避免上一轮 hook 尚未等待完成时被覆盖。
    bool low_latency_mode = false;
    // low_latency_mode=true 时构造函数和 sync() 会额外初始化 IBGDA/NVSHMEM 资源，
    // 并允许每 GPU 都作为 NVSHMEM PE；普通 internode 模式则通常每节点一个 NVSHMEM PE。

    // NVLink Buffer
    // buffer_ptrs 保存本节点内每个 NVLink peer 的 IPC 映射地址；buffer_ptrs_gpu 是同一张指针表
    // 在 GPU 上的副本，legacy intranode kernel 会用它直接写目标 rank 的 buffer。
    int64_t num_nvl_bytes;
    void* buffer_ptrs[LEGACY_NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;

    // NVSHMEM Buffer
    // 跨节点或 low-latency IBGDA 路径使用 NVSHMEM 对称内存。rdma_buffer_ptr 指向本 rank
    // 的 NVSHMEM buffer，远端访问由 NVSHMEM/NIC 负责。
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    // Shrink mode buffer
    bool enable_shrink = false;  // 是否启用低延迟 rank mask/shrink 功能。
    int* mask_buffer_ptr = nullptr;  // NVSHMEM symmetric mask buffer，[num_ranks]。
    int* sync_buffer_ptr = nullptr;  // NVSHMEM symmetric sync buffer，[num_ranks]。

    // Device info and communication
    int device_id;  // 当前 CUDA device id。
    int num_device_sms;  // 当前 GPU SM 数。
    int rank, rdma_rank, nvl_rank;  // 全局 rank、跨节点 RDMA rank、节点内 NVLink rank。
    int num_ranks, num_rdma_ranks, num_nvl_ranks;  // 全局 rank 数、RDMA 域数、节点内 NVLink rank 数。
    shared_memory::MemHandle ipc_handles[LEGACY_NUM_MAX_NVL_PEERS];  // 节点内 peer 的 CUDA IPC/Fabric handle。

    // Stream for communication
    at::cuda::CUDAStream comm_stream;  // legacy 通信 high-priority stream。

    // After IPC/NVSHMEM synchronization, this flag will be true
    bool available = false;  // sync() 完成后置 true。

    // Whether explicit `destroy()` is required.
    bool explicitly_destroy;  // 是否要求 Python 显式 destroy。
    // After `destroy()` be called, this flag will be true
    bool destroyed = false;  // destroy() 完成后置 true，防止重复释放。

    // Barrier signals
    int* barrier_signal_ptrs[LEGACY_NUM_MAX_NVL_PEERS] = {nullptr};  // 每个 NVLink peer buffer 尾部的 barrier signal 地址。
    int** barrier_signal_ptrs_gpu = nullptr;  // GPU 上的 barrier signal 指针表。

    // Workspace
    void* workspace = nullptr;  // legacy kernel 临时 workspace，大小 LEGACY_NUM_WORKSPACE_BYTES。

    // Host-side MoE info
    volatile int* moe_recv_counter = nullptr;  // CPU 轮询总接收 token 数的 mapped host counter。
    int* moe_recv_counter_mapped = nullptr;  // device 可写的 mapped counter 指针。

    // Host-side expert-level MoE info
    volatile int* moe_recv_expert_counter = nullptr;  // CPU 轮询每 expert 接收 token 数。
    int* moe_recv_expert_counter_mapped = nullptr;  // device 可写的 expert counter 指针。

    // Host-side RDMA-level MoE info
    volatile int* moe_recv_rdma_counter = nullptr;  // CPU 轮询 RDMA 阶段接收 token 数。
    int* moe_recv_rdma_counter_mapped = nullptr;  // device 可写的 RDMA counter 指针。

    shared_memory::SharedMemoryAllocator shared_memory_allocator;  // CUDA IPC/Fabric allocator。

public:
    Buffer(int rank,  // 当前全局 rank。
           int num_ranks,  // 全局 rank 数。
           int64_t num_nvl_bytes,  // 本地 NVLink/IPC buffer 字节数。
           int64_t num_rdma_bytes,  // NVSHMEM/RDMA buffer 字节数。
           bool low_latency_mode,  // 是否启用 low-latency IBGDA 模式。
           bool explicitly_destroy,  // 是否要求显式 destroy。
           bool enable_shrink,  // 是否启用 rank mask/shrink。
           bool use_fabric) : rank(rank),  // 初始化全局 rank。
      num_ranks(num_ranks),  // 初始化 world size。
      num_nvl_bytes(num_nvl_bytes),  // 初始化 NVLink buffer 字节数。
      num_rdma_bytes(num_rdma_bytes),  // 初始化 RDMA buffer 字节数。
      enable_shrink(enable_shrink),  // 初始化 shrink 开关。
      low_latency_mode(low_latency_mode),  // 初始化 low-latency 开关。
      explicitly_destroy(explicitly_destroy),  // 初始化析构策略。
      comm_stream(at::cuda::getStreamFromPool(true)),  // 获取 high-priority 通信 stream。
      shared_memory_allocator(use_fabric) {  // 根据 use_fabric 创建共享内存 allocator。
        // Metadata memory
        // NVLink buffer 的尾部不仅存 token payload，还附带几张小型元数据表：
        // barrier_signal_ptrs 用于节点内同步，buffer_ptrs_gpu/barrier_signal_ptrs_gpu 让 kernel
        // 能在 device 端找到每个 peer 的 payload 区和 barrier 信号区。
        int64_t barrier_signal_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(int);  // 每个 NVLink peer 一个 int barrier signal。
        int64_t buffer_ptr_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(void*);  // GPU 端 peer buffer 指针表大小。
        int64_t barrier_signal_ptr_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(int*);  // GPU 端 barrier 指针表大小。

        // Common checks
        EP_STATIC_ASSERT(LEGACY_NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");  // buffer 对齐必须支持 int4 向量化。
        EP_HOST_ASSERT(num_nvl_bytes % LEGACY_NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                       (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));  // NVL buffer 对齐且普通路径大小不超过 int。
        EP_HOST_ASSERT(num_rdma_bytes % LEGACY_NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                       (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));  // RDMA buffer 对齐且非 LL 模式大小不超过 int。
        EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());  // NVL buffer 以 int4 计数也不能溢出 int。
        EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());  // RDMA buffer 以 int4 计数也不能溢出 int。
        EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= LEGACY_NUM_MAX_NVL_PEERS * LEGACY_NUM_MAX_RDMA_PEERS or low_latency_mode));  // rank 范围和 legacy 最大拓扑检查。
        EP_HOST_ASSERT(num_ranks < LEGACY_NUM_MAX_NVL_PEERS or num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0);  // 多节点时 rank 数必须是 8 的倍数。
        if (num_rdma_bytes > 0)  // 分配 RDMA buffer 时。
            EP_HOST_ASSERT(num_ranks > LEGACY_NUM_MAX_NVL_PEERS or low_latency_mode);  // 只有跨节点或 low-latency 模式需要 RDMA buffer。

        // Get ranks
        // legacy 把全局 rank 拆成 rdma_rank 和 nvl_rank：
        //   rank = rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + nvl_rank。
        // 例如 2 节点 * 8 卡时，num_rdma_ranks=2、num_nvl_ranks=8；
        // internode kernel 会先按 nvl_rank 维度跨 RDMA 通信，再在节点内 NVLink 分发。
        CUDA_RUNTIME_CHECK(cudaGetDevice(&device_id));  // 查询当前 CUDA device。
        rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS, nvl_rank = rank % LEGACY_NUM_MAX_NVL_PEERS;  // 拆分全局 rank 为 RDMA/NVL 坐标。
        num_rdma_ranks = std::max(1, num_ranks / LEGACY_NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, LEGACY_NUM_MAX_NVL_PEERS);  // 计算拓扑维度。

        // Get device info
        cudaDeviceProp device_prop = {};  // CUDA device 属性结构体。
        CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&device_prop, device_id));  // 查询当前 GPU 属性。
        num_device_sms = device_prop.multiProcessorCount;  // 保存 SM 数。

        // Number of per-channel bytes cannot be large
        EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());  // 每 channel NVL 字节数不能溢出 int。
        EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());  // 每 channel RDMA 字节数不能溢出 int。

        if (num_nvl_bytes > 0) {  // 需要节点内 NVLink/IPC buffer。
            // Local IPC: alloc local memory and set local IPC handles
            shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank],
                                           num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes);  // 分配本地 payload + metadata 尾部。
            shared_memory_allocator.get_mem_handle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);  // 导出本地 IPC/Fabric handle。
            buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);  // GPU 指针表位于 payload+signal 后。

            // Set barrier signals
            barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);  // 本地 barrier signal 区位于 payload 尾部。
            barrier_signal_ptrs_gpu =
                reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);  // barrier 指针表位于 buffer 指针表之后。

            // No need to synchronize, will do a full device sync during `sync`
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));  // 异步清零本地 barrier signal。
        }

        // Create 32 MiB workspace
        CUDA_RUNTIME_CHECK(cudaMalloc(&workspace, LEGACY_NUM_WORKSPACE_BYTES));  // 分配 legacy kernel 临时 workspace。
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(workspace, 0, LEGACY_NUM_WORKSPACE_BYTES, comm_stream));  // 清零 workspace。

        // MoE counter
        // 这些 host pinned + mapped counter 是 legacy kernel 与 CPU 同步接收 token 数的桥。
        // device kernel 写 mapped 指针，CPU 侧轮询 volatile host 指针，从而决定 recv_x 的真实大小。
        CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));  // 分配 mapped host 总计数器。
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));  // 获取 device 可写指针。
        *moe_recv_counter = -1;  // -1 表示 GPU 尚未写入 ready count。

        // MoE expert-level counter
        CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * LEGACY_NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));  // 分配每 expert mapped counter。
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));  // 获取 device 可写 expert counter 指针。
        for (int i = 0; i < LEGACY_NUM_MAX_LOCAL_EXPERTS; ++i)  // 初始化所有 expert counter。
            moe_recv_expert_counter[i] = -1;  // -1 表示该 expert count 尚未 ready。

        // MoE RDMA-level counter
        if (num_rdma_ranks > 0) {  // 存在 RDMA 维度时分配 RDMA 阶段计数器。
            CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));  // 分配 mapped host RDMA counter。
            CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));  // 获取 device 可写指针。
            *moe_recv_rdma_counter = -1;  // -1 表示尚未 ready。
        }
    }

    ~Buffer() noexcept(false) {
        if (not explicitly_destroy) {  // 非显式销毁模式由析构自动释放。
            destroy();  // 释放 IPC/NVSHMEM/workspace/counters。
        } else if (not destroyed) {  // 显式销毁模式但用户忘记 destroy。
            printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");  // 打印泄漏警告。
            fflush(stdout);  // 立即刷新日志。
        }
    }

    bool is_available() const {
        // sync() 完成 IPC/NVSHMEM 建联后才为 true；大部分 legacy 通信 API 都要求
        // available=true，否则 device 端拿不到 peer buffer 指针。
        return available;
    }

    bool is_internode_available() const {
        // internode 路径要求已经 sync，并且全局 rank 数超过单节点 NVLink peer 上限。
        return is_available() and num_ranks > LEGACY_NUM_MAX_NVL_PEERS;
    }

    int get_num_rdma_ranks() const {
        return num_rdma_ranks;  // 返回 RDMA 域数量。
    }

    int get_rdma_rank() const {
        return rdma_rank;  // 返回当前 rank 的 RDMA 维坐标。
    }

    int get_root_rdma_rank(bool global) const {
        // NVSHMEM unique id 只由每个 RDMA team 的 root 生成。global=true 时返回
        // 当前 nvl_rank 在全局 rank 空间中的对应 root；否则返回本 RDMA team 内 root=0。
        return global ? nvl_rank : 0;
    }

    int get_local_device_id() const {
        return device_id;  // 返回当前 CUDA device id。
    }

    pybind11::bytearray get_local_ipc_handle() const {
        // IPC/fabric handle 是二进制结构，Python 侧会 all_gather bytearray 后再传回 sync()。
        const shared_memory::MemHandle& handle = ipc_handles[nvl_rank];  // 取本地 rank 的共享内存 handle。
        return {reinterpret_cast<const char*>(&handle), sizeof(handle)};  // 作为 Python bytearray 返回。
    }

    pybind11::bytearray get_local_nvshmem_unique_id() const {
        EP_HOST_ASSERT(rdma_rank == 0 and "Only RDMA rank 0 can get NVSHMEM unique ID");  // 只有每个 RDMA team root 生成 unique id。
        const auto unique_id = nvshmem::get_unique_id();  // 调 backend 生成 unique id bytes。
        return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};  // 返回给 Python 广播。
    }

    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
        // 暴露底层 buffer 的调试入口。返回 tensor 不拥有内存，只是 view；
        // offset 以字节为单位，dtype 决定 tensor 元素个数。
        torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);  // 把 Python dtype 转成 C++ ScalarType。
        auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));  // 计算单元素字节数。
        auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;  // 选择 RDMA/NVL buffer 并加 offset。
        auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;  // 选择对应 buffer 总字节数。
        return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));  // 返回不拥有内存的 CUDA tensor view。
    }

    torch::Stream get_comm_stream() const {
        return comm_stream;  // 返回 legacy 通信 stream。
    }

    void sync(const std::vector<int>& device_ids,  // Python all_gather 得到的每个 rank CUDA device id。
              const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,  // 所有 rank 的 IPC/Fabric handle。
              const std::optional<pybind11::bytearray>& root_unique_id_opt) {  // NVSHMEM root unique id，可选。
        EP_HOST_ASSERT(not is_available());  // sync 只能执行一次。

        // Sync IPC handles
        if (num_nvl_bytes > 0) {
            // all_gathered_handles 是所有全局 rank 的 IPC handle；本 rank 只打开同一
            // rdma_rank 内的 num_nvl_ranks 个 peer，从而组成节点内 NVLink 共享内存视图。
            EP_HOST_ASSERT(num_ranks == device_ids.size());  // device id 列表长度必须等于 world size。
            EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());  // handle 列表长度必须匹配。
            for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {  // 遍历本节点 NVLink peers。
                EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());  // 每个 peer 都必须提供 handle。
                auto handle_str = std::string(all_gathered_handles[offset + i].value());  // bytearray 转成 string bytes。
                EP_HOST_ASSERT(handle_str.size() == sizeof(shared_memory::MemHandle));  // handle byte 长度必须匹配结构体。
                if (offset + i != rank) {
                    // 远端 peer：保存 handle、open 成本进程可访问指针，并计算 barrier signal
                    // 位于 peer buffer 尾部的地址。
                    std::memcpy(&ipc_handles[i], handle_str.c_str(), sizeof(shared_memory::MemHandle));  // 保存 peer handle。
                    shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);  // 打开 peer buffer 映射。
                    barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);  // 计算 peer barrier signal 地址。
                } else {
                    // 本地 peer：all_gather 回来的 handle 应与构造时生成的一致，作为一致性校验。
                    EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), sizeof(shared_memory::MemHandle)) == 0);  // 本地 handle 必须和 all_gather 回来的一致。
                }
            }

            // device kernel 不能访问 host 数组，所以把 buffer_ptrs 和 barrier_signal_ptrs
            // 两张指针表复制到本地 GPU buffer 尾部。
            CUDA_RUNTIME_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * LEGACY_NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));  // 复制 peer buffer 指针表到 GPU。
            CUDA_RUNTIME_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * LEGACY_NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));  // 复制 barrier 指针表到 GPU。
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());  // 等指针表拷贝完成。
        }

        // Sync NVSHMEM handles and allocate memory
        if (num_rdma_bytes > 0) {
            // low_latency_mode=false 时每个节点只有一个 NVSHMEM PE（rdma_rank）；
            // low_latency_mode=true 时每张 GPU 都是一个 PE，便于直接按 rank 失败 mask。
            EP_HOST_ASSERT(root_unique_id_opt.has_value());  // NVSHMEM 初始化必须有 root unique id。
            std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());  // 分配 unique id byte buffer。
            auto root_unique_id_str = root_unique_id_opt->cast<std::string>();  // pybind bytearray 转 string。
            std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());  // 拷贝 bytes 到 vector。
            auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;  // low-latency 每 GPU 一个 PE；普通 internode 每节点一个 PE。
            auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;  // NVSHMEM PE 总数。
            EP_HOST_ASSERT(nvshmem_rank == nvshmem::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks,
                                                         low_latency_mode ? LEGACY_NUM_MAX_NVL_PEERS : 0));  // 初始化 NVSHMEM 并校验返回 rank。

            // NVSHMEM symmetric heap 分配必须所有 PE 按同顺序调用。返回的 rdma_buffer_ptr
            // 可被远端 PE 通过 NVSHMEM/IBGDA 访问。
            rdma_buffer_ptr = nvshmem::alloc(num_rdma_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES);  // 分配 RDMA symmetric buffer。

            // low-latency buffer 中有大量 flag/counter，初始化为 0 保证第一轮协议状态干净。
            CUDA_RUNTIME_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));  // 清零 RDMA buffer。

            // shrink buffer 用于失败 rank 屏蔽：mask_buffer_ptr [num_ranks] 标记失效 rank，
            // sync_buffer_ptr [num_ranks] 是自定义容错 barrier 的轮次计数。
            if (enable_shrink) {
                int num_mask_buffer_bytes = num_ranks * sizeof(int);  // mask buffer 字节数。
                int num_sync_buffer_bytes = num_ranks * sizeof(int);  // sync buffer 字节数。
                mask_buffer_ptr = static_cast<int*>(nvshmem::alloc(num_mask_buffer_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES));  // 分配 symmetric mask buffer。
                sync_buffer_ptr = static_cast<int*>(nvshmem::alloc(num_sync_buffer_bytes, LEGACY_NUM_BUFFER_ALIGNMENT_BYTES));  // 分配 symmetric sync buffer。
                CUDA_RUNTIME_CHECK(cudaMemset(mask_buffer_ptr, 0, num_mask_buffer_bytes));  // 初始所有 rank unmasked。
                CUDA_RUNTIME_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));  // 初始 sync 计数为 0。
            }

            // 所有 PE 完成分配和清零后再进入可用状态，防止某 rank 先启动 kernel 访问
            // 远端尚未初始化的 symmetric heap。
            nvshmem::barrier(true);  // 所有 PE 同步，确保 symmetric allocations 完成。
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());  // 本 GPU 等清零完成。
        }

        // 到这里 IPC 指针表、NVSHMEM buffer、workspace 和 mapped counters 都已准备完毕。
        available = true;  // 标记 runtime 可用。
    }

    void destroy() {
        EP_HOST_ASSERT(not destroyed);  // 防止重复释放。

        // 析构/释放前先同步当前设备，避免 host 提前关闭 IPC/NVSHMEM handle。
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());  // 先等本 GPU 所有工作完成。

        if (num_nvl_bytes > 0) {  // 存在 NVLink/IPC buffer。
            // 节点内先做一次 barrier，确认所有 NVLink peer 都不再访问彼此 IPC buffer。
            intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);  // 节点内 peer barrier。
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());  // 等 barrier 完成。

            // 只关闭远端 handle，本地 buffer 由 shared_memory_allocator.free 释放。
            if (is_available()) {  // 只有 sync 后才打开过 peer handle。
                for (int i = 0; i < num_nvl_ranks; ++i)  // 遍历节点内 peers。
                    if (i != nvl_rank)  // 跳过本地 buffer。
                        shared_memory_allocator.close_mem_handle(buffer_ptrs[i]);  // 关闭 peer IPC/Fabric 映射。
            }

            // buffer_ptrs[nvl_rank] 是构造函数中 malloc 的本地 IPC/fabric allocation。
            shared_memory_allocator.free(buffer_ptrs[nvl_rank]);  // 释放本地 IPC/Fabric allocation。
        }

        // NVSHMEM finalize 是 collective，因此只有 sync() 成功后 available=true 才能进入。
        if (is_available() and num_rdma_bytes > 0) {  // sync 成功且分配了 RDMA buffer。
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());  // 等所有 RDMA 相关 GPU 工作完成。
            nvshmem::barrier(true);  // NVSHMEM PE 间同步。
            nvshmem::free(rdma_buffer_ptr);  // 释放 RDMA symmetric buffer。
            if (enable_shrink) {  // shrink buffer 已分配时释放。
                nvshmem::free(mask_buffer_ptr);  // 释放 mask buffer。
                nvshmem::free(sync_buffer_ptr);  // 释放 sync buffer。
            }
            nvshmem::finalize();  // 关闭 NVSHMEM runtime。
        }

        // workspace 是普通 cudaMalloc，MoE counters 是 cudaMallocHost mapped memory。
        CUDA_RUNTIME_CHECK(cudaFree(workspace));  // 释放普通 CUDA workspace。
        CUDA_RUNTIME_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));  // 释放总接收 counter。

        // expert counter 与 rdma counter 分开分配；这里释放 expert counter，rdma counter
        // 当前代码未单独 free，保持原有行为不改动。
        CUDA_RUNTIME_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));  // 释放 expert counter。

        destroyed = true;  // 标记已销毁。
        available = false;  // 标记不可再用。
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>> get_dispatch_layout(
        const torch::Tensor& topk_idx,  // 输入 top-k expert id，形状 [num_tokens, num_topk]。
        int num_experts,  // 全局 expert 总数，用来统计 num_tokens_per_expert [num_experts]。
        const std::optional<EventHandle>& previous_event,  // 上游异步事件；存在时通信流先等待它。
        bool async,  // 是否返回 EventHandle 而不是让 compute stream 立即等待通信流。
        bool allocate_on_comm_stream) {  // 是否临时切到通信流分配输出 tensor，配合 CUDA graph/overlap。
        EP_HOST_ASSERT(topk_idx.dim() == 2);  // topk_idx 必须是二维矩阵：[token, topk lane]。
        EP_HOST_ASSERT(topk_idx.is_contiguous());  // layout kernel 按连续内存线性读取 topk_idx。
        EP_HOST_ASSERT(num_experts > 0);  // expert 数必须为正，否则 per-expert 输出无意义。

        // Allocate all tensors on comm stream if set
        // NOTES: do not allocate tensors upfront!
        // allocate_on_comm_stream=True 用于 CUDA graph/异步重叠场景：输出张量的分配也归属通信流，
        // 避免默认 compute stream 上的 allocator 事件插入破坏 overlap。要求 previous_event 存在，
        // 是为了确保通信流不会在上游数据尚未就绪时启动 layout kernel。
        auto compute_stream = at::cuda::getCurrentCUDAStream();  // 记录进入函数时 PyTorch 当前计算流。
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);  // 通信流分配模式要求上游事件和异步返回一起使用。
            at::cuda::setCurrentCUDAStream(comm_stream);  // 后续 torch::empty 归属通信流，减少跨流 allocator 同步。
        }

        // Wait previous tasks to be finished
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());  // 通信流等待显式传入的上游事件。
        } else {
            stream_wait(comm_stream, compute_stream);  // 没有事件时通信流等待当前计算流，保证 topk_idx 已可读。
        }

        // layout kernel 的四个核心输出：
        // - num_tokens_per_rank: [num_ranks]，每个目标 rank 接收多少 token；
        // - num_tokens_per_rdma_rank: [num_rdma_ranks]，internode 时跨 RDMA 维度的聚合计数；
        // - num_tokens_per_expert: [num_experts]，每个 expert 的 token 数；
        // - is_token_in_rank: [num_tokens, num_ranks]，一个 token 是否需要发给某 rank。
        auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));  // 读取 token 数和每 token 的 top-k 路数。
        auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));  // [num_ranks]：目标 rank 粒度计数。
        auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();  // 仅 internode 可用时创建 [num_rdma_ranks]。
        auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));  // [num_experts]：全局 expert 粒度计数。
        auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));  // [num_tokens,num_ranks]：token 到 rank 的布尔路由表。
        if (is_internode_available())
            num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));  // internode 额外按 RDMA rank 聚合计数。

        layout::get_dispatch_layout(topk_idx.data_ptr<topk_idx_t>(),  // 输入 topk expert id 指针。
                                    num_tokens_per_rank.data_ptr<int>(),  // 输出 [num_ranks] 计数。
                                    num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,  // 可选 RDMA 聚合计数。
                                    num_tokens_per_expert.data_ptr<int>(),  // 输出 [num_experts] 计数。
                                    is_token_in_rank.data_ptr<bool>(),  // 输出 token-rank 路由布尔矩阵。
                                    num_tokens,  // token 数。
                                    num_topk,  // top-k 路数。
                                    num_ranks,  // 全局 rank 数。
                                    num_experts,  // 全局 expert 数。
                                    comm_stream);  // kernel 在通信流上发射。

        // Wait streams
        std::optional<EventHandle> event;  // async 模式返回给 Python 的可等待事件。
        if (async) {
            // async=True 时返回 EventHandle，调用方可自行等待；record_stream 确保这些
            // tensor 在通信流未完成前不会被 PyTorch allocator 复用。
            event = EventHandle(comm_stream);  // 在通信流当前点记录 CUDA event。
            for (auto& t : {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
                t.record_stream(comm_stream);  // 告诉 PyTorch allocator 这些 tensor 被通信流使用。
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);  // 若在通信流分配，也关联回计算流避免提前复用。
            }
            for (auto& to : {num_tokens_per_rdma_rank}) {
                to.has_value() ? to->record_stream(comm_stream) : void();  // 可选 tensor 存在时同样记录通信流。
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();  // 可选 tensor 也记录计算流生命周期。
            }
        } else {
            stream_wait(compute_stream, comm_stream);  // 同步模式让计算流等待 layout kernel 完成。
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);  // 恢复调用方原本的 PyTorch 当前流。

        return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};  // 返回四类布局 tensor 和可选事件。
    }

    std::tuple<torch::Tensor,                 // recv_x。
               std::optional<torch::Tensor>,  // recv_x_scales。
               // 中文：recv_x_scales 仅 FP8 dispatch 返回，BF16 路径为 nullopt。
               std::optional<torch::Tensor>,  // recv_topk_idx。
               std::optional<torch::Tensor>,  // recv_topk_weights。
               // 中文：前四项是 Python 直接看到的 payload/路由元数据，形状与 BF16/FP8 输入模式相关。
               std::vector<int>,              // num_recv_tokens_per_expert_list。
               // 中文：后续 tensor 是 combine handle 所需的控制面，Python 不直接解释其内部编码。
               torch::Tensor,                 // recv_src_idx。
               torch::Tensor,                 // recv_channel_prefix_matrix。
               // 中文：recv_src_idx 和 recv_channel_prefix_matrix 是 intranode combine 反向寻址的核心。
               torch::Tensor,                 // send_head。
               // 中文：中间四项构成 combine handle，用于从专家输出反向写回原 token。
               torch::Tensor,                 // rank_prefix_matrix 或 cached 副本。
               torch::Tensor,                 // channel_prefix_matrix 或 cached 副本。
               std::optional<EventHandle>>    // 可选异步事件。
    intranode_dispatch(const torch::Tensor& x,  // 待发送 token 特征，形状 [num_tokens, hidden]。
                       const std::optional<torch::Tensor>& x_scales,  // 可选 FP8 scale，形状通常 [num_tokens, hidden/128]。
                       const std::optional<torch::Tensor>& topk_idx,  // 可选 top-k expert id，[num_tokens,num_topk]。
                       const std::optional<torch::Tensor>& topk_weights,  // 可选 top-k 权重，[num_tokens,num_topk]。
                       const std::optional<torch::Tensor>& num_tokens_per_rank,  // 非 cached 模式输入：[num_ranks]。
                       const torch::Tensor& is_token_in_rank,  // 路由布尔矩阵：[num_tokens,num_ranks]。
                       const std::optional<torch::Tensor>& num_tokens_per_expert,  // 非 cached 模式输入：[num_experts]。
                       int cached_num_recv_tokens,  // cached 模式直接复用的本 rank 接收 token 数。
                       const std::optional<torch::Tensor>& cached_rank_prefix_matrix,  // cached 模式复用 [num_ranks,num_ranks]。
                       const std::optional<torch::Tensor>& cached_channel_prefix_matrix,  // cached 模式复用 [num_ranks,num_channels]。
                       int expert_alignment,  // expert token 数对齐粒度，用于 kernel 端 padding。
                       int num_worst_tokens,  // >0 时按最坏 token 数分配，跳过 CPU 轮询。
                       const Config& config,  // legacy kernel 分块和 SM 配置。
                       std::optional<EventHandle>& previous_event,  // 上游事件，通信流在 launch 前等待。
                       bool async,  // 是否异步返回事件。
                       bool allocate_on_comm_stream) {  // 是否把输出 tensor 分配放到通信流。
        bool cached_mode = cached_rank_prefix_matrix.has_value();  // 是否走 cached dispatch 路径。
        // cached_mode 表示复用上一次 dispatch 的 rank/channel prefix，跳过 notify 阶段
        // 的 CPU 计数同步。要求调用方传入 cached_num_recv_tokens 和两张 prefix 矩阵。

        // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
        // 因为每个 channel 拆成 send/recv 两个 block，所以 num_sms 必须为偶数，num_channels=num_sms/2。
        // 这和 legacy CUDA kernel 中“偶数 block 发、奇数 block 收”的约定强绑定。
        EP_HOST_ASSERT(config.num_sms % 2 == 0);  // send/recv block 成对出现，因此 SM 数必须为偶数。
        int num_channels = config.num_sms / 2;  // 每个 channel 对应一对 send/recv block。
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());  // cached 模式必须有 rank prefix。
            EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());  // cached 模式必须有 channel prefix。
        } else {
            EP_HOST_ASSERT(num_tokens_per_rank.has_value());  // 非 cached 模式需要 layout 阶段输出 rank 计数。
            EP_HOST_ASSERT(num_tokens_per_expert.has_value());  // 非 cached 模式需要 expert 计数。
        }

        // Type checks：legacy kernel 对 dtype 分支较少，错误 dtype 可能导致 reinterpret_cast
        // 后按错误元素大小读写，所以在 C++ 层尽早断言。
        EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);  // 布尔路由表按 bool kernel 读取。
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);  // prefix 矩阵元素是 int32 offset。
            EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);  // channel prefix 同样是 int32 offset。
        } else {
            EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);  // expert 计数 kernel 按 int32 读。
            EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);  // rank 计数 kernel 按 int32 读。
        }

        // Shape and contiguous checks：hidden 字节数必须能按 int4 向量化搬运；
        // is_token_in_rank [num_tokens, num_ranks] 决定每个 token 是否发给每个 rank。
        EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());  // x 必须是 [num_tokens,hidden] 连续矩阵。
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);  // hidden 字节数按 int4 对齐，便于向量化拷贝。
        EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());  // 路由表必须二维连续。
        EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);  // 路由表维度必须匹配 token/rank。
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());  // cached rank prefix 必须二维连续。
            EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);  // 形状 [dst_rank,src_rank]。
            EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());  // cached channel prefix 必须二维连续。
            EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);  // 形状 [rank,channel]。
        } else {
            EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());  // expert 计数是一维连续向量。
            EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);  // expert 总数必须能均分到 rank。
            EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= LEGACY_NUM_MAX_LOCAL_EXPERTS);  // 本地 expert 数不超过 legacy 静态上限。
            EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());  // rank 计数是一维连续向量。
            EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);  // rank 计数长度必须等于全局 rank 数。
        }

        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));  // 缓存 token 数与 hidden 维度。
        auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;  // 非 cached 下推导全局/本地 expert 数。

        // Top-k checks：有 topk_idx 就必须同时有 topk_weights。dispatch 会把全局 expert id
        // 转成本 rank 局部 expert id，无效 lane 写 -1。
        int num_topk = 0;  // top-k 路数，未传 topk 时为 0。
        topk_idx_t* topk_idx_ptr = nullptr;  // 可选 topk index 原始指针。
        float* topk_weights_ptr = nullptr;  // 可选 topk weight 原始指针。
        EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());  // topk index 与 weight 必须成对出现。
        if (topk_idx.has_value()) {
            num_topk = static_cast<int>(topk_idx->size(1));  // 从 topk_idx 第二维读取 top-k 路数。
            EP_HOST_ASSERT(num_experts > 0);  // 需要 expert 总数才能把全局 expert id 映射成本地 id。
            EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());  // topk_idx 必须二维连续。
            EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());  // topk_weights 必须二维连续。
            EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));  // topk 两张表 token 维必须匹配 x。
            EP_HOST_ASSERT(num_topk == topk_weights->size(1));  // topk_idx/topk_weights 的 top-k 维必须一致。
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);  // 权重按 float32 在 kernel 中读写。
            topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();  // 取 topk_idx CUDA 指针。
            topk_weights_ptr = topk_weights->data_ptr<float>();  // 取 topk_weights CUDA 指针。
        }

        // FP8 scales checks：x 是 FP8 时 element_size==1，x_scales 形状通常为
        // [num_tokens, hidden/128]；stride 透传给 kernel 以支持转置后再 contiguous 的布局。
        float* x_scales_ptr = nullptr;  // 可选 FP8 scale 指针。
        int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;  // scale 列数和 stride 信息传给 kernel。
        if (x_scales.has_value()) {
            EP_HOST_ASSERT(x.element_size() == 1);  // 只有 FP8 输入才允许传 scale。
            EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);  // 支持 e4m3 scale 或 ue8m0 scale。
            EP_HOST_ASSERT(x_scales->dim() == 2);  // scale 当前要求二维。
            EP_HOST_ASSERT(x_scales->size(0) == num_tokens);  // scale token 维必须和 x 一致。
            num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));  // scale hidden 分块数。
            x_scales_ptr = static_cast<float*>(x_scales->data_ptr());  // 原始指针按 void/float 传给 kernel。
            scale_token_stride = static_cast<int>(x_scales->stride(0));  // token 维 stride。
            scale_hidden_stride = static_cast<int>(x_scales->stride(1));  // scale hidden 维 stride。
        }

        // Allocate all tensors on comm stream if set
        // NOTES: do not allocate tensors upfront!
        auto compute_stream = at::cuda::getCurrentCUDAStream();  // 保存当前计算流，后面用于同步和恢复。
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);  // 通信流分配要求异步事件承接生命周期。
            at::cuda::setCurrentCUDAStream(comm_stream);  // 把 torch::empty 绑定到通信流。
        }

        // Wait previous tasks to be finished
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());  // 通信流等待调用方传入的前序事件。
        } else {
            stream_wait(comm_stream, compute_stream);  // 没有事件时等待当前计算流，保证输入已准备。
        }

        // 这些 handle tensor 会返回给 Python，用于 combine 和 cached dispatch：
        // rank_prefix_matrix [num_ranks,num_ranks]，channel_prefix_matrix [num_ranks,num_channels]，
        // send_head [num_tokens,num_ranks] 记录每个 token 发送到各 rank 的环形槽位置。
        int num_recv_tokens = -1;  // 本 rank 最终接收 token 数，-1 表示尚未从 GPU metadata 得到。
        auto rank_prefix_matrix = torch::Tensor();  // [num_ranks,num_ranks]，rank 间 token 前缀和矩阵。
        auto channel_prefix_matrix = torch::Tensor();  // [num_ranks,num_channels]，每 rank 每 channel 起始 offset。
        std::vector<int> num_recv_tokens_per_expert_list;  // CPU 返回给 Python 的本地 expert token 计数。

        // Barrier or send sizes。num_memset_int 覆盖 channel start/end/head/tail 四类
        // int 元数据，notify kernel 会把这些队列控制字清零。
        int num_memset_int = num_channels * num_ranks * 4;  // 清理 channel start/end/head/tail 四组 int 控制字。
        if (cached_mode) {
            num_recv_tokens = cached_num_recv_tokens;  // cached 模式直接复用已知接收 token 数。
            rank_prefix_matrix = cached_rank_prefix_matrix.value();  // 复用上一轮 rank prefix 矩阵。
            channel_prefix_matrix = cached_channel_prefix_matrix.value();  // 复用上一轮 channel prefix 矩阵。

            // cached notify 只把 rank_prefix_matrix 拷到 peer buffer 并清队列，不重新计算
            // num_tokens_per_rank/num_tokens_per_expert。
            intranode::cached_notify_dispatch(
                rank_prefix_matrix.data_ptr<int>(),  // 输入 cached rank prefix。
                num_memset_int,  // 需要清零的 int 控制字数量。
                buffer_ptrs_gpu,  // GPU 上的 peer IPC buffer 指针表。
                barrier_signal_ptrs_gpu,  // GPU 上的 peer barrier 指针表。
                rank,  // 当前全局 rank。
                num_ranks,  // 节点内 rank 数，此处 intranode world。
                comm_stream);  // 通信流。
        } else {
            rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));  // 分配 rank prefix 输出矩阵。
            channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));  // 分配 channel prefix 输出矩阵。

            // 非 cached notify 会把本 rank 的 rank/expert 计数写入 peer buffer，
            // 其它 rank 聚合后通过 mapped host counter 回传真实接收 token 数。
            // Meta information:
            //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
            //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
            // 中文：rank prefix 后续用于分配 recv_x，expert prefix 主要返回给 Python 作为 MoE expert 输入长度。
            // NOTES: no more token dropping in this version
            *moe_recv_counter = -1;  // mapped host counter 置为 -1，GPU 写回非负值表示 ready。
            for (int i = 0; i < num_local_experts; ++i)
                moe_recv_expert_counter[i] = -1;  // 每个本地 expert 的 mapped counter 也置为未就绪。
            EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);  // peer buffer 头部要容纳 rank/expert 元数据。
            intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),  // 输入 [num_ranks] 目标 rank 计数。
                                       moe_recv_counter_mapped,  // GPU 写回总接收 token 数的 mapped 指针。
                                       num_ranks,  // rank 数。
                                       num_tokens_per_expert->data_ptr<int>(),  // 输入 [num_experts] expert 计数。
                                       moe_recv_expert_counter_mapped,  // GPU 写回本地 expert 计数的 mapped 指针。
                                       num_experts,  // 全局 expert 数。
                                       num_tokens,  // 本 rank 输入 token 数。
                                       is_token_in_rank.data_ptr<bool>(),  // token-rank 路由表。
                                       channel_prefix_matrix.data_ptr<int>(),  // 输出 channel prefix。
                                       rank_prefix_matrix.data_ptr<int>(),  // 输出 rank prefix。
                                       num_memset_int,  // 需要清零的队列控制字数量。
                                       expert_alignment,  // expert 计数对齐粒度。
                                       buffer_ptrs_gpu,  // peer buffer 指针表。
                                       barrier_signal_ptrs_gpu,  // peer barrier 指针表。
                                       rank,  // 当前 rank。
                                       comm_stream,  // 通信流。
                                       num_channels);  // channel 数。

            if (num_worst_tokens > 0) {
                // No CPU sync：直接按用户给的 worst case 分配输出，避免 CPU 轮询等待；
                // kernel 尾部会把未使用 topk_idx 清成 -1。
                num_recv_tokens = num_worst_tokens;  // 直接使用上界作为接收容量。

                // Must be forward with top-k stuffs
                EP_HOST_ASSERT(topk_idx.has_value());  // worst-token 模式只用于带 topk 的 forward。
                EP_HOST_ASSERT(topk_weights.has_value());  // topk 权重也必须存在。
            } else {
                // CPU 轮询 mapped counter：moe_recv_counter 给总 token 数，
                // moe_recv_expert_counter [num_local_experts] 给每个本地 expert token 数。
                auto start_time = std::chrono::high_resolution_clock::now();  // 记录轮询开始时间，用于超时保护。
                while (true) {
                    // Read total count
                    num_recv_tokens = static_cast<int>(*moe_recv_counter);  // 读取 GPU 写回的总接收 token 数。

                    // Read per-expert count
                    bool ready = (num_recv_tokens >= 0);  // 总数非负说明总计数已就绪。
                    for (int i = 0; i < num_local_experts and ready; ++i)
                        ready &= moe_recv_expert_counter[i] >= 0;  // 每个本地 expert 计数都需非负。

                    if (ready)
                        break;  // 所有 metadata 就绪后退出 CPU 轮询。

                    // Timeout check
                    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                        LEGACY_NUM_CPU_TIMEOUT_SECS)
                        throw std::runtime_error("DeepEP error: CPU recv timeout");  // 防止 peer 挂死时 Python 永久卡住。
                }
                num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);  // 拷贝成 Python 可返回的 vector。
            }
        }

        // 分配输出张量。recv_x [num_recv_tokens, hidden]；recv_src_idx [num_recv_tokens]
        // 记录源 token id；recv_channel_prefix_matrix 和 send_head 是 combine/cached 路径所需 metadata。
        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());  // 接收后的 token 特征 [recv_tokens,hidden]。
        auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));  // 每个接收 token 对应原始 token index。
        auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
             recv_x_scales = std::optional<torch::Tensor>();  // 可选接收 topk/scales 输出。
        auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));  // combine 需要的接收侧 channel prefix。
        auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));  // [num_tokens,num_ranks]，记录发送队列槽位。

        // optional tensor 只有在对应输入存在时分配；kernel 通过 nullptr 判断是否处理 topk/scales。
        topk_idx_t* recv_topk_idx_ptr = nullptr;  // 接收 topk_idx 输出指针。
        float* recv_topk_weights_ptr = nullptr;  // 接收 topk_weights 输出指针。
        float* recv_x_scales_ptr = nullptr;  // 接收 FP8 scales 输出指针。
        if (topk_idx.has_value()) {
            recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());  // 接收侧 topk_idx [recv_tokens,num_topk]。
            recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());  // 接收侧 topk_weights [recv_tokens,num_topk]。
            recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();  // 取 topk_idx 输出指针。
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();  // 取 topk_weights 输出指针。
        }
        if (x_scales.has_value()) {
            recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                                 : torch::empty({num_recv_tokens, num_scales}, x_scales->options());  // 接收侧 FP8 scales，维度跟输入 scale 类型对应。
            recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());  // 取 scale 输出指针。
        }

        // Dispatch 前做一次严格容量检查，逐项对应 intranode.cu 中 channel buffer 的实际布局。
        // 这类检查虽然长，但能在 host 侧提前发现 config chunk size 太大导致的 buffer 越界。
        EP_HOST_ASSERT(
            num_ranks * num_ranks * sizeof(int) +                                                                     // rank 前缀矩阵。
                num_channels * num_ranks * sizeof(int) +                                                              // channel 起始 offset。
                num_channels * num_ranks * sizeof(int) +                                                              // channel 结束 offset。
                num_channels * num_ranks * sizeof(int) * 2 +                                                          // 队列 head/tail。
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +  // payload 数据 buffer。
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                     // 源 token index buffer。
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +   // top-k id buffer。
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +        // top-k 权重 buffer。
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales        // FP8 scale buffer。
            <= num_nvl_bytes);
        intranode::dispatch(recv_x.data_ptr(),  // 输出接收 token 特征。
                            recv_x_scales_ptr,  // 输出接收 FP8 scale，可为 nullptr。
                            recv_src_idx.data_ptr<int>(),  // 输出源 token index。
                            recv_topk_idx_ptr,  // 输出接收 topk_idx，可为 nullptr。
                            recv_topk_weights_ptr,  // 输出接收 topk_weights，可为 nullptr。
                            recv_channel_prefix_matrix.data_ptr<int>(),  // 输出接收侧 channel prefix。
                            send_head.data_ptr<int>(),  // 输出发送槽位 metadata。
                            x.data_ptr(),  // 输入 token 特征。
                            x_scales_ptr,  // 输入 FP8 scale，可为 nullptr。
                            topk_idx_ptr,  // 输入 topk_idx，可为 nullptr。
                            topk_weights_ptr,  // 输入 topk_weights，可为 nullptr。
                            is_token_in_rank.data_ptr<bool>(),  // 输入路由布尔表。
                            channel_prefix_matrix.data_ptr<int>(),  // notify 阶段计算出的发送 channel prefix。
                            num_tokens,  // 输入 token 数。
                            num_worst_tokens,  // worst-case 接收容量；0 表示使用真实计数。
                            static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),  // hidden 以 int4 为单位的向量化长度。
                            num_topk,  // top-k 路数。
                            num_experts,  // 全局 expert 数。
                            num_scales,  // FP8 scale 数。
                            scale_token_stride,  // scale token stride。
                            scale_hidden_stride,  // scale hidden stride。
                            buffer_ptrs_gpu,  // peer buffer 指针表。
                            rank,  // 当前 rank。
                            num_ranks,  // 节点内 rank 数。
                            comm_stream,  // 通信流。
                            config.num_sms,  // kernel 使用的 SM/block 数。
                            config.num_max_nvl_chunked_send_tokens,  // 每 channel NVL send chunk 上限。
                            config.num_max_nvl_chunked_recv_tokens);  // 每 channel NVL recv chunk 上限。

        // Wait streams
        std::optional<EventHandle> event;  // async 模式返回给 Python 的通信完成事件。
        if (async) {
            event = EventHandle(comm_stream);  // 在当前通信流位置记录事件。
            for (auto& t : {x,
                            is_token_in_rank,
                            // 中文：rank/channel prefix 是 intranode cached dispatch 与 combine 共享的布局表。
                            rank_prefix_matrix,
                            channel_prefix_matrix,
                            // 中文：recv_x 及 handle tensor 都可能在 Python 当前流继续被引用，必须绑定通信流生命周期。
                            recv_x,
                            // 中文：recv_src_idx/recv_channel_prefix_matrix/send_head 都会进入 Python handle。
                            recv_src_idx,
                            recv_channel_prefix_matrix,
                            send_head}) {
                t.record_stream(comm_stream);  // 输入/输出 tensor 都标记被通信流使用。
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);  // 通信流分配时也绑定回计算流生命周期。
            }
            for (auto& to : {x_scales,  // 可选输入 FP8 scale。
                             topk_idx,  // 可选 top-k id。
                             topk_weights,  // 可选 top-k weight。
                             num_tokens_per_rank,  // 可选 rank 计数。
                             num_tokens_per_expert,  // 可选 expert 计数。
                             cached_channel_prefix_matrix,  // cached channel prefix。
                             cached_rank_prefix_matrix,  // cached rank prefix。
                             recv_topk_idx,  // 可选输出 topk_idx。
                             recv_topk_weights,  // 可选输出 topk_weights。
                             recv_x_scales}) {  // 可选输出 FP8 scale。
                to.has_value() ? to->record_stream(comm_stream) : void();  // 可选 tensor 存在时记录通信流。
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();  // 可选 tensor 也关联计算流。
            }
        } else {
            stream_wait(compute_stream, comm_stream);  // 同步模式让计算流等待 dispatch 完成。
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);  // 恢复原始当前流。

        // Return values
        return {recv_x,
                recv_x_scales,  // 可选接收 FP8 scales。
                recv_topk_idx,  // 可选接收 topk_idx。
                recv_topk_weights,  // 可选接收 topk_weights。
                num_recv_tokens_per_expert_list,  // CPU 侧本地 expert 接收计数。
                rank_prefix_matrix,  // combine/cached dispatch 复用的 rank prefix。
                channel_prefix_matrix,  // dispatch 发送侧 channel prefix。
                recv_channel_prefix_matrix,  // combine 使用的接收侧 channel prefix。
                recv_src_idx,  // combine 回传所需源 token index。
                send_head,  // combine/cached dispatch 所需发送槽位。
                event};  // 可选异步事件。
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> intranode_combine(
        const torch::Tensor& x,  // expert 计算后的 token 特征，[num_tokens,hidden]。
        const std::optional<torch::Tensor>& topk_weights,  // 可选 top-k 权重，[num_tokens,num_topk]。
        const std::optional<torch::Tensor>& bias_0,  // 可选 bias tensor 0，[num_recv_tokens,hidden]。
        const std::optional<torch::Tensor>& bias_1,  // 可选 bias tensor 1，[num_recv_tokens,hidden]。
        const torch::Tensor& src_idx,  // dispatch 返回的源 token index，[num_tokens]。
        const torch::Tensor& rank_prefix_matrix,  // dispatch 返回的 rank prefix，[num_ranks,num_ranks]。
        const torch::Tensor& channel_prefix_matrix,  // dispatch 返回的 recv channel prefix，[num_ranks,num_channels]。
        const torch::Tensor& send_head,  // dispatch 返回的发送槽位，[num_recv_tokens,num_ranks]。
        const Config& config,  // legacy combine kernel 配置。
        const std::optional<EventHandle>& previous_event,  // 上游事件。
        bool async,  // 是否异步返回 EventHandle。
        bool allocate_on_comm_stream) {  // 是否在通信流上分配输出。
        EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());  // x 必须为二维连续矩阵。
        EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);  // src_idx 是 int32 一维表。
        EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);  // send_head 是 int32 二维矩阵。
        EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                       rank_prefix_matrix.scalar_type() == torch::kInt32);  // rank prefix 是 int32 二维矩阵。
        EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and
                       channel_prefix_matrix.scalar_type() == torch::kInt32);  // channel prefix 是 int32 二维矩阵。

        // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
        EP_HOST_ASSERT(config.num_sms % 2 == 0);  // combine 也沿用 send/recv block 成对约定。
        int num_channels = config.num_sms / 2;  // channel 数等于 block 对数。

        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));  // 本地 expert 输出 token 数和 hidden。
        auto num_recv_tokens = static_cast<int>(send_head.size(0));  // combine 后原始 rank 应接收的 token 数。
        EP_HOST_ASSERT(src_idx.size(0) == num_tokens);  // 每个 expert 输出 token 都必须有源 index。
        EP_HOST_ASSERT(send_head.size(1) == num_ranks);  // send_head 第二维按 rank 展开。
        EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);  // rank prefix 形状校验。
        EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);  // channel prefix 形状校验。
        EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);  // hidden 字节数必须 int4 对齐。

        // Allocate all tensors on comm stream if set
        // NOTES: do not allocate tensors upfront!
        auto compute_stream = at::cuda::getCurrentCUDAStream();  // 保存原计算流。
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);  // 通信流分配要求异步事件。
            at::cuda::setCurrentCUDAStream(comm_stream);  // 输出 tensor 在通信流上分配。
        }

        // Wait previous tasks to be finished
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());  // 通信流等待前序事件。
        } else {
            stream_wait(comm_stream, compute_stream);  // 或等待当前计算流。
        }

        int num_topk = 0;  // top-k 路数。
        auto recv_topk_weights = std::optional<torch::Tensor>();  // combine 后回到原 token 顺序的 topk 权重。
        float* topk_weights_ptr = nullptr;  // 输入 topk_weights 指针。
        float* recv_topk_weights_ptr = nullptr;  // 输出 topk_weights 指针。
        if (topk_weights.has_value()) {
            EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());  // topk 权重必须二维连续。
            EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);  // topk 权重 token 维匹配 expert 输出。
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);  // topk 权重按 float32 处理。
            num_topk = static_cast<int>(topk_weights->size(1));  // 提取 top-k 路数。
            topk_weights_ptr = topk_weights->data_ptr<float>();  // 输入权重 CUDA 指针。
            recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());  // 输出权重 [num_recv_tokens,num_topk]。
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();  // 输出权重 CUDA 指针。
        }

        // Launch barrier and reset queue head and tail
        EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);  // buffer 至少能容纳 head/tail 两组队列控制字。
        intranode::cached_notify_combine(buffer_ptrs_gpu,  // peer buffer 指针表。
                                         send_head.data_ptr<int>(),  // 输入 dispatch 阶段记录的发送槽位。
                                         num_channels,  // channel 数。
                                         num_recv_tokens,  // combine 输出 token 数。
                                         num_channels * num_ranks * 2,  // 需要清零的 head/tail int 数。
                                         barrier_signal_ptrs_gpu,  // peer barrier 指针表。
                                         rank,  // 当前 rank。
                                         num_ranks,  // 节点内 rank 数。
                                         comm_stream);  // 通信流。

        // Assign bias pointers
        auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});  // 用数组统一处理两个可选 bias。
        void* bias_ptrs[2] = {nullptr, nullptr};  // 传给 CUDA kernel 的 bias 指针数组。
        for (int i = 0; i < 2; ++i)
            if (bias_opts[i].has_value()) {
                auto bias = bias_opts[i].value();  // 取出第 i 个 bias tensor。
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());  // bias 必须二维连续。
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());  // bias dtype 要和 x 一致，便于 kernel 直接相加。
                EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and bias.size(1) == hidden);  // bias 形状 [num_recv_tokens,hidden]。
                bias_ptrs[i] = bias.data_ptr();  // 保存原始 CUDA 指针。
            }

        // Combine data
        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());  // combine 输出，回到源 rank/token 布局。
        EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
                           // 中文：buffer 布局依次包含 head/tail、payload、src_idx 和可选 topk_weights ring。
                           num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * x.element_size() +  // Data buffer
                           num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +             // Source index buffer
                           // 中文：topk weight ring 的最后一维为 num_topk，只有请求权重 combine 时才真正读取。
                           num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)  // Top-k weight buffer
                       <= num_nvl_bytes);
        intranode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),  // 把 torch dtype 转为 CUDA dtype enum。
                           recv_x.data_ptr(),  // 输出 combine 后 token 特征。
                           recv_topk_weights_ptr,  // 输出 combine 后 topk 权重，可为 nullptr。
                           x.data_ptr(),  // 输入 expert 输出 token 特征。
                           topk_weights_ptr,  // 输入 topk 权重，可为 nullptr。
                           bias_ptrs[0],  // 可选 bias_0。
                           bias_ptrs[1],  // 可选 bias_1。
                           src_idx.data_ptr<int>(),  // dispatch 源 token index。
                           rank_prefix_matrix.data_ptr<int>(),  // rank prefix metadata。
                           channel_prefix_matrix.data_ptr<int>(),  // channel prefix metadata。
                           send_head.data_ptr<int>(),  // 发送槽位 metadata。
                           num_tokens,  // 输入 token 数。
                           num_recv_tokens,  // 输出 token 数。
                           hidden,  // hidden 维度。
                           num_topk,  // top-k 路数。
                           buffer_ptrs_gpu,  // peer buffer 指针表。
                           rank,  // 当前 rank。
                           num_ranks,  // 节点内 rank 数。
                           comm_stream,  // 通信流。
                           config.num_sms,  // kernel block/SM 配置。
                           config.num_max_nvl_chunked_send_tokens,  // send chunk token 上限。
                           config.num_max_nvl_chunked_recv_tokens);  // recv chunk token 上限。

        // Wait streams
        std::optional<EventHandle> event;  // async 模式的返回事件。
        if (async) {
            event = EventHandle(comm_stream);  // 记录通信流当前位置。
            for (auto& t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
                t.record_stream(comm_stream);  // 必选 tensor 记录通信流使用关系。
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);  // 通信流分配时也记录计算流。
            }
            for (auto& to : {topk_weights, recv_topk_weights, bias_0, bias_1}) {
                to.has_value() ? to->record_stream(comm_stream) : void();  // 可选 tensor 存在时记录通信流。
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();  // 可选 tensor 也记录计算流。
            }
        } else {
            stream_wait(compute_stream, comm_stream);  // 同步模式让计算流等待 combine 完成。
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);  // 恢复调用方原计算流。

        return {recv_x, recv_topk_weights, event};  // 返回 combine 输出、可选权重和异步事件。
    }

    std::tuple<torch::Tensor,                 // 接收 token 特征 recv_x。
               std::optional<torch::Tensor>,  // 可选接收 scale recv_x_scales。
               std::optional<torch::Tensor>,  // 可选接收 topk_idx。
               std::optional<torch::Tensor>,  // 可选接收 topk_weights。
               std::vector<int>,              // 本地 expert 接收计数列表。
               torch::Tensor,                 // 源 metadata recv_src_meta。
               torch::Tensor,                 // RDMA channel prefix 矩阵。
               std::optional<torch::Tensor>,  // cached RDMA channel prefix。
               torch::Tensor,                 // RDMA rank prefix。
               std::optional<torch::Tensor>,  // cached RDMA rank prefix。
               torch::Tensor,                 // 全局/NVL channel prefix。
               std::optional<torch::Tensor>,  // cached 全局/NVL channel prefix。
               std::optional<torch::Tensor>,  // RDMA send head。
               std::optional<torch::Tensor>,  // NVL send head。
               std::optional<EventHandle>>    // 可选异步事件。
    internode_dispatch(const torch::Tensor& x,  // 待发送 token 特征，[num_tokens,hidden]。
                       const std::optional<torch::Tensor>& x_scales,  // 可选 FP8 scales，[num_tokens,num_scales]。
                       const std::optional<torch::Tensor>& topk_idx,  // 可选 top-k expert id，[num_tokens,num_topk]。
                       const std::optional<torch::Tensor>& topk_weights,  // 可选 top-k 权重，[num_tokens,num_topk]。
                       const std::optional<torch::Tensor>& num_tokens_per_rank,  // 非 cached 输入：[num_ranks]。
                       const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,  // 非 cached 输入：[num_rdma_ranks]。
                       const torch::Tensor& is_token_in_rank,  // 路由布尔表：[num_tokens,num_ranks]。
                       const std::optional<torch::Tensor>& num_tokens_per_expert,  // 非 cached 输入：[num_experts]。
                       int cached_num_recv_tokens,  // cached 模式复用的最终接收 token 数。
                       int cached_num_rdma_recv_tokens,  // cached 模式复用的 RDMA 阶段接收 token 数。
                       const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,  // cached RDMA channel prefix：[num_rdma_ranks,num_channels]。
                       const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,  // cached RDMA rank prefix：[num_rdma_ranks]。
                       const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,  // cached 全局 channel prefix：[num_ranks,num_channels]。
                       const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,  // cached 全局 rank prefix：[num_ranks]。
                       int expert_alignment,  // expert 接收计数对齐粒度。
                       int num_worst_tokens,  // >0 时按 worst case 容量跳过部分动态计数。
                       const Config& config,  // RDMA/NVL chunk 和 SM 配置。
                       std::optional<EventHandle>& previous_event,  // 上游事件。
                       bool async,  // 是否异步返回。
                       bool allocate_on_comm_stream) {  // 是否在通信流上分配输出。
        // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
        // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
        // unless we release GIL here.
        pybind11::gil_scoped_release release;  // CPU busy-wait 期间释放 GIL，避免阻塞其它 Python 线程。

        const int num_channels = config.num_sms / 2;  // internode 也将 SM 分为 send/recv channel 对。
        EP_HOST_ASSERT(config.num_sms % 2 == 0);  // num_sms 必须为偶数才能成对分配 send/recv block。
        EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= LEGACY_NUM_MAX_RDMA_PEERS);  // RDMA peer 数必须在静态上限内。

        bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();  // 是否复用上一轮 internode metadata。
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());  // cached 模式必须有 RDMA channel prefix。
            EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());  // cached 模式必须有 RDMA rank prefix。
            EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());  // cached 模式必须有全局 channel prefix。
            EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());  // cached 模式必须有全局 rank prefix。
        } else {
            EP_HOST_ASSERT(num_tokens_per_rank.has_value());  // 非 cached 模式需要 rank 计数。
            EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());  // 非 cached 模式需要 RDMA rank 计数。
            EP_HOST_ASSERT(num_tokens_per_expert.has_value());  // 非 cached 模式需要 expert 计数。
        }

        // Type checks
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);  // RDMA channel prefix 是 int32 offset。
            EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);  // RDMA rank prefix 是 int32 offset。
            EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);  // 全局 channel prefix 是 int32 offset。
            EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);  // 全局 rank prefix 是 int32 offset。
        } else {
            EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);  // rank 计数按 int32 读取。
            EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);  // RDMA rank 计数按 int32 读取。
            EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);  // expert 计数按 int32 读取。
        }

        // Shape and contiguous checks
        EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());  // x 必须是二维连续 token 矩阵。
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);  // hidden 字节数必须能 int4 向量化搬运。
        if (cached_mode) {
            EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());  // RDMA channel prefix 二维连续。
            EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
                           cached_rdma_channel_prefix_matrix->size(1) == num_channels);  // 形状 [num_rdma_ranks,num_channels]。
            EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());  // RDMA prefix 一维连续。
            EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);  // RDMA prefix 长度等于 RDMA rank 数。
            EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());  // 全局 channel prefix 二维连续。
            EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and
                           cached_gbl_channel_prefix_matrix->size(1) == num_channels);  // 形状 [num_ranks,num_channels]。
            EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());  // 全局 rank prefix 一维连续。
            EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);  // 全局 rank prefix 长度等于 rank 数。
        } else {
            EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());  // rank 计数一维连续。
            EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());  // RDMA rank 计数一维连续。
            EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());  // expert 计数一维连续。
            EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);  // rank 计数长度必须等于全局 rank 数。
            EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);  // RDMA 计数长度必须等于 RDMA rank 数。
            EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);  // expert 总数必须能按 rank 均分。
            EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= LEGACY_NUM_MAX_LOCAL_EXPERTS);  // 本地 expert 数不能超过静态上限。
        }

        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
             hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));  // hidden_int4 是 kernel 搬运循环的向量长度。
        auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;  // 非 cached 推导 expert 数。

        // Top-k checks
        int num_topk = 0;  // top-k 路数，未传 topk 时保持 0。
        topk_idx_t* topk_idx_ptr = nullptr;  // topk_idx 输入指针，可为空。
        float* topk_weights_ptr = nullptr;  // topk 权重输入指针，可为空。
        EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());  // topk_idx 和 topk_weights 必须同时存在或同时不存在。
        if (topk_idx.has_value()) {
            num_topk = static_cast<int>(topk_idx->size(1));  // 读取 top-k 路数。
            EP_HOST_ASSERT(num_experts > 0);  // 需要 expert 总数做 expert id 映射。
            EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());  // topk_idx 必须二维连续。
            EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());  // topk_weights 必须二维连续。
            EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));  // topk token 维匹配 x。
            EP_HOST_ASSERT(num_topk == topk_weights->size(1));  // top-k 维一致。
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);  // 权重按 float32 处理。
            topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();  // 获取 topk_idx CUDA 指针。
            topk_weights_ptr = topk_weights->data_ptr<float>();  // 获取 topk_weights CUDA 指针。
        }

        // FP8 scales checks
        float* x_scales_ptr = nullptr;  // FP8 scale 输入指针，可为空。
        int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;  // scale 维度和 stride。
        if (x_scales.has_value()) {
            EP_HOST_ASSERT(x.element_size() == 1);  // 只有 FP8 输入才允许携带 scale。
            EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);  // 支持 fp32 scale 或 ue8m0 int scale。
            EP_HOST_ASSERT(x_scales->dim() == 2);  // scale 当前按二维矩阵处理。
            EP_HOST_ASSERT(x_scales->size(0) == num_tokens);  // scale token 维必须匹配 x。
            num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));  // scale 列数。
            x_scales_ptr = static_cast<float*>(x_scales->data_ptr());  // 获取 scale 原始指针。
            scale_token_stride = static_cast<int>(x_scales->stride(0));  // token 维 stride。
            scale_hidden_stride = static_cast<int>(x_scales->stride(1));  // hidden/scale 维 stride。
        }

        // Allocate all tensors on comm stream if set
        // NOTES: do not allocate tensors upfront!
        auto compute_stream = at::cuda::getCurrentCUDAStream();  // 保存调用方当前计算流。
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);  // 通信流分配要求异步事件保障生命周期。
            at::cuda::setCurrentCUDAStream(comm_stream);  // 后续分配发生在通信流上。
        }

        // Wait previous tasks to be finished
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());  // 通信流等待前序事件。
        } else {
            stream_wait(comm_stream, compute_stream);  // 或等待当前计算流。
        }

        // Create handles (only return for non-cached mode)
        int num_recv_tokens = -1, num_rdma_recv_tokens = -1;  // 最终接收 token 数和 RDMA 中间层接收 token 数。
        auto rdma_channel_prefix_matrix = torch::Tensor();  // [num_rdma_ranks,num_channels]，RDMA 阶段 channel prefix。
        auto recv_rdma_rank_prefix_sum = torch::Tensor();  // [num_rdma_ranks]，本 rank 从每个 RDMA rank 接收的前缀和。
        auto gbl_channel_prefix_matrix = torch::Tensor();  // [num_ranks,num_channels]，全局 NVL 阶段 channel prefix。
        auto recv_gbl_rank_prefix_sum = torch::Tensor();  // [num_ranks]，最终从每个全局 rank 接收的前缀和。
        std::vector<int> num_recv_tokens_per_expert_list;  // 返回 Python 的本地 expert 接收计数。

        // Barrier or send sizes
        if (cached_mode) {
            num_recv_tokens = cached_num_recv_tokens;  // 复用最终接收 token 数。
            num_rdma_recv_tokens = cached_num_rdma_recv_tokens;  // 复用 RDMA 中间接收 token 数。
            rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();  // 复用 RDMA channel prefix。
            recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();  // 复用 RDMA rank prefix。
            gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();  // 复用全局 channel prefix。
            recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();  // 复用全局 rank prefix。

            // Just a barrier and clean flags
            internode::cached_notify(hidden_int4,  // hidden 的 int4 向量长度。
                                     num_scales,  // FP8 scale 数。
                                     num_topk,  // dispatch 携带的 topk index 数。
                                     num_topk,  // dispatch 携带的 topk weight 数。
                                     num_ranks,  // 全局 rank 数。
                                     num_channels,  // channel 数。
                                     0,  // combine token 数，此处 dispatch cached notify 不需要。
                                     nullptr,  // combine RDMA head，此处不用。
                                     nullptr,  // RDMA channel prefix，此处只做清理/同步。
                                     nullptr,  // RDMA rank prefix，此处不用。
                                     nullptr,  // NVL head，此处不用。
                                     rdma_buffer_ptr,  // NVSHMEM symmetric RDMA buffer。
                                     config.num_max_rdma_chunked_recv_tokens,  // RDMA recv chunk 上限。
                                     buffer_ptrs_gpu,  // NVL peer buffer 指针表。
                                     config.num_max_nvl_chunked_recv_tokens,  // NVL recv chunk 上限。
                                     barrier_signal_ptrs_gpu,  // NVL peer barrier 指针表。
                                     rank,  // 当前 rank。
                                     comm_stream,  // 通信流。
                                     config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),  // RDMA buffer hint。
                                     num_nvl_bytes,  // NVL buffer 总字节数。
                                     true,  // true 表示 dispatch notify 语义。
                                     low_latency_mode);  // 是否复用 low-latency RDMA PE 编号约定。
        } else {
            rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));  // 分配 RDMA channel prefix。
            recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));  // 分配 RDMA rank prefix。
            gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));  // 分配全局 channel prefix。
            recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));  // 分配全局 rank prefix。

            // Send sizes
            *moe_recv_counter = -1;  // 总接收 token mapped counter 置未就绪。
            *moe_recv_rdma_counter = -1;  // RDMA 阶段接收 token mapped counter 置未就绪。
            for (int i = 0; i < num_local_experts; ++i)
                moe_recv_expert_counter[i] = -1;  // 每个本地 expert counter 置未就绪。
            internode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),  // 输入全局 rank 计数。
                                       moe_recv_counter_mapped,  // GPU 写回最终接收 token 数。
                                       num_ranks,  // 全局 rank 数。
                                       num_tokens_per_rdma_rank->data_ptr<int>(),  // 输入 RDMA rank 计数。
                                       moe_recv_rdma_counter_mapped,  // GPU 写回 RDMA 阶段接收 token 数。
                                       num_tokens_per_expert->data_ptr<int>(),  // 输入 expert 计数。
                                       moe_recv_expert_counter_mapped,  // GPU 写回本地 expert 计数。
                                       num_experts,  // 全局 expert 数。
                                       is_token_in_rank.data_ptr<bool>(),  // token-rank 路由表。
                                       num_tokens,  // 输入 token 数。
                                       num_worst_tokens,  // worst case token 容量。
                                       num_channels,  // channel 数。
                                       hidden_int4,  // hidden 的 int4 长度。
                                       num_scales,  // FP8 scale 数。
                                       num_topk,  // top-k 路数。
                                       expert_alignment,  // expert token 对齐粒度。
                                       rdma_channel_prefix_matrix.data_ptr<int>(),  // 输出 RDMA channel prefix。
                                       recv_rdma_rank_prefix_sum.data_ptr<int>(),  // 输出 RDMA rank prefix。
                                       gbl_channel_prefix_matrix.data_ptr<int>(),  // 输出全局 channel prefix。
                                       recv_gbl_rank_prefix_sum.data_ptr<int>(),  // 输出全局 rank prefix。
                                       rdma_buffer_ptr,  // RDMA symmetric buffer。
                                       config.num_max_rdma_chunked_recv_tokens,  // RDMA recv chunk 上限。
                                       buffer_ptrs_gpu,  // NVL peer buffer 指针表。
                                       config.num_max_nvl_chunked_recv_tokens,  // NVL recv chunk 上限。
                                       barrier_signal_ptrs_gpu,  // NVL barrier 指针表。
                                       rank,  // 当前 rank。
                                       comm_stream,  // 通信流。
                                       config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),  // RDMA buffer hint。
                                       num_nvl_bytes,  // NVL buffer 字节数。
                                       low_latency_mode);  // 是否处于 low-latency PE 编号模式。

            // Synchronize total received tokens and tokens per expert
            auto start_time = std::chrono::high_resolution_clock::now();  // 记录 CPU 轮询开始时间。
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);  // 读取最终接收 token 数。
                num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);  // 读取 RDMA 阶段接收 token 数。

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);  // 两个总计数均非负才算初步 ready。
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;  // 每个本地 expert counter 也必须 ready。

                if (ready)
                    break;  // 所有 metadata 写回后退出轮询。

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                    LEGACY_NUM_CPU_TIMEOUT_SECS) {
                    printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);  // 超时时打印总计数便于定位卡住 rank。
                    for (int i = 0; i < num_local_experts; ++i)
                        printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);  // 打印每个 expert counter。
                    throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");  // 抛错而不是无限等待。
                }
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);  // 拷贝 expert 计数给 Python。
        }

        // Allocate new tensors
        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());  // 最终接收 token 特征 [num_recv_tokens,hidden]。
        auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
             recv_x_scales = std::optional<torch::Tensor>();  // 可选接收 topk/scales 输出。
        auto recv_src_meta = std::optional<torch::Tensor>();  // 非 cached 返回 source meta，combine 需要它回传。
        auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();  // 非 cached 的接收侧 RDMA channel prefix。
        auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();  // 非 cached 的接收侧全局 channel prefix。
        auto send_rdma_head = std::optional<torch::Tensor>();  // [num_tokens,num_rdma_ranks]，RDMA 发送槽位。
        auto send_nvl_head = std::optional<torch::Tensor>();  // [num_rdma_recv_tokens,LEGACY_NUM_MAX_NVL_PEERS]，NVL 发送槽位。
        if (not cached_mode) {
            recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));  // 每个接收 token 的源 rank/offset 元数据。
            recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));  // combine 侧 RDMA channel prefix。
            recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));  // combine 侧全局 channel prefix。
            send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));  // dispatch RDMA send head。
            send_nvl_head = torch::empty({num_rdma_recv_tokens, LEGACY_NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));  // dispatch NVL send head。
        }

        // Assign pointers
        topk_idx_t* recv_topk_idx_ptr = nullptr;  // 接收 topk_idx 输出指针。
        float* recv_topk_weights_ptr = nullptr;  // 接收 topk_weights 输出指针。
        float* recv_x_scales_ptr = nullptr;  // 接收 FP8 scales 输出指针。
        if (topk_idx.has_value()) {
            recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());  // 接收 topk_idx [num_recv_tokens,num_topk]。
            recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());  // 接收 topk 权重 [num_recv_tokens,num_topk]。
            recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();  // 获取输出 topk_idx 指针。
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();  // 获取输出 topk_weights 指针。
        }
        if (x_scales.has_value()) {
            recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                                 : torch::empty({num_recv_tokens, num_scales}, x_scales->options());  // 接收 FP8 scales。
            recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());  // 获取输出 scales 指针。
        }

        // Launch data dispatch
        // NOTES: the buffer size checks are moved into the `.cu` file
        internode::dispatch(recv_x.data_ptr(),  // 输出最终接收 token 特征。
                            recv_x_scales_ptr,  // 输出接收 scales，可为 nullptr。
                            recv_topk_idx_ptr,  // 输出接收 topk_idx，可为 nullptr。
                            recv_topk_weights_ptr,  // 输出接收 topk_weights，可为 nullptr。
                            cached_mode ? nullptr : recv_src_meta->data_ptr(),  // 非 cached 输出 source meta，cached combine 已有旧 meta。
                            x.data_ptr(),  // 输入 token 特征。
                            x_scales_ptr,  // 输入 FP8 scales，可为 nullptr。
                            topk_idx_ptr,  // 输入 topk_idx，可为 nullptr。
                            topk_weights_ptr,  // 输入 topk_weights，可为 nullptr。
                            cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),  // 非 cached 输出 RDMA send head。
                            cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),  // 非 cached 输出 NVL send head。
                            cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),  // 非 cached 输出接收 RDMA channel prefix。
                            cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),  // 非 cached 输出接收全局 channel prefix。
                            rdma_channel_prefix_matrix.data_ptr<int>(),  // 输入 RDMA channel prefix。
                            recv_rdma_rank_prefix_sum.data_ptr<int>(),  // 输入 RDMA rank prefix。
                            gbl_channel_prefix_matrix.data_ptr<int>(),  // 输入全局 channel prefix。
                            recv_gbl_rank_prefix_sum.data_ptr<int>(),  // 输入全局 rank prefix。
                            is_token_in_rank.data_ptr<bool>(),  // 输入 token-rank 路由表。
                            num_tokens,  // 输入 token 数。
                            num_worst_tokens,  // worst-case 容量。
                            hidden_int4,  // hidden 的 int4 向量长度。
                            num_scales,  // FP8 scale 数。
                            num_topk,  // top-k 路数。
                            num_experts,  // 全局 expert 数。
                            scale_token_stride,  // scale token stride。
                            scale_hidden_stride,  // scale hidden stride。
                            rdma_buffer_ptr,  // RDMA symmetric buffer。
                            config.num_max_rdma_chunked_send_tokens,  // RDMA send chunk 上限。
                            config.num_max_rdma_chunked_recv_tokens,  // RDMA recv chunk 上限。
                            buffer_ptrs_gpu,  // NVL peer buffer 指针表。
                            config.num_max_nvl_chunked_send_tokens,  // NVL send chunk 上限。
                            config.num_max_nvl_chunked_recv_tokens,  // NVL recv chunk 上限。
                            rank,  // 当前 rank。
                            num_ranks,  // 全局 rank 数。
                            cached_mode,  // 是否走 cached kernel 路径。
                            comm_stream,  // 通信流。
                            num_channels,  // channel 数。
                            low_latency_mode);  // 是否使用 low-latency NVSHMEM PE 编号。

        // Wait streams
        std::optional<EventHandle> event;  // async 模式返回事件。
        if (async) {
            event = EventHandle(comm_stream);  // 在通信流当前位置记录事件。
            for (auto& t : {x,
                            is_token_in_rank,
                            recv_x,
                            // 中文：RDMA prefix、global prefix 与 send head 共同组成 internode dispatch handle。
                            rdma_channel_prefix_matrix,
                            // 中文：internode dispatch 的 handle 同时包含 RDMA prefix 和 NVL/global prefix。
                            recv_rdma_rank_prefix_sum,
                            gbl_channel_prefix_matrix,
                            // 中文：recv_gbl_rank_prefix_sum 描述最终落到各 global rank 的 token 前缀。
                            recv_gbl_rank_prefix_sum}) {
                t.record_stream(comm_stream);  // 必选 tensor 记录通信流使用关系。
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);  // 通信流分配时关联计算流。
            }
            for (auto& to : {x_scales,  // 可选输入 FP8 scale。
                             topk_idx,  // 可选 top-k id。
                             topk_weights,  // 可选 top-k weight。
                             num_tokens_per_rank,  // 可选全局 rank 计数。
                             num_tokens_per_rdma_rank,  // 可选 RDMA rank 计数。
                             num_tokens_per_expert,  // 可选 expert 计数。
                             // 中文：cached_* 参数必须来自同一 handle；混用不同路由轮次会导致反向 combine 错位。
                             cached_rdma_channel_prefix_matrix,  // cached RDMA channel prefix。
                             cached_recv_rdma_rank_prefix_sum,  // cached RDMA rank prefix。
                             // 中文：cached_gbl_* 对应节点内 NVL/global rank 的第二级 prefix。
                             cached_gbl_channel_prefix_matrix,  // cached NVL/global channel prefix。
                             cached_recv_gbl_rank_prefix_sum,  // cached global rank prefix。
                             // 中文：下面三个 optional 输出由 wrapper 按 with_topk/FP8 模式决定是否分配。
                             recv_topk_idx,  // 可选输出 topk_idx。
                             recv_topk_weights,  // 可选输出 topk_weights。
                             recv_x_scales,  // 可选输出 FP8 scale。
                             recv_rdma_channel_prefix_matrix,  // 可选输出 RDMA channel prefix。
                             recv_gbl_channel_prefix_matrix,  // 可选输出 global channel prefix。
                             send_rdma_head,  // 可选 RDMA send head。
                             send_nvl_head,  // 可选 NVL send head。
                             recv_src_meta}) {  // 可选源 metadata。
                to.has_value() ? to->record_stream(comm_stream) : void();  // 可选 tensor 存在时记录通信流。
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();  // 可选 tensor 也记录计算流。
            }
        } else {
            stream_wait(compute_stream, comm_stream);  // 同步模式下计算流等待 internode dispatch 完成。
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);  // 恢复调用方当前流。

        // Return values
        return {recv_x,
                recv_x_scales,  // 可选接收 scales。
                recv_topk_idx,  // 可选接收 topk_idx。
                recv_topk_weights,  // 可选接收 topk_weights。
                num_recv_tokens_per_expert_list,  // CPU expert 计数。
                rdma_channel_prefix_matrix,  // dispatch/cached 复用的 RDMA channel prefix。
                gbl_channel_prefix_matrix,  // dispatch/cached 复用的全局 channel prefix。
                recv_rdma_channel_prefix_matrix,  // combine 使用的接收 RDMA channel prefix。
                recv_rdma_rank_prefix_sum,  // combine 使用的 RDMA rank prefix。
                recv_gbl_channel_prefix_matrix,  // combine 使用的接收全局 channel prefix。
                recv_gbl_rank_prefix_sum,  // combine 使用的全局 rank prefix。
                recv_src_meta,  // combine 回传源位置所需 source meta。
                send_rdma_head,  // combine/cached 所需 RDMA send head。
                send_nvl_head,  // combine/cached 所需 NVL send head。
                event};  // 可选异步事件。
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> internode_combine(
        const torch::Tensor& x,  // expert 计算后的 token 特征，[num_tokens,hidden]。
        const std::optional<torch::Tensor>& topk_weights,  // 可选 top-k 权重，[num_tokens,num_topk]。
        const std::optional<torch::Tensor>& bias_0,  // 可选 bias 0，[num_combined_tokens,hidden]。
        const std::optional<torch::Tensor>& bias_1,  // 可选 bias 1，[num_combined_tokens,hidden]。
        const torch::Tensor& src_meta,  // dispatch 生成的源 token 元数据，[num_tokens,source_meta_bytes]。
        const torch::Tensor& is_combined_token_in_rank,  // combine 路由布尔表，[num_combined_tokens,num_ranks]。
        const torch::Tensor& rdma_channel_prefix_matrix,  // RDMA channel prefix，[num_rdma_ranks,num_channels]。
        const torch::Tensor& rdma_rank_prefix_sum,  // RDMA rank prefix，[num_rdma_ranks]。
        const torch::Tensor& gbl_channel_prefix_matrix,  // 全局 channel prefix，[num_ranks,num_channels]。
        const torch::Tensor& combined_rdma_head,  // combine RDMA send head，[num_combined_tokens,num_rdma_ranks]。
        const torch::Tensor& combined_nvl_head,  // combine NVL send head，[*,LEGACY_NUM_MAX_NVL_PEERS]。
        const Config& config,  // internode combine kernel 配置。
        const std::optional<EventHandle>& previous_event,  // 上游事件。
        bool async,  // 是否异步返回事件。
        bool allocate_on_comm_stream) {  // 是否在通信流分配输出。
        const int num_channels = config.num_sms / 2;  // channel 数由 SM 数的一半得到。
        EP_HOST_ASSERT(config.num_sms % 2 == 0);  // send/recv block 成对，SM 数必须偶数。

        // Shape and contiguous checks
        EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());  // x 必须二维连续。
        EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);  // source meta 是 byte 矩阵。
        EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and
                       is_combined_token_in_rank.scalar_type() == torch::kBool);  // combine 路由表是 bool 矩阵。
        EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and
                       rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);  // RDMA channel prefix 是 int32 矩阵。
        EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and
                       rdma_rank_prefix_sum.scalar_type() == torch::kInt32);  // RDMA rank prefix 是 int32 向量。
        EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and
                       gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);  // 全局 channel prefix 是 int32 矩阵。
        EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and
                       combined_rdma_head.scalar_type() == torch::kInt32);  // RDMA head 是 int32 矩阵。
        EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and combined_nvl_head.scalar_type() == torch::kInt32);  // NVL head 是 int32 矩阵。

        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
             hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));  // hidden 的 int4 向量长度。
        auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));  // combine 输出 token 数。
        EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);  // hidden 字节数必须 int4 对齐。
        EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());  // source meta 第二维必须等于 kernel 约定字节数。
        EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);  // combine 路由表第二维按全局 rank 展开。
        EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);  // RDMA channel prefix 形状校验。
        EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);  // RDMA rank prefix 长度校验。
        EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);  // 全局 channel prefix 形状校验。
        EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and
                       combined_rdma_head.size(1) == num_rdma_ranks);  // RDMA head 形状 [num_combined_tokens,num_rdma_ranks]。
        EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == LEGACY_NUM_MAX_NVL_PEERS);  // NVL head 第二维固定为最大 NVL peer 数。

        // Allocate all tensors on comm stream if set
        // NOTES: do not allocate tensors upfront!
        auto compute_stream = at::cuda::getCurrentCUDAStream();  // 保存当前计算流。
        if (allocate_on_comm_stream) {
            EP_HOST_ASSERT(previous_event.has_value() and async);  // 通信流分配模式需要异步事件。
            at::cuda::setCurrentCUDAStream(comm_stream);  // 切换当前流，后续分配归属通信流。
        }

        // Wait previous tasks to be finished
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());  // 通信流等待上游事件。
        } else {
            stream_wait(comm_stream, compute_stream);  // 或等待当前计算流。
        }

        // Top-k checks
        int num_topk = 0;  // top-k 路数。
        auto combined_topk_weights = std::optional<torch::Tensor>();  // combine 后的 topk 权重输出。
        float* topk_weights_ptr = nullptr;  // 输入 topk 权重指针。
        float* combined_topk_weights_ptr = nullptr;  // 输出 topk 权重指针。
        if (topk_weights.has_value()) {
            EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());  // topk 权重必须二维连续。
            EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);  // topk 权重 token 维匹配输入 x。
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);  // 权重 dtype 固定 float32。
            num_topk = static_cast<int>(topk_weights->size(1));  // 读取 top-k 路数。
            topk_weights_ptr = topk_weights->data_ptr<float>();  // 输入权重 CUDA 指针。
            combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());  // 输出权重 [num_combined_tokens,num_topk]。
            combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();  // 输出权重 CUDA 指针。
        }

        // Extra check for avoid-dead-lock design
        EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);  // NVL recv chunk 需能按 RDMA rank 均分，避免某一路长期占满。
        EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);  // send chunk 不超过均分容量，降低死锁风险。

        // Launch barrier and reset queue head and tail
        internode::cached_notify(hidden_int4,  // hidden 的 int4 长度。
                                 0,  // combine 不传 FP8 scale 元数据。
                                 0,  // combine 不传 topk_idx。
                                 num_topk,  // combine 可传 topk_weights。
                                 num_ranks,  // 全局 rank 数。
                                 num_channels,  // channel 数。
                                 num_combined_tokens,  // combine 输出 token 数。
                                 // 中文：combined_*_head 记录每个 token 发往各 rank 的 ring head，用于 cached notify。
                                 combined_rdma_head.data_ptr<int>(),  // combine RDMA send head。
                                 rdma_channel_prefix_matrix.data_ptr<int>(),  // RDMA channel prefix。
                                 rdma_rank_prefix_sum.data_ptr<int>(),  // RDMA rank prefix。
                                 // 中文：combined_nvl_head 配合 combined_rdma_head 让 cached_notify 能修正两级 head。
                                 combined_nvl_head.data_ptr<int>(),  // combine NVL send head。
                                 rdma_buffer_ptr,  // RDMA symmetric buffer。
                                 config.num_max_rdma_chunked_recv_tokens,  // RDMA recv chunk 上限。
                                 buffer_ptrs_gpu,  // NVL peer buffer 指针表。
                                 config.num_max_nvl_chunked_recv_tokens,  // NVL recv chunk 上限。
                                 barrier_signal_ptrs_gpu,  // NVL barrier 指针表。
                                 rank,  // 当前 rank。
                                 comm_stream,  // 通信流。
                                 config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),  // RDMA buffer hint。
                                 num_nvl_bytes,  // NVL buffer 字节数。
                                 false,  // false 表示 combine notify 语义。
                                 low_latency_mode);  // 是否处于 low-latency PE 编号模式。

        // Assign bias pointers
        auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});  // 将两个可选 bias 放入数组统一校验。
        void* bias_ptrs[2] = {nullptr, nullptr};  // CUDA kernel 接收的 bias 指针数组。
        for (int i = 0; i < 2; ++i)
            if (bias_opts[i].has_value()) {
                auto bias = bias_opts[i].value();  // 取第 i 个 bias tensor。
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());  // bias 必须二维连续。
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());  // bias dtype 与 x 一致。
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);  // bias 形状 [num_combined_tokens,hidden]。
                bias_ptrs[i] = bias.data_ptr();  // 保存 bias CUDA 指针。
            }

        // Launch data combine
        auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());  // combine 输出 token 特征。
        // internode combine 使用 dispatch 保存的 src_meta 和两级 prefix，把 expert 输出经 RDMA/NVL 两段送回源 token。
        internode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),  // torch dtype 转 CUDA dtype。
                           combined_x.data_ptr(),  // 输出 combine 特征。
                           combined_topk_weights_ptr,  // 输出 combine topk 权重，可为 nullptr。
                           is_combined_token_in_rank.data_ptr<bool>(),  // combine token-rank 路由表。
                           x.data_ptr(),  // 输入 expert 输出特征。
                           topk_weights_ptr,  // 输入 topk 权重，可为 nullptr。
                           bias_ptrs[0],  // 可选 bias_0。
                           bias_ptrs[1],  // 可选 bias_1。
                           combined_rdma_head.data_ptr<int>(),  // RDMA send head。
                           combined_nvl_head.data_ptr<int>(),  // NVL send head。
                           src_meta.data_ptr(),  // dispatch source meta。
                           // 中文：src_meta 中编码原 rank、原 token 和 top-k slot，是 combine 写回位置的根依据。
                           rdma_channel_prefix_matrix.data_ptr<int>(),  // RDMA channel prefix。
                           rdma_rank_prefix_sum.data_ptr<int>(),  // RDMA rank prefix。
                           gbl_channel_prefix_matrix.data_ptr<int>(),  // 全局 channel prefix。
                           num_tokens,  // 输入 token 数。
                           num_combined_tokens,  // 输出 token 数。
                           hidden,  // hidden 维度。
                           num_topk,  // top-k 路数。
                           rdma_buffer_ptr,  // RDMA symmetric buffer。
                           config.num_max_rdma_chunked_send_tokens,  // RDMA send chunk 上限。
                           config.num_max_rdma_chunked_recv_tokens,  // RDMA recv chunk 上限。
                           buffer_ptrs_gpu,  // NVL peer buffer 指针表。
                           config.num_max_nvl_chunked_send_tokens,  // NVL send chunk 上限。
                           config.num_max_nvl_chunked_recv_tokens,  // NVL recv chunk 上限。
                           rank,  // 当前 rank。
                           num_ranks,  // 全局 rank 数。
                           comm_stream,  // 通信流。
                           num_channels,  // channel 数。
                           low_latency_mode);  // 是否使用 low-latency PE 编号约定。

        // Wait streams
        std::optional<EventHandle> event;  // async 模式返回事件。
        if (async) {
            event = EventHandle(comm_stream);  // 在通信流当前位置记录事件。
            for (auto& t : {x,
                            src_meta,
                            is_combined_token_in_rank,
                            // 中文：is_combined_token_in_rank 与 prefix 矩阵共同决定哪些 rank 需要回收贡献。
                            rdma_channel_prefix_matrix,
                            // 中文：combine 反向路径要保活 src_meta、两级 prefix 与返回 head，直到通信事件完成。
                            rdma_rank_prefix_sum,
                            gbl_channel_prefix_matrix,
                            // 中文：combined_x 和两个 head tensor 是 combine kernel 写出的主要结果/control plane。
                            combined_x,
                            combined_rdma_head,
                            combined_nvl_head}) {
                t.record_stream(comm_stream);  // 必选 tensor 记录通信流使用关系。
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);  // 通信流分配时关联计算流。
            }
            for (auto& to : {topk_weights, combined_topk_weights, bias_0, bias_1}) {
                to.has_value() ? to->record_stream(comm_stream) : void();  // 可选 tensor 存在时记录通信流。
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();  // 可选 tensor 也记录计算流。
            }
        } else {
            stream_wait(compute_stream, comm_stream);  // 同步模式下计算流等待通信流完成。
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);  // 恢复调用方当前流。

        // Return values
        return {combined_x, combined_topk_weights, event};  // 返回 combine 输出、可选权重和事件。
    }

    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
        EP_HOST_ASSERT(low_latency_mode);  // 只有 low-latency 模式才有对应 RDMA buffer layout。

        auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);  // 根据容量参数解析 RDMA 双缓冲布局。
        auto clean_meta_0 = layout.buffers[0].clean_meta();  // 第 0 个 ping-pong buffer 需要清理的 metadata 区。
        auto clean_meta_1 = layout.buffers[1].clean_meta();  // 第 1 个 ping-pong buffer 需要清理的 metadata 区。

        auto check_boundary = [=, this](void* ptr, size_t num_bytes) {
            auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);  // 计算清理区域相对 RDMA buffer 的偏移。
            EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);  // 防止 layout 计算越过 symmetric buffer。
        };  // 边界检查 lambda 只在 host 侧执行。
        check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));  // 检查 buffer0 metadata 清理范围。
        check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));  // 检查 buffer1 metadata 清理范围。

        internode_ll::clean_low_latency_buffer(clean_meta_0.first,  // buffer0 metadata 指针。
                                               clean_meta_0.second,  // buffer0 metadata int 数。
                                               clean_meta_1.first,  // buffer1 metadata 指针。
                                               clean_meta_1.second,  // buffer1 metadata int 数。
                                               rank,  // 当前 rank。
                                               num_ranks,  // 全局 rank 数。
                                               mask_buffer_ptr,  // shrink 模式 rank mask，可为空取决于 kernel 逻辑。
                                               sync_buffer_ptr,  // shrink 模式同步 buffer。
                                               at::cuda::getCurrentCUDAStream());  // 使用当前 PyTorch 流清理。
    }

    std::tuple<torch::Tensor,                         // packed_recv_x，固定容量专家接收 buffer。
               std::optional<torch::Tensor>,          // packed_recv_x_scales，FP8/UE8M0 scale buffer。
               torch::Tensor,                         // packed_recv_count，[num_local_experts]。
               torch::Tensor,                         // packed_recv_src_info，combine 反向寻址 metadata。
               torch::Tensor,                         // packed_recv_layout_range，每个 expert 的有效 slot 范围。
               std::optional<EventHandle>,            // async 模式事件。
               std::optional<std::function<void()>>>  // return_recv_hook=True 时的接收等待 hook。
    low_latency_dispatch(const torch::Tensor& x,  // BF16 输入 token 特征，[num_tokens,hidden]。
                         const torch::Tensor& topk_idx,  // top-k expert id，[num_tokens,num_topk]。
                         const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,  // 可选诊断累计计数，[num_local_experts]。
                         const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,  // 可选等待成本统计，[num_ranks]。
                         int num_max_dispatch_tokens_per_rank,  // 每 rank 最大 dispatch token 容量。
                         int num_experts,  // 全局 expert 数。
                         bool use_fp8,  // 是否在 dispatch 中把 BF16 转成 FP8。
                         bool round_scale,  // FP8 scale 是否按 kernel 约定取整。
                         bool use_ue8m0,  // 是否使用 UE8M0 scale 存储。
                         bool async,  // 是否异步返回 EventHandle。
                         bool return_recv_hook) {  // 是否只先发 send phase，并返回后续 recv phase hook。
        EP_HOST_ASSERT(low_latency_mode);  // low-latency API 只能在 low_latency_mode buffer 上使用。

        // Tensor checks
        // By default using `ptp128c` FP8 cast
        EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);  // low-latency dispatch 输入固定 BF16 二维连续矩阵。
        EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);  // hidden 需满足 int4 搬运和 FP8 scale block 对齐。
        EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());  // topk_idx 必须二维连续。
        EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);  // token 数匹配且不超过容量。
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);  // topk_idx dtype 必须匹配编译期 topk_idx_t。
        EP_HOST_ASSERT(num_experts % num_ranks == 0);  // expert 必须能按 rank 均分。

        // Diagnosis tensors
        if (cumulative_local_expert_recv_stats.has_value()) {
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);  // 诊断累计计数按 int32 累加。
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());  // stats 是一维连续向量。
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);  // 长度等于本地 expert 数。
        }
        if (dispatch_wait_recv_cost_stats.has_value()) {
            EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);  // 等待成本用 int64 cycle/time 计数。
            EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());  // stats 是一维连续向量。
            EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);  // 每个 rank 一项等待统计。
        }

        auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));  // 输入 token 数与 hidden 维度。
        auto num_topk = static_cast<int>(topk_idx.size(1));  // top-k 路数。
        auto num_local_experts = num_experts / num_ranks;  // 本 rank 拥有的 expert 数。

        // Buffer control
        LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);  // 根据容量解析 RDMA 双缓冲 layout。
        EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);  // 确认 layout 不超过实际 RDMA buffer。
        auto buffer = layout.buffers[low_latency_buffer_idx];  // 当前轮使用的 ping-pong buffer。
        auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];  // 翻转索引并取得下一轮需要预清理的 buffer。

        // Wait previous tasks to be finished
        // NOTES: the hook mode will always use the default stream
        auto compute_stream = at::cuda::getCurrentCUDAStream();  // 当前 PyTorch 计算流。
        auto launch_stream = return_recv_hook ? compute_stream : comm_stream;  // hook 模式在当前流发 send，普通模式走通信流。
        EP_HOST_ASSERT(not(async and return_recv_hook));  // hook 模式由调用方手动触发 recv，不与 EventHandle 异步模式混用。
        if (not return_recv_hook)
            stream_wait(launch_stream, compute_stream);  // 普通模式通信流先等待计算流输入准备好。

        // Allocate packed tensors
        auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                          x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));  // 输出按 [local_expert,rank*capacity,hidden] 打包。
        auto packed_recv_src_info =
            torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));  // 每个接收槽对应源 token 信息。
        auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));  // 每 expert 每 rank 的 token 范围。
        auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));  // 每个本地 expert 实际接收 token 数。

        // Allocate column-majored scales
        auto packed_recv_x_scales = std::optional<torch::Tensor>();  // FP8 模式下的 packed scales。
        void* packed_recv_x_scales_ptr = nullptr;  // scales 原始指针，可为空。
        EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");  // TMA 要求 token 维 4 对齐。

        if (use_fp8) {
            // TODO: support unaligned cases
            EP_HOST_ASSERT(hidden % 512 == 0);  // 当前 FP8 scale layout 要求 hidden 512 对齐。
            if (not use_ue8m0) {
                packed_recv_x_scales = torch::empty({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank},
                                                    torch::dtype(torch::kFloat32).device(torch::kCUDA));  // e4m3 scale 原始布局 [expert,scale_block,token_slot]。
            } else {
                EP_HOST_ASSERT(round_scale);  // UE8M0 scale 需要 round_scale 打开。
                packed_recv_x_scales = torch::empty({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank},
                                                    torch::dtype(torch::kInt).device(torch::kCUDA));  // UE8M0 scale 原始布局 [expert,scale_block,token_slot]。
            }
            packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);  // 转成 [expert,token_slot,scale_block]，匹配后续 GEMM/TMA 读取。
            packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();  // 获取 scales 输出指针。
        }

        // Kernel launch
        auto next_clean_meta = next_buffer.clean_meta();  // 当前 kernel 顺手清理下一轮 ping-pong buffer metadata。
        auto launcher = [=, this](int phases) {
            internode_ll::dispatch(
                packed_recv_x.data_ptr(),  // 输出 packed token。
                packed_recv_x_scales_ptr,  // 输出 packed scales，可为空。
                packed_recv_src_info.data_ptr<int>(),  // 输出源 token 信息。
                packed_recv_layout_range.data_ptr<int64_t>(),  // 输出每 expert/rank 的布局范围。
                packed_recv_count.data_ptr<int>(),  // 输出每 expert 接收计数。
                mask_buffer_ptr,  // shrink 模式 rank mask。
                cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,  // 可选累计诊断计数。
                dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,  // 可选等待成本统计。
                buffer.dispatch_rdma_recv_data_buffer,  // 当前 buffer 的 RDMA 接收数据区。
                buffer.dispatch_rdma_recv_count_buffer,  // 当前 buffer 的 RDMA 接收计数区。
                buffer.dispatch_rdma_send_buffer,  // 当前 buffer 的 RDMA 发送区。
                x.data_ptr(),  // 输入 BF16 token。
                topk_idx.data_ptr<topk_idx_t>(),  // 输入 topk expert id。
                next_clean_meta.first,  // 下一轮 buffer 需要清理的 metadata 指针。
                next_clean_meta.second,  // 下一轮 buffer 需要清理的 int 数。
                num_tokens,  // 输入 token 数。
                hidden,  // hidden 维度。
                num_max_dispatch_tokens_per_rank,  // 每 rank token 容量。
                num_topk,  // top-k 路数。
                num_experts,  // 全局 expert 数。
                rank,  // 当前 rank。
                num_ranks,  // 全局 rank 数。
                use_fp8,  // 是否输出 FP8 packed token。
                round_scale,  // 是否对 scale 做 round。
                use_ue8m0,  // 是否使用 UE8M0 scale。
                workspace,  // 临时 workspace。
                num_device_sms,  // 设备 SM 数，用于 kernel 内调度。
                launch_stream,  // 发射流。
                phases);  // SEND/RECV phase bitmask。
        };  // launcher 捕获所有 tensor，hook 模式下可延迟执行 recv phase。
        launcher(return_recv_hook ? LEGACY_LOW_LATENCY_SEND_PHASE : (LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE));  // 普通模式一次执行 send+recv，hook 模式只执行 send。

        // Wait streams
        std::optional<EventHandle> event;  // async 模式返回事件。
        if (async) {
            // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
            // so in Python API, we must wrap all tensors into the event handle.
            event = EventHandle(launch_stream);  // 记录 launch_stream 当前位置。
        } else if (not return_recv_hook) {
            stream_wait(compute_stream, launch_stream);  // 非 hook 同步模式让计算流等待 kernel 完成。
        }

        // Receiver callback
        std::optional<std::function<void()>> recv_hook = std::nullopt;  // hook 模式返回给 Python 的 recv phase 闭包。
        if (return_recv_hook)
            recv_hook = [=]() { launcher(LEGACY_LOW_LATENCY_RECV_PHASE); };  // 调用闭包时只执行 recv phase。

        // Return values
        return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};  // 返回 packed 输出、metadata、事件和可选 hook。
    }

    std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> low_latency_combine(
        const torch::Tensor& x,  // packed expert 输出，[num_local_experts,num_ranks*capacity,hidden]。
        const torch::Tensor& topk_idx,  // 原始 token topk expert id，[num_combined_tokens,num_topk]。
        const torch::Tensor& topk_weights,  // 原始 token topk 权重，[num_combined_tokens,num_topk]。
        const torch::Tensor& src_info,  // dispatch 返回的源信息，[num_local_experts,num_ranks*capacity]。
        const torch::Tensor& layout_range,  // dispatch 返回的布局范围，[num_local_experts,num_ranks]。
        const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,  // 可选 combine 等待成本统计，[num_ranks]。
        int num_max_dispatch_tokens_per_rank,  // 每 rank dispatch token 容量。
        int num_experts,  // 全局 expert 数。
        bool use_logfmt,  // 是否按 log 格式处理权重/归约。
        bool zero_copy,  // 是否使用 zero-copy combine 路径。
        bool async,  // 是否异步返回事件。
        bool return_recv_hook,  // 是否返回 recv phase hook。
        const std::optional<torch::Tensor>& out = std::nullopt) {  // 可选外部输出 buffer。
        EP_HOST_ASSERT(low_latency_mode);  // 只能在 low-latency buffer 上调用。

        // Tensor checks
        EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);  // packed expert 输出必须是 BF16 三维连续。
        EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);  // 第一维是本地 expert 数。
        EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);  // 第二维是 rank*容量的 packed token 槽。
        EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);  // hidden 满足 int4 搬运和 128 对齐。
        EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());  // topk_idx 二维连续。
        EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));  // topk_idx/weights 形状一致。
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);  // topk_idx dtype 匹配编译期类型。
        EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());  // topk_weights 二维连续。
        EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);  // combine token 数不超过每 rank 容量。
        EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);  // topk 权重固定 float32。
        EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());  // src_info 二维连续。
        EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));  // src_info dtype/本地 expert 维匹配。
        EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());  // layout_range 二维连续。
        EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);  // layout_range 用 int64 表示 range。
        EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);  // layout_range 形状 [local_expert,num_ranks]。

        if (combine_wait_recv_cost_stats.has_value()) {
            EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);  // 等待成本统计用 int64。
            EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());  // stats 一维连续。
            EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks);  // 每个 rank 一项统计。
        }

        auto hidden = static_cast<int>(x.size(2));  // hidden 维度。
        auto num_topk = static_cast<int>(topk_weights.size(1));  // top-k 路数。
        auto num_combined_tokens = static_cast<int>(topk_weights.size(0));  // combine 输出 token 数。

        // Buffer control
        LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);  // 解析 low-latency RDMA layout。
        EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);  // layout 总大小不能超过 RDMA buffer。
        auto buffer = layout.buffers[low_latency_buffer_idx];  // 当前 combine 使用的 buffer。
        auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];  // 翻转双缓冲索引并选中下一轮 buffer。

        // Wait previous tasks to be finished
        // NOTES: the hook mode will always use the default stream
        auto compute_stream = at::cuda::getCurrentCUDAStream();  // 当前 PyTorch 计算流。
        auto launch_stream = return_recv_hook ? compute_stream : comm_stream;  // hook 模式使用计算流，普通模式使用通信流。
        EP_HOST_ASSERT(not(async and return_recv_hook));  // hook 模式和 async EventHandle 模式互斥。
        if (not return_recv_hook)
            stream_wait(launch_stream, compute_stream);  // 通信流等待计算流上的输入准备完成。

        // Allocate output tensor
        torch::Tensor combined_x;  // combine 输出 tensor。
        if (out.has_value()) {
            EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());  // 外部 out 必须二维连续。
            EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);  // out 形状 [num_combined_tokens,hidden]。
            EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());  // out dtype 与 x 一致。
            combined_x = out.value();  // 复用调用方传入的输出 buffer。
        } else {
            combined_x = torch::empty({num_combined_tokens, hidden}, x.options());  // 自行分配输出 buffer。
        }

        // Kernel launch
        auto next_clean_meta = next_buffer.clean_meta();  // 当前 kernel 顺手清理下一轮 buffer metadata。
        auto launcher = [=, this](int phases) {
            internode_ll::combine(combined_x.data_ptr(),  // 输出 combine token。
                                  buffer.combine_rdma_recv_data_buffer,  // RDMA 接收数据区。
                                  buffer.combine_rdma_recv_flag_buffer,  // RDMA 接收 flag 区。
                                  buffer.combine_rdma_send_buffer,  // RDMA 发送区。
                                  x.data_ptr(),  // 输入 packed expert 输出。
                                  topk_idx.data_ptr<topk_idx_t>(),  // 输入 topk expert id。
                                  topk_weights.data_ptr<float>(),  // 输入 topk 权重。
                                  src_info.data_ptr<int>(),  // 输入源 token 信息。
                                  layout_range.data_ptr<int64_t>(),  // 输入 layout range。
                                  mask_buffer_ptr,  // shrink 模式 rank mask。
                                  combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,  // 可选等待统计。
                                  next_clean_meta.first,  // 下一轮 buffer 清理指针。
                                  next_clean_meta.second,  // 下一轮 buffer 清理 int 数。
                                  num_combined_tokens,  // 输出 token 数。
                                  hidden,  // hidden 维度。
                                  num_max_dispatch_tokens_per_rank,  // 每 rank 容量。
                                  num_topk,  // top-k 路数。
                                  num_experts,  // 全局 expert 数。
                                  rank,  // 当前 rank。
                                  num_ranks,  // 全局 rank 数。
                                  use_logfmt,  // 是否使用 log 格式权重处理。
                                  workspace,  // 临时 workspace。
                                  num_device_sms,  // 设备 SM 数。
                                  launch_stream,  // 发射流。
                                  phases,  // SEND/RECV phase bitmask。
                                  zero_copy);  // 是否启用 zero-copy 路径。
        };  // launcher 用于普通模式一次执行或 hook 模式分阶段执行。
        launcher(return_recv_hook ? LEGACY_LOW_LATENCY_SEND_PHASE : (LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE));  // 根据 hook 模式选择先执行哪些 phase。

        // Wait streams
        std::optional<EventHandle> event;  // async 模式返回事件。
        if (async) {
            // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
            // so in Python API, we must wrap all tensors into the event handle.
            event = EventHandle(launch_stream);  // 记录 launch_stream 当前位置。
        } else if (not return_recv_hook) {
            stream_wait(compute_stream, launch_stream);  // 同步模式让计算流等待 combine 完成。
        }

        // Receiver callback
        std::optional<std::function<void()>> recv_hook = std::nullopt;  // hook 模式返回的 recv phase 闭包。
        if (return_recv_hook)
            recv_hook = [=]() { launcher(LEGACY_LOW_LATENCY_RECV_PHASE); };  // 调用时执行 recv phase。

        // Return values
        return {combined_x, event, recv_hook};  // 返回输出、事件和可选 hook。
    }

    torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
        LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);  // 按当前容量参数解析 low-latency layout。

        auto buffer = layout.buffers[low_latency_buffer_idx];  // 取下一次 combine 将使用的 ping-pong buffer。
        auto dtype = torch::kBFloat16;  // combine send buffer 暴露为 BF16 tensor view。
        auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));  // 每条 combine message 对应的 BF16 元素数。

        EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);  // message 字节数必须能整除 BF16 元素大小。
        return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,  // 返回 RDMA send buffer 的非 owning tensor view。
                                {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},  // 逻辑形状 [local_expert,rank*capacity,hidden]。
                                {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},  // stride 跳过每条 message 中的 flag/header 区。
                                torch::TensorOptions().dtype(dtype).device(torch::kCUDA));  // CUDA BF16 tensor 选项。
    }

    void low_latency_update_mask_buffer(int rank_to_mask, bool mask) const {
        EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");  // 只有 enable_shrink 时才分配 mask buffer。
        EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < num_ranks);  // 待屏蔽 rank 必须在 world 范围内。
        internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, at::cuda::getCurrentCUDAStream());  // 在当前流更新 mask 状态。
    }

    void low_latency_query_mask_buffer(const torch::Tensor& mask_status) const {
        EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");  // 查询也要求 shrink buffer 存在。
        EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);  // 输出 tensor 长度等于 rank 数且 dtype int32。

        internode_ll::query_mask_buffer(
            mask_buffer_ptr, num_ranks, static_cast<int*>(mask_status.data_ptr()), at::cuda::getCurrentCUDAStream());  // 把 mask buffer 拷/写到用户 tensor。
    }

    void low_latency_clean_mask_buffer() const {
        EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");  // 清理要求 shrink buffer 存在。
        internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, at::cuda::getCurrentCUDAStream());  // 在当前流把所有 rank mask 清零。
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<Config>(m, "Config")  // 向 Python 暴露 legacy.Config。
        .def(pybind11::init<int, int, int, int, int>(),  // 绑定 Config 构造函数。
             py::arg("num_sms") = 20,  // 默认使用 20 个 SM/block。
             py::arg("num_max_nvl_chunked_send_tokens") = 6,  // 默认 NVL send chunk token 上限。
             py::arg("num_max_nvl_chunked_recv_tokens") = 256,  // 默认 NVL recv chunk token 上限。
             py::arg("num_max_rdma_chunked_send_tokens") = 6,  // 默认 RDMA send chunk token 上限。
             py::arg("num_max_rdma_chunked_recv_tokens") = 256)  // 默认 RDMA recv chunk token 上限。
        .def("get_nvl_buffer_size_hint", &Config::get_nvl_buffer_size_hint)  // 暴露 NVL buffer size hint。
        .def("get_rdma_buffer_size_hint", &Config::get_rdma_buffer_size_hint);  // 暴露 RDMA buffer size hint。
    m.def("get_low_latency_rdma_size_hint", &get_low_latency_rdma_size_hint);  // 暴露 low-latency RDMA buffer size 计算。

    pybind11::class_<EventHandle>(m, "EventHandle")  // 向 Python 暴露事件句柄。
        .def(pybind11::init<>())  // 绑定默认构造。
        .def("current_stream_wait", &EventHandle::current_stream_wait);  // 绑定当前流等待事件的方法。

    pybind11::class_<Buffer>(m, "Buffer")  // 向 Python 暴露 legacy.Buffer 主类。
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool>())  // 绑定 Buffer 构造函数。
        .def("is_available", &Buffer::is_available)  // 查询 sync 后是否可用。
        .def("get_num_rdma_ranks", &Buffer::get_num_rdma_ranks)  // 查询 RDMA rank 数。
        .def("get_rdma_rank", &Buffer::get_rdma_rank)  // 查询当前 RDMA rank。
        .def("get_root_rdma_rank", &Buffer::get_root_rdma_rank)  // 查询 root RDMA rank。
        .def("get_local_device_id", &Buffer::get_local_device_id)  // 查询当前 CUDA device id。
        .def("get_local_ipc_handle", &Buffer::get_local_ipc_handle)  // 获取本地 IPC/Fabric handle。
        .def("get_local_nvshmem_unique_id", &Buffer::get_local_nvshmem_unique_id)  // 获取 NVSHMEM unique id。
        .def("get_local_buffer_tensor", &Buffer::get_local_buffer_tensor)  // 获取本地 buffer tensor view。
        .def("get_comm_stream", &Buffer::get_comm_stream)  // 获取通信流对象。
        .def("sync", &Buffer::sync)  // 绑定跨 rank handle/NVSHMEM 同步。
        .def("destroy", &Buffer::destroy)  // 绑定显式释放。
        .def("get_dispatch_layout", &Buffer::get_dispatch_layout)  // 绑定布局统计 API。
        .def("intranode_dispatch", &Buffer::intranode_dispatch)  // 绑定节点内 dispatch。
        .def("intranode_combine", &Buffer::intranode_combine)  // 绑定节点内 combine。
        .def("internode_dispatch", &Buffer::internode_dispatch)  // 绑定跨节点 dispatch。
        .def("internode_combine", &Buffer::internode_combine)  // 绑定跨节点 combine。
        .def("clean_low_latency_buffer", &Buffer::clean_low_latency_buffer)  // 绑定 low-latency buffer 清理。
        .def("low_latency_dispatch", &Buffer::low_latency_dispatch)  // 绑定 low-latency dispatch。
        .def("low_latency_combine", &Buffer::low_latency_combine)  // 绑定 low-latency combine。
        .def("low_latency_update_mask_buffer", &Buffer::low_latency_update_mask_buffer)  // 绑定 shrink mask 更新。
        .def("low_latency_query_mask_buffer", &Buffer::low_latency_query_mask_buffer)  // 绑定 shrink mask 查询。
        .def("low_latency_clean_mask_buffer", &Buffer::low_latency_clean_mask_buffer)  // 绑定 shrink mask 清理。
        .def("get_next_low_latency_combine_buffer", &Buffer::get_next_low_latency_combine_buffer);  // 绑定 zero-copy combine buffer view。
}

}  // namespace deep_ep::legacy
