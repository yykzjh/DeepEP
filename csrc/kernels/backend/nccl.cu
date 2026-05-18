#include <cstring>   // 使用 std::memcpy 在 NCCL unique id 与字节数组之间转换。
#include <memory>    // 保持与 api.cuh/上下文对象实现一致；智能指针类型在相关声明中可能出现。
#include <vector>    // unique id 字节数组、NVLink peer window 指针列表使用 std::vector。
#include <string>    // pybind11::bytearray 转 std::string 后再 memcpy 到 ncclUniqueId。
#include <optional>  // 与 backend API 头中的可选 stream 声明保持一致。
#include <pybind11/pytypes.h>  // 使用 pybind11::bytearray 作为 Python 可广播的 unique id 容器。
#include <pybind11/stl.h>      // 允许 pybind11 自动转换 STL 容器。
#include <sstream>             // 预留字符串拼接能力，和同一后端模块的诊断风格保持一致。

#include <nccl.h>              // host 侧 NCCL communicator、window、unique id API。
#include <nccl_device/core.h>  // device communicator/Gin/LSA 拓扑查询所需的 NCCL device API。

#include <deep_ep/common/compiled.cuh>   // 全局编译配置，例如是否启用 SM90/Gin 相关路径。
#include <deep_ep/common/exception.cuh>  // NCCL_CHECK/EP_HOST_ASSERT 等错误检查宏。

#include "api.cuh"                  // 声明本文件实现的 deep_ep::nccl backend API。
#include "../../utils/system.hpp"   // get_env 读取 EP_BUFFER_DEBUG、EP_DISABLE_GIN 等环境变量。


namespace deep_ep::nccl {  // NCCL Gin / symmetric memory backend 命名空间。
// 该 backend 是 V2 Elastic 的核心运行时支撑：host 侧创建 communicator/window，device 侧 kernel 通过 ncclDevComm_t 发 Gin 请求。

pybind11::bytearray get_local_unique_id() {  // 生成可在 Python 进程间广播的 NCCL unique id。
    // NCCL unique id 必须从 root rank 广播到其他 rank。这里转成 bytearray，
    // Python 侧可以通过 torch.distributed.broadcast_object_list 之类的方式传递。
    ncclUniqueId unique_id;
    // ncclUniqueId 是固定大小 POD 结构，不能直接跨 Python 进程序列化。
    NCCL_CHECK(ncclGetUniqueId(&unique_id));
    // 只有 root rank 调用 ncclGetUniqueId，其他 rank 应接收 root 广播的字节。
    std::vector<char> result(sizeof(ncclUniqueId));
    // 用 char buffer 保留完整二进制内容，不能使用 C 字符串语义。
    std::memcpy(result.data(), &unique_id, sizeof(ncclUniqueId));
    // memcpy 后 result 的长度等于 NCCL unique id 的真实字节数。
    return {result.data(), result.size()};
    // 返回 bytearray，Python 侧可以安全广播并传回 create_nccl_comm。
}

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,  // root rank 广播来的 unique id 字节。
                         const int& num_ranks, const int& rank_idx) {  // world size 与当前 rank。
    // Copy unique ID
    ncclUniqueId root_unique_id;
    // root_unique_id 是传给 ncclCommInitRank 的二进制结构。
    const auto root_unique_id_str = root_unique_id_bytes.cast<std::string>();
    // pybind11::bytearray 转成 std::string 只是作为字节容器使用，不按文本解析。
    std::memcpy(&root_unique_id, root_unique_id_str.c_str(), sizeof(ncclUniqueId));
    // 从 Python 字节恢复原始 ncclUniqueId。

    // 返回值用 int64_t 装裸 ncclComm_t 指针，Python 层只保存句柄，不直接解引用。
    ncclComm_t comm;
    // host communicator 指针由 NCCL 创建，后续封装成 int64_t 句柄。
    NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, root_unique_id, rank_idx));
    // 所有 rank 使用同一个 unique id、各自 rank_idx 共同创建 communicator。
    if (get_env<int>("EP_BUFFER_DEBUG"))
        // 调试模式打印 communicator 创建信息，便于确认 rank/world 配置。
        printf("New NCCL host communicator created (%d/%d)\n", rank_idx, num_ranks);
    return reinterpret_cast<int64_t>(comm);
    // pybind11 传裸指针不方便，这里转 int64_t 由 Python Buffer 保存。
}

void destroy_nccl_comm(const int64_t& nccl_comm) {  // 销毁/中止 Python 保存的 NCCL host communicator。
    // Python Buffer finalize 时调用；这里使用 Abort 而非 Destroy，容忍异常路径快速收尾。
    NCCL_CHECK(ncclCommAbort(reinterpret_cast<ncclComm_t>(nccl_comm)));
    // 把 int64_t 句柄还原为 ncclComm_t。
    if (get_env<int>("EP_BUFFER_DEBUG"))
        // 调试模式打印销毁事件。
        printf("NCCL host communicator aborted\n");
}

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm) {  // 查询物理 RDMA/NVLink 二维域大小。
    // physical domain 来自 NCCL team：world rank 总数 / LSA(NVLink) rank 数 =
    // RDMA/rail 域数量。返回 (num_rdma_ranks, num_nvl_ranks)。
    const auto comm = reinterpret_cast<ncclComm_t>(nccl_comm);
    // Python 传入的是 int64_t 句柄，这里还原成 NCCL communicator。
    const int num_ranks = ncclTeamWorld(comm).nRanks, num_nvl_ranks = ncclTeamLsa(comm).nRanks;
    // world team 是全局 rank 数；LSA team 通常对应同节点 NVLink/高速域 rank 数。
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0);
    // 物理拓扑假设 world 可被 LSA 均匀切分。
    return {num_ranks / num_nvl_ranks, num_nvl_ranks};
    // 第一维是 RDMA 域数量，第二维是每个 RDMA 域内的 NVLink rank 数。
}

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode) {  // 根据 hybrid 开关返回逻辑拓扑。
    // allow_hybrid_mode=false 时把所有 rank 视为一个 scale-up 域，禁用 direct RDMA
    // kernel；true 时 scaleout=RDMA 域数量，scaleup=单节点 NVLink rank 数。
    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm);
    // 先读取物理二维拓扑，再根据 allow_hybrid_mode 决定逻辑二维拓扑。
    return {allow_hybrid_mode ? num_rdma_ranks : 1,
            allow_hybrid_mode ? num_nvl_ranks : num_rdma_ranks * num_nvl_ranks};
    // hybrid 开启时逻辑 scaleout=RDMA 域、scaleup=NVLink 域；关闭时所有 rank 都在 scaleup。
}

NCCLSymmetricMemoryContext::NCCLSymmetricMemoryContext(const int64_t& nccl_comm,  // Python 层持有的 host communicator 句柄。
                                                       const int& num_ranks, const int& rank_idx,  // world size 与当前 rank。
                                                       const size_t& size, const size_t& alignment,  // 对称窗口大小与对齐要求。
                                                       const bool& allow_hybrid_mode,  // 是否启用 RDMA scaleout + NVLink scaleup hybrid 拓扑。
                                                       const int& sl_idx, const int& num_allocated_qps):  // Gin traffic class 与 QP/context 数。
    rank_idx(rank_idx), num_ranks(num_ranks), num_allocated_qps(num_allocated_qps) {
    // 构造函数完成 NCCL device communicator + symmetric memory window 的全部初始化。
    if (get_env("EP_BUFFER_DEBUG", 0)) {
        // 调试模式打印实际加载的 NCCL 动态库版本，排查运行时/编译时版本不一致问题。
        int nccl_version;
        // NCCL 版本编码通常是 major*10000 + minor*100 + patch。
        NCCL_CHECK(ncclGetVersion(&nccl_version));
        // 查询运行时加载库版本。
        printf("DeepEP initialized with NCCL version: %d.%d.%d (loaded library)\n",
               nccl_version / 10000, (nccl_version % 10000) / 100, nccl_version % 100);
        // 打印成可读 major.minor.patch。
    }

    // 复用 Python/Buffer 层创建的 host communicator，在其上创建 device communicator
    // 和 symmetric memory window。comm 的生命周期由 Buffer finalize 统一管理。
    comm = reinterpret_cast<ncclComm_t>(nccl_comm);
    // 这里不拥有 host communicator 的生命周期，只保存句柄供 window/dev_comm 使用。

    // Print number of allocated QPs
    if (get_env<int>("EP_BUFFER_DEBUG"))
        // num_allocated_qps 影响 device Gin 上可并发的 RDMA context 数。
        printf("EP NCCL device communicator has %d allocated QPs\n", num_allocated_qps);

    // 查询当前 NCCL 是否支持 GIN。hybrid 模式要求 railed GIN，非 hybrid 模式要求
    // full GIN；否则 device kernel 无法直接发起 RDMA get/put/signal。
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    // NCCL 要求用初始化宏创建 properties 结构。
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    // 查询当前 communicator 可用的 Gin 能力。
    EP_HOST_ASSERT(
        (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE and
        // 中文：hybrid 模式需要 railed Gin，direct 模式需要普通 Gin；缺失时无法在 device 端发起通信。
        "NCCL GIN is unavailable. This is usually due to a network configuration issue, "
        "such as `allow_hybrid_mode=0` (disable direct RDMA kernels) in multi-plane network.");
    // hybrid 用 rail Gin，非 hybrid 用 full Gin；没有 Gin 时 Elastic kernel 无法设备端通信。

    // device communicator 把 host ncclComm_t 转成 kernel 可用的 ncclDevComm_t。
    // ginSignalCount = num_ranks + 4：前 num_ranks 通常给 PP recv 信号，额外槽位
    // 给 send 侧释放/自定义 barrier 使用。
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    // device communicator 的创建需求，默认全零/默认值。
    if (num_ranks > 1 and get_env("EP_DISABLE_GIN", 0) == 0) {
        // 单 rank 或显式禁用 GIN 时不申请 Gin context，避免不必要资源占用。
        reqs.ginContextCount = num_allocated_qps;
        // 申请的 Gin context/QP 数，由 Buffer 初始化参数传入。
        reqs.ginExclusiveContexts = true;
        // 独占 context 避免与其他 NCCL 操作争用通信资源。
        reqs.ginQueueDepth = 1024;
        // 每个 Gin 队列深度，影响设备端 outstanding 请求容量。
        reqs.ginTrafficClass = sl_idx;
        // service level / traffic class，用于网络 QoS。
        // Customized RDMA barrier needs extra signals
        reqs.ginSignalCount = num_ranks + 2 * 2;
        // 前 num_ranks 个 signal 可给 rank 级同步，额外 4 个给自定义 RDMA barrier/双缓冲。
        reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL: NCCL_GIN_CONNECTION_FULL;
        // hybrid 使用 rail 连接匹配多 rail 网络，非 hybrid 使用 full 连接。
    }
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));
    // 创建 kernel 可直接使用的 ncclDevComm_t。

    // dev_comm 初始化后才能可靠知道 LSA/NVLink 域大小。rank_idx 被拆成
    // rdma_rank_idx 与 nvl_rank_idx，后续 elastic buffer 用它们计算 scaleout/scaleup。
    num_nvl_ranks = dev_comm.lsaSize, nvl_rank_idx = dev_comm.lsaRank;
    // LSA size/rank 来自 device communicator，比 host team 查询更贴近 kernel 视角。
    num_rdma_ranks = num_ranks / num_nvl_ranks, rdma_rank_idx = rank_idx / num_nvl_ranks;
    // 把全局 rank 映射到二维物理坐标 (rdma_rank_idx, nvl_rank_idx)。
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0 and nvl_rank_idx == rank_idx % num_nvl_ranks);
    // 校验 LSA rank 与全局 rank 的低位坐标一致。
    EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);
    // 校验二维坐标能还原全局 rank。

    // Calculate scaleout/up domain size
    if (allow_hybrid_mode) {
        // hybrid 模式把跨节点和节点内通信分开建模。
        num_scaleout_ranks = num_rdma_ranks, num_scaleup_ranks = num_nvl_ranks;
        // scaleout 是 RDMA 域数量，scaleup 是每个域内 NVLink rank 数。
        scaleout_rank_idx = rdma_rank_idx, scaleup_rank_idx = nvl_rank_idx;
        // 当前 rank 的逻辑坐标等于物理二维坐标。
    } else {
        // 非 hybrid 模式把所有 rank 视为单一 scaleup 域。
        num_scaleout_ranks = 1, num_scaleup_ranks = num_ranks;
        // 没有跨节点直连维度，Elastic kernel 会使用另一套通信策略。
        scaleout_rank_idx = 0, scaleup_rank_idx = rank_idx;
        // scaleout 坐标固定为 0，scaleup 坐标等于全局 rank。
    }
    is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;
    // 如果逻辑 scaleup 域等于物理 NVLink 域，hybrid kernel 可使用更快的 NVLink 路径。

    // 创建 NCCL symmetric window：raw_window_ptr 是本 rank 分配到的窗口地址，
    // mapped_window_ptr 是当前 GPU 通过 LSA 访问本 rank window 的映射地址。
    // NOTES: `ncclCommWindowRegister` is collective: it internally calls bootstrapBarrier
    // across all ranks, so no explicit barrier is needed after this call.
    NCCL_CHECK(ncclMemAlloc(&raw_window_ptr, size));
    // 分配 NCCL 管理的 symmetric memory，所有 rank 需以相同 size/alignment 进入。
    NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, size, &window, NCCL_WIN_DEFAULT));
    // 把 raw_window_ptr 注册到 communicator 的 window，供 peer 访问。
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));
    // 获取当前 rank 在 LSA 地址空间中的 mapped pointer，kernel 使用这个地址作为本地窗口基址。

    // Get LSA pointers for all LSA peers
    // TODO: check whether this is correct for network with RDMA
    nvl_window_ptrs.resize(num_nvl_ranks);
    // 保存所有 NVLink peer 的 window pointer，get_sym_ptr 需要它们做 offset 映射。
    for (int i = 0; i < num_nvl_ranks; ++ i)
        // 逐个查询 LSA peer 的 mapped pointer。
        NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]));

    // TODO: push NCCL team to support aligned allocation
    EP_HOST_ASSERT(size % alignment == 0);
    // Buffer 切分要求总大小按 alignment 对齐。
    EP_HOST_ASSERT(reinterpret_cast<uint64_t>(raw_window_ptr) % alignment == 0);
    // 原始 window 指针也必须满足 alignment。
    EP_HOST_ASSERT(reinterpret_cast<uint64_t>(mapped_window_ptr) % alignment == 0);
    // kernel 实际使用的 mapped pointer 同样必须满足 alignment。
}

void* NCCLSymmetricMemoryContext::get_sym_ptr(void* ptr, const int& dst_rank_idx) const {  // 把本地窗口指针映射到 NVLink peer 对称地址。
    // 仅用于 NVLink/LSA peer：保持相同 offset，替换成 dst_rank_idx 对应的窗口基址。
    const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(mapped_window_ptr);
    // 计算 ptr 相对当前 rank window 基址的字节偏移。
    return static_cast<uint8_t*>(nvl_window_ptrs[dst_rank_idx]) + offset;
    // 把相同 offset 加到目标 NVLink peer 的 window 基址上，得到对称地址。
}

void NCCLSymmetricMemoryContext::finalize() const {  // 释放 window、symmetric memory 和 device communicator。
    // 释放顺序和初始化相反：先注销 window，再 free symmetric memory，最后销毁
    // device communicator，避免 kernel 或 NCCL 内部仍持有 window 元数据。
    NCCL_CHECK(ncclCommWindowDeregister(comm, window));
    // 注销 communicator 上的 symmetric window。
    NCCL_CHECK(ncclMemFree(raw_window_ptr));
    // 释放 NCCL symmetric memory 原始分配。

    // Destroy device communicator
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    // 销毁 device communicator，释放 Gin context/signal 等设备端资源。
}

}  // namespace deep_ep::nccl
