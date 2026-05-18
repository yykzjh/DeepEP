#include <cstring>   // std::memcpy 用于 NVSHMEM unique id 与字节数组互转。
#include <optional>  // barrier 可选绑定 cudaStream_t。
#include <vector>    // unique id 以 std::vector<uint8_t> 返回给 Python。

#include <deep_ep/common/compiled.cuh>   // 全局编译配置和基础类型。
#include <deep_ep/common/exception.cuh>  // CUDA_RUNTIME_CHECK/EP_HOST_ASSERT 等检查宏。

#include <nvshmem.h>  // NVSHMEM host API、team、symmetric heap 分配函数。
// 这个 backend 只服务 legacy internode kernel；Elastic/V2 路径主要改用 NCCL device Gin。

namespace deep_ep::nvshmem {  // legacy/V1 NVSHMEM backend 命名空间。
// legacy/V1 跨节点路径的 NVSHMEM 后端封装命名空间。

// legacy internode 路径仍使用 NVSHMEM。cpu_rdma_team 是按 team_split_stride 切出的
// CPU 侧 RDMA 子 team，用于跨节点但排除同节点 NVLink rank 的同步/通信场景。
nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
// cpu_rdma_team 保存按 stride 切出的 RDMA 子 team；未创建时为 INVALID。
nvshmem_team_config_t cpu_rdma_team_config;
// 创建子 team 所需的配置结构，当前使用默认值。

std::vector<uint8_t> get_unique_id() {  // 生成 NVSHMEM unique id 并以字节数组形式返回。
    // 和 NCCL unique id 类似，NVSHMEM unique id 由 root PE 生成并广播给所有 PE。
    nvshmemx_uniqueid_t unique_id;  // NVSHMEM bootstrap 唯一 ID。
    // NVSHMEM unique id 是固定大小的二进制结构。
    nvshmemx_get_uniqueid(&unique_id);
    // root PE 生成 unique id，调用方负责广播给其他 rank。
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));  // 返回给 Python 的二进制容器。
    // 用 uint8_t vector 保存原始字节，避免文本编码问题。
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    // 复制完整二进制内容。
    return result;
    // 返回给 Python 侧做跨进程广播。
    // 例如 rank0 调用本函数后，把 result 广播给所有 rank，再由 init() 还原成 unique id。
}

void* alloc(const size_t& size, const size_t& alignment) {  // 分配 NVSHMEM symmetric heap 内存。
    // NVSHMEM symmetric allocation：所有 PE 必须按相同顺序调用，返回的指针可被
    // nvshmem put/get 在远端解析。
    return nvshmem_align(alignment, size);
    // 返回对称 heap 指针；所有 PE 必须以相同调用顺序分配才能远端寻址。
    // 这里不额外保存 size，因为 NVSHMEM runtime 自己管理 symmetric heap 元数据。
}

void free(void* ptr) {  // 释放 NVSHMEM symmetric heap 指针。
    // 释放 NVSHMEM symmetric heap 上的内存。
    nvshmem_free(ptr);
    // 调用顺序也应在所有 PE 上一致，避免 symmetric heap 状态不匹配。
    // 与 alloc 一样，这个 free 是 collective 语义的一部分，不能只在部分 rank 上调用。
}

void barrier(const bool& with_cpu_sync,  // 是否让 CPU 在 barrier 前后同步等待。
             const std::optional<cudaStream_t>& stream_opt = std::nullopt) {  // 可选 CUDA stream 版本 barrier。
    // with_cpu_sync=true 时提供“host 也确认同步完成”的强语义；false 时只维护 stream 顺序。
    // Wait all streams to finish on this GPU
    if (with_cpu_sync)
        // CPU 同步模式先等待本 GPU 上已有工作完成，再进入全局 barrier。
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        // 这一步把当前设备上先前所有 stream 的异步工作收束到 barrier 之前。

    // on_stream 版本只把 barrier kernel 放进指定 CUDA stream；如果 with_cpu_sync=false，
    // CPU 不等待它完成，调用方依赖 stream 顺序保证后续 kernel 在 barrier 后执行。
    if (stream_opt.has_value()) {
        // stream 版本把 barrier 排入 CUDA stream，适合和 kernel 顺序组合。
        nvshmemx_barrier_all_on_stream(stream_opt.value());
    } else {
        // 无 stream 时使用 CPU 侧阻塞 barrier。
        nvshmem_barrier_all();
    }

    // Let CPU wait
    if (with_cpu_sync)
        // CPU 同步模式在 barrier 后再次等待，确保 barrier kernel 完成。
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        // 第二次 synchronize 确保 NVSHMEM barrier 自身也已结束，host 才继续返回。
}

int init(const std::vector<uint8_t>& root_unique_id_val,  // root rank 广播来的 NVSHMEM unique id 字节。
         const int& rank,  // 当前 PE/rank。
         const int& num_ranks,  // 全局 PE 数。
         const int& team_split_stride) {  // 按 local rank 切 RDMA 子 team 的 stride。
    // NVSHMEM 初始化需要所有 rank 使用相同 unique id，并传入各自 PE 编号。
    // team_split_stride>0 时按 stride 建子 team，例如 8 卡节点且 stride=8，
    // 不同节点上相同 local rank 会落入同一个 RDMA 子 team。
    nvshmemx_uniqueid_t root_unique_id;  // 还原后的 NVSHMEM unique id。
    // 从 Python 广播来的字节恢复成 NVSHMEM unique id。
    nvshmemx_init_attr_t attr;  // NVSHMEM init 属性结构。
    // NVSHMEM 初始化参数结构。
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    // 拷贝 unique id 原始字节。
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    // 把 rank/world size/unique id 写入 init attr。
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    // 使用 unique id 模式初始化 NVSHMEM runtime。
    // 这里没有 MPI bootstrap；DeepEP 由 Python/torch.distributed 负责 unique id 广播。

    // Create sub-RDMA teams
    if (team_split_stride > 0 and num_ranks > team_split_stride) {
        // 多节点场景才需要把相同 local rank 切到一个 RDMA 子 team。
        EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        // 当前实现只允许创建一次子 team。
        EP_HOST_ASSERT(num_ranks % team_split_stride == 0);
        // world size 必须能被节点内 rank 数整除。
        EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD,
                                                  rank % team_split_stride,
                                                  // 中文：起点是本地 rank，步长是节点内 GPU 数，得到同 local_rank 的跨节点 team。
                                                  team_split_stride,
                                                  num_ranks / team_split_stride,
                                                  // 中文：team 大小等于节点数，成员是所有节点上同一 local_rank 的 GPU。
                                                  &cpu_rdma_team_config,
                                                  0,
                                                  &cpu_rdma_team) == 0);
        // 以 rank%stride 为起点、stride 为步长创建 team，例如 local_rank 相同的跨节点 PE。
        EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
        // 确认子 team 创建成功。
    }

    // Wait all GPUs to get ready
    barrier(true);
    // 初始化完成后全局同步，确保所有 PE 都能安全进入 symmetric allocation/通信。
    return nvshmem_my_pe();
    // 返回 NVSHMEM runtime 看到的 PE 编号，通常应等于传入 rank。
    // 如果返回值与 rank 不一致，说明 bootstrap 或 rank 映射出现问题。
}

void finalize() {  // 销毁子 team 并结束 NVSHMEM runtime。
    // finalize 前做一次全局 barrier，确保没有 rank 仍在使用 symmetric heap 或子 team。
    barrier(true);
    // finalize 前同步所有 PE，防止某个 PE 仍在访问 symmetric heap。
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
        // 如果创建过 RDMA 子 team，先销毁它。
        nvshmem_team_destroy(cpu_rdma_team);
        // 释放 team 资源。
        cpu_rdma_team = NVSHMEM_TEAM_INVALID;
        // 重置全局句柄，避免重复 finalize 时误用。
    }
    nvshmem_finalize();
    // 关闭 NVSHMEM runtime。
    // finalize 后不应再调用 alloc/barrier，否则 NVSHMEM runtime 状态已不可用。
}

}  // namespace deep_ep::nvshmem
// NVSHMEM backend 命名空间结束。
