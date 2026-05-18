#pragma once  // Elastic barrier device 实现会被 JIT 翻译单元 include，防止重复定义。

#include <deep_ep/common/comm.cuh>      // gpu_barrier 和通信同步辅助函数。
#include <deep_ep/common/compiled.cuh>  // 全局编译配置和基础类型。
#include <deep_ep/common/layout.cuh>    // WorkspaceLayout 描述 workspace 中控制区布局。
#include <deep_ep/common/ptx.cuh>       // PTX load/store/barrier 辅助。


namespace deep_ep::elastic {
// Elastic/V2 device kernel 命名空间。

template <bool kIsScaleupNVLink,
          int kNumSMs, int kNumThreads,
          // 中文：scaleout/scaleup rank 数在模板期固定，便于编译器展开 barrier 分支。
          int kNumScaleoutRanks, int kNumScaleupRanks,
          // 中文：timeout cycles 用于设备端自旋等待诊断，避免无限 hang。
int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(kNumThreads, 1)
// 中文：barrier_impl 的运行期参数只包含 communicator/window、workspace 和本 rank 二维坐标。
barrier_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* workspace,
             const int scaleout_rank_idx, const int scaleup_rank_idx) {
    // barrier_impl 是 ElasticBuffer.expert_parallel_barrier 等同步接口最终启动的 device kernel。
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    // sm_idx 对应参与 barrier 的 block/SM，thread_idx 是 block 内线程编号。

    // barrier 只使用 WorkspaceLayout 的控制区：signal/counter/PP 计数等固定空间，
    // 不访问按 expert 展开的 prefix-sum 数组。因此 num_experts 传 0 即可，避免为
    // barrier 额外携带 expert 数量。scaleout_rank_idx/scaleup_rank_idx 合起来定位
    // 当前全局 rank，例如 scaleout=1、scaleup=3 表示第二个节点内第 4 张卡。
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, 0);
    // workspace_layout 解析 workspace 控制区，num_experts=0 表示不使用 expert prefix 数组。
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0);
    // barrier 只需要一个 Gin context，这里使用 qp/context 0。
    // gpu_barrier 内部按 kIsScaleupNVLink 选择 NVLink symmetric pointer 或 GIN signal；
    // kFlushAllAllocatedQPs 表示退出前刷新所有已分配 QP，保证后续 dispatch/combine
    // 看到的 workspace 状态已经跨 rank 可见。
    comm::gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs, kNumTimeoutCycles, comm::kKernelBarrierTag, false, false, false>(
            gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    // 参数中的 false/false/false 分别关闭额外模式，保持这是一个通用 kernel barrier。
}

} // namespace deep_ep::elastic
// Elastic device kernel 命名空间结束。
