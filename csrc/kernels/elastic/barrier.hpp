#pragma once  // BarrierRuntime 是 header-only JIT wrapper，防止重复定义。

#include <nccl.h>         // 使用 ncclWindow_t 等 host/device 共享句柄类型。
#include <nccl_device.h>  // 使用 ncclDevComm_t，elastic kernel 通过它做设备端同步。

#include <deep_ep/common/compiled.cuh>   // 全局编译配置和基础类型。
#include <deep_ep/common/exception.cuh>  // EP_HOST_ASSERT/EP_CUDA_UNIFIED_CHECK 等错误检查宏。

#include "../../jit/compiler.hpp"        // jit::compiler 负责生成/缓存/加载 barrier cubin。
#include "../../jit/launch_runtime.hpp"  // LaunchRuntime 负责生成源码、构造 launch config、发起 kernel。

namespace deep_ep::elastic {  // Elastic/V2 host wrapper 命名空间。
// Elastic/V2 通信路径命名空间。

// BarrierRuntime 是 elastic 模式下的 JIT host wrapper：Python/C++ 侧只负责传入
// NCCL device communicator、对称窗口和 workspace，真正的同步逻辑在
// deep_ep/impls/barrier.cuh 中按 rank 数量实例化。这里把 num_scaleout_ranks、
// num_scaleup_ranks 等值写进模板参数，是为了让 device 端循环边界在编译期固定，
// 例如 scaleout=2、scaleup=4 时 kernel 会生成针对 8 个全局 rank 的轻量 barrier。
class BarrierRuntime final : public jit::LaunchRuntime<BarrierRuntime> {  // barrier JIT runtime 封装。
    // CRTP wrapper：generate_impl 生成 CUDA 源码，launch_impl 绑定实际 kernel 参数。
public:
    struct Args {  // barrier JIT/launch 参数包。
        // Args 同时包含模板参数和运行时参数；generate_impl 只读取模板参数部分。
        // Templated arguments
        bool is_scaleup_nvlink;  // scaleup 是否走 NVLink 快路径。
        // scaleup 域是否是 NVLink，本值进入模板以选择 device 端快速路径。
        int num_scaleout_ranks, num_scaleup_ranks;  // 二维拓扑维度。
        // 逻辑二维拓扑大小；模板化后 kernel 循环边界固定。
        int64_t num_timeout_cycles;  // GPU clock64 超时阈值。
        // GPU clock64() 超时阈值；模板化后 device 端无需从参数区加载。

        // Parameters
        ncclDevComm_t nccl_dev_comm;  // device communicator。
        // NCCL device communicator，barrier_impl 通过它访问 Gin/LSA 设备资源。
        ncclWindow_t nccl_window;  // symmetric memory window。
        // NCCL symmetric memory window，barrier workspace 位于该 window 内。
        void* workspace;  // barrier workspace。
        // barrier 使用的对称 workspace 指针。
        int scaleout_rank_idx, scaleup_rank_idx;  // 当前 rank 坐标。
        // 当前 rank 在逻辑二维拓扑中的坐标。

        jit::LaunchArgs launch_args;  // launch 配置。
        // grid/block/cooperative 等 launch 配置。
    };

    static std::string generate_impl(const Args& args) {  // 生成 barrier_impl 实例源码。
        // 生成一个极小 CUDA 翻译单元：include device 实现并显式实例化当前模板。
        return fmt::format(R"(
#include <deep_ep/impls/barrier.cuh>

using namespace deep_ep::elastic;

// JIT 生成文件：只实例化当前拓扑/timeout 的 barrier kernel。
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&barrier_impl<{}, {}, {}, {}, {}, {}>);
}}
	)",                        args.is_scaleup_nvlink,  // scaleup 是否 NVLink。
	                           args.launch_args.grid_dim.first, args.launch_args.num_threads,  // SM 数和线程数。
	                           args.num_scaleout_ranks, args.num_scaleup_ranks,  // 二维拓扑。
	                           args.num_timeout_cycles);  // 超时周期。
        // 模板参数顺序必须和 barrier_impl 声明一致，否则 JIT 会实例化错误 kernel。
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {  // 启动 barrier kernel。
        // 把运行时参数按 barrier_impl 的形参顺序传给统一 launch_kernel。
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            // 中文：workspace 提供 barrier signal 存储，后两个 rank idx 是二维逻辑拓扑坐标。
            args.workspace, args.scaleout_rank_idx, args.scaleup_rank_idx
        ));
        // EP_CUDA_UNIFIED_CHECK 会根据 handle.hpp 分支映射到 runtime 或 driver 错误检查。
    }
};

static void launch_barrier(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,  // communicator/window。
                           void* workspace,  // barrier workspace。
                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 当前 rank 坐标。
                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,  // 拓扑维度。
                           const int64_t& num_timeout_cycles,  // 超时周期。
                           const bool& is_scaleup_nvlink,  // scaleup 是否 NVLink。
                           const at::cuda::CUDAStream& stream) {  // CUDA stream。
    // 每个 block 固定 512 线程，实际参与同步的 rank 数由模板参数控制；多出的线程
    // 会在 kernel 内自然空转。这样 JIT cache key 更稳定，也避免频繁生成不同线程数版本。
    constexpr auto kNumThreads = 512;  // barrier 固定线程数。
    // 512 线程给 scaleout/scaleup barrier 留足并行度，同时保持一个固定 JIT 实例形状。

    // scaleout_ranks > 1 表示跨节点/跨 scale-out 组的 hybrid barrier，需要一个 SM
    // 处理 scale-up 域、另一个 SM 处理 scale-out 域；纯 NVLink scale-up barrier 只需 1 个 SM。
    const auto num_sms = num_scaleout_ranks > 1 ? 2 : 1;  // hybrid barrier 需要两个 SM。
    // 跨 scaleout 时用 2 个 block 分别处理不同通信维度；纯本地同步用 1 个 block。
    const BarrierRuntime::Args args = {  // 聚合模板和运行时参数。
        // 聚合模板参数、运行时参数和 launch 参数，后续 generate/build/launch 都使用同一份 Args。
        .is_scaleup_nvlink = is_scaleup_nvlink,
        // 当前 scaleup 域是否 NVLink。
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        // 逻辑拓扑尺寸。
        .num_timeout_cycles = num_timeout_cycles,
        // 超时周期数。
        // 中文：后续三个指针/坐标都是 device kernel 运行期参数，不参与模板实例化。
        .nccl_dev_comm = nccl_dev_comm,
        // device communicator。
        // 中文：device communicator 提供 kernel 内 Gin/barrier 通信入口。
        .nccl_window = nccl_window,
        // symmetric window。
        // 中文：workspace 是各 rank 的 barrier signal 区，kernel 内按逻辑坐标索引。
        // 中文：nccl_dev_comm/nccl_window 共同让 device kernel 能访问 NCCL symmetric memory。
        .workspace = workspace,
        // barrier workspace。
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        // 当前 rank 逻辑坐标。
        // 中文：launch_args 的 shared memory 为 0，第四个参数 1 表示 cooperative launch。
        .launch_args = jit::LaunchArgs(num_sms, kNumThreads, 0, 1, true)};
        // barrier 使用 cooperative launch，因为 device 端可能调用 grid sync。
    const auto code = BarrierRuntime::generate(args);  // 生成 JIT 源码。
    // 生成带 include hash 的 CUDA 源码。
    const auto runtime = jit::compiler->build("barrier", code);  // 编译或命中 cache。
    // 根据源码/flags hash 编译或复用 cubin。
    BarrierRuntime::launch(runtime, args, stream);
    // 在指定 PyTorch CUDA stream 上启动 barrier kernel。
}

}  // namespace deep_ep::elastic
