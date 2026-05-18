#pragma once  // PP send/recv host wrapper 是 header-only JIT 代码，防止重复定义。

#include <nccl.h>         // 使用 ncclWindow_t。
#include <nccl_device.h>  // 使用 ncclDevComm_t 和 device Gin 能力。

#include <deep_ep/common/compiled.cuh>   // 全局编译配置。
#include <deep_ep/common/exception.cuh>  // EP_CUDA_UNIFIED_CHECK 等错误检查宏。
#include <deep_ep/common/layout.cuh>     // PP device 实现复用 layout/align helper。

#include "../../jit/compiler.hpp"        // JIT 编译和 cache。
#include "../../jit/launch_runtime.hpp"  // JIT launch CRTP 基类。

namespace deep_ep::elastic {
// Elastic/V2 pipeline parallel send/recv wrapper 命名空间。

// Pipeline parallel send/recv 的 host wrapper。它不同于 EP dispatch/combine：
// 这里传输的是一整块连续 tensor 字节流，buffer 被划分为 num_max_inflight_tensors
// 个环形槽位，每个槽位最大 num_max_tensor_bytes。send/recv 通过 workspace 中的
// signal 协议握手，避免覆盖对端尚未消费的数据。
class PPSendRuntime final : public jit::LaunchRuntime<PPSendRuntime> {
    // PPSendRuntime 生成并启动 pp_send_impl。
public:
    struct Args {
        // Args 前半是模板参数，后半是实际 kernel 参数。
        // Templated arguments
        int num_ranks;  // PP 通信所在 world 的 rank 总数。
        // 总 rank 数，发送端 signal 表按 rank 维索引。
        int num_smem_bytes;  // 每个 block 的 shared staging 字节数。
        // kernel 使用的动态 shared memory 字节数，也作为模板参数参与内部分块。
        int64_t num_timeout_cycles;  // 等待远端信号的 GPU cycle 超时。
        // 发送等待空槽/对端确认时的 GPU cycle 超时阈值。

        // Parameters
        ncclDevComm_t nccl_dev_comm;  // device 端 NCCL communicator。
        // NCCL device communicator。
        ncclWindow_t nccl_window;  // 对称内存 window。
        // NCCL symmetric memory window。
        void* x;  // 本次要发送的 tensor 起始地址。
        // 待发送 tensor 起始地址。
        int64_t num_x_bytes;  // 本次 tensor 实际字节数。
        // 待发送 tensor 字节数。
        void* buffer;  // PP payload 环形缓冲。
        // PP 环形 buffer，按 inflight slot 存放 tensor payload。
        void* workspace;  // PP 控制信号区。
        // PP signal/workspace 区，保存 send/recv 轮次和槽位状态。
        int rank_idx, dst_rank_idx;  // 当前 rank 与目标 rank。
        // 当前 rank 和目标 rank。
        int64_t num_max_tensor_bytes;  // 单个 inflight slot 最大容量。
        // 单个 inflight 槽位可容纳的最大 tensor 字节数。
        int num_max_inflight_tensors;  // 环形 slot 数。
        // 环形槽位数量，允许多个 tensor 在途。

        jit::LaunchArgs launch_args;  // send kernel launch 配置。
        // launch 配置。
    };

    static std::string generate_impl(const Args& args) {
        // 生成 pp_send_impl 的模板实例。
        return fmt::format(R"(
#include <deep_ep/impls/pp_send_recv.cuh>

using namespace deep_ep::elastic;

// JIT 生成文件：实例化 pipeline-parallel send kernel。
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pp_send_impl<{}, {}, {}, {}>);
}}
	)", args.launch_args.grid_dim.first,  // SM/block 数。
	    args.num_ranks,  // PP 通信 rank 数。
	    args.num_smem_bytes,  // 每 block shared memory 字节数。
	    args.num_timeout_cycles);  // 超时周期。
        // grid.x、rank 数、shared memory、timeout 都进入模板，减少 device 参数访问。
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        // 按 pp_send_impl 声明顺序传入 kernel 参数。
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,  // device communicator 和 symmetric window。
            args.x, args.num_x_bytes,             // 发送 tensor 基址与实际字节数。
            args.buffer, args.workspace,           // 环形 payload buffer 与信号 workspace。
            args.rank_idx, args.dst_rank_idx,      // 当前 rank 和目标 rank。
            args.num_max_tensor_bytes, args.num_max_inflight_tensors  // 单槽容量和环形槽位数。
        ));
        // send kernel 完成表示 payload 已提交/发送协议完成。
    }
};

static void launch_pp_send(const ncclDevComm_t& nccl_dev_comm,  // NCCL device communicator，device 端用它访问 Gin。
                           const ncclWindow_t& nccl_window,     // symmetric window，buffer/workspace 地址属于该窗口。
                           void* x, const int64_t& num_x_bytes, // 待发送 tensor 和实际字节数。
                           void* buffer, void* workspace,       // payload 环形 buffer 与控制面 workspace。
                           const int& rank_idx, const int& dst_rank_idx, const int& num_ranks, // 当前/目标/总 rank。
                           const int64_t& num_max_tensor_bytes, // 每个 inflight slot 允许的最大 tensor 字节数。
                           const int num_max_inflight_tensors,  // 环形槽位数量，允许多个 PP tensor 同时在途。
                           const int& num_sms,                  // 并行发送使用的 SM 数。
                           const int64_t& num_timeout_cycles,   // 等待槽位或信号的超时周期。
                           const int& num_smem_bytes,           // 每个 block 的 shared memory staging 大小。
                           const at::cuda::CUDAStream& stream) { // CUDA launch stream。
    // 发送端每个 SM 只有一个 warp 参与 TMA/Gin 搬运，shared memory 大小用于暂存
    // 分片 payload；num_sms 控制同一个 tensor 被切成多少个并行搬运通道。
    const PPSendRuntime::Args args = {
        // 构造 send kernel JIT 参数。
        .num_ranks = num_ranks,
        // 总 rank 数。
        .num_smem_bytes = num_smem_bytes,
        // 动态 shared memory 大小。
        .num_timeout_cycles = num_timeout_cycles,
        // 超时阈值。
        // 中文：下面的 communicator/window 与 x 指针是 PP send kernel 的运行期资源。
        .nccl_dev_comm = nccl_dev_comm,
        // device communicator。
        // 中文：device communicator 用于在 kernel 内访问 Gin signal 和 symmetric window。
        .nccl_window = nccl_window,
        // symmetric window。
        // 中文：x 指向用户待发送 tensor，kernel 会按 nbytes 直接搬运。
        .x = x,
        // 中文：dst_rank_idx 只允许 PP ring 的相邻 rank，由 C++ runtime 负责校验。
        // 输入 tensor。
        .num_x_bytes = num_x_bytes,
        // 输入 tensor 字节数。
        .buffer = buffer,
        // 环形 payload buffer。
        .workspace = workspace,
        // signal/workspace。
        .rank_idx = rank_idx,
        // 当前 rank。
        .dst_rank_idx = dst_rank_idx,
        // 目标 rank。
        .num_max_tensor_bytes = num_max_tensor_bytes,
        // 每槽最大字节数。
        .num_max_inflight_tensors = num_max_inflight_tensors,
        // 最大在途 tensor 数。
        .launch_args = jit::LaunchArgs(num_sms, 32, num_smem_bytes, 1, true)
        // 每个 block 一个 warp，cooperative=true 以支持 device 端 grid/protocol 同步。
    };
    const auto code = PPSendRuntime::generate(args);
    // 生成源码。
    const auto runtime = jit::compiler->build("pp_send", code);
    // 编译或复用 pp_send cubin。
    PPSendRuntime::launch(runtime, args, stream);
    // 启动发送 kernel。
}

class PPRecvRuntime final : public jit::LaunchRuntime<PPRecvRuntime> {
    // PPRecvRuntime 生成并启动 pp_recv_impl。
public:
    struct Args {
        // recv kernel 的模板参数和运行时参数。
        // Templated arguments
        int num_ranks;  // PP 通信所在 world 的 rank 总数。
        // 总 rank 数。
        int num_smem_bytes;  // 每个 block 的 shared staging 字节数。
        // 动态 shared memory 大小。
        int64_t num_timeout_cycles;  // 等待发送方数据/信号的超时。
        // 等待发送方信号/数据的超时阈值。

        // Parameters
        ncclDevComm_t nccl_dev_comm;  // device 端 NCCL communicator。
        // NCCL device communicator。
        ncclWindow_t nccl_window;  // 对称内存 window。
        // NCCL symmetric memory window。
        void* x;  // 接收输出 tensor 起始地址。
        // 接收输出 tensor 地址。
        int64_t num_x_bytes;  // 期望接收的 tensor 字节数。
        // 接收 tensor 期望字节数。
        void* buffer;  // PP payload 环形缓冲。
        // 环形 payload buffer。
        void* workspace;  // PP 控制信号区。
        // signal/workspace 区。
        int rank_idx;  // 当前 rank。
        // 当前 rank。
        int src_rank_idx;  // 源 rank。
        // 源 rank。
        int64_t num_max_tensor_bytes;  // 单个 inflight slot 最大容量。
        // 每个 inflight 槽位的最大字节数。
        int num_max_inflight_tensors;  // 环形 slot 数。
        // 环形槽位数量。

        jit::LaunchArgs launch_args;  // recv kernel launch 配置。
        // launch 配置。
    };

    static std::string generate_impl(const Args& args) {
        // 生成 pp_recv_impl 的模板实例。
        return fmt::format(R"(
#include <deep_ep/impls/pp_send_recv.cuh>

using namespace deep_ep::elastic;

// JIT 生成文件：实例化 pipeline-parallel recv kernel。
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pp_recv_impl<{}, {}, {}, {}>);
}}
	)", args.launch_args.grid_dim.first,  // SM/block 数。
	    args.num_ranks,  // PP 通信 rank 数。
	    args.num_smem_bytes,  // shared memory 字节数。
	    args.num_timeout_cycles);  // 超时周期。
        // 模板参数必须与 send 端一致，保证槽位和 signal 语义配对。
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        // 按 pp_recv_impl 声明顺序传参。
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,  // device communicator 和 symmetric window。
            args.x, args.num_x_bytes,             // 接收 tensor 基址与期望字节数。
            args.buffer, args.workspace,           // 环形 payload buffer 与信号 workspace。
            args.rank_idx, args.src_rank_idx,      // 当前 rank 和源 rank。
            args.num_max_tensor_bytes,             // 单个 inflight slot 的容量。
            args.num_max_inflight_tensors          // 环形槽位数量。
        ));
        // recv kernel 完成后，x 中数据对同一 stream 后续算子可见。
    }
};

static void launch_pp_recv(const ncclDevComm_t& nccl_dev_comm,  // NCCL device communicator。
                           const ncclWindow_t& nccl_window,     // symmetric window。
                           void* x,                             // 接收输出 tensor 基址。
                           const int64_t& num_x_bytes,          // 期望接收字节数。
                           void* buffer, void* workspace,       // payload buffer 与控制面 workspace。
                           const int& rank_idx, const int& src_rank_idx, const int& num_ranks, // 当前/源/总 rank。
                           const int64_t& num_max_tensor_bytes, // 每个 inflight slot 最大字节数。
                           const int& num_max_inflight_tensors, // 环形槽位数量。
                           const int& num_sms,                  // 并行接收使用的 SM 数。
                           const int64_t& num_timeout_cycles,   // 等待发送方信号/数据的超时周期。
                           const int& num_smem_bytes,           // shared memory staging 大小。
                           const at::cuda::CUDAStream& stream) { // CUDA launch stream。
    // 接收端和发送端使用相同的模板参数与槽位计算方式，保证双方对第 k 个 inflight
    // tensor 的 buffer offset 有一致理解；否则 signal 成功也会读写错地址。
    const PPRecvRuntime::Args args = {
        // 构造 recv kernel JIT 参数。
        .num_ranks = num_ranks,
        // 总 rank 数。
        .num_smem_bytes = num_smem_bytes,
        // 动态 shared memory 大小。
        .num_timeout_cycles = num_timeout_cycles,
        // 超时阈值。
        // 中文：recv 侧同样使用 NCCL symmetric window 查找对端环形 buffer。
        .nccl_dev_comm = nccl_dev_comm,
        // device communicator。
        // 中文：recv kernel 用 communicator 等待发送端 signal 递增。
        .nccl_window = nccl_window,
        // symmetric window。
        // 中文：x 指向用户预分配输出 tensor，PP recv 会原地写入。
        .x = x,
        // 中文：src_rank_idx 同样对应 PP ring 相邻源 rank。
        // 输出 tensor。
        .num_x_bytes = num_x_bytes,
        // 输出 tensor 字节数。
        .buffer = buffer,
        // 环形 payload buffer。
        .workspace = workspace,
        // signal/workspace。
        .rank_idx = rank_idx,
        // 当前 rank。
        .src_rank_idx = src_rank_idx,
        // 源 rank。
        .num_max_tensor_bytes = num_max_tensor_bytes,
        // 每槽最大字节数。
        .num_max_inflight_tensors = num_max_inflight_tensors,
        // 最大在途 tensor 数。
        .launch_args = jit::LaunchArgs(num_sms, 32, num_smem_bytes, 1, true)
        // 每个 block 一个 warp，cooperative=true 对齐 device 协议同步需求。
    };
    const auto code = PPRecvRuntime::generate(args);
    // 生成源码。
    const auto runtime = jit::compiler->build("pp_recv", code);
    // 编译或复用 pp_recv cubin。
    PPRecvRuntime::launch(runtime, args, stream);
    // 启动接收 kernel。
}

}  // namespace deep_ep::elastic
// Elastic PP send/recv wrapper 命名空间结束。
