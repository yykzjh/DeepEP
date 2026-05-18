#pragma once  // Engram host wrapper 是 header-only JIT 代码，防止重复定义。

#include <nccl.h>         // 使用 ncclWindow_t 等 NCCL host/device 句柄类型。
#include <nccl_device.h>  // 使用 ncclDevComm_t 和 ncclGinRequest_t。

#include <deep_ep/common/compiled.cuh>   // 全局编译配置。
#include <deep_ep/common/exception.cuh>  // EP_CUDA_UNIFIED_CHECK 等错误检查宏。
#include <deep_ep/common/layout.cuh>     // Engram device 实现可能复用布局工具。

#include "../../jit/compiler.hpp"        // JIT 编译/缓存入口。
#include "../../jit/launch_runtime.hpp"  // JIT launch CRTP 基类。

namespace deep_ep::elastic {  // Elastic/V2 host 侧 launch wrapper 命名空间。
// Elastic/V2 通信与远端取数命名空间。

// EngramFetchRuntime 封装远端 KV/embedding 取数 kernel。storage 是 NCCL symmetric memory
// 中所有 rank 对等可见的存储区，indices 是长度 num_tokens 的全局 entry id：
//   global_entry = rank_idx * num_entries_per_rank + local_entry
// kernel 会把每个 entry 的 hidden 个元素搬到 fetched，输出维度为 [num_tokens, hidden]。
// num_qps 控制并发 GIN 请求队列数量，last_gin_requests 保存每个 QP 的最后一个异步请求句柄。
class EngramFetchRuntime final : public jit::LaunchRuntime<EngramFetchRuntime> {  // fetch kernel 的 JIT runtime 封装。
    // EngramFetchRuntime 负责生成并启动真正执行远端 fetch 的 kernel。
public:
    struct Args {  // JIT 生成和 kernel launch 共享的参数包。
        // Args 中前半部分是模板参数，后半部分是 kernel 运行时参数。
        // Templated arguments
        int num_entries_per_rank;  // 每 rank entry 数。
        // 每个 rank 的 entry 数；全局 index 用它拆成 rank/local entry。
        int hidden;  // 每个 entry 的 hidden 维度。
        // 每个 entry 的向量长度，输出 fetched 形状为 [num_tokens, hidden]。
        int num_ranks;  // 全局 rank 数。
        // 总 rank 数，用于边界检查和 rank 映射。
        int num_qps;  // Gin QP/block 数。
        // 使用多少个 Gin QP/request lane 并行取数。

        // Parameters
        ncclDevComm_t nccl_dev_comm;  // device 侧 NCCL communicator。
        // NCCL device communicator，kernel 通过它发起 Gin 操作。
        ncclWindow_t nccl_window;  // NCCL symmetric memory window。
        // symmetric memory window，storage 位于所有 rank 对称可见区域。
        void* storage;  // entry 存储基址。
        // 远端/本地 entry 存储基址，逻辑形状为 [num_entries_per_rank, hidden] 每 rank 一份。
        void* fetched;  // fetch 输出基址。
        // 输出 buffer，逻辑形状为 [num_tokens, hidden]。
        int* indices;  // 全局 entry id 数组。
        // 输入全局 entry id 数组，长度 num_tokens。
        ncclGinRequest_t* last_gin_requests;  // 每个 QP 的最后请求句柄。
        // 每个 QP 的最后一个异步 Gin 请求，wait kernel 会读取并等待。
        int num_tokens;  // 本次 fetch entry 数。
        // 要 fetch 的 token/entry 数。

        jit::LaunchArgs launch_args;  // grid/block/smem 配置。
        // grid/block/smem 等 launch 配置。
    };

    static std::string generate_impl(const Args& args) {  // 生成 fetch kernel 的 CUDA 源码。
        // 生成针对当前 hidden/rank/QP 配置的 engram_fetch_impl 模板实例。
        // JIT 源码只实例化一个函数指针，真正 kernel 定义来自 deep_ep/include/deep_ep/impls/engram_fetch.cuh。
        return fmt::format(R"(
#include <deep_ep/impls/engram_fetch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&engram_fetch_impl<{}, {}, {}, {}, {}>);
}}
		)", args.num_qps,  // 模板参数 1：QP 数。
    args.num_entries_per_rank, args.hidden,
    args.num_ranks, args.launch_args.num_threads);
        // args.launch_args.num_threads 也进入模板，使 device 端循环展开/边界固定。
        // 这些模板值变化会生成不同 cubin，因此 compiler cache key 需要包含 include 和源码内容。
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {  // 启动已编译 fetch kernel。
        // 按 device kernel 形参顺序传入运行时参数。
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,  // NCCL device communicator 和 symmetric window。
            args.storage, args.fetched,  // 远端 storage 和本地 fetched 输出。
            args.indices,  // 全局 entry id 列表。
            args.last_gin_requests,  // 每 QP/rank 的最后请求句柄。
            args.num_tokens  // fetch token 数。
        ));
        // fetch kernel 只提交/推进请求，完成性由后续 wait kernel 确认。
        // 因此 Python 侧 engram_fetch 返回 hook，hook 内部会 launch wait kernel 并返回 fetched。
    }
};

static void launch_engram_fetch(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,  // communicator/window。
                                void* storage, void* fetched,  // 输入 storage 和输出 fetched。
                                int* indices,  // 全局 entry id。
                                ncclGinRequest_t* last_gin_requests,  // 请求完成句柄数组。
                                const int& num_entries_per_rank, const int& hidden,  // entry 数和 hidden 维度。
                                const int& num_tokens,  // fetch token 数。
                                const int& num_ranks, const int& num_qps,  // rank 数和 QP 数。
                                const at::cuda::CUDAStream& stream) {  // launch 使用的 CUDA stream。
    // 一个 block 对应一个 QP，1024 线程用于把 hidden 维切片并行搬运；如果 hidden 小于
    // 1024，则高编号线程不会访问数据。把 hidden 写入模板后，编译器可消除很多除法和边界分支。
    constexpr int kNumEngramFetchThreads = 1024;  // fetch kernel 固定线程数。
    // 固定 1024 线程便于覆盖大 hidden；实际有效工作由 hidden/num_tokens 控制。

    // 生成的代码只 include egram_fetch.cuh 并显式取 kernel 函数指针；这样 JIT 编译器
    // 能仅实例化当前形状需要的模板，避免把所有 hidden/rank 组合编进二进制。
    const EngramFetchRuntime::Args args = {  // 聚合生成源码和 launch 所需参数。
        // 构造 fetch kernel 的完整 JIT 参数。
        .num_entries_per_rank = num_entries_per_rank,
        // 每 rank entry 数。
        .hidden = hidden,
        // 单 entry 隐藏维度。
        .num_ranks = num_ranks,
        // 总 rank 数。
        .num_qps = num_qps,
        // 并发 Gin request/QP 数。
        .nccl_dev_comm = nccl_dev_comm,
        // device communicator。
        // 中文：Engram fetch 通过 NCCL Gin 从远端 symmetric storage 拉取 entry。
        .nccl_window = nccl_window,
        // symmetric window。
        .storage = storage,
        // entry 存储基址。
        .fetched = fetched,
        // fetch 输出基址。
        .indices = indices,
        // 全局 entry id 数组。
        .last_gin_requests = last_gin_requests,
        // 每 QP 最后请求句柄数组。
        .num_tokens = num_tokens,
        // fetch 数量。
        .launch_args = jit::LaunchArgs(num_qps, kNumEngramFetchThreads)};
        // grid.x=num_qps，一个 block 对应一个 QP。
    const auto code = EngramFetchRuntime::generate(args);  // 生成完整 JIT 源码。
    // 生成带 include hash 的源码。
    const auto runtime = jit::compiler->build("engram_fetch", code);  // 编译或命中 JIT cache。
    // 编译或复用 engram_fetch cubin。
    // build 名称相同但源码不同会走 JIT cache 的内容 hash，不会错误复用其它 hidden/rank 配置。
    EngramFetchRuntime::launch(runtime, args, stream);
    // 在指定 stream 上启动 fetch kernel。
}

class EngramFetchWaitRuntime final : public jit::LaunchRuntime<EngramFetchWaitRuntime> {  // wait kernel 的 JIT runtime 封装。
    // WaitRuntime 负责等待 fetch kernel 提交的所有 Gin 请求完成。
public:
    struct Args {  // wait kernel 参数包。
        // wait kernel 的 JIT 参数集合。
        // Templated arguments
        int num_ranks;  // rank 数。
        // 总 rank 数，wait_impl 用它遍历/验证请求来源。
        // 中文：wait kernel 不关心 token 数，只需要知道每个 QP 上有多少 rank request 需要等待。

        ncclDevComm_t nccl_dev_comm;  // device communicator。
        // NCCL device communicator。
        // 中文：wait kernel 通过 communicator 等待 Gin request 完成。
        ncclWindow_t nccl_window;  // symmetric window。
        // symmetric window。
        // 中文：window 用于把本 rank 指针解析成可由 Gin get 访问的 symmetric 地址。
        // 中文：wait kernel 不重新发起通信，只等待 fetch kernel 记录的最后请求。
        ncclGinRequest_t* last_gin_requests;  // fetch 写出的最后请求句柄。
        // fetch kernel 写入的每 QP 最后请求句柄。

        jit::LaunchArgs launch_args;  // launch 配置。
        // launch 配置。
    };

    static std::string generate_impl(const Args& args) {  // 生成 wait kernel 源码。
        // 生成 wait kernel 的模板实例。
        // wait kernel 不需要 hidden/num_tokens，只需要知道 rank 数和线程数来遍历 request。
        return fmt::format(R"(
#include <deep_ep/impls/engram_fetch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    // 中文：wait_impl 只依赖 rank 数和线程数两个模板参数。
    auto ptr = reinterpret_cast<void*>(&engram_fetch_wait_impl<{}, {}>);
}}
)", args.num_ranks, args.launch_args.num_threads);
        // num_threads 进入模板，让 wait_impl 可按固定线程数展开。
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {  // 启动 wait kernel。
        // 绑定 wait kernel 的运行时参数。
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            // 中文：参数顺序必须和 engram_fetch_wait_impl 的形参顺序一致。
            args.last_gin_requests
        ));
        // wait kernel 完成后，fetched buffer 对同一 stream 后续算子可见。
        // last_gin_requests 是 fetch kernel 写出的数组；wait 只等待每个 QP 最后一条请求即可覆盖此前请求。
    }
};

static void launch_engram_fetch_wait(ncclGinRequest_t* last_gin_requests,  // fetch 阶段写出的请求句柄。
                                     const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,  // communicator/window。
                                     const int& num_ranks, const int& num_qps,  // rank 数和 QP 数。
                                     const at::cuda::CUDAStream& stream) {  // wait kernel 所在 CUDA stream。
    // fetch kernel 只提交 GIN 请求；wait kernel 在同一 stream 上等待所有 QP 完成，
    // 因而调用方可以把 wait 作为显式同步点插入到后续使用 fetched 的算子前。
    constexpr int kNumEngramFetchWaitThreads = 1024;  // wait kernel 固定线程数。
    // wait kernel 也固定 1024 线程，按 QP/rank 请求槽分摊等待。

    // Generate, build and launch
    const EngramFetchWaitRuntime::Args args = {  // wait kernel JIT/launch 参数。
        // 构造 wait kernel 的 JIT 参数。
        .num_ranks = num_ranks,
        // 总 rank 数。
        .nccl_dev_comm = nccl_dev_comm,
        // device communicator。
        // 中文：fetch/wait 使用同一个 NCCL device communicator 与 window。
        .nccl_window = nccl_window,
        // symmetric window。
        .last_gin_requests = last_gin_requests,
        // 请求句柄数组。
        // 中文：num_qps 个 block/CTA 分别等待对应 QP 的最后请求。
        .launch_args = jit::LaunchArgs(num_qps, kNumEngramFetchWaitThreads)};
        // grid.x=num_qps，与 fetch kernel 的 QP 划分一致。
    const auto code = EngramFetchWaitRuntime::generate(args);  // 生成 wait 源码。
    // 生成源码。
    const auto runtime = jit::compiler->build("engram_fetch_wait", code);  // 编译或复用 wait cubin。
    // 编译或复用 wait cubin。
    // wait 与 fetch 分开编译，是为了允许调用方把 fetch 和本地计算重叠，只在 hook 中等待。
    EngramFetchWaitRuntime::launch(runtime, args, stream);
    // 在同一 stream 上等待 fetch 完成。
}

}  // namespace deep_ep::elastic
