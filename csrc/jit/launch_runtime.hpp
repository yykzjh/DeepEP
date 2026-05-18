#pragma once  // LaunchRuntime 是模板化 header-only 代码，防止重复定义。

#include <ATen/cuda/CUDAContext.h>  // 读取 PyTorch 当前 CUDA stream。

#include <deep_ep/common/exception.cuh>  // CUDA/host 侧错误检查宏。

#include "../utils/format.hpp"   // fmt::format 用于拼接 include hash 注释和 debug 输出。
#include "../utils/system.hpp"   // get_env 读取 EP_JIT_DEBUG。
#include "compiler.hpp"          // Compiler 负责把 generate 得到的源码编译成 cubin。
#include "include_parser.hpp"    // include_parser 负责把 include 头文件内容纳入 hash。

namespace deep_ep::jit {  // DeepEP JIT 子系统命名空间。

struct LaunchArgs {
    // 所有 JIT kernel 共用的 launch 形状描述：
    // - grid_dim：通常 x 维表示参与通信的 SM 数或 QP 数；
    // - num_threads：block 内线程数，常见为 num_warps * 32；
    // - smem_size：动态共享内存字节数，device kernel 通过 extern __shared__ 接收；
    // - cooperative/cluster_dim/pdl_enabled：SM90/Hopper 相关 launch 语义。
    // host wrapper 只需填好 Args，LaunchRuntime 会统一把这些字段转换成 CUDA driver launch config。
    std::pair<int, int> grid_dim;  // grid 的 x/y 维；z 维固定为 1。
    int num_threads;               // 每个 block 的线程数。
    int smem_size;                 // 动态共享内存大小，单位字节。
    int cluster_dim;               // thread block cluster 的 x 维大小，1 表示不启用 cluster。
    bool cooperative;              // 是否启用 cooperative launch，device 侧 grid sync 依赖它。
    bool pdl_enabled;              // 是否允许 programmatic dependent launch。

    LaunchArgs(const int& grid_dim_x, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1, const bool& cooperative = false, const bool& pdl_enabled = false):
        // 常用构造函数：只有 x 维 grid。多数通信 kernel 用 grid.x 表示 SM/channel/QP 数。
        grid_dim({grid_dim_x, 1}), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim), cooperative(cooperative), pdl_enabled(pdl_enabled) {}  // 保存一维 grid launch 配置。

    LaunchArgs(const std::pair<int, int>& grid_dim, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1, const bool& cooperative = false, const bool& pdl_enabled = false):
        // 少数 kernel 需要二维 grid，例如某些批量任务；保留 pair 形式。
        grid_dim(grid_dim), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim), cooperative(cooperative), pdl_enabled(pdl_enabled) {}  // 保存二维 grid launch 配置。
};

template <typename Derived>  // CRTP：Derived 提供 generate_impl 和 launch_impl。
class LaunchRuntime {        // 封装所有 JIT kernel 共享的源码生成、include hash 和 CUDA launch 流程。
public:
    template <typename Args>
    static std::string generate(const Args& args) {
        auto code = Derived::generate_impl(args);  // 由具体 wrapper 生成包含模板实例化的 CUDA 源码。

        // generate_impl 通常只生成一个“小翻译单元”：
        //   #include <deep_ep/impls/xxx.cuh>
        //   auto ptr = reinterpret_cast<void*>(&xxx_impl<模板参数...>);
        // include_parser 会计算被 include 的头文件 hash，并把它拼到 code 前面。
        // 这样当 deep_ep/include/deep_ep/impls 或 common 头文件变化时，即使模板参数不变，
        // JIT cache key 也会变化，避免继续使用旧 cubin。
        // NOTES: we require that `generate_impl`'s includes never change
        static std::string include_hash;  // 每个 Derived 类型缓存 include 内容 hash。
        if (include_hash.empty())         // 首次生成该类 kernel 时解析 include 依赖。
            // include_hash 是 static 局部变量，按 Derived 类型各自缓存一次。假设
            // generate_impl 的 include 集合稳定，只需首次解析 include 依赖。
            include_hash = include_parser->get_hash_value(code);  // 计算源码中 include 到的项目头文件 hash。

        // TODO: optimize string concat performance
        code = fmt::format("// Includes' hash value: {}\n{}", include_hash, code);  // 把 include hash 放进源码，进而进入 compiler cache key。
        if (get_env<int>("EP_JIT_DEBUG", 0))                                        // debug 模式打印完整生成源码。
            printf("Generated kernel code:\n%s\n", code.c_str());                   // 输出源码便于复现 NVCC 编译。
        return code;                                                                // 返回最终参与 hash 和编译的源码。
    }

    template <typename Args>
    static void launch(const std::shared_ptr<KernelRuntime>& kernel_runtime, const Args& args,
                       const std::optional<at::cuda::CUDAStream>& stream_opt = std::nullopt) {
        // Derived::launch_impl 只负责把业务参数传给 kernel；grid/block/smem/cluster/cooperative
        // 的配置都集中在这里，避免每个 wrapper 重复写 CUDA driver API 调用。
        const auto kernel = kernel_runtime->kernel;                         // 从 KernelRuntime 中取已加载 kernel handle。
        const auto stream = stream_opt.value_or(at::cuda::getCurrentCUDAStream());  // 未显式传 stream 时使用 PyTorch 当前 stream。
        const LaunchArgs& launch_args = args.launch_args;                   // 业务 Args 中必须包含 launch_args 字段。

        // dim3 必须在 construct_launch_config 前构造好；cluster/cooperative/pdl 等属性
        // 会被 handle.hpp 中的 driver/runtime API 包装进 LaunchConfigHandle。
        const dim3& grid_dim = {static_cast<unsigned>(launch_args.grid_dim.first),   // grid.x 来自 LaunchArgs。
                                static_cast<unsigned>(launch_args.grid_dim.second),  // grid.y 来自 LaunchArgs。
                                1};                                                  // grid.z 固定为 1。
        const dim3& block_dim = {static_cast<unsigned>(launch_args.num_threads), 1, 1};  // block 只使用 x 维线程。
        auto config = construct_launch_config(kernel, stream, launch_args.smem_size,     // 构造 CUDA runtime/driver 统一 launch config。
                                              grid_dim, block_dim, launch_args.cluster_dim,  // 传入 grid、block、cluster 维度。
                                              launch_args.cooperative, launch_args.pdl_enabled);  // 传入 cooperative 和 PDL 开关。

        // Launch in the derived class
        // 中文：真正的 launch_impl 由具体 Runtime 子类实现，这里只统一构造配置和调试日志。
        if (get_env<int>("EP_JIT_DEBUG")) {
            printf("Launch kernel with {%d, %d} x %d (cooperative: %d), shared memory: %d bytes, cluster: %d, stream: %ld\n",
                   launch_args.grid_dim.first, launch_args.grid_dim.second, launch_args.num_threads,
                   // 中文：调试日志打印 grid/block/cooperative/smem/cluster/stream，便于复现 launch 配置。
                   launch_args.cooperative,
                   // 中文：后续参数继续打印动态 shared memory、cluster 维度和 stream id。
                   launch_args.smem_size, launch_args.cluster_dim, stream.id());  // 打印 launch 形状和 stream id。
        }
        Derived::launch_impl(kernel, config, args);  // 由具体 wrapper 把 kernel 参数打包并调用 launch_kernel。
    }
};

} // namespace deep_ep::jit  // 结束 DeepEP JIT 子系统命名空间。
