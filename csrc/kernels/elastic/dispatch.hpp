#pragma once  // Elastic dispatch host wrapper 是 header-only JIT 代码，防止重复定义。

#include <nccl.h>         // 使用 ncclWindow_t 等 NCCL 句柄类型。
#include <nccl_device.h>  // 使用 ncclDevComm_t，device kernel 通过它访问 Gin/LSA。

#include <deep_ep/common/compiled.cuh>   // topk_idx_t、sf_pack_t 等全局编译类型。
#include <deep_ep/common/exception.cuh>  // EP_HOST_ASSERT/EP_CUDA_UNIFIED_CHECK 等检查宏。

#include "../../jit/compiler.hpp"        // JIT 编译器和 cubin cache。
#include "../../jit/launch_runtime.hpp"  // CRTP launch wrapper 基类。

namespace deep_ep::elastic {  // Elastic/V2 dispatch host wrapper 命名空间。
// Elastic/V2 dispatch 相关 host wrapper 命名空间。

class DispatchPrologueRuntime final : public jit::LaunchRuntime<DispatchPrologueRuntime> {  // deterministic prologue JIT runtime。
    // deterministic prologue 只计算稳定 slot，不搬运 payload。
public:
    struct Args {  // prologue JIT/launch 参数包。
        // JIT Args 同时保存模板参数和 kernel 运行时参数。
        // Templated arguments
        int num_warps;  // prologue warp 数。
        // prologue kernel 内用于扫描 topk 的 warp 数。
        int num_ranks;  // rank 数。
        // 总 rank 数，rank_count_buffer 按这个维度统计。
        int num_max_tokens_per_rank;  // 每 rank 最大 token 容量。
        // 每个 rank 可接收的最大 token 数，用于 slot 越界保护。
        int num_experts, num_topk;  // expert 数和 top-k。
        // 全局 expert 数和每 token top-k 数。

        // Parameters
        topk_idx_t* topk_idx;  // 输入 top-k expert id。
        // 输入 top-k expert id，逻辑形状 [num_tokens, num_topk]。
        int* rank_count_buffer;  // rank 计数临时 buffer。
        // 输出/临时 rank 计数 buffer，统计目标 rank 的 token 数。
        int* dst_buffer_slot_idx;  // 输出稳定 slot。
        // 输出 slot index，逻辑形状 [num_tokens, num_topk]。
        int num_tokens;  // token 数。
        // 当前 batch token 数。
        int rank_idx;  // 当前 rank。
        // 当前 rank 编号，用于排除/定位自身。

        jit::LaunchArgs launch_args;  // launch 配置。
        // launch 形状和 shared memory 配置。
    };

    static std::string generate_impl(const Args& args) {  // 生成 prologue kernel 源码。
        // 生成 dispatch_deterministic_prologue_impl 的模板实例源码。
        // 生成源码只做两件事：include device 实现头，并取模板函数地址强制 nvcc 实例化。
        // 真正的参数绑定发生在 launch_impl 中，因此这里的 raw string 不包含运行时指针。
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_deterministic_prologue.cuh>

using namespace deep_ep::elastic;

// JIT 生成文件：仅取模板函数地址，强制 nvcc 实例化当前 prologue 参数组合。
static void __instantiate_kernel() {{
    // 中文：ptr 不会被调用；取地址本身就足够让 NVCC 生成对应模板实例。
    auto ptr = reinterpret_cast<void*>(&dispatch_deterministic_prologue_impl<{}, {}, {}, {}, {}, {}>);
}}
)",
                           args.launch_args.grid_dim.first,  // 模板 kNumSMs，对应 cooperative grid.x。
                           args.num_warps,  // 模板 kNumWarps。
                           args.num_ranks,  // 模板 kNumScaleupRanks。
                           args.num_max_tokens_per_rank,  // 模板 kNumMaxTokensPerRank。
                           args.num_experts, args.num_topk);  // 模板 expert/top-k 形状。
        // grid.x、warp 数、rank/expert/topk 均进入模板，让 prologue 走静态循环边界。
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {  // 启动 prologue kernel。
        // 按 prologue kernel 声明顺序传入运行时参数。
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel,
                                                 config,
                                                 args.topk_idx,  // 输入 top-k expert id。
                                                 args.rank_count_buffer,  // 临时 rank 计数 buffer。
                                                 args.dst_buffer_slot_idx,  // 输出确定性 slot。
                                                 args.num_tokens,  // token 数。
                                                 args.rank_idx));  // 当前 scaleup rank。
        // prologue 完成后 dst_buffer_slot_idx 可被主 dispatch 复用。
    }
};

static void launch_dispatch_deterministic_prologue(topk_idx_t* topk_idx, int* rank_count_buffer,  // 输入 topk 和 rank 计数 buffer。
                                                   int* dst_buffer_slot_idx,  // 输出稳定 slot。
                                                   const int& num_tokens, const int& num_max_tokens_per_rank,  // token 数和容量。
                                                   const int& num_experts, const int& num_topk,  // expert/top-k 形状。
                                                   const int& rank_idx, const int& num_ranks,  // 当前 rank 和总 rank。
                                                   const int& num_sms, const int& num_smem_bytes,  // launch SM 数和 shared bytes。
                                                   const at::cuda::CUDAStream& stream) {  // CUDA stream。
    // deterministic prologue 只负责“布局确定”，不搬运 token payload：
    // - topk_idx: [num_tokens, num_topk]；
    // - rank_count_buffer: 统计每个目标 rank 需要接收多少 token；
    // - dst_buffer_slot_idx: 为每个 token/top-k 选择稳定的目标 slot。
    // 后续 dispatch_impl 若 reuse_slot_indices=true，就可以复用这些 slot，避免非确定性 atomic 顺序。
    constexpr auto num_warps = 8;  // prologue 固定 warp 数。
    // 固定 8 个 warp 扫描 topk/rank 计数，JIT 实例数量更稳定。
    constexpr auto num_threads = num_warps * 32;  // block 线程数。
    // block 线程数等于 warp 数 * 32。
    EP_HOST_ASSERT((2 * num_warps + 1) * num_ranks * sizeof(int) <= num_smem_bytes and
                   "Insufficient shared memory");
    // prologue shared memory 需要保存 rank 计数前缀/临时区，不足时直接报错。

    // Generate, build and launch
    const DispatchPrologueRuntime::Args args = {  // 构造 JIT/launch 参数。
        // 汇总模板参数和运行时参数。
        .num_warps = num_warps,
        // prologue warp 数。
        .num_ranks = num_ranks,
        // rank 总数。
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        // 每 rank 最大 token 容量。
        .num_experts = num_experts, .num_topk = num_topk,
        // expert 数和 top-k。
        .topk_idx = topk_idx,
        // top-k 输入。
        .rank_count_buffer = rank_count_buffer,
        // rank 计数 buffer。
        .dst_buffer_slot_idx = dst_buffer_slot_idx,
        // 输出稳定 slot。
        .num_tokens = num_tokens,
        // token 数。
        .rank_idx = rank_idx,
        // 当前 rank。
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, true)};
        // cooperative=true，因为 device prologue 可能使用 grid 级同步。
    const auto code = DispatchPrologueRuntime::generate(args);  // 生成源码。
    // 生成带 include hash 的源码。
    const auto runtime = jit::compiler->build("dispatch_deterministic_prologue", code);  // 编译或复用 cubin。
    // 编译或复用 prologue cubin。
    DispatchPrologueRuntime::launch(runtime, args, stream);
    // 在指定 stream 上启动 prologue。
}

class DispatchRuntime final : public jit::LaunchRuntime<DispatchRuntime> {  // 主 dispatch JIT runtime。
    // DispatchRuntime 生成并启动单层 dispatch_impl 或 hybrid_dispatch_impl。
public:
    struct Args {  // 主 dispatch 参数包。
        // 主 dispatch 的参数很多：模板参数决定通信拓扑/布局/warp 划分，运行时参数是实际 buffer 指针。
        // Templated arguments
        bool is_scaleup_nvlink;  // scaleup 是否 NVLink。
        // 单层路径下 scaleup 域是否 NVLink，影响 device 端 peer 访问策略。
        bool do_cpu_sync;  // 是否执行 CPU sync 输出计数。
        // 是否在某些同步点使用 CPU-visible workspace 做 host 协调。
        bool reuse_slot_indices;  // 是否复用 slot。
        // 是否复用 deterministic/cached slot，避免 device 端重新 atomic 分配 slot。
        int num_notify_warps;  // notify warp 数。
        // 负责统计/通知 token count 的 warp 数。
        int num_dispatch_warps; // For hybrid dispatch  // 单层 payload warp 数。
        // 单层 dispatch 中负责 payload 搬运的 warp 数。
        int num_scaleout_warps, num_forward_warps; // For direct dispatch  // hybrid scaleout/forward warp 数。
        // hybrid dispatch 中跨节点 scaleout warp 和节点内 forward warp 数。
        int num_scaleout_ranks, num_scaleup_ranks;  // 二维拓扑维度。
        // 逻辑二维拓扑大小。
        int num_hidden_bytes, num_sf_packs;  // hidden 字节数和 SF pack 数。
        // token hidden payload 字节数和 FP8 scale factor pack 数。
        int num_max_tokens_per_rank;  // 每 rank token 容量。
        // 每个目标 rank 的最大 token 容量。
        int num_experts, num_topk, expert_alignment;  // expert/top-k/对齐。
        // expert 数、top-k 和 expert token 对齐粒度。
        int num_qps;  // Gin QP 数。
        // Gin/NCCL QP 数量，用于设备端通信通道分配。
        int64_t num_timeout_cycles;  // 超时周期。
        // GPU clock64 超时阈值。

        // Parameters
        void* x; sf_pack_t* sf; topk_idx_t* topk_idx; float* topk_weights;  // 输入张量指针。
        // 输入 token、可选 scale factor、top-k expert id 和 top-k 权重。
        topk_idx_t* copied_topk_idx;  // 可选 topk 副本输出。
        // dispatch 可把 topk_idx 复制到接收侧 metadata，供后续 combine/epilogue 使用。
        int* cumulative_local_expert_recv_stats;  // 本地 expert 统计。
        // 可选累计统计：每个本地 expert 接收 token 数。
        int* psum_num_recv_tokens_per_scaleup_rank;  // rank 前缀输出。
        // 输出 prefix-sum：按 scaleup rank 统计接收 token，用于 EPHandle。
        int* psum_num_recv_tokens_per_expert;  // expert 前缀输出。
        // 输出 prefix-sum：按 expert 统计接收 token，用于专家输入布局。
        int* dst_buffer_slot_idx;  // slot index。
        // deterministic/cached slot index，reuse_slot_indices=true 时使用。
        int* token_metadata_at_forward;  // hybrid forward metadata。
        // hybrid forward 阶段的 token metadata，中间转发时记录来源和目标。
        int num_tokens;  // token 数。
        // 当前 batch token 数。
        int sf_token_stride, sf_hidden_stride;  // SF tensor stride。
        // scale factor 的 token/hidden stride，用于支持非紧凑 scale 布局。
        ncclDevComm_t nccl_dev_comm;  // device communicator。
        // NCCL device communicator。
        // 中文：后续 window/buffer/workspace 共同构成 dispatch 主 kernel 的通信上下文。
        ncclWindow_t nccl_window;  // symmetric window。
        // NCCL symmetric memory window。
        void* buffer;  // payload buffer。
        // 通信 payload buffer。
        void* workspace; void* mapped_host_workspace;  // GPU/host workspace。
        // GPU workspace 和可能映射到 host/CPU 可见的 workspace。
        int scaleout_rank_idx, scaleup_rank_idx;  // 当前二维 rank 坐标。
        // 当前 rank 的逻辑二维坐标。

        jit::LaunchArgs launch_args;  // launch 配置。
        // launch 配置。
    };

    static std::string generate_impl(const Args& args) {  // 根据 Args 生成主 dispatch 源码。
        // 根据拓扑选择 include 的 device 头和模板函数名。
        std::string header_name, func_name;  // 要 include 的头和实例化函数名。
        // header_name 是 deep_ep/impls/<header>.cuh，func_name 是具体模板实例字符串。
        if (args.num_scaleout_ranks == 1) {
            // 单层拓扑：所有 rank 都在同一个 scaleup 域，通常是节点内 NVLink 或单一通信域。
            // 生成 dispatch.cuh 中的 dispatch_impl<...> 实例。
            header_name = "dispatch";  // 单层 dispatch device 头。
            // 单层路径使用 dispatch.cuh。
            func_name = fmt::format("dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",  // 单层模板实例。
                args.is_scaleup_nvlink,  // 是否启用 NVLink direct/symmetric 路径。
                args.do_cpu_sync,  // 是否把计数同步到 host mapped workspace。
                args.reuse_slot_indices,  // 是否复用外部 slot index。
                args.launch_args.grid_dim.first,  // kNumSMs。
                args.num_notify_warps, args.num_dispatch_warps,  // notify/payload warp 数。
                args.num_scaleup_ranks,  // 单层 rank 数。
                args.num_hidden_bytes, args.num_sf_packs,  // token payload/scale 布局。
                args.num_max_tokens_per_rank,  // 每 rank 容量。
                args.num_experts, args.num_topk, args.expert_alignment,  // expert/top-k/对齐。
                args.num_qps, args.num_timeout_cycles);  // Gin QP 数与超时周期。
            // dispatch_impl 模板参数顺序与 deep_ep/impls/dispatch.cuh 中声明严格一致。
        } else {
            // 两层拓扑：scaleout 跨 RDMA，scaleup 走 NVLink。hybrid_dispatch_impl 会把 dispatch
            // 拆成 scaleout warps 和 forward warps，先跨节点，再在节点内转发到目标专家。
            header_name = "hybrid_dispatch";  // hybrid dispatch device 头。
            // 两层拓扑使用 hybrid_dispatch.cuh。
            func_name = fmt::format("hybrid_dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                args.do_cpu_sync,  // 是否 CPU sync。
                args.reuse_slot_indices,  // 是否复用 slot。
                args.launch_args.grid_dim.first,  // kNumSMs。
                args.num_notify_warps, args.num_scaleout_warps, args.num_forward_warps,  // 三类 warp 数。
                args.num_scaleout_ranks, args.num_scaleup_ranks,  // 二维拓扑。
                args.num_hidden_bytes, args.num_sf_packs,  // token layout。
                args.num_max_tokens_per_rank,  // 每 rank 容量。
                args.num_experts, args.num_topk, args.expert_alignment,  // expert/top-k/对齐。
                args.num_qps, args.num_timeout_cycles);  // QP/超时。
            // hybrid_dispatch_impl 多出 scaleout/forward warp 划分和二维拓扑参数。
        }

        return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>

using namespace deep_ep::elastic;

// JIT 生成文件：按 host wrapper 选择的拓扑实例化 dispatch/hybrid_dispatch kernel。
static void __instantiate_kernel() {{
    // 中文：JIT 生成的 TU 通过保存函数指针触发模板实例化，避免链接未使用符号。
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
        // 生成一个只 include 对应实现并实例化函数指针的小翻译单元。
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        // 单层和 hybrid kernel 形参不同，因此 launch 参数也分两支。
        if (args.num_scaleout_ranks == 1) {
            // 单层 dispatch 不需要 scaleout_rank_idx/token_metadata_at_forward。
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
                kernel, config,
                args.x, args.sf, args.topk_idx, args.topk_weights,  // 输入 token、scale、路由和权重。
                args.copied_topk_idx,  // 可选复制后的 topk_idx，用于 cached/deterministic 路径保留路由。
                args.cumulative_local_expert_recv_stats,  // 可选本地 expert 累计接收统计。
                args.psum_num_recv_tokens_per_scaleup_rank,  // scaleup rank prefix sum 输出。
                args.psum_num_recv_tokens_per_expert,  // expert prefix sum 输出。
                args.dst_buffer_slot_idx,  // deterministic prologue 生成的目标 slot，可为空。
                args.num_tokens,  // 当前 batch token 数。
                args.sf_token_stride, args.sf_hidden_stride,  // FP8 scale 的 token/hidden stride。
                args.nccl_dev_comm, args.nccl_window,  // NCCL device communicator 和 window。
                args.buffer,  // 通信 buffer。
                args.workspace, args.mapped_host_workspace,  // device workspace 和 host-mapped workspace。
                args.scaleup_rank_idx));  // 当前 rank 在 scaleup 域内的编号。
            // scaleup_rank_idx 足以描述当前 rank 在单层域中的位置。
        } else {
            // hybrid dispatch 需要额外传 token_metadata_at_forward 和二维 rank 坐标。
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
                kernel, config,
                args.x, args.sf, args.topk_idx, args.topk_weights,  // 输入 token payload/scale/路由/权重。
                args.copied_topk_idx,  // cached 路由副本。
                args.cumulative_local_expert_recv_stats,  // 本地 expert 累计统计。
                args.psum_num_recv_tokens_per_scaleup_rank,  // scaleup rank prefix 输出。
                args.psum_num_recv_tokens_per_expert,  // expert prefix 输出。
                args.dst_buffer_slot_idx,  // 可复用 slot index。
                args.token_metadata_at_forward,  // hybrid forward 阶段生成的 token metadata。
                args.num_tokens,  // 当前 batch token 数。
                args.sf_token_stride, args.sf_hidden_stride,  // FP8 scale stride。
                args.nccl_dev_comm, args.nccl_window,  // NCCL Gin/Window 句柄。
                args.buffer,  // 通信 buffer。
                args.workspace, args.mapped_host_workspace,  // device/host workspace。
                args.scaleout_rank_idx, args.scaleup_rank_idx  // 二维拓扑坐标。
            ));
            // scaleout_rank_idx/scaleup_rank_idx 共同定位当前 rank。
        }
    }
};

constexpr int kNumNotifyWarps = 4;
// 非 cached dispatch 默认使用 4 个 notify warp 做 token 计数和远端通知。

static int get_num_notify_smem_bytes(const int& num_ranks, const int& num_experts) {
    // notify shared memory 存 rank/expert 计数，按 notify warp 总 lane 数对齐。
    return math::align(num_ranks + num_experts, kNumNotifyWarps * 32) * sizeof(int);
    // 返回字节数，供 launch_dispatch 从总 smem 中扣除。
}

static layout::TokenLayout get_dispatch_token_layout(
    const int& hidden, const int& elem_size, const int& num_sf_packs, const int& num_topk) {
    // dispatch token layout 包含 hidden payload、scale factor、topk_idx/topk_weights metadata。
    return layout::TokenLayout(hidden * elem_size, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
    // 第四个参数 true 表示 layout 中包含 dispatch 需要的 topk_idx 元数据。
}

static void launch_dispatch(void* x, void* sf,  // 输入 payload 与可选 scale factor。
                            topk_idx_t* topk_idx, float* topk_weights,  // 输入路由矩阵和权重，形状 [T,K]。
                            topk_idx_t* copied_topk_idx,  // 输出/缓存路由副本，可为空。
                            int* cumulative_local_expert_recv_stats,  // 可选本地 expert 累计统计。
                            int* psum_num_recv_tokens_per_scaleup_rank,  // 输出 scaleup rank prefix。
                            int* psum_num_recv_tokens_per_expert,  // 输出 expert prefix。
                            int* dst_buffer_slot_idx,  // deterministic slot index buffer。
                            int* token_metadata_at_forward,  // hybrid forward metadata buffer。
                            const int& num_tokens, const int& num_max_tokens_per_rank,  // 当前 token 数和每 rank 容量上界。
                            const int& hidden, const int& elem_size,  // hidden 元素数与单元素字节数。
                            const int& num_sf_packs, const int& sf_token_stride, const int& sf_hidden_stride,  // FP8 scale layout。
                            const int& num_experts, const int& num_topk, const int& expert_alignment,  // expert/topk/对齐粒度。
                            const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,  // NCCL device communicator/window。
                            void* buffer,  // dispatch 通信 buffer。
                            void* workspace, void* mapped_host_workspace,  // device workspace 与 host-mapped workspace。
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 当前 rank 二维坐标。
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,  // scaleout/scaleup 维度大小。
                            const bool& is_scaleup_nvlink,  // scaleup 维是否走 NVLink。
                            const int& num_sms, const int& num_channels_per_sm,  // kernel 使用 SM 数和每 SM channel 数。
                            const int& num_smem_bytes,  // 每 block 可用 dynamic shared memory。
                            const int& num_qps, const int64_t& num_timeout_cycles,  // Gin QP 数和设备端超时周期。
                            const bool& cached_mode,  // 是否复用已有 layout。
                            const bool& deterministic,  // 是否启用 deterministic prologue slot。
                            const bool& do_cpu_sync,  // 是否需要 CPU 读取接收 token 数。
                            const at::cuda::CUDAStream& stream) {  // CUDA launch stream。
    // 这是 Elastic dispatch 的 host wrapper。它把 C++ runtime 中的运行时参数固化成模板参数：
    //   x: [num_tokens, hidden] 的 BF16/FP8 token payload；
    //   sf: FP8 scale factor，可为空；
    //   topk_idx/topk_weights: [num_tokens, num_topk]；
    //   psum_num_recv_tokens_per_*: dispatch 结束后写入，作为 Python EPHandle 的核心元数据。
    // 这些参数一旦进入模板实例，device 端就能用 constexpr 分支和静态布局减少运行时判断。
    // Cached mode does not support expert token counting
    if (cached_mode)
        // cached 模式复用旧布局，不重新统计 expert token，因此统计输出必须为空。
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats == nullptr);

    // Utils
    const auto num_ranks = num_scaleout_ranks * num_scaleup_ranks;
    // 逻辑总 rank 数 = scaleout 维度 * scaleup 维度。

    // Notify warps
    // notify warps 负责统计每个 rank/expert 的 token 数并通知远端；cached 模式已经有这些信息，
    // 所以可以把 notify warps 置 0，只保留 payload 搬运，提高 replay/backward 场景效率。
    // TODO: why don't we use 4 notify warps?
    const int num_notify_warps = cached_mode ? 0 : kNumNotifyWarps;
    // cached 模式不需要 notify warp；普通模式使用固定 4 个。
    const bool reuse_slot_indices = cached_mode or deterministic;
    // cached 和 deterministic 都要求复用外部提供/预先计算的 slot index。
    const int num_notify_smem_bytes = cached_mode ? 0 : get_num_notify_smem_bytes(num_ranks, num_experts);
    // notify warp 的 shared memory 需求；cached 模式为 0。
    EP_HOST_ASSERT(num_notify_warps % 4 == 0);
    // device 实现按 4-warp 组组织 notify 逻辑，warp 数必须是 4 的倍数。

    // Other warps
    int num_dispatch_warps = 0;
    // 单层 dispatch payload warp 数，只有 num_scaleout_ranks==1 时使用。
    int num_scaleout_warps = 0, num_forward_warps = 0;
    // hybrid dispatch 跨节点和节点内转发 warp 数。
    int num_threads = 0;
    // 最终 blockDim.x。

    // Maximize shared memory utilization
    // 单层 dispatch 中，每个 dispatch warp 使用一块 token layout 共享内存作为 TMA staging buffer。
    // 因此可用 warp 数受动态 shared memory 限制；同时限制总 warp 数不超过约 512/num_sms，
    // 避免过多 warp 在同一批 SM 上带来调度和寄存器压力。
    if (num_scaleout_ranks == 1) {
        // 单层路径：notify warp + dispatch warp 都在同一组 rank 域内工作。
        // Too many warps may cause performance degrade, so we limit the total warps into 512
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);
        // 计算一个 token 在 shared staging 中需要多少字节。
        num_dispatch_warps = std::min<int>(std::min<int>(
            (num_smem_bytes - num_notify_smem_bytes) / token_layout.get_num_bytes<true>(), 32 - num_notify_warps),
            // 中文：总 dispatch warp 数再按 SM 数限制到约 512 个 warp，避免过度占用。
            math::ceil_div(512, num_sms));
        // dispatch warp 数受三重限制：shared memory、block 最多 32 warp、全 grid 约 512 warp。
        num_threads = (num_notify_warps + num_dispatch_warps) * 32;
        // blockDim.x = notify warp + payload warp。
    } else {  // 中文：hybrid 模式使用另一套 warp 规划，拆分 scaleout/forward 角色。
        // Hybrid kernels
        // 中文：hybrid 会把跨节点 scaleout 与节点内 forward 拆给不同 warp 组。
        // hybrid 模式每个 SM 分配 num_channels_per_sm 条 scaleout 与 forward channel。
        // scaleout_warps 面向 RDMA，forward_warps 面向节点内 scaleup 转发；两者配对可以隐藏
        // 跨节点延迟，但当前实现不支持 deterministic prologue。
        // Some unimplemented assertions
        EP_HOST_ASSERT(not deterministic);
        // hybrid deterministic prologue 尚未实现，防止调用方误以为 slot 稳定。

        num_scaleout_warps = num_channels_per_sm;
        // 每个 SM 分配 num_channels_per_sm 个跨节点通道。
        num_forward_warps = num_channels_per_sm;
        // 每个跨节点通道配一个节点内 forward 通道。
        num_threads = (num_notify_warps + num_scaleout_warps + num_forward_warps) * 32;
        // hybrid block 线程数由 notify + scaleout + forward 三类 warp 组成。
    }

    // Generate, build and launch
    const DispatchRuntime::Args args = {
        // 聚合所有模板参数和运行时指针，后续 generate/build/launch 共享这一份。
        .is_scaleup_nvlink = is_scaleup_nvlink,
        // scaleup 域是否 NVLink。
        .do_cpu_sync = do_cpu_sync,
        // 是否启用 CPU 同步语义。
        .reuse_slot_indices = reuse_slot_indices,
        // 是否复用 slot。
        .num_notify_warps = num_notify_warps,
        // notify warp 数。
        .num_dispatch_warps = num_dispatch_warps,
        // 单层 dispatch warp 数。
        .num_scaleout_warps = num_scaleout_warps, .num_forward_warps = num_forward_warps,
        // hybrid 两类 warp 数。
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        // 逻辑二维拓扑。
        .num_hidden_bytes = hidden * elem_size, .num_sf_packs = num_sf_packs,
        // token payload 字节数和 scale pack 数。
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        // 每 rank 最大 token。
        .num_experts = num_experts, .num_topk = num_topk, .expert_alignment = expert_alignment,
        // expert/topk/对齐参数。
        .num_qps = num_qps, .num_timeout_cycles = num_timeout_cycles,
        // 通信 QP 数和超时周期。
        .x = x, .sf = static_cast<sf_pack_t*>(sf), .topk_idx = topk_idx, .topk_weights = topk_weights,
        // 输入 token、scale、路由和权重。
        // x: [num_tokens, hidden]；sf: FP8 时为 [num_tokens, hidden_sf]；
        // topk_idx/topk_weights: [num_tokens, num_topk]，device kernel 的 lane0..topk-1 直接读取。
        .copied_topk_idx = copied_topk_idx,
        // 可选复制后的 topk_idx 输出。
        // copied_topk_idx 只服务 Python/测试或后续模块读取，不参与通信 slot 分配。
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats,
        // 可选接收统计。
        // 统计维度为 [num_local_experts]，记录未 padding 的真实接收 token 数。
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        // scaleup rank 维 prefix sum 输出。
        // dispatch_copy_epilogue 用它把 scaleup_buffer 中按 rank 排列的 token 映射到连续 recv_x。
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,
        // expert 维 prefix sum 输出。
        // expanded dispatch 用它为每个本地 expert 分配输出行；最后一项可推导 expanded 总行数。
        .dst_buffer_slot_idx = dst_buffer_slot_idx,
        // slot index 输入/输出。
        // 逻辑形状 [num_tokens,num_topk]；deterministic/cached 模式读它，普通模式写它。
        .token_metadata_at_forward = token_metadata_at_forward,
        // hybrid forward metadata。
        // hybrid combine 会重放该 metadata，将 scale-up 中间结果跨 scale-out 发回源 rank。
        .num_tokens = num_tokens,
        // token 数。
        .sf_token_stride = sf_token_stride, .sf_hidden_stride = sf_hidden_stride,
        // scale factor stride。
        .nccl_dev_comm = nccl_dev_comm, .nccl_window = nccl_window,
        // NCCL device communicator 和 window。
        .buffer = buffer,
        // 通信 buffer。
        // buffer 是 NCCL window 中的 payload 区，dispatch_impl 写入，copy_epilogue/后续 combine 再读取。
        .workspace = workspace, .mapped_host_workspace = mapped_host_workspace,
        // GPU/host mapped workspace。
        // workspace 保存计数、barrier、tail、链表等控制面；mapped_host_workspace 用于 CPU sync 路径。
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        // 当前 rank 逻辑坐标。
        // NOTES: make cluster dim 2 to overlap with clustered computation kernels
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 2 - (num_sms % 2), true)};
        // 偶数 SM 数时 cluster_dim=2，否则为 1；cooperative=true 支持 grid sync。
        // cooperative launch 是 notify/reduction 和 barrier 协议的前提，cluster_dim 则用于和集群化计算 kernel 更好重叠。
    const auto code = DispatchRuntime::generate(args);
    // 生成带 include hash 的源码。
    const auto runtime = jit::compiler->build("dispatch", code);
    // 编译或复用 dispatch cubin。
    DispatchRuntime::launch(runtime, args, stream);
    // 启动主 dispatch kernel。
}

class DispatchCopyEpilogueRuntime final : public jit::LaunchRuntime<DispatchCopyEpilogueRuntime> {
    // Copy epilogue 把通信 buffer 中按 rank/expert/channel 排列的数据整理成 Python 期望的 recv_* 输出。
public:
    struct Args {
        // epilogue 的模板参数控制布局展开方式、拓扑和 token layout。
        // Templated arguments
        bool do_expand, cached_mode;
        // do_expand 表示是否展开到 dense expert layout；cached_mode 表示复用已有 layout。
        int num_channels;
        // dispatch 使用的通信 channel 数，linked list 也按 channel 组织。
        int num_warps;
        // epilogue 每个 block 使用的 warp 数。
        int num_scaleout_ranks, num_scaleup_ranks;
        // 逻辑二维拓扑。
        int num_hidden_bytes, num_sf_packs;
        // 每 token hidden 字节数和 scale pack 数。
        int num_max_tokens_per_rank;
        // 每 rank 最大 token 容量。
        int num_experts, num_topk;
        // expert 数和 top-k 数。

        // Parameters
        void *buffer, *workspace;
        // dispatch 通信 buffer 和 workspace。
        int* psum_num_recv_tokens_per_scaleup_rank;
        // scaleup rank 维 prefix sum，定位每个 rank 的接收区间。
        int* psum_num_recv_tokens_per_expert;
        // expert 维 prefix sum，定位每个 expert 的接收区间。
        void* recv_x; void* recv_sf;
        // 输出 token payload 和可选 scale factor。
        topk_idx_t* recv_topk_idx; float* recv_topk_weights;
        // 输出接收 token 对应的 topk_idx/topk_weights。
        int* recv_src_metadata;
        // 输出源 token metadata，combine 会据此把专家结果送回原 token。
        int* channel_linked_list;
        // channel 链表，记录每个 channel 收到的 token 顺序。
        int num_recv_tokens;
        // 当前 rank 总接收 token 数。
        int recv_sf_token_stride, recv_sf_hidden_stride;
        // 输出 scale factor 的 token/hidden stride。
        int scaleout_rank_idx, scaleup_rank_idx;
        // 当前 rank 逻辑坐标。

        jit::LaunchArgs launch_args;
        // launch 配置。
    };

    static std::string generate_impl(const Args& args) {
        // 生成 dispatch_copy_epilogue_impl 模板实例。
        // copy epilogue 是 dispatch 的第二阶段：主 kernel 已经把 token 写入 buffer/workspace，
        // 这里再按 prefix sum 把它们整理成用户可见的 recv_x/recv_topk/handle metadata。
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_copy_epilogue.cuh>

using namespace deep_ep::elastic;

// JIT 生成文件：实例化 copy epilogue，把通信 buffer 整理成 Python 可见输出。
static void __instantiate_kernel() {{
    // 中文：copy epilogue 模板参数必须与主 dispatch 返回的 layout 完全一致。
    auto ptr = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
	)",
                           args.do_expand, args.cached_mode,  // 是否 expanded、是否 cached。
                           args.launch_args.grid_dim.first, args.num_channels, args.num_warps,  // SM/channel/warp 数。
                           args.num_scaleout_ranks, args.num_scaleup_ranks,  // 二维拓扑。
                           args.num_hidden_bytes, args.num_sf_packs,  // token payload/scale 布局。
                           args.num_max_tokens_per_rank,  // 每 rank 容量。
                           args.num_experts, args.num_topk);  // expert/top-k 形状。
        // 所有布局相关参数都进入模板，device 端可静态计算 offset。
        // cached_mode 进入模板，是因为 cached dispatch 不需要重新写某些 metadata。
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        // 按 epilogue kernel 形参顺序传参。
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.buffer, args.workspace,  // dispatch payload buffer 和控制面 workspace。
                                                 args.psum_num_recv_tokens_per_scaleup_rank,  // rank prefix 输入。
                                                 args.psum_num_recv_tokens_per_expert,  // expert prefix 输入。
                                                 args.recv_x, args.recv_sf, args.recv_topk_idx, args.recv_topk_weights,  // 输出 payload/scale/路由/权重。
                                                 args.recv_src_metadata,  // 输出源 metadata。
                                                 args.channel_linked_list,  // 输出 channel 链表。
                                                 args.num_recv_tokens,  // 当前 rank 接收 token 数。
                                                 args.recv_sf_token_stride, args.recv_sf_hidden_stride,  // 输出 scale stride。
                                                 args.scaleout_rank_idx, args.scaleup_rank_idx));  // 当前 rank 二维坐标。
        // epilogue 完成后 recv_x/recv_* 元数据即可交给专家计算或 Python handle。
        // 该 kernel 和主 dispatch 在同一 stream 上顺序执行，天然依赖主 dispatch 的写入完成。
    }
};

static void launch_dispatch_copy_epilogue(void* buffer, void* workspace,
                                          int* psum_num_recv_tokens_per_scaleup_rank,  // dispatch 主 kernel 输出的 rank prefix。
                                          int* psum_num_recv_tokens_per_expert,  // dispatch 主 kernel 输出的 expert prefix。
                                          void* recv_x, void* recv_sf,  // epilogue 输出 token payload 和可选 FP8 scale。
                                          topk_idx_t* recv_topk_idx, float* recv_topk_weights,  // epilogue 输出路由和权重元数据。
                                          int* recv_src_metadata,  // epilogue 输出源 token/top-k/rank metadata。
                                          int* channel_linked_list,  // hybrid combine 复用的 channel 链表。
                                          const int& num_recv_tokens, const int& num_max_tokens_per_rank,  // 本 rank 接收 token 数和容量上界。
                                          const int& num_hidden_bytes,  // 单 token hidden payload 字节数。
                                          const int& num_sf_packs, const int& recv_sf_token_stride, const int& recv_sf_hidden_stride,  // FP8 scale 输出布局。
                                          const int& num_experts, const int& num_topk,  // expert/top-k 形状。
                                          const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 当前 rank 二维坐标。
                                          const int& num_scaleout_ranks, const int& num_scaleup_ranks,  // 二维拓扑大小。
                                          const int& num_sms, const int& num_smem_bytes,  // epilogue 使用的 SM 数和 dynamic smem。
                                          const int& num_channels,  // dispatch 主 kernel 使用的通信 channel 数。
                                          const bool& do_expand, const bool& cached_mode,  // 是否输出 expanded layout、是否 cached。
                                          const at::cuda::CUDAStream& stream) {  // CUDA launch stream。
    // Maximize shared memory utilization
    const auto token_layout = layout::TokenLayout(num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
    // epilogue 每个 warp 在 shared memory 暂存一个 dispatch token layout。
    // with_metadata=true，因为 epilogue 要读取 topk、source token 和 linked-list 字段。
    const auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);
    // warp 数受 shared memory 和 block 最大 32 warp 限制。
    // num_warps 越多，单 block 可并行整理更多 token；但每个 warp 都需要一个完整 token staging。
    const auto num_threads = num_warps * 32;
    // blockDim.x。

    // Generate, build and launch
    const DispatchCopyEpilogueRuntime::Args args = {
        // 聚合 epilogue 模板参数和运行时参数。
        .do_expand = do_expand, .cached_mode = cached_mode,
        // 是否展开输出、是否 cached。
        // do_expand=true 时输出按 top-k slot 展开，combine 会按 expanded layout 反向处理。
        .num_channels = num_channels, .num_warps = num_warps,
        // channel 数和 warp 数。
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        // 逻辑拓扑。
        .num_hidden_bytes = num_hidden_bytes, .num_sf_packs = num_sf_packs,
        // token payload 布局参数。
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        // 每 rank 最大 token。
        .num_experts = num_experts, .num_topk = num_topk,
        // expert/top-k。
        .buffer = buffer, .workspace = workspace,
        // 输入通信 buffer/workspace。
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        // scaleup rank prefix sum。
        // 用于把每个 scaleup rank 的接收区间映射到 recv_x 的连续区间。
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,
        // expert prefix sum。
        // 用于把 token 按本地 expert 聚类，方便后续 expert GEMM 直接按区间读取。
        .recv_x = recv_x, .recv_sf = recv_sf,
        // 输出 payload/scale。
        .recv_topk_idx = recv_topk_idx, .recv_topk_weights = recv_topk_weights,
        // 输出路由元数据。
        .recv_src_metadata = recv_src_metadata,
        // 输出源 token metadata。
        // combine 依赖 recv_src_metadata 找回原 rank、原 token 和 top-k 槽。
        .channel_linked_list = channel_linked_list,
        // channel 链表。
        // hybrid 模式下链表记录每个 channel 的 token 顺序，combine forward 会重放它。
        .num_recv_tokens = num_recv_tokens,
        // 接收 token 数。
        .recv_sf_token_stride = recv_sf_token_stride, .recv_sf_hidden_stride = recv_sf_hidden_stride,
        // 输出 scale stride。
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        // 当前 rank 坐标。
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, false, true)};
        // epilogue 不需要 cooperative，但启用 PDL 以便和前后 kernel 串接减少空泡。
    const auto code = DispatchCopyEpilogueRuntime::generate(args);
    // 生成源码。
    const auto runtime = jit::compiler->build("dispatch_copy_epilogue", code);
    // 编译或复用 epilogue cubin。
    // epilogue 与 dispatch_impl 分开编译，允许它使用不同 shared memory 和 PDL launch 配置。
    DispatchCopyEpilogueRuntime::launch(runtime, args, stream);
    // 启动 copy epilogue。
}

}  // namespace deep_ep::elastic
// Elastic dispatch wrapper 命名空间结束。
