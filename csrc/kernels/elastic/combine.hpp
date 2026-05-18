#pragma once  // Elastic combine host wrapper 是 header-only JIT 代码，防止重复定义。

#include <nccl.h>  // 使用 ncclDevComm_t/ncclWindow_t 等 NCCL 句柄类型。

#include <deep_ep/common/compiled.cuh>   // nv_bfloat16、topk_idx_t 等全局类型。
#include <deep_ep/common/exception.cuh>  // EP_CUDA_UNIFIED_CHECK 等错误检查宏。

#include "../../jit/compiler.hpp"        // JIT 编译/cache 入口。
#include "../../jit/launch_runtime.hpp"  // CRTP launch wrapper。

namespace deep_ep::elastic {
// Elastic/V2 combine host wrapper 命名空间。

// CombineRuntime 是 dispatch 的反向路径：专家计算后的 x/topk_weights 会根据
// dispatch 阶段保存的 src_metadata 返回原始 token 所属 rank。num_scaleout_ranks==1
// 走单层 combine.cuh；跨节点 hybrid 模式走 hybrid_combine.cuh，先在 scale-up 域聚合，
// 再跨 scale-out 域转发。
class CombineRuntime final : public jit::LaunchRuntime<CombineRuntime> {
    // CombineRuntime 负责生成并启动 combine 或 hybrid_combine kernel。
public:
    struct Args {
        // Args 前半是模板参数，后半是运行时 buffer/metadata 指针。
        // Templated arguments
        bool is_scaleup_nvlink;  // 单层路径下 scaleup 域是否可走 NVLink fast path。
        // 单层 combine 中 scaleup 域是否 NVLink。
        bool use_expanded_layout, allow_multiple_reduction;  // 专家输出布局与多路规约策略。
        // 是否使用展开 expert layout；是否允许多个 scaleup rank 对同一 token 贡献规约。
        int num_scaleup_warps, num_forward_warps;  // hybrid 中本地规约/跨域转发 warp 数。
        // hybrid combine 中本地规约 warp 和跨 scaleout 转发 warp 数。
        int num_scaleout_ranks, num_scaleup_ranks;  // 逻辑二维拓扑大小。
        // 逻辑二维拓扑。
        int hidden;  // token hidden 元素数，元素类型固定 nv_bfloat16。
        // hidden 维度，combine 目前按 BF16 元素处理。
        int num_max_tokens_per_rank;  // 每个 rank 的通信 buffer token 容量。
        // 每 rank 最大 token 容量。
        int num_experts;  // 全局 expert 数。
        // 全局 expert 数。
        int num_topk;  // 每个原始 token 的路由 expert 数。
        // 每 token top-k 数。
        int num_qps;  // NCCL Gin QP 数。
        // Gin/NCCL QP 数。
        int64_t num_timeout_cycles;  // 设备端等待通信信号的超时周期。
        // GPU clock64 超时阈值。

        // Parameters
        nv_bfloat16* x;  // 专家输出 BF16 payload。
        // 专家输出 token，来源于 dispatch 后专家计算，逻辑上按 expert/rank/token/hidden 排布。
        float* topk_weights;  // 对应 top-k 权重。
        // dispatch 携带过来的 top-k 权重，用于 combine 加权求和。
        int* src_metadata;  // 源 token metadata。
        // dispatch epilogue 生成的源 token metadata，combine 用它把结果送回原 token。
        int* psum_num_recv_tokens_per_scaleup_rank;  // dispatch 阶段 rank prefix。
        // dispatch 输出的 scaleup rank 维 prefix sum，用来遍历本 rank 收到的 token。
        int* token_metadata_at_forward;  // hybrid forward metadata。
        // hybrid 转发阶段使用的 token metadata。
        int* channel_linked_list;  // hybrid channel linked-list。
        // hybrid 通信 channel 链表。
        ncclDevComm_t nccl_dev_comm;  // device communicator。
        // NCCL device communicator。
        // 中文：下面的 communicator/window/buffer/workspace 是 device kernel 访问 symmetric memory 的入口。
        ncclWindow_t nccl_window;  // symmetric memory window。
        // NCCL symmetric memory window。
        void* buffer;  // combine 通信缓冲区。
        // combine 通信 buffer。
        void* workspace;  // combine 临时工作区。
        // combine workspace。
        int scaleout_rank_idx, scaleup_rank_idx;  // 当前 rank 二维坐标。
        // 当前 rank 在逻辑二维拓扑中的坐标。
        int num_reduced_tokens;  // 本 rank 参与 combine 的 token 数。
        // 当前 combine 需要规约/返回的 token 数。

        jit::LaunchArgs launch_args;  // grid/block/smem/cluster 配置。
        // launch 配置。
    };

    static std::string generate_impl(const Args& args) {
        // 根据拓扑选择单层 combine.cuh 或 hybrid_combine.cuh。
        std::string header_name, func_name;
        // header_name 是 include 名，func_name 是待实例化的模板函数。
        // 单节点路径只需要一个通信层，模板里的 num_ranks = scaleup * scaleout；
        // hybrid 路径需要额外的 scaleup/forward warp 划分，因此使用另一套 kernel 入口。
        if (args.num_scaleout_ranks == 1) {
            header_name = "combine";
            // 单层路径使用 combine_impl。
            // 单层模板参数顺序必须和 deep_ep/impls/combine.cuh 的 combine_impl 声明一致；
            // 这里把运行时拓扑固化后，device 端可以静态选择 rank layout 与 vector reduce。
            func_name = fmt::format("combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                    args.is_scaleup_nvlink,  // 单层是否可走 NVLink。
                                    args.use_expanded_layout, args.allow_multiple_reduction,  // 输出布局和多路规约开关。
                                    args.launch_args.grid_dim.first,  // kNumSMs。
                                    args.launch_args.num_threads / 32,  // 每 block warp 数。
                                    args.num_scaleup_ranks * args.num_scaleout_ranks,  // 单层总 rank 数。
                                    args.hidden,  // hidden 元素数。
                                    args.num_max_tokens_per_rank,  // 每 rank 容量。
                                    args.num_experts,  // 全局 expert 数。
                                    args.num_topk,  // top-k 数。
                                    args.num_qps, args.num_timeout_cycles);  // QP/超时。
            // num_ranks 在单层模板中等于 scaleup*scaleout。
        } else {
            header_name = "hybrid_combine";
            // 两层拓扑使用 hybrid_combine_impl。
            // hybrid 模板多出 scaleup/forward warp 数与二维 rank 维度，用于生成两层通信状态机。
            func_name = fmt::format("hybrid_combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                    args.use_expanded_layout, args.allow_multiple_reduction,  // expanded 与多路规约开关。
                                    args.launch_args.grid_dim.first,  // kNumSMs。
                                    args.num_scaleup_warps, args.num_forward_warps,  // 本地归并/跨域转发 warp 数。
                                    args.num_scaleout_ranks, args.num_scaleup_ranks,  // 二维拓扑。
                                    args.hidden,  // hidden 元素数。
                                    args.num_max_tokens_per_rank,  // 每 rank 容量。
                                    args.num_experts,  // expert 数。
                                    args.num_topk,  // top-k 数。
                                    args.num_qps,  // QP 数。
                                    args.num_timeout_cycles);  // 超时周期。
            // hybrid 模板额外包含 scaleup/forward warp 划分和二维拓扑。
        }
        // 生成的小翻译单元只负责 include 目标实现并实例化模板，实际 launch 仍在 launch_impl 中传入。
        return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>

using namespace deep_ep::elastic;

// JIT 生成文件：按单层或 hybrid 拓扑实例化 combine 主 kernel。
static void __instantiate_kernel() {{
    // 中文：取函数地址即可强制编译器实例化对应模板，不需要真正调用。
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
        // 生成极小翻译单元，JIT 只实例化当前形状的 combine kernel。
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        // 单层和 hybrid kernel 参数列表不同，按拓扑分支传参。
        if (args.num_scaleout_ranks == 1) {
            // 单层 combine 不需要 forward metadata/channel linked list。
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,  // expert 输出和对应 top-k 权重。
                                                     args.src_metadata, args.psum_num_recv_tokens_per_scaleup_rank,  // 源 token metadata 与 rank prefix。
                                                     args.nccl_dev_comm, args.nccl_window,  // NCCL device communicator/window。
                                                     args.buffer, args.workspace,  // combine 通信 buffer 与 workspace。
                                                     args.scaleup_rank_idx,  // 单层拓扑中的当前 rank。
                                                     args.num_reduced_tokens));  // 本 rank 需要规约回源 token 的数量。
            // scaleup_rank_idx 足以定位当前 rank。
        } else {
            // hybrid combine 需要中间转发 metadata 和二维 rank 坐标。
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,  // expert 输出和权重。
                                                     args.src_metadata,  // dispatch epilogue 保存的源 token 信息。
                                                     args.psum_num_recv_tokens_per_scaleup_rank,  // scaleup rank prefix。
                                                     args.token_metadata_at_forward,  // hybrid forward 阶段 metadata。
                                                     args.channel_linked_list,  // dispatch 生成的 channel 链表。
                                                     args.nccl_dev_comm, args.nccl_window,  // NCCL Gin/Window 句柄。
                                                     args.buffer, args.workspace,  // 通信 buffer 和 workspace。
                                                     args.scaleout_rank_idx, args.scaleup_rank_idx,  // 当前 rank 二维坐标。
                                                     args.num_reduced_tokens));  // 待规约 token 数。
            // scaleout/scaleup 坐标共同决定通信目标。
        }
    }
};

    static layout::TokenLayout get_combine_token_layout(  // 构造 combine 通信 token layout。
        const int& hidden, const int& elem_size, const int& num_topk) {
    // combine 的通信 buffer 不需要 topk_idx，但需要保存 topk_weights；TokenLayout 中
    // topk_idx_bytes=0、num_topk=num_topk。单个 token 大小约为 hidden*sizeof(bf16)+num_topk*sizeof(float)。
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
    // 最后一个 false 表示 combine layout 不携带 dispatch 的 topk_idx metadata。
}

static void* launch_combine(void* x,  // expert 计算后的 BF16 输出，逻辑形状由 dispatch epilogue 决定。
                            void* topk_weights,  // 原 token 的 top-k 权重，用于加权规约。
                            int* src_metadata,  // 每个接收 token 对应的源 rank/token/top-k metadata。
                            int* psum_num_recv_tokens_per_scaleup_rank,  // scaleup rank 接收 token prefix。
                            int* token_metadata_at_forward,  // hybrid forward metadata，单层路径可为空。
                            int* channel_linked_list,  // hybrid channel 链表，描述跨 scaleout 转发顺序。
                            const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,  // NCCL device communicator/window。
                            void* buffer, void* workspace,  // combine 通信 buffer 与 workspace。
                            const int& num_reduced_tokens, const int& num_max_tokens_per_rank,  // 待规约 token 数和每 rank 容量。
                            const int& hidden,  // hidden 元素数，combine 固定按 BF16 处理。
                            const int& num_experts, const int& num_topk,  // expert 总数和 top-k 路数。
                            const int& num_qps, const int64_t& num_timeout_cycles,  // Gin QP 数和超时周期。
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,  // 二维拓扑大小。
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 当前 rank 二维坐标。
                            const bool& is_scaleup_nvlink,  // scaleup 维是否 NVLink。
                            const int& num_sms, const int& num_smem_bytes,  // kernel 使用 SM 数和 shared memory 字节数。
                            const int& num_channels,  // hybrid dispatch/combine 对齐使用的 channel 数。
                            const bool& use_expanded_layout, const bool& allow_multiple_reduction,  // layout 和规约策略开关。
                            const at::cuda::CUDAStream& stream) {  // CUDA launch stream。
    // 每个 warp 在 shared memory 中暂存一个 token 的 x/topk_weights，因此能启动的 warp 数
    // 由 num_smem_bytes / token_layout_bytes 决定，最多 32 个 warp。hidden 越大，单 SM
    // 可并发处理的 token 越少，这是 combine 性能模型里的主要约束。
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
    // 计算 combine 通信 buffer 中一个 token 的布局大小。
    auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);
    // 初始 warp 数由 shared memory 容量决定，最多 32 个 warp。

    // hybrid combine 中每个 SM 的 warp 被一分为二：scaleup_warps 负责从本节点专家结果
    // 读出并归并，forward_warps 负责把跨 scaleout 的结果继续转发到目标节点。
    int num_scaleup_warps = 0, num_forward_warps = 0;
    // 单层 combine 不使用这两个值，保持 0；hybrid 下会设置成 channel/SM。
    if (num_scaleout_ranks > 1) {
        // hybrid combine 需要和 dispatch 使用相同 channel/SM 配置，保证 channel_linked_list 对齐。
        EP_HOST_ASSERT(num_channels % num_sms == 0 and
                       "Invalid number of channels or SMs, you may use a different SM count than dispatch");
        // channel 数必须能均匀分到每个 SM。
        EP_HOST_ASSERT(num_channels / num_sms <= 16);
        // 每 SM channel 数限制为 16，避免 block warp 数/共享内存超出实现假设。

        num_scaleup_warps = num_forward_warps = num_channels / num_sms;
        // 一条 scaleup 规约通道配一条 forward 通道。
        num_warps = num_scaleup_warps + num_forward_warps;
        // hybrid 下总 warp 数由两类通道相加得到。
        EP_HOST_ASSERT(num_warps * token_layout.get_num_bytes<true>() <= num_smem_bytes and
                       "Invalid combine SM count, please try to match your dispatch config");
        // 检查 shared memory 是否足够容纳所有 warp 的 token staging。
    }

    // cluster_dim 使用 1 或 2：当 num_sms 为奇数时取 1，偶数时取 2，用来和 clustered
    // GEMM/计算 kernel 更好地重叠，避免通信 kernel 独占所有 cluster 调度资源。
    const auto num_threads = num_warps * 32;
    // blockDim.x = warp 数 * 32。
    const CombineRuntime::Args args = {
        // 聚合 combine 的模板参数和运行时参数。
        .is_scaleup_nvlink = is_scaleup_nvlink,
        // scaleup 是否 NVLink。
        .use_expanded_layout = use_expanded_layout,
        // 是否使用展开 expert layout。
        .allow_multiple_reduction = allow_multiple_reduction,
        // 是否允许多路贡献同 token。
        .num_scaleup_warps = num_scaleup_warps, .num_forward_warps = num_forward_warps,
        // hybrid warp 划分。
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        // 逻辑拓扑。
        .hidden = hidden,
        // hidden 维度。
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        // 每 rank 最大 token。
        .num_experts = num_experts,
        // expert 数。
        .num_topk = num_topk,
        // top-k 数。
        .num_qps = num_qps, .num_timeout_cycles = num_timeout_cycles,
        // QP 数和超时。
        .x = static_cast<nv_bfloat16*>(x),
        // 专家输出，转换为 BF16 指针。
        .topk_weights = static_cast<float*>(topk_weights),
        // top-k 权重。
        .src_metadata = src_metadata,
        // 源 token metadata。
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        // rank prefix sum。
        // 中文：hybrid combine 需要 forward metadata 和 channel 链表来追踪跨 scaleout token。
        .token_metadata_at_forward = token_metadata_at_forward,
        // hybrid forward metadata。
        .channel_linked_list = channel_linked_list,
        // channel 链表。
        .nccl_dev_comm = nccl_dev_comm, .nccl_window = nccl_window,
        // NCCL device communicator 和 symmetric window。
        .buffer = buffer, .workspace = workspace,
        // 通信 buffer 和 workspace。
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        // 当前 rank 逻辑坐标。
        .num_reduced_tokens = num_reduced_tokens,
        // 本次 reduce token 数。
        // NOTES: make cluster dim 2 to overlap with clustered computation kernels
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 2 - (num_sms % 2), true)
        // cooperative=true 支持 grid sync；偶数 num_sms 时 cluster_dim=2。
    };
    const auto code = CombineRuntime::generate(args);
    // 生成带 include hash 的源码。
    const auto runtime = jit::compiler->build("combine", code);
    // 编译或复用 combine cubin。
    CombineRuntime::launch(runtime, args, stream);
    // 启动 combine kernel。

    // 返回后续 reduce epilogue 应该读取的 buffer 起始地址。单节点模式 reduce_buffer 就是
    // buffer 本身；hybrid 模式下 buffer 前半段是 scale-up 中间区，需要跳过后再读最终归并区。
    if (num_scaleout_ranks == 1)
        // 单层路径没有额外 scaleup 中间区，reduce epilogue 直接读 buffer。
        return buffer;

    // 如果允许一个 token 被多个 scale-up rank 同时贡献，scaleup_buffer 使用 rank layout；
    // 否则使用 topk layout。这个选择必须和 hybrid_combine.cuh 写入布局保持一致。
    // 中文：rank layout 方便同一 token 多 rank 聚合，topk layout 方便按 top-k 槽逐个转发。
    const bool is_scaleup_buffer_rank_layout =
        allow_multiple_reduction ? (num_scaleup_ranks <= num_topk) : false;
    // 多路规约时如果 scaleup rank 数不超过 top-k，按 rank 布局更紧凑/直接。
    const auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, 
        // 中文：第二维根据 rank-layout 或 topk-layout 切换，决定 reduce buffer 的索引方式。
        is_scaleup_buffer_rank_layout ? num_scaleup_ranks : num_topk,
        num_scaleout_ranks * num_max_tokens_per_rank,
        buffer);
    // 构造 scaleup 中间 buffer layout，用它计算最终 reduce buffer 的起始位置。
    return scaleup_buffer.get_buffer_end_ptr();
    // hybrid 最终 reduce buffer 紧跟 scaleup_buffer 之后。
}

class CombineReduceEpilogueRuntime final : public jit::LaunchRuntime<CombineReduceEpilogueRuntime> {
    // reduce epilogue 把 combine 通信 buffer 中的多路贡献最终规约成 combined_x/topk 输出。
public:
    struct Args {
        // epilogue 模板参数描述拓扑和输出布局。
        // Templated arguments
        bool use_expanded_layout, allow_multiple_reduction;  // layout/规约策略需与主 combine 一致。
        // 与 combine 主 kernel 保持一致的布局/多路规约开关。
        int num_scaleout_ranks, num_scaleup_ranks;  // 逻辑二维拓扑。
        // 逻辑二维拓扑。
        int hidden;  // 输出 hidden 元素数。
        // hidden 维度。
        int num_max_tokens_per_rank;  // 每 rank 容量上界。
        // 每 rank 最大 token。
        int num_experts, num_topk;  // expert/top-k 维度。
        // expert/top-k 数。

        // Parameters
        nv_bfloat16* combined_x;  // 最终 combine 输出 payload。
        // 最终输出 token，逻辑形状 [num_combined_tokens, hidden]。
        float* combined_topk_weights;  // 最终输出权重，可为空。
        // 可选输出合并后的 top-k 权重。
        topk_idx_t* combined_topk_idx;  // 最终输出 top-k index，可为空。
        // 可选输出合并后的 top-k expert id。
        void* reduce_buffer;  // 主 combine 返回的 reduce buffer 起点。
        // combine 主 kernel 返回的 reduce buffer 起点。
        void* bias_0;  // 可选 bias 输入 0。
        // 可选 bias 0，epilogue 内按实现语义加到输出。
        void* bias_1;  // 可选 bias 输入 1。
        // 可选 bias 1。
        int num_combined_tokens;  // 需要写出的最终 token 数。
        // 需要写出的 token 数。
        int scaleout_rank_idx, scaleup_rank_idx;  // 当前 rank 二维坐标。
        // 当前 rank 逻辑坐标。

        jit::LaunchArgs launch_args;  // epilogue launch 配置。
        // launch 配置。
    };

    static std::string generate_impl(const Args& args) {
        // 生成 combine_reduce_epilogue_impl 模板实例。
        return fmt::format(R"(
#include <deep_ep/impls/combine_reduce_epilogue.cuh>

using namespace deep_ep::elastic;

// JIT 生成文件：实例化 reduce epilogue，把临时规约结果写回 combined_x。
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
	)",                        args.use_expanded_layout, args.allow_multiple_reduction,  // expanded 与多路规约开关。
	                           args.launch_args.grid_dim.first,  // kNumSMs。
	                           args.launch_args.num_threads / 32,  // 每 block warp 数。
	                           args.num_scaleout_ranks, args.num_scaleup_ranks,  // 二维拓扑。
	                           args.hidden,  // hidden 元素数。
	                           args.num_max_tokens_per_rank,  // 每 rank 容量。
	                           args.num_experts, args.num_topk);  // expert/top-k 形状。
        // grid.x/warp 数/拓扑/layout 参数都进入模板，device 端 offset 静态化。
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        // 按 epilogue kernel 声明顺序传参。
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.combined_x,  // 最终输出 tensor [num_combined_tokens, hidden]。
                                                 args.combined_topk_weights,  // 可选输出 top-k 权重。
                                                 args.combined_topk_idx,  // 可选输出 top-k expert id。
                                                 args.reduce_buffer,  // combine 主 kernel 生成的 reduce buffer。
                                                 args.bias_0, args.bias_1,  // 可选 bias，epilogue 在写 combined_x 前叠加。
                                                 args.num_combined_tokens,  // 输出 token 数。
                                                 args.scaleout_rank_idx, args.scaleup_rank_idx));  // 当前 rank 二维坐标。
        // epilogue 完成后 combined_x/combined_topk_* 对后续计算可见。
    }
};

static void launch_combine_reduce_epilogue(void* combined_x,  // 最终输出 BF16 tensor [num_combined_tokens, hidden]。
                                           float* combined_topk_weights,  // 可选输出权重 [num_combined_tokens, num_topk]。
                                           topk_idx_t* combined_topk_idx,  // 可选输出 top-k expert id。
                                           const int& num_combined_tokens, const int& num_max_tokens_per_rank,  // 输出 token 数和容量。
                                           const int& hidden,  // hidden 元素数。
                                           const int& num_experts, const int& num_topk,  // expert/top-k 维度。
                                           void* reduce_buffer,  // combine 主 kernel 生成的临时规约 buffer。
                                           void* bias_0, void* bias_1,  // 可选 bias 输入。
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,  // 二维拓扑大小。
                                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,  // 当前 rank 坐标。
                                           const int& num_sms, const int& num_smem_bytes,  // epilogue 使用 SM 数和 shared memory。
                                           const bool& use_expanded_layout, const bool& allow_multiple_reduction,  // layout/规约开关需与主 combine 一致。
                                           const at::cuda::CUDAStream& stream) {  // CUDA launch stream。
    // Maximize shared memory utilization
    // Too many warps may cause performance degrade, so we limit into 1024
    // 中文：这里把 shared memory 尽量切成 token staging，但限制总 warp 防止调度开销过大。
    const auto token_layout = layout::TokenLayout(hidden * sizeof(nv_bfloat16), 0, 0, false);
    // reduce epilogue 只需要 hidden payload，不再携带 top-k metadata。
    const auto num_warps = std::min<int>(num_smem_bytes / token_layout.get_num_bytes<false>(), 32);
    // 根据 shared memory 能容纳多少 token staging 选择 warp 数，最多 32。
    const auto num_threads = num_warps * 32;
    // blockDim.x。

    // Generate, build and launch
    const CombineReduceEpilogueRuntime::Args args = {
        // 聚合 reduce epilogue 参数。
        .use_expanded_layout = use_expanded_layout,
        // 是否使用展开布局。
        .allow_multiple_reduction = allow_multiple_reduction,
        // 是否允许多路规约。
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        // 逻辑拓扑。
        .hidden = hidden,
        // hidden 维度。
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        // 每 rank 最大 token。
        .num_experts = num_experts, .num_topk = num_topk,
        // expert/top-k 数。
        .combined_x = static_cast<nv_bfloat16*>(combined_x),
        // 最终输出 BF16 指针。
        .combined_topk_weights = combined_topk_weights,
        // 输出 top-k 权重。
        .combined_topk_idx = combined_topk_idx,
        // 输出 top-k index。
        .reduce_buffer = reduce_buffer,
        // 输入 reduce buffer。
        .bias_0 = bias_0, .bias_1 = bias_1,
        // 可选 bias 指针。
        .num_combined_tokens = num_combined_tokens,
        // 输出 token 数。
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        // 当前 rank 坐标。
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, false, true)
        // epilogue 不需要 cooperative，启用 PDL 以减少 kernel 间调度空泡。
    };
    const auto code = CombineReduceEpilogueRuntime::generate(args);
    // 生成源码。
    const auto runtime = jit::compiler->build("combine_reduce_epilogue", code);
    // 编译或复用 epilogue cubin。
    CombineReduceEpilogueRuntime::launch(runtime, args, stream);
    // 启动 reduce epilogue。
}

}  // namespace deep_ep::elastic
// Elastic combine wrapper 命名空间结束。
