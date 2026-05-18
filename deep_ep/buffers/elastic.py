import os  # 读取调试开关、RDMA service level 覆盖等环境变量。
import math  # 计算 buffer size hint、PP/Engram 字节数时使用乘积和对齐。
import torch  # Python API 输入输出均为 CUDA Tensor，并使用 torch.Stream/Event。
import torch.distributed as dist  # ElasticBuffer 基于 PyTorch ProcessGroup 建立 NCCL communicator。
from typing import Callable, Optional, Tuple, Union, List, Sequence  # 标注 dispatch/combine/PP/AGRS API 的复杂返回类型。
from contextlib import contextmanager  # agrs_new_session 使用 contextmanager 封装 session 生命周期。

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # C++/CUDA 扩展模块，ElasticBuffer 的 runtime 和 kernel 调用在这里。
# noinspection PyUnresolvedReferences
from deep_ep._C import EventHandle  # C++ CUDA event 句柄，用于异步 overlap。

from ..utils.event import EventOverlap  # Python 侧事件包装，支持等待/回调。
from ..utils.math import ceil_div, align  # 形状计算：scale-factor 列数、expert token 对齐。
from ..utils.semantic import value_or, weak_lru  # 默认值 helper 与弱引用 LRU 缓存。
from ..utils.envs import (  # 导入 Elastic 初始化和调度前需要的环境/拓扑检查 helper。
    check_fast_rdma_atomic_support,  # 自动选择 QP 数时判断网卡 atomic 能力。
    check_nvlink_connections, check_torch_deterministic,  # 初始化/dispatch 前的环境安全检查。
    get_nvlink_gbs, get_rdma_gbs  # 性能日志或 size hint 估算时可用的带宽探测。
)  # envs helper 分组导入。
from ..utils.comm import get_nccl_comm_handle  # 将 torch ProcessGroup 转成 C++ 可用 NCCL communicator handle。


class EPHandle:  # dispatch 返回并传给 combine 的路由元数据句柄。
    """
    Communication handle returned by `ElasticBuffer.dispatch`.
    中文：EPHandle 是 Elastic 路径中 dispatch/combine 之间最重要的 Python 侧契约对象。
    Can be reused as a cached handle in subsequent `ElasticBuffer.dispatch` calls to skip layout recomputation,
    and is consumed by `ElasticBuffer.combine` to reverse the token routing.
    中文：它不是通信资源本身，而是一组 Python/CUDA 元数据，用来把 dispatch 的布局传给 combine。

    Attributes:
        中文速览：EPHandle 保存 dispatch 到 combine 的完整路由契约，cached 模式也复用它。
        do_expand: whether the expanding (one-token-per-expert-slot) layout is used.
        num_experts: the number of all experts.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        中文：do_expand/num_experts/expert_alignment 三者共同决定本地 expert 输入区间如何切分。
        num_max_tokens_per_rank: the maximum number of tokens per rank, all the ranks must hold the same value.
        num_sms: the SM count used during dispatch (reused in combine).
        topk_idx: cloned top-k expert indices from dispatch, `[num_tokens, num_topk]`.
        中文：上述字段决定 expert layout、容量和 combine 复用的 SM 配置。
        psum_num_recv_tokens_per_scaleup_rank: inclusive prefix sum of deduplicated received token counts
            per scaleup rank, shape `[num_scaleup_ranks]`. A token is counted once per rank even if
            中文：这里的“去重”以目标 rank 为粒度；同一 token 命中同 rank 内多个 expert 仍只占一行 recv_x。
            multiple of its top-k experts land on the same rank. The last element equals the total number
            of received tokens.
        中文：rank prefix 最后一项是本 rank 接收 token 总数，用来分配 recv_x 行数。
        psum_num_recv_tokens_per_expert: prefix sum of alignment-padded received token counts per local
            expert, shape `[num_local_experts]`. Each expert's count is padded to `expert_alignment`.
            中文：这一行强调 prefix 已经按 expert_alignment 对齐，不能直接当真实 token 数。
            In non-expand mode, this is the inclusive prefix sum. In expand mode, `psum[i]` equals
            中文：非 expand 可直接用相邻 prefix 求区间；expand 需要额外考虑当前 expert 的真实未对齐计数。
            the aligned cumulative count of experts before `i` plus the actual (unaligned) token count
            中文：例如 expert_alignment=16 且前一 expert 真实 5 个 token，则下一 expert 起点会按 16 对齐。
            of expert `i` — so `psum[i] - align(psum[i-1], expert_alignment)` recovers the real
            count for expert `i`, and `align(psum[i], expert_alignment)` gives expert `i+1`'s
            starting offset.
        中文：expert prefix 的 expanded 语义较特殊，读取专家区间时必须按 expert_alignment 反推起点。
        num_recv_tokens_per_expert_list: Python list of per-expert received token counts (CPU-side).
        recv_src_metadata: source token indices and buffer slot indices.
        dst_buffer_slot_idx: destination buffer slot indices from dispatch.
        中文：这三个字段分别给 Python 统计、combine 源寻址和远端写入 slot 寻址使用。
        token_metadata_at_forward: per-channel forwarded token metadata (hybrid mode only).
        channel_linked_list: per-channel per-scaleup-peer linked list (hybrid mode only).
        num_recv_tokens: the total number of received tokens.
        中文：metadata/slot/linked-list 是 combine 反向寻址所需的关键数组，不能跨 batch 混用。

    中文字段说明：
        do_expand 表示 dispatch epilogue 是否把一个 token 的多个本地 expert 命中拆成多行；
        num_experts / expert_alignment 与 C++ 的 expert prefix-sum 逻辑绑定，决定专家输入区间；
        topk_idx 是 dispatch 时克隆的路由矩阵，形状为 [num_tokens, num_topk]，cached dispatch
            和 combine 都依赖它保持不变；
        psum_num_recv_tokens_per_scaleup_rank 是 scale-up rank 维 inclusive prefix，最后一个元素
            等于本 rank 总接收 token 数；
        psum_num_recv_tokens_per_expert 是本地 expert 维 prefix，expanded 和非 expanded 模式下的
            对齐语义不同，读取专家区间时必须配合 expert_alignment；
        recv_src_metadata / dst_buffer_slot_idx / token_metadata_at_forward / channel_linked_list
            共同描述反向 combine 的寻址关系，不能跨不同路由 batch 混用。
    """

    def __init__(self,  # 初始化 handle，保存 dispatch 产生的所有元数据。
                 do_expand: bool,  # 是否使用 expand layout；expand 会把 token 的 top-k expert 槽展开。
                 num_experts: int, expert_alignment: int,  # 全局 expert 总数，以及每个本地 expert token 数对齐粒度。
                 num_max_tokens_per_rank: int,  # 每 rank 最大 token 数，用于 metadata 编码和 handle 校验。
                 num_sms: int,  # dispatch 使用的 SM 数，combine 默认复用。
                 topk_idx: torch.Tensor,  # dispatch 时克隆的 top-k 路由矩阵 [num_tokens, num_topk]。
                 num_recv_tokens_per_expert_list: list,  # CPU list，每个本地 expert 的真实接收 token 数。
                 psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,  # scaleup rank 维度接收 token 前缀和 [num_scaleup_ranks]。
                 psum_num_recv_tokens_per_expert: torch.Tensor,  # 本地 expert 维度接收 token 前缀和 [num_local_experts]。
                 recv_src_metadata: torch.Tensor,  # combine 回写所需的源 token/top-k/buffer slot metadata。
                 dst_buffer_slot_idx: torch.Tensor,  # dispatch 写入远端 buffer slot 的索引。
                 token_metadata_at_forward: Optional[torch.Tensor],  # hybrid forward 额外 metadata，非 hybrid 时为 None。
                 channel_linked_list: Optional[torch.Tensor]):  # hybrid channel 链表，非 hybrid 时为 None。
        # EPHandle 是 dispatch 与 combine 之间的“路由契约”：
        # - Python 层保存 topk_idx、prefix-sum、slot index 等元数据；
        # - C++/CUDA combine 会根据这些元数据把 expert 计算后的 token 写回原 rank/原 token。
        # 一个典型例子：num_tokens=4096、num_topk=8、num_experts=256、num_ranks=32 时，
        # topk_idx 形状为 [4096, 8]，每个 expert 属于某个 rank。dispatch 之后 recv_src_metadata
        # 会记录“当前接收 token 来自哪个原始 token、哪个 top-k 槽、哪个 buffer slot”，combine 必须
        # 使用同一份 metadata；如果用户原地修改 topk_idx，就会导致 combine 写回错误位置。
        # 因此默认由 C++ 返回 cloned_topk_idx，避免用户后续修改输入张量破坏 handle 语义。
        # NOTES: remember to copy the original users' input to prevent uncasual modifications on them
        assert topk_idx is not None  # combine 必须依赖 dispatch 时克隆的 topk_idx。

        self.do_expand = do_expand  # 是否使用 expand layout：一个 token 的多个 expert 槽会展开成多行。
        self.num_experts = num_experts  # 全局 expert 总数。
        self.expert_alignment = expert_alignment  # 每个本地 expert 接收 token 数的对齐粒度。
        self.num_max_tokens_per_rank = num_max_tokens_per_rank  # 每 rank 最大 token 数，用于 src token id 编码。
        self.num_sms = num_sms  # dispatch 使用的 SM 数，combine 复用相同配置。
        self.topk_idx = topk_idx  # 克隆后的路由矩阵 [num_tokens,num_topk]。
        self.psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank  # scaleup rank 接收 token 前缀和 [num_scaleup_ranks]。
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert  # 本地 expert 接收 token 前缀和 [num_local_experts]。
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list  # CPU list 形式的 per-expert 接收计数。
        self.recv_src_metadata = recv_src_metadata  # 每个接收 token 的源信息，形状 [num_recv_tokens,...] 由 C++ layout 决定。
        self.dst_buffer_slot_idx = dst_buffer_slot_idx  # dispatch 写入目标 buffer slot 的索引。
        self.token_metadata_at_forward = token_metadata_at_forward  # hybrid forward metadata，可为 None。
        self.channel_linked_list = channel_linked_list  # hybrid channel 链表 metadata，可为 None。

        # recv_src_metadata 的第 0 维等于本 rank 接收到的 token 条数。若 dispatch 使用无 CPU sync
        # 或复用 cached handle，这个值是从已知 metadata 推断出的；精确的 per-expert count 仍以
        # num_recv_tokens_per_expert_list / prefix-sum 为准。
        self.num_recv_tokens = recv_src_metadata.shape[0]  # 当前 rank 实际接收到的 token 行数。


class ElasticBuffer:  # 新版弹性通信 buffer，统一承载 EP/Engram/PP/AGRS 功能。
    """
    The elastic communication buffer, which supports:
        - high-throughput expert-parallel all-to-all (dispatch and combine, using NVLink and/or RDMA)
        中文：这是 MoE token 路由主路径，dispatch 发给专家，combine 把专家输出送回原 token。
        - Engram (remote KV cache fetch, using RDMA)
        中文：EP 和 Engram 都可能使用 RDMA，但前者按 token 路由，后者按存储 entry 拉取。
        - pipeline-parallel send/recv (PP, using NVLink)
        - all-gather reduce-scatter (AGRS, using NVLink)
    中文：ElasticBuffer 是新版统一通信入口，同一块 symmetric buffer 支撑 EP、Engram、PP 和 AGRS。
    "Elastic" refers to the flexibility of underlying memory: currently GPU-only, with CPU and mixed
        (GPU+CPU) backends on the roadmap

    Attributes:
        中文速览：下面字段描述 ElasticBuffer 的通信域、逻辑拓扑和 C++ runtime 生命周期。
        group: the communication group.
        rank_idx: the rank index.
        num_ranks: the number of ranks in the group.
        中文：group/rank_idx/num_ranks 与 torch.distributed ProcessGroup 一一对应。
        allow_hybrid_mode: whether to enable hybrid mode for multi-node communication. Hybrid mode uses
            hierarchical RDMA + NVLink communication to achieve higher bandwidth, and is more friendly
            to multi-plane/multi-rail networks.
        中文：hybrid 模式把跨节点 RDMA 与节点内 NVLink 分层，适合多机多卡。
        allow_multiple_reduction: whether to allow multiple reductions in combine. If disabled,
            only one reduction will be done in the combine epilogue for best precision, 
            but it may increase data transfer size.
        中文：allow_multiple_reduction 在通信量和规约精度之间取舍。
        prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            If enabled, we tend to use fewer SMs.
        中文：减少通信 SM 可给专家计算让出资源，但可能降低纯通信吞吐。
        num_bytes: the total buffer size in bytes.
        num_max_tokens_per_rank: the default maximum tokens per rank.
        中文：prefer_overlap_with_compute 会影响默认 SM 数；num_bytes/num_max_tokens_per_rank 决定容量。
        num_scaleout_ranks: the number of scaleout ranks.
        num_scaleup_ranks: the number of scaleup ranks.
        中文：scaleout 通常对应跨节点域，scaleup 通常对应节点内高速互联域。
        scaleout_rank_idx: the scaleout rank index of this rank.
        scaleup_rank_idx: the scaleup rank index of this rank.
        中文：scaleout/scaleup 是 kernel 使用的二维逻辑坐标，不一定等同物理 rank 编号。
        num_rdma_ranks: the number of physical RDMA ranks.
        num_nvlink_ranks: the number of physical NVLink ranks.
        中文：这两个字段用于对照硬件拓扑，排查 logical domain 是否符合预期。
        runtime: the C++ runtime.
        中文：num_rdma_ranks/num_nvlink_ranks 是硬件域诊断字段，runtime 是 pybind C++ 对象。

    中文字段说明：
        group/rank_idx/num_ranks 定义 Elastic EP 通信域；
        num_scaleout_ranks/num_scaleup_ranks 是 DeepEP kernel 采用的逻辑二维拓扑，hybrid 模式
            通常对应 RDMA 维和 NVLink 维；
        num_rdma_ranks/num_nvlink_ranks 是 NCCL/硬件看到的物理域，用于诊断逻辑域是否合理；
        num_bytes 是 NCCL symmetric window 中为 payload、workspace、PP/Engram/AGRS 等功能预留的总容量；
        runtime 是 pybind C++ 对象，持有 NCCL communicator、window、device communicator 和通信 stream，
            Python 方法大多只是检查参数并把 Tensor 指针转交给它。
    """

    def __init__(self,  # 构造 ElasticBuffer 并初始化 C++ runtime。
                 group: dist.ProcessGroup,  # PyTorch distributed ProcessGroup，定义 EP rank 集合。
                 # Provide `num_bytes`
                 num_bytes: Optional[int] = None,  # 显式 buffer 字节数；传 None 时按 MoE shape 自动估算。
                 # Or provide MoE settings (BF16 by default)
                 num_max_tokens_per_rank: int = 0,  # 每 rank 最大 token 数，用于 size hint 和 dispatch 默认值。
                 hidden: int = 0,  # token hidden 维度；BF16 路径每 token payload 字节为 hidden*2。
                 num_topk: int = 0,  # 每 token top-k expert 数；0 表示按 rank 数保守估算。
                 use_fp8_dispatch: bool = False,  # 是否按 FP8 dispatch 估算 buffer，FP8 会额外存 scale factor。
                 # Configs
                 deterministic: bool = False,  # 是否使用确定性路由/拷贝算法；会影响 kernel 路径和性能。
                 allow_hybrid_mode: bool = True,  # 是否允许 RDMA+NVLink 分层 hybrid 通信。
                 allow_multiple_reduction: bool = True,  # combine 是否允许多次归约以减少传输。
                 prefer_overlap_with_compute: bool = True,  # 是否偏向少占 SM，给专家 GEMM 留计算资源。
                 sl_idx: int = 3,  # RDMA service level index，可被 EP_OVERRIDE_RDMA_SL 覆盖。
                 num_allocated_qps: int = 0,  # RDMA/Gin QP 数；0 表示按硬件能力自动选择。
                 num_cpu_timeout_secs: int = 300, num_gpu_timeout_secs: int = 100,  # CPU/GPU 两类超时阈值。
                 explicitly_destroy: bool = False):  # 是否要求用户显式 destroy，避免析构期释放资源。
        """
        Initialize the elastic communication buffer.

        Arguments:
            中文：如果直接传 num_bytes，就跳过 MoE shape 估算；否则需要给 token/hidden/topk 等形状。
            group: the communication group.
            num_bytes: the total buffer size in bytes, if set, overrides MoE-based calculation.
            num_max_tokens_per_rank: the maximum number of tokens per rank, used for buffer size calculation.
            中文：num_max_tokens_per_rank 是容量上界，不一定等于每次 dispatch 的真实 token 数。
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            中文：hidden/num_topk/num_max_tokens_per_rank 会传给 C++ size calculator，估算 dispatch/combine 所需字节。
            use_fp8_dispatch: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            deterministic: whether to use deterministic routing algorithms.
            allow_hybrid_mode: whether to enable hybrid mode.
            中文：deterministic 影响路由稳定性，hybrid 影响跨节点是否走分层 RDMA+NVLink。
            allow_multiple_reduction: whether to allow multiple reductions in combine.
            prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            中文：deterministic/hybrid/reduction/overlap 这些开关会改变 JIT kernel 模板实例和 SM 分配。
            sl_idx: the RDMA service level index, can be overridden by `EP_OVERRIDE_RDMA_SL` env var.
            num_allocated_qps: the number of QPs to allocate for RDMA (0 for automatic).
            num_cpu_timeout_secs: CPU-side timeout in seconds for CPU sync.
            中文：CPU timeout 主要覆盖 host 侧同步/初始化等待，不等同于 GPU kernel 内 timeout。
            num_gpu_timeout_secs: GPU-side timeout in seconds for GPU operations.
            中文：RDMA SL、QP 数和超时配置最终进入 C++ runtime，用于通信资源初始化和故障诊断。
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
            中文：长生命周期服务建议显式 destroy，避免 Python 析构时机和 CUDA/NCCL 资源释放交错。
        """
        # Some useful utilities
        self.group = group  # 保存 PyTorch ProcessGroup。
        self.rank_idx = group.rank()  # 当前 EP rank。
        self.num_ranks = group.size()  # EP world size。
        self.allow_hybrid_mode = allow_hybrid_mode  # 是否允许 RDMA+NVLink 分层通信。
        self.allow_multiple_reduction = allow_multiple_reduction  # combine 是否允许多阶段归约。
        self.prefer_overlap_with_compute = prefer_overlap_with_compute  # 是否倾向少用 SM 以给计算留资源。
        self.nccl_comm_handle = get_nccl_comm_handle(group)  # C++ 可消费的 NCCL communicator handle。

        # Calculate buffer size
        # 如果调用者没有显式给 num_bytes，就用 MoE 配置估算通信 buffer 的最坏需求。
        # 这里的 hidden/num_topk/num_max_tokens_per_rank 与 C++ 的
        # csrc/elastic/buffer.hpp::calculate_buffer_size 联动：C++ 会同时估算 dispatch layout
        # 和 combine layout 所需字节数，并取较大值。FP8 dispatch 会额外考虑 scale factor。
        if num_bytes is None:  # 未显式给 buffer size 时按 MoE 配置自动估算。
            # NOTES: we allow `num_topk == 0`, as the buffer size can also be calculated by number of ranks (maybe bigger though)
            num_bytes = _C.calculate_elastic_buffer_size(  # 调用 C++ size calculator，与 kernel 内 layout 保持一致。
                self.nccl_comm_handle.get(),  # NCCL communicator handle，用于查询拓扑/域划分。
                num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,  # MoE shape 与 FP8 配置。
                allow_hybrid_mode, allow_multiple_reduction)  # 通信/归约模式影响 metadata 和 buffer 需求。
        if os.environ.get('EP_BUFFER_DEBUG', 0):  # 调试开关开启时打印每 rank buffer 大小。
            print(f'Initializing EP elastic buffer with {num_bytes} bytes at rank EP {group.rank()}/{group.size()}')  # 帮助排查 size hint。
        self.num_bytes = num_bytes  # 保存最终 buffer 字节数。

        # Store default values
        self.num_max_tokens_per_rank = num_max_tokens_per_rank  # 保存默认 max tokens，dispatch 未显式传入时可复用。

        # Check PCIe GPUs
        check_nvlink_connections(group)  # 初始化前检查本地 NVLink 连接是否满足假设。

        # RDMA SL
        if 'EP_OVERRIDE_RDMA_SL' in os.environ:  # 用户可以通过环境变量覆盖 RDMA service level。
            sl_idx = int(os.environ['EP_OVERRIDE_RDMA_SL'])  # 将字符串 SL 转成 int。

        # Automatic maximum QP count allowed
        # QP(queue pair) 是 RDMA/Gin 发送请求使用的通道资源。Hybrid 模式下每个 SM/channel
        # 需要更细粒度的 QP 分配来减少争用，所以预留数量明显多于 direct 模式。
        # 65/129 的差异与网卡是否支持 fast RDMA atomic 有关；notify warp 还会额外占一个 QP。
        # TODO(tianr22): revise the QP count in consideration of Engram
        if num_allocated_qps == 0:  # 0 表示自动选择 QP 数。
            # Hybrid mode will consume more QPs
            # The extra QP is for notify warps
            if self.allow_hybrid_mode:  # hybrid 模式需要更多 QP 支撑多 channel/notify。
                num_allocated_qps = 65 if check_fast_rdma_atomic_support() else 129  # fast atomic 可用时 QP 需求更低。
            else:  # direct 模式 QP 需求较小。
                num_allocated_qps = 17  # 16 个数据 QP + 1 个 notify/控制 QP。
        self.num_allocated_qps = num_allocated_qps  # 保存实际分配 QP 数，测试会打印该值。

        # Create CPP handle
        self.explicitly_destroy = explicitly_destroy  # 保存析构策略；为 True 时用户必须显式 destroy。
        self.runtime = _C.ElasticBuffer(group.rank(), group.size(),  # 创建 C++ ElasticBuffer runtime，传入 rank/world。
                                        self.nccl_comm_handle.get(),  # NCCL communicator handle，C++ 用于拓扑和窗口注册。
                                        num_bytes,  # symmetric/workspace buffer 总字节数。
                                        deterministic,  # 是否使用确定性路由算法。
                                        allow_hybrid_mode,  # 是否允许 hybrid RDMA+NVLink 路径。
                                        allow_multiple_reduction,  # combine 是否允许多阶段 reduction。
                                        prefer_overlap_with_compute,  # 是否倾向减少通信 SM 占用。
                                        sl_idx, num_allocated_qps,  # RDMA service level 与 QP 数。
                                        num_cpu_timeout_secs, num_gpu_timeout_secs,  # CPU/GPU 侧超时时间。
                                        self.explicitly_destroy)  # C++ runtime 是否采用显式销毁策略。

        # Logical rank indices
        # 逻辑域把所有 EP rank 拆成 scaleout 与 scaleup 两层：
        # - scaleout：跨节点/跨 RDMA 域；
        # - scaleup：节点内或 NVLink 域。
        # 例如 EP=16 且每节点 8 卡时，通常 num_scaleout_ranks=2、num_scaleup_ranks=8。
        # dispatch/combine 的 hybrid kernel 会利用这个层次先走 RDMA，再走 NVLink 转发。
        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()  # 逻辑域大小 (scaleout, scaleup)。
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks  # 当前 rank 在 scaleout 维度的坐标。
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks  # 当前 rank 在 scaleup 维度的坐标。

        # Physical rank indices
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()  # 物理域大小 (RDMA ranks, NVLink ranks)。

        # C++ 构造函数会注册 NCCL symmetric memory/window，并把 workspace 指针映射给 GPU。
        # 这里做 device sync + group barrier + device sync，确保所有 rank 的 window 都完成创建，
        # 否则某个 rank 提前发起 Gin/NVLink 访问时，远端映射可能还不可见。
        torch.cuda.synchronize()  # 等本 rank C++ 初始化相关 GPU 工作完成。
        group.barrier()  # 所有 rank 在 CPU 侧同步，确保 symmetric window 全局就绪。
        torch.cuda.synchronize()  # barrier 后再同步一次，避免后续 kernel 看到未完成的初始化工作。

    def destroy(self) -> None:  # 显式释放 ElasticBuffer runtime。
        """
        Destroy the C++ runtime and release resources. Requires `explicitly_destroy=True` at construction.
        """
        assert self.explicitly_destroy  # 未启用显式销毁模式时不允许手动 destroy，防止双重释放。

        if self.runtime is not None:  # 允许 destroy 幂等地被上层保护调用。
            # C++ destroy 会做 GPU barrier、释放 mapped host workspace，并注销 NCCL window。
            # Python 侧随后把 runtime/nccl_comm_handle 置空，避免用户误复用已释放的句柄。
            self.runtime.destroy()  # 调用 C++ 销毁通信窗口、workspace、QP 等资源。
            self.runtime = None  # Cannot use anymore；清空 runtime 防止继续调用。
            self.nccl_comm_handle = None  # 清空 communicator handle 引用。

    @staticmethod
    # 中文：静态容量估算函数不依赖已有 ElasticBuffer 实例，便于用户先预分配足够的通信区。
    def get_buffer_size_hint(group: dist.ProcessGroup,
                             num_max_tokens_per_rank: int, hidden: int,
                             num_topk: int = 0, use_fp8_dispatch: bool = False,
                             # 中文：use_fp8_dispatch 会额外估算 scale-factor 存储空间。
                             # 中文：下面两个布尔开关只影响容量估算的模式选择，不会改变现有 group。
                             allow_hybrid_mode: bool = True,
                             # 中文：allow_multiple_reduction 会影响 combine 中间 buffer 的容量估计。
                             allow_multiple_reduction: bool = True) -> int:
        """
        Get a recommended buffer size (in bytes) for the given MoE settings, without constructing the buffer.

        Arguments:
            中文：该函数只做容量估算，不分配 symmetric buffer，也不会创建 ElasticBuffer。
            group: the communication group.
            num_max_tokens_per_rank: the maximum number of tokens per rank.
            hidden: the hidden dimension of each token.
            中文：hidden 以元素数计，不是字节数；C++ 会结合 dtype 推导 payload bytes。
            num_topk: the number of top-k experts per token.
            中文：num_max_tokens_per_rank/hidden/num_topk 决定 dispatch 和 combine payload 上限。
            use_fp8_dispatch: whether to use FP8 for dispatch.
            allow_hybrid_mode: whether to enable hybrid mode.
            allow_multiple_reduction: whether to allow multiple reductions in combine.
            中文：FP8、hybrid、multiple reduction 都会改变 buffer 分段大小。

        Returns:
            size: the recommended buffer size in bytes.
            中文：返回值单位为 byte，通常直接作为 ElasticBuffer(num_bytes=...) 的输入。
        """
        # 这个静态函数只需要现有 process group，不会创建 ElasticBuffer。它会通过
        # get_nccl_comm_handle(group) 复用/创建 NCCL communicator，并调用 C++ 侧
        # calculate_elastic_buffer_size 根据物理/逻辑拓扑估算字节数。
        return _C.calculate_elastic_buffer_size(  # 返回 C++ 估算出的最小推荐字节数。
            get_nccl_comm_handle(group).get(),  # C++ 需要 communicator 推断拓扑。
            num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,  # MoE shape 与 FP8 配置。
            allow_hybrid_mode, allow_multiple_reduction)  # 通信模式和归约模式影响 buffer layout。

    @staticmethod
    def get_engram_storage_size_hint(num_entries: int, hidden: int,
                                     num_max_tokens_per_rank: int,
                                     # 中文：dtype 影响每条 entry 的字节数以及是否需要 scale-factor pack。
                                     dtype: torch.dtype = torch.bfloat16) -> int:
        """
        (Experimental) Get a minimum buffer size requirement for Engram storage.

        Arguments:
            中文：Engram storage 预留远端可 fetch 的 entry 区和接收 token 的临时区。
            num_entries: the number of entries in the Engram storage.
            hidden: the hidden dimension of each entry.
            num_max_tokens_per_rank: the maximum number of tokens per rank (reserved for receive space).
            中文：额外预留的 token 空间用于 fetch 返回暂存，不属于持久 storage entry。
            dtype: the data type, defaults to `torch.bfloat16`.
            中文：num_entries*hidden*dtype_size 是主体容量；低精度 dtype 会额外估算 scale-factor pack。

        Returns:
            size: the recommended Engram storage size in bytes.
            中文：返回值单位为 byte，可直接作为 Engram 相关 buffer 规划的下界参考。
        """
        # Engram 把远端 KV/embedding-like 存储放入同一块 symmetric buffer。
        # 每条 entry 的数据形状为 [hidden]；若 dtype 是 FP8/更低位宽，还需要为每 32 个 hidden
        # 元素保存一个 scale factor pack。每条 entry 按 32B 对齐，是为了匹配 device 端 LDG.256
        # 访问粒度，减少非对齐访存带来的额外事务。
        # TODO: refactor all APIs to allow more parallelism
        # 中文：后续如果 API 支持更多并行 fetch，storage hint 可能需要额外预留请求队列空间。
        # TODO: consider FP4
        num_sf_packs = ceil_div(hidden, 32) if dtype.itemsize <= 1 else 0  # 低于/等于 1 字节 dtype 需要 scale-factor pack。
        # NOTES: we align per-entry size with 32 bytes (LDG.256)
        num_bytes_per_entry = align(hidden * dtype.itemsize + num_sf_packs * 4, 32)  # 单 entry 字节数按 32B 对齐。
        return num_bytes_per_entry * (num_entries + num_max_tokens_per_rank)  # storage entries 加上接收暂存 token entries。

    @staticmethod
    # 中文：PP 容量估算只需要单 tensor 最大字节数和并发槽数，不需要知道 rank 拓扑。
    def get_pp_buffer_size_hint(num_max_tensor_bytes: int,
                                # 中文：inflight 槽数越大，PP pipeline 可同时悬挂的 send/recv 越多。
                                num_max_inflight_tensors: int) -> int:
        """
        (Experimental) Get a minimum buffer size requirement for pipeline-parallel (PP) send/recv.

        Arguments:
            中文：PP buffer 只按 tensor 字节流估算，不关心 MoE expert/top-k 维度。
            num_max_tensor_bytes: the maximum tensor size in bytes per send/recv operation.
            num_max_inflight_tensors: the maximum number of in-flight tensors at once.
            中文：inflight 数越大，ring staging 槽位越多，适合 pipeline overlap 更深的模型。

        Returns:
            size: the recommended PP buffer size in bytes.
            中文：返回值包含 prev/next 两个方向，以及 send/recv 两类槽位的容量。
        """
        # PP 使用一块环形 staging buffer，不存任何 MoE 元数据。每个槽只存连续 tensor
        # 字节流；32B 对齐对应 device kernel 中 TMA/LDG.256 的最小友好粒度。
        # Align with `LDG.256`
        num_max_tensor_bytes = align(num_max_tensor_bytes, 32)  # 每条 tensor 消息向上对齐到 32B。

        # 乘以 2*2：两个邻居方向(prev/next)，每个方向又分 send staging 和 recv landing。
        return num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2  # 返回 PP 环形 buffer 总容量。

    @staticmethod
    # 中文：AGRS 的容量模型以 session 为单位，因此接口只暴露 session 最大字节数。
    def get_agrs_buffer_size_hint(group: dist.ProcessGroup,
                                  # 中文：session bytes 应覆盖一次 AGRS 会话内所有 gather 切片总和。
                                  num_max_session_bytes: int) -> int:
        """
        (Experimental) Get a minimum buffer size requirement for all-gather reduce-scatter (AGRS) sessions.

        Arguments:
            中文：AGRS 以 session 为单位，调用方需要把单次 session 内所有 tensor 的总字节数估进去。
            group: the communication group.
            num_max_session_bytes: the maximum total bytes of all gathered tensors in a single session.

        Returns:
            size: the recommended AGRS buffer size in bytes.
            中文：当前实现直接返回 session 容量，因为 AGRS runtime 不额外推导 rank 倍数。
        """
        # AGRS 的 session buffer 已经按“每个 tensor 的所有 rank 切片连续”组织，
        # 因此 hint 就是用户预估的 session 总容量；真正的 rank 倍数由用户传入的
        # num_max_session_bytes 自己包含。
        return num_max_session_bytes  # AGRS 不额外放大，调用方传入的 session bytes 已包含 world/inflight 需求。

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:  # 执行 ElasticBuffer 全局 barrier。
        """
        Perform a GPU-level barrier across all ranks, optionally with CPU synchronization.

        Arguments:
            中文：barrier 用于显式同步 ElasticBuffer 内部通信状态，常见于 benchmark 或资源切换点。
            use_comm_stream: whether to use the communication stream (otherwise uses the current compute stream).
            with_cpu_sync: whether to also call `cudaDeviceSynchronize` before and after the barrier.
        """
        # C++ barrier 可以放在通信 stream 或当前 stream 上；Python 默认使用 comm_stream，
        # 这样用户当前 stream 只需要等待最终 event，不必承载通信 kernel。
        self.runtime.barrier(use_comm_stream, with_cpu_sync)  # 委托 C++ runtime 在指定 stream 上发起 barrier。

    @staticmethod
    # 中文：拆 handle 时保留返回类型的逐项结构，方便 IDE 跳转到 EPHandle 字段定义。
    def _unpack_handle(handle: Optional[EPHandle] = None) \
        -> Tuple[Optional[int], Optional[list],
                 Optional[torch.Tensor], Optional[torch.Tensor],  # 中文：对应 rank/expert prefix 与源 metadata。
                 Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # C++ runtime.dispatch 支持两种模式：
        # - handle=None：重新计算布局、统计接收 token、生成 metadata；
        # - handle!=None：复用上一次 dispatch 的布局，只重新搬运新的 x payload。
        # 这个 helper 把 EPHandle 拆成 C++ 需要的 cached_* 参数，None 会逐个传空值。
        # 注意 cached 模式不需要重新传 topk_idx：topk_idx 已经被 handle 克隆保存，
        # C++ 只需要知道“上一轮每个 token/专家最终落到哪个 slot”，因此只拆 prefix 和 slot metadata。
        # 这和 combine 的依赖链一致：combine 也从 handle 中读取 recv_src_metadata/dst_buffer_slot_idx 来反向写回。
        if handle is None:  # 没有 cached handle 时所有 cached 参数都传 None。
            return None, None, None, None, None, None, None  # C++ 会重新计算 dispatch layout。
        return (handle.num_recv_tokens,  # cached 接收 token 总数。
                handle.num_recv_tokens_per_expert_list,  # cached 每 expert 接收计数。
                handle.psum_num_recv_tokens_per_scaleup_rank,  # cached scaleup rank 前缀和。
                handle.psum_num_recv_tokens_per_expert,  # cached expert 前缀和。
                handle.dst_buffer_slot_idx,  # cached 目标 buffer slot。
                handle.token_metadata_at_forward,  # cached hybrid forward metadata。
                handle.channel_linked_list)  # cached hybrid channel 链表。

    @staticmethod
    def capture() -> EventHandle:  # 捕获当前 CUDA stream 的事件。
        # 与 legacy.EventOverlap.capture 不同，这里直接返回 C++ EventHandle，外层 API 再包装 EventOverlap。
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event_handle: the captured event handle.
        """
        # EventHandle 保存当前 CUDA stream 上的事件，后续 dispatch/combine 的
        # previous_event 参数可让通信 stream 等待这个点，从而和用户计算安全衔接。
        # 典型用法：先在 compute stream 上准备 x/topk，再把 capture() 结果传给 dispatch。
        return EventHandle()  # 返回 C++ event handle，可作为 previous_event 传入通信 API。

    def get_comm_stream(self) -> torch.Stream:  # 获取 C++ runtime 的通信 stream。
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        ts: torch.Stream = self.runtime.get_comm_stream()  # 取出底层 stream 元数据。
        # C++ 返回 torch::Stream，Python 侧重新包装成 torch.cuda.Stream，保留 stream_id、
        # device_index 和 device_type，方便用户显式在该 stream 上排队工作。
        return torch.cuda.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)  # 重新构造成 Python CUDA stream。

    def get_physical_domain_size(self) -> Tuple[int, int]:  # 查询物理通信域大小。
        # 物理域由硬件拓扑决定，不受 allow_hybrid_mode 等软件策略直接改变。
        # 例如同一节点 8 卡 NVSwitch 可能返回 (1,8)，两节点各 8 卡通常返回 (2,8)。
        """
        Get the physical domain sizes (RDMA ranks and NVLink ranks).

        Returns:
            中文：物理域用于判断硬件拓扑，例如跨节点 RDMA 域数和每节点 NVLink rank 数。
            num_rdma_ranks: the number of physical RDMA ranks.
            num_nvlink_ranks: the number of physical NVLink ranks.
        """
        # physical domain 来自 NCCL/拓扑探测，通常等价于 RDMA 域数和每域 NVLink rank 数。
        return self.runtime.get_physical_domain_size()  # 返回 (num_rdma_ranks, num_nvlink_ranks)。

    def get_logical_domain_size(self) -> Tuple[int, int]:  # 查询逻辑通信域大小。
        # 逻辑域由 C++ runtime 按当前配置规划，hybrid 模式会倾向保留 scaleout/scaleup 两层；
        # direct 或特殊拓扑下可能把其中一层压成 1，使后续 kernel 跳过对应 barrier/转发。
        """
        Get the logical domain sizes (scaleout ranks and scaleup ranks).

        Returns:
            中文：逻辑域是 kernel 调度维度，可能因 hybrid/direct 策略与物理域不同。
            num_scaleout_ranks: the number of logical scaleout ranks.
            num_scaleup_ranks: the number of logical scaleup ranks.
        """
        # logical domain 是 Elastic kernel 规划后的 scaleout/scaleup 拆分，可能和物理域不同。
        return self.runtime.get_logical_domain_size()  # 返回 (num_scaleout_ranks, num_scaleup_ranks)。

    def engram_write(self, storage: torch.Tensor) -> None:  # 将本 rank Engram storage 写入 symmetric buffer。
        """
        (Experimental) Write Engram storage data into the buffer.
        This call includes a barrier before and after the write to ensure visibility.

        Arguments:
            中文：storage 是本 rank 持有的 Engram 条目表，所有 rank 的 hidden 维必须一致。
            storage: `[num_entries, hidden]` with `torch.bfloat16`, the Engram storage tensor.
        """
        # TODO: support FP8
        # 该调用会触发两次 barrier：写前等待旧 fetch，写后让所有 rank 的 storage 可见。
        # storage 的第 0 维是本 rank 持有的 entry 数，所有 rank 必须一致。
        # storage 逻辑形状为 [num_entries, hidden]，entry id 由 engram_fetch 的 indices 指定。
        self.runtime.engram_write(storage)  # C++ 负责 barrier、copy 和可见性同步。

    def engram_fetch(self, indices: torch.Tensor, num_qps: int = 0) -> Callable:  # 发起远端 Engram entry 读取。
        """
        (Experimental) Fetch Engram entries from remote ranks via RDMA.
        Returns a callable that, when invoked, waits for the RDMA gets to complete and returns the fetched tensor.

        Arguments:
            中文：indices 是待拉取 entry id 的一维列表，C++ 会映射到远端 rank 和 entry offset。
            indices: `[num_tokens]` with `torch.int`, the entry indices to fetch.
            num_qps: the number of QPs to use (0 for all allocated QPs).

        Returns:
            hook: a callable that blocks until data arrives and returns the fetched tensor
                with shape `[num_tokens, hidden]` and type `torch.bfloat16`.
            中文：hook() 的返回 tensor 行顺序与 indices 行顺序一致。
        """
        # 返回 hook 而不是直接同步，是为了让调用方可以先发起远端读取，再运行一段本地
        # 计算，最后在真正需要 fetched 时调用 hook() 等待。
        # indices 是全局 entry id，C++ 会根据 entry id 映射到远端 rank 和 rank 内 offset。
        return self.runtime.engram_fetch(indices, num_qps)  # 返回等待 hook，hook() 产出 [num_tokens, hidden]。

    def pp_set_config(self, num_max_tensor_bytes: int, num_max_inflight_tensors: int):  # 配置 PP send/recv 环形 buffer。
        """
        (Experimental) Configure pipeline-parallel (PP) send/recv parameters. Includes a barrier to flush previous operations.

        Arguments:
            中文：这两个参数必须覆盖后续所有 pp_send/pp_recv 的最大字节数和并发深度。
            num_max_tensor_bytes: the maximum tensor size in bytes per send/recv operation.
            num_max_inflight_tensors: the maximum number of in-flight tensors at once.
        """
        # 必须先配置再 pp_send/pp_recv。C++ 会把 max bytes 对齐到 32B，并检查 buffer
        # 是否足够容纳 prev/next 双方向、send/recv 双区域的 inflight 槽。
        # num_max_inflight_tensors 是 PP ring 中未完成 send/recv 的最大槽数，过小会阻塞流水。
        # 这个配置带 barrier，是因为它会重置 PP ring 的 head/tail/signal；若旧 send/recv 未完成，
        # 直接覆盖配置会让另一端还在使用的槽位被新 session 解释。
        self.runtime.pp_set_config(num_max_tensor_bytes, num_max_inflight_tensors)  # C++ 保存容量和槽位配置。

    def pp_send(self, t: torch.Tensor, dst_rank_idx: int, num_sms: int = 0) -> None:  # 向 PP 环形邻居发送 tensor。
        """
        (Experimental) Send a tensor to an adjacent rank in the PP ring (prev or next rank only).

        Arguments:
            中文：PP send/recv 是按字节流搬运，因此 t 必须 contiguous。
            t: the tensor to send, must be contiguous and fit within `num_max_tensor_bytes`.
            中文：发送端和接收端必须约定相同 nbytes，否则接收端会按自己的 tensor 大小解释字节流。
            dst_rank_idx: the destination rank index (must be prev or next rank in the ring).
            num_sms: the number of SMs to use (0 for all SMs).
        """
        # PP send 是 fire-and-forget：它把 kernel 排到当前 stream，内部通过 signal 等待
        # 目标 recv 槽可用。需要和其它工作同步时，用户应使用 CUDA stream/event 语义。
        # dst_rank_idx 只允许相邻 PP rank，C++ 会检查它是否等于 ring 上的 prev/next；
        # 这使 buffer 只需要维护两个方向的槽位，而不是 full all-to-all。
        self.runtime.pp_send(t, dst_rank_idx, num_sms)  # C++ kernel 把 t 写入目标方向的 ring buffer。

    def pp_recv(self, t: torch.Tensor, src_rank_idx: int, num_sms: int = 0) -> None:  # 从 PP 环形邻居接收 tensor。
        """
        (Experimental) Receive a tensor from an adjacent rank in the PP ring (prev or next rank only).

        Arguments:
            中文：接收 tensor 的 dtype/shape 只影响本地解释，通信层只检查 nbytes。
            t: the output tensor to receive into, must be contiguous and fit within `num_max_tensor_bytes`.
            中文：t 是预分配输出 buffer，PP recv 不会重新分配或返回新 tensor。
            src_rank_idx: the source rank index (must be prev or next rank in the ring).
            num_sms: the number of SMs to use (0 for all SMs).
        """
        # PP recv 会等待源 rank 对应 signal 到达，再把 ring buffer 槽拷贝到 t。
        # t 的 shape 不参与通信协议，只有 nbytes 参与容量检查。
        # 因此发送端和接收端必须在模型并行调度层保证 nbytes 一致；DeepEP 只按字节流搬运。
        self.runtime.pp_recv(t, src_rank_idx, num_sms)  # C++ kernel 等待 signal 并把数据写入 t。

    def create_agrs_session(self) -> None:  # 创建 AGRS session。
        """
        (Experimental) Begin a new all-gather reduce-scatter (AGRS) session. Must be paired with `destroy_agrs_session`.

        """
        # session 是 AGRS 的内存复用边界。session 内可以多次 all_gather，直到
        # destroy_agrs_session 统一通知 peer 这些槽位可释放。
        # 设计成显式 session 是为了支持一段计算中连续发起多个 all_gather，
        # 并在最后统一等待/释放，减少每次 gather 都全局同步的开销。
        self.runtime.create_agrs_session()  # C++ 初始化 session 内部 cursor/signal 状态。

    def destroy_agrs_session(self) -> None:  # 销毁当前 AGRS session。
        """
        (Experimental) End the current AGRS session. Waits for the compute stream, signals session completion to all peers.

        """
        # C++ 会写/等 peer 的 session signal。返回后，当前 session 的所有 buffer 槽
        # 都不会再被远端访问，可以开始下一轮 session。
        self.runtime.destroy_agrs_session()  # C++ 等待远端访问完成并释放 session 槽。

    @contextmanager
    def agrs_new_session(self, enabled: bool = True):  # AGRS session 的上下文管理器封装。
        """
        (Experimental) Context manager that wraps `create_agrs_session` and `destroy_agrs_session`.

        Arguments:
            enabled: if `False`, the context manager is a no-op.
        """
        if not enabled:  # 允许调用方临时关闭 session 管理逻辑。
            # enabled=False 用于 benchmark/调试时复用同一段代码路径，但不实际创建 session。
            yield  # 不创建 session，直接执行用户代码块。
            return  # 退出 no-op 分支，避免继续创建/销毁 session。

        self.runtime.create_agrs_session()  # 进入上下文时创建 AGRS session。
        try:  # 用 try/finally 确保 session 尽量被销毁。
            # yield 内部通常会调用 all_gather 或 agrs_get_inplace_tensor；finally 保证
            # 即使用户代码抛异常，也尽量结束 session，避免后续 buffer 状态卡住。
            yield  # 交还控制权给用户代码，通常会执行 all_gather。
        finally:  # 无论用户代码正常或异常退出都走这里。
            self.runtime.destroy_agrs_session()  # 结束 session 并释放/同步槽位。

    def agrs_set_config(self, num_max_session_bytes: int,  # 一个 session 内所有 all-gather 汇总后的最大总字节数。
                        num_max_all_gathers_per_session: int) -> None:  # 一个 session 内最多允许多少次 all-gather。
        """
        (Experimental) Configure AGRS session parameters. Includes a barrier to flush previous operations.

        Arguments:
            中文：两个配置值一起约束 AGRS session 内可复用的 staging 空间和操作次数。
            num_max_session_bytes: the maximum total bytes of gathered tensors per session.
            num_max_all_gathers_per_session: the maximum number of all-gather operations per session.
        """
        # AGRS 目前要求所有 tensor nbytes 32B 对齐；num_max_session_bytes 应至少等于
        # 一个 session 内所有 all_gather 输入字节 * num_ranks 的总和。
        self.runtime.agrs_set_config(num_max_session_bytes, num_max_all_gathers_per_session)  # C++ 保存 session 最大字节数和最大 all_gather 数。

    # noinspection PyTypeChecker
    def agrs_get_inplace_tensor(self,  # 从 AGRS buffer 申请本 rank inplace 输入 tensor。
                                shapes: Union[Tuple[int, ...], torch.Size, Sequence[Union[Tuple[int, ...], torch.Size]]],  # 单个 shape 或多个 shape。
                                dtype: torch.dtype) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:  # 目标 tensor dtype。
        """
        (Experimental) Get in-place tensor(s) from the AGRS buffer for this rank's slot, without copying.
        Must be called within an active AGRS session.

        Arguments:
            shapes: the shape(s) of tensor(s) to allocate. Pass a single shape tuple, or a sequence of shape tuples
                中文：batched mode 会在同一 rank slot 内顺序切分多个 tensor 视图，不做额外拷贝。
                for batched mode.
            dtype: the data type for the tensor(s).

        Returns:
            中文：返回 tensor 直接指向 AGRS symmetric buffer，写入它等价于写入本 rank 的 all-gather 输入槽。
            tensor: a single tensor if a single shape is given, or a tuple of tensors for batched mode.
        """
        is_batched_mode = isinstance(shapes[0], tuple)  # shapes[0] 是 tuple 时表示传入了多个 shape。
        if not is_batched_mode:  # 单 tensor 模式。
            shapes = (shapes, )  # 统一转成 tuple[shape]，便于后续循环。
        # C++ 返回的是 uint8 view，Python 再按 dtype 和 shape 解释。这样 C++ 不需要理解
        # 任意 PyTorch dtype/shape，只按字节数做 buffer 分配。
        tensors = self.runtime.agrs_get_inplace_tensor(  # C++ 按字节数返回一组 uint8 tensor view。
            (math.prod(shape) * dtype.itemsize for shape in shapes)  # 每个 shape 对应的字节数。
        )  # tensors 长度等于 shapes 长度。
        out = tuple(tensor.view(dtype).view(shape) for tensor, shape in zip(tensors, shapes, strict=True))  # 将 uint8 view 解释成目标 dtype/shape；zip(strict=True) 确保数量一致。
        return out if is_batched_mode else out[0]  # 批量输入返回 tuple，单输入返回单个 Tensor。

    def all_gather(self, t: Union[torch.Tensor, Sequence[torch.Tensor]]):  # 在当前 AGRS session 内执行 all_gather；t 可为单 Tensor 或 Tensor 序列。
        """
        (Experimental) Perform an all-gather operation within an active AGRS session.
        Each rank's data is gathered to all ranks via NVLink symmetric memory.

        Arguments:
            t: a single tensor or a sequence of tensors to all-gather. Each tensor must be contiguous and
                CUDA-allocated, with `nbytes` aligned to 32 bytes.

        Returns:
            For a single tensor: `(gathered, handle)` where `gathered` has an extra leading dimension of
                中文：gathered 的第 0 维是 rank 维，后续维度保持输入 tensor 原形状。
                `num_ranks`, and `handle` is a callable to wait for data arrival.
            For a sequence: `(*gathered_tensors, handle)` with one gathered tensor per input.
        """
        if isinstance(t, torch.Tensor):  # 单 Tensor API。
            # 单 tensor 模式保持 API 简洁：返回 gathered, handle；C++ 内部仍统一按
            # vector<torch::Tensor> 处理。
            tensors, handle = self.runtime.all_gather((t,))  # C++ 统一按 tuple/list 处理，返回 outputs 和 handle。
            return tensors[0], handle  # 单 Tensor 模式拆出第一个输出。

        # Batched
        # batched 模式返回 (*gathered_tensors, handle)，调用者通常先拿到多个 view，
        # 再统一调用 handle() 等待所有 peer 数据到齐。
        tensors, handle = self.runtime.all_gather(t)  # 批量 all_gather，输入是 Sequence[Tensor]。
        return *tensors, handle  # Python 返回多个 gathered tensor 加最后一个等待 handle。

    @weak_lru(maxsize=None)
    def get_theoretical_num_sms(self, num_experts: int, num_topk: int,  # expert 总数和每 token top-k 数，用于估计触达 rank 数。
                                num_scaleout_topk: int = 0,  # 预留给 group-limited gate；当前必须为 0。
                                rdma_gbs: float = 0, nvlink_gbs: float = 0,  # 手动指定 RDMA/NVLink GB/s；0 表示自动探测。
                                # TODO: use different values for other architectures
                                sm_read_gbs: float = 200, sm_write_gbs: float = 50) -> int:  # 单 SM HBM 读/写带宽估计。
        """
        Estimate the optimal number of SMs for dispatch/combine kernels based on bandwidth modeling.
        The result is cached. This assumes a balanced gate distribution.

        Arguments:
            中文：这个模型用通信带宽和单 SM HBM 读写带宽估算 dispatch/combine 应占用多少 SM。
            num_experts: the number of all experts.
            num_topk: the number of top-k experts per token.
            num_scaleout_topk: reserved for balanced gate (must be 0 currently).
            中文：当前实现假设 balanced gate，num_scaleout_topk 只是预留参数。
            rdma_gbs: the RDMA bandwidth in GB/s (0 for auto-detect).
            nvlink_gbs: the NVLink bandwidth in GB/s (0 for auto-detect).
            中文：传 0 时会调用环境探测函数读取硬件理论带宽并套用经验折扣。
            sm_read_gbs: the per-SM HBM read bandwidth in GB/s.
            sm_write_gbs: the per-SM HBM write bandwidth in GB/s.
            中文：RDMA/NVLink 带宽决定通信瓶颈，sm_read/write 决定 kernel 能否喂满链路。

        Returns:
            num_sms: the recommended SM count (even, at least 4).
            中文：返回值会向偶数和最小 4 SM 对齐，便于 kernel 成对组织 channel。
        """
        # TODO: support `do_expand` and `allow_multiple_reduction`

        # The `1` in this function means scale-up traffic
        # i.e. the HBM read volume of the dispatch copy epilogue, equals to "the number of tokens" * "num_expected_topk" * "data size per token"
        # 中文：这里把一次 scale-up 读流量近似为每个 token 的期望 top-k 次 payload 读取。
        # NOTES: this is for balanced gate
        # For V3.0's group-limited gate, please do not use this function
        # TODO: support this
        assert num_scaleout_topk == 0  # 当前模型尚不支持 group-limited gate 的 scaleout_topk。

        # Get bandwidth
        if rdma_gbs == 0 and self.num_rdma_ranks > 1:  # 未手动给 RDMA 带宽且存在 RDMA 域时自动探测。
            rdma_gbs = get_rdma_gbs()  # 单位 GB/s。
        if nvlink_gbs == 0:  # 未手动给 NVLink 带宽时自动探测。
            nvlink_gbs = get_nvlink_gbs()  # 单位 GB/s。

        # Initial count
        # NOTES: we don't count HBM traffic
        sm_read, sm_write = 0, 0  # 归一化 HBM 读/写工作量。
        rdma_traffic, nvlink_traffic = 0, 0  # 归一化 RDMA/NVLink 链路流量。

        def get_expected_topk(num_groups: int) -> float:  # 估计 top-k 会覆盖多少个 group/rank。
            # 期望一个 token 会触达多少个 group。假设 gating 在 num_experts 中均匀抽 num_topk 个 expert，
            # 某个 group 完全没被选中的概率为 C(E - E/G, K) / C(E, K)，所以至少选中一次的概率为
            # 1 - 上式；乘以 group 数即期望触达 group 数。
            # 例：64 experts、8 ranks、topk=8 时，这里估计一个 token 平均要发往多少个 rank，
            # 进而估计 NVLink/RDMA 与 SM 读写的相对瓶颈。
            assert num_experts % num_groups == 0  # expert 必须能均匀分给 groups。
            return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk) / math.comb(num_experts, num_topk))  # 返回期望触达 group 数。

        # Expected top-k scale-out ranks
        num_expected_scaleout_topk = get_expected_topk(self.num_scaleout_ranks) if self.num_scaleout_ranks > 1 else 0  # 期望触达 scaleout rank 数。

        # Expected top-k scale-up ranks
        num_expected_topk = get_expected_topk(self.num_ranks)  # 期望触达全局 rank 数。

        # Read tokens
        sm_read += 1 / num_expected_topk  # 每个目标 rank 平均分摊一次 token 读取。

        # NOTES: we don't consider the skip-send-buffer cases (all selections fall in the local)
        if self.num_scaleout_ranks > 1:  # hybrid 多 scaleout 场景。
            # Scaleup warps: write send buffer
            sm_write += 1 / num_expected_topk  # scaleup warp 写 send buffer。

            # Scaleout traffic
            sm_write += (1 / num_expected_topk) * (num_expected_scaleout_topk / self.num_scaleout_ranks)  # Local bypass；本 scaleout 内不走 RDMA。
            rdma_traffic += (1 / num_expected_topk) * (num_expected_scaleout_topk * (1 - 1 / self.num_scaleout_ranks))  # 跨 scaleout 的归一化 RDMA 流量。

            # Forward warps
            sm_read += num_expected_scaleout_topk / num_expected_topk  # forward warp 从 RDMA buffer 读取。
            sm_write += 1  # Issue scaleup；向 scaleup/NVLink 域写入。
            nvlink_traffic += 1 - (1 / self.num_scaleup_ranks)  # 除本地 bypass 外的 NVLink 流量。
        else:  # 单 scaleout 或 direct 模式。
            # Write send buffer
            if self.num_rdma_ranks > 1:  # 存在跨 RDMA rank 时需要写 send buffer。
                sm_write += 1 / num_expected_topk  # 写 send buffer 的归一化 HBM 工作量。

            # Issue NVLink
            sm_write += self.num_nvlink_ranks / self.num_ranks  # NVLink 域内 issue 的写工作量。

            # NVLink and RDMA traffic
            nvlink_traffic += self.num_nvlink_ranks / self.num_ranks * (1 - 1 / self.num_nvlink_ranks)  # Except local bypass；本地 rank 不产生链路流量。
            rdma_traffic += (self.num_ranks - self.num_nvlink_ranks) / self.num_ranks  # 跨 NVLink 域部分走 RDMA。

        # Found the bounded one
        # 用“通信流量 / 链路带宽”估算 NVLink 或 RDMA 哪个更容易成为瓶颈，然后反推需要多少 SM
        # 来提供足够的 HBM 读写吞吐。如果 prefer_overlap_with_compute=True，则倾向少占 SM，
        # 让 MoE expert GEMM 有更多计算资源；否则至少使用 64 SM 追求通信峰值。
        if self.num_scaleout_ranks > 1 and (rdma_traffic / rdma_gbs) > (nvlink_traffic / nvlink_gbs):  # RDMA 归一化耗时更大时。
            bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs  # 以 RDMA 作为瓶颈链路。
        else:  # 否则以 NVLink 作为瓶颈或无 scaleout。
            bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs  # 以 NVLink 作为瓶颈链路。

        # Calculate SM count
        # NOTES: will try to use more SMs if not overlap with compute
        num_device_sms = torch.cuda.get_device_properties('cuda').multi_processor_count  # 当前 GPU 总 SM 数。
        num_sms = num_device_sms  # No traffic, e.g., EP=1；无通信流量时默认全 SM。
        if bounded_traffic > 0:  # 存在瓶颈链路流量时按 HBM 读写需求估算。
            num_sms = max(  # 读和写两种 HBM 需求取更大者。
                bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,  # 满足读带宽所需 SM 数。
                bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,  # 满足写带宽所需 SM 数。
            )  # num_sms 此时是浮点估计值。
        num_sms = align(max(4, math.ceil(num_sms * 1.25)), 2)  # 加 25% safety margin，并保证至少 4 且为偶数。
        num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)  # 不追求 overlap 时至少使用 64 SM。
        num_sms = min(num_sms, num_device_sms)  # 不超过 GPU 物理 SM 数。

        # Summary
        if os.environ.get('EP_BUFFER_DEBUG', 0):  # 调试模式打印模型中间量。
            print(f'EP SM approximation: '  # 打印 SM 估算标题。
                  f'{sm_read=}, {sm_write=}, {rdma_traffic=}, {nvlink_traffic=}, '  # 打印 HBM 与链路归一化流量。
                  f'{rdma_gbs=}, {nvlink_gbs=}, '  # 打印探测或传入的链路带宽。
                  f'{num_expected_scaleout_topk=}, {num_expected_topk=}, '  # 打印期望触达 scaleout/global rank 数。
                  f'{bounded_traffic=}, {bounded_gbs=}, {num_sms=}')  # 打印瓶颈链路与最终 SM 数。
        return num_sms  # 返回最终推荐 SM 数。

    def get_theoretical_num_qps(self, num_sms: int) -> int:  # 基于 SM 数和模式估算 QP 数；返回值会被 num_allocated_qps 截断。
        """
        Estimate the optimal number of RDMA QPs based on SM count and mode.

        Arguments:
            中文：num_sms 越大通常需要更多 QP 分摊 doorbell/队列压力。
            num_sms: the number of SMs used for the dispatch/combine kernel.

        Returns:
            中文：返回值不会超过构造 ElasticBuffer 时实际申请到的 QP 数。
            num_qps: the recommended QP count, capped by `num_allocated_qps`.
        """
        # For direct mode, we encourage less QPs to reduce DB ringing overhead
        num_qps = min(num_sms, 8 + 1)  # direct 模式最多 8 个数据 QP 加 1 个 notify QP。

        # For hybrid mode, we encourage every channel (and notify) to have an independent QP
        if self.allow_hybrid_mode:  # hybrid 模式每个 SM/channel 倾向独立 QP。
            num_qps = num_sms * 16 + 1  # 16 个 channel per SM 加一个 notify QP。

        return min(num_qps, self.num_allocated_qps)  # 不能超过初始化时实际分配的 QP 数。

    def dispatch(self,  # 执行 EP dispatch，将 token 发送到 expert 所在 rank。
                 x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # BF16 Tensor [T,H] 或 FP8 (data,scale) tuple。
                 topk_idx: Optional[torch.Tensor] = None,  # 路由矩阵 [T,K]；cached handle 模式必须为 None。
                 topk_weights: Optional[torch.Tensor] = None,  # 路由权重 [T,K]；可选，dispatch 侧可一并搬运。
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,  # 可选累计 per-expert 接收统计 [num_local_experts]。
                 num_experts: Optional[int] = None,  # 全局 expert 总数；cached 模式可从 handle 推断。
                 num_max_tokens_per_rank: Optional[int] = None,  # 每 rank 最大 token 数；用于 metadata 编码。
                 expert_alignment: Optional[int] = None,  # 每个本地 expert 接收 token slot 的对齐粒度。
                 num_sms: int = 0, num_qps: int = 0,  # 通信 kernel 使用 SM/QP 数；0 表示自动估算。
                 previous_event: Optional[EventHandle] = None,  # dispatch kernel 前需要等待的事件。
                 previous_event_before_epilogue: Optional[EventHandle] = None,  # copy epilogue 前需要等待的事件。
                 async_with_compute_stream: bool = False,  # True 时当前 compute stream 不等待通信完成。
                 allocate_on_comm_stream: bool = False,  # True 时输出 tensor 归属通信 stream，适合异步 overlap。
                 handle: Optional[EPHandle] = None,  # cached dispatch handle；提供时复用布局 metadata。
                 do_handle_copy: bool = True,  # 是否克隆 topk_idx 放入 handle，防止用户后续原地修改。
                 do_cpu_sync: Optional[bool] = None,  # 是否 CPU 同步拿精确接收计数；None 使用默认策略。
                 do_expand: bool = False,  # 是否展开 top-k 槽为独立 token 行。
                 use_tma_aligned_col_major_sf: bool = False) \
            -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # recv_x，BF16 Tensor 或 FP8 tuple。
                     Optional[torch.Tensor], Optional[torch.Tensor],  # recv_topk_idx 和 recv_topk_weights。
                     EPHandle, EventOverlap]:  # 返回 handle 与异步事件。
        """
        Dispatch tokens to different ranks. Supports both single-node and multi-node settings.
        中文：该 API 是 Elastic 路径主入口，会按拓扑选择 direct 或 hybrid kernel。
            SM and QP counts are automatically determined if not specified.

        Arguments:
            中文：dispatch 支持 BF16 或 FP8 输入，并可选择普通、cached、deterministic/expand 等路径。
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be
                `[num_tokens, hidden]`, and type must be `torch.bfloat16`; for the second type (FP8 mode),
                中文：FP8 tuple 的 data 与 scale 第一维都必须等于 num_tokens。
                the first element of the tuple must be `[num_tokens, hidden]` with type `torch.float8_e4m3fn`,
                the second is the scale factors.
            中文：FP8 输入时 x 是二元组，第二项 scale factor 的 layout 会影响 TMA 对齐。
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.
                中文：-1 槽不会产生发送目标，但仍保留在 top-k 维中以维持 shape。
                Must be `None` if `handle` is provided.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
                Must be `None` if `handle` is provided.
            中文：cached handle 模式下 topk_idx/topk_weights 必须为空，因为路由由 handle 固定。
            cumulative_local_expert_recv_stats: `[num_local_experts]` with `torch.int`, a cumulative expert count
                tensor for statistics, useful for online EP load balance monitoring.
            中文：统计张量由 kernel 原地累加，形状只覆盖本地 expert，不影响通信正确性。
            num_experts: the number of all experts. Inferred from `handle` if provided.
            num_max_tokens_per_rank: the maximum number of tokens per rank. Inferred from constructor default
                or `handle` if provided.
            中文：使用 cached handle 时，这两个容量/拓扑参数必须与 handle 记录的 dispatch 布局一致。
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_sms: the number of SMs to use (0 for automatic via `get_theoretical_num_sms`).
            num_qps: the number of RDMA QPs to use (0 for automatic via `get_theoretical_num_qps`).
            中文：num_sms/num_qps 为 0 时按拓扑和 overlap 偏好自动估算。
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            previous_event_before_epilogue: the event to wait before actually executing the copy epilogue.
            中文：previous_event_before_epilogue 可把主通信和 copy epilogue 的等待点拆开。
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                中文：开启后调用方需要使用返回 EventOverlap 或 stream 依赖保证数据可见。
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.
            中文：async/allocate_on_comm_stream 共同决定返回 tensor 的 stream 归属和事件等待方式。
            handle: an optional cached `EPHandle` from a previous dispatch, if set, the CPU will reuse the layout
                information to save some time. `topk_idx` and `topk_weights` must be `None`.
            do_handle_copy: whether to clone `topk_idx` in the returned handle (to prevent user modification).
            中文：handle/do_handle_copy 控制 layout 复用和 topk_idx 防修改保护。
            do_cpu_sync: whether to synchronize with CPU to get exact received token counts.
                `None` defaults to `True` unless `handle` is provided.
            中文：cached handle 模式不再重新同步接收计数，而是复用 handle 里的容量信息。
            do_expand: whether to use the expanding layout (one slot per expert per token).
            use_tma_aligned_col_major_sf: whether to use TMA-aligned column-major layout for scale factors.
            中文：do_cpu_sync/expand/scale-factor layout 会影响返回形状、EPHandle 内容和后续 combine 读取方式。

        Returns:
            中文：返回接收 token、可选路由 metadata、EPHandle 和通信事件。
            recv_x: received tokens, the same type and tuple as the input `x`
            recv_topk_idx: received expert indices
            中文：recv_topk_idx 的行数与 recv_x 对齐，记录接收 token 原始 top-k expert id。
            recv_topk_weights: received expert weights (`None` if `topk_weights` was not provided).
            handle: the returned communication handle.
            中文：handle 必须保留到 combine；recv_topk_* 主要服务非 expanded 或调试/上层逻辑。
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
        """
        check_torch_deterministic()  # 防止 PyTorch deterministic 初始化 kernel 与通信 stream 干扰。

        # Automatic decide SM and QP count
        # cached 模式下 topk_idx=None，所以必须从 handle.topk_idx 读取 K；非 cached 模式直接读取用户输入。
        # num_experts 在非 cached 模式应由调用者传入；cached 模式若传 None，会在下面从 handle 里补齐。
        # 这里先用 top-k 宽度估 SM，是因为理论模型只需要 K 与 rank/expert 数来估计通信扇出。
        num_topk = (handle.topk_idx if topk_idx is None else topk_idx).shape[1]  # top-k 宽度 K；cached 模式从 handle 取。
        # num_sms 控制 CUDA kernel channel 并行度，num_qps 控制 Gin/RDMA 并发队列数量。
        num_sms = self.get_theoretical_num_sms(num_experts, num_topk) if num_sms == 0 else num_sms  # 未指定时自动估算 SM 数。
        num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps  # 未指定时按 SM 数估算 QP。
        assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'  # 运行时使用 QP 不能超过初始化分配数量。

        # Unpack SF
        # sf 可能是普通 [T,H/128] scale，也可能由 use_tma_aligned_col_major_sf 要求成 TMA 友好布局。
        # BF16 输入保持单 Tensor；FP8 输入必须是 (x_fp8, scale_factor) 二元组，C++ 根据 sf 是否为空选择 kernel 路径。
        x, sf = x if isinstance(x, tuple) else (x, None)  # FP8 输入拆成数据 [T,H] 与 scale-factor；BF16 sf=None。

        # Unpack handles
        # Reuse some values if possible
        if handle is not None:  # cached handle 模式。
            # cached handle 模式用于“路由不变、token payload 改变”的场景，例如反向传播或重复 replay。
            # 此时 topk_idx/topk_weights 必须为空，因为路由已经由 handle.topk_idx 与 metadata 固定；
            # 同时不能再做 CPU sync 获取新布局，否则会与“复用布局”的语义冲突。
            assert topk_idx is None and topk_weights is None  # 复用 handle 时不能提供新的路由。
            assert do_cpu_sync is None or not do_cpu_sync, 'Cannot do CPU sync with cached handle'  # cached 模式不重新做 CPU 布局同步。
            topk_idx = handle.topk_idx  # 使用 handle 中保存的 cloned topk_idx。
            num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, handle.num_max_tokens_per_rank)  # 复用 handle 的 max tokens。
            num_experts = value_or(num_experts, handle.num_experts)  # 复用 handle 的 expert 总数。
            expert_alignment = value_or(expert_alignment, handle.expert_alignment)  # 复用 handle 的 expert 对齐。
            do_cpu_sync = False  # cached 模式固定不做 CPU sync。

            # Should be aligned with the handle context
            # 这三个值参与 metadata 编码和 buffer stride，任何一个变化都会让 cached layout 地址解释错误。
            assert (num_experts, expert_alignment, num_max_tokens_per_rank) == (  # 调用方覆盖值必须与 handle 创建时一致。
                handle.num_experts, handle.expert_alignment, handle.num_max_tokens_per_rank  # 不一致会破坏 metadata 解释。
            )  # 使用括号续行，避免反斜杠后追加注释破坏 Python 语法。
        (cached_num_recv_tokens, cached_num_recv_tokens_per_expert_list,
         cached_psum_num_recv_tokens_per_scaleup_rank, cached_psum_num_recv_tokens_per_expert,
         cached_dst_buffer_slot_idx,
         # 中文：hybrid cached dispatch 额外复用 forward metadata 与 channel linked list。
         cached_token_metadata_at_forward,
         cached_channel_linked_list) = self._unpack_handle(handle)  # 将 cached handle 拆成 C++ 参数；handle=None 时全为 None。

        # Some default values
        # expert_alignment 会让每个本地 expert 的接收 token 数按固定粒度对齐，便于后续 GEMM
        # 以更规则的 batch/stride 读取。例如某 expert 实际收到 37 个 token，alignment=8 时，
        # prefix-sum 中会按 40 个 slot 预留，真实有效 token 数仍由 per-expert count 表示。
        # num_max_tokens_per_rank 是 metadata 编码上界：source token id 与 slot index 都依赖这个容量，
        # 因此同一个 handle 生命周期内不能随意改变。
        num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, self.num_max_tokens_per_rank)  # 默认使用构造时的最大 token 数。
        expert_alignment = value_or(expert_alignment, 1)  # 默认不额外对齐每 expert token 数。
        # CPU sync=True 会返回精确 per-expert list；False 更适合 CUDA graph/低同步开销场景。
        do_cpu_sync = value_or(do_cpu_sync, True)  # 默认做 CPU sync 获取精确接收计数。

        # Do dispatch
        # 这里跨过 Python/C++ 边界，进入 csrc/elastic/buffer.hpp::dispatch。
        # 返回值中 recv_x/recv_sf 是实际给 expert 计算的数据；其余 tensor 构成 EPHandle。
        # token_metadata_at_forward/channel_linked_list 只在 hybrid scaleout+scaleup 路径下存在，
        # 用于记录跨 RDMA 转发后，在 scaleup 域内继续分发的链表关系。
        (recv_x, recv_sf,  # recv_x: [recv_tokens,H] 或 expand layout；recv_sf 为 FP8 scale-factor。
         recv_topk_idx, recv_topk_weights,  # 接收侧路由和权重，形状通常 [recv_tokens,K]。
         cloned_topk_idx,  # C++ 克隆的原始 topk_idx，用于 handle 防止用户修改。
         num_recv_tokens_per_expert_list,  # CPU list，长度为本地 expert 数。
         psum_num_recv_tokens_per_scaleup_rank,  # scaleup rank 前缀和。
         psum_num_recv_tokens_per_expert,  # expert 前缀和/对齐前缀和。
         recv_src_metadata,  # combine 回写所需源 token metadata。
         dst_buffer_slot_idx,  # dispatch 写目标 buffer slot 索引。
         token_metadata_at_forward,  # hybrid forward metadata。
         channel_linked_list,  # hybrid channel 链表。
         event) = self.runtime.dispatch(x, sf, topk_idx, topk_weights,  # 调用 C++ dispatch。
                                        cumulative_local_expert_recv_stats,  # 可选累计 per-expert 接收统计。
                                        cached_num_recv_tokens,  # cached 接收 token 数。
                                        cached_num_recv_tokens_per_expert_list,  # cached per-expert 计数。
                                        cached_psum_num_recv_tokens_per_scaleup_rank,  # cached scaleup 前缀和。
                                        cached_psum_num_recv_tokens_per_expert,  # cached expert 前缀和。
                                        cached_dst_buffer_slot_idx,  # cached 目标 slot。
                                        cached_token_metadata_at_forward,  # cached hybrid metadata。
                                        cached_channel_linked_list,  # cached channel 链表。
                                        num_max_tokens_per_rank,  # 每 rank 最大 token 数。
                                        num_experts, expert_alignment,  # expert 总数和对齐粒度。
                                        num_sms, num_qps,  # kernel 使用的 SM/QP 数。
                                        previous_event,  # 通信开始前等待的事件。
                                        previous_event_before_epilogue,  # epilogue 前等待的事件。
                                        async_with_compute_stream, allocate_on_comm_stream,  # 异步和内存归属策略。
                                        do_handle_copy, do_cpu_sync, do_expand,  # handle copy、CPU sync、expand layout 开关。
                                        use_tma_aligned_col_major_sf)  # FP8 scale-factor 是否用 TMA 对齐列主布局。
        if handle is None:  # 非 cached 模式需要创建新的 handle 返回给用户。
            # EPHandle 是 combine 的唯一布局来源；cached dispatch 也复用它避免重复做布局计算。
            # do_handle_copy=False 会把用户原始 topk_idx 放进 handle，省一次 clone，但用户必须保证
            # 后续不会原地改动 topk_idx；否则 combine/cached dispatch 会按被改坏的路由解释 metadata。
            handle = EPHandle(do_expand,  # 保存 expand layout 标记。
                              num_experts, expert_alignment,  # 保存 expert 配置。
                              num_max_tokens_per_rank,  # 保存全局 token id 编码上界。
                              num_sms,  # 保存 SM 数供 combine 默认复用。
                              cloned_topk_idx if do_handle_copy else topk_idx,  # 保存克隆或原 topk_idx。
                              num_recv_tokens_per_expert_list,  # 保存 per-expert 接收计数。
                              psum_num_recv_tokens_per_scaleup_rank,  # 保存 scaleup 前缀和。
                              psum_num_recv_tokens_per_expert,  # 保存 expert 前缀和。
                              recv_src_metadata,  # 保存源 token metadata。
                              dst_buffer_slot_idx,  # 保存目标 slot。
                              token_metadata_at_forward,  # 保存 hybrid forward metadata。
                              channel_linked_list)  # 保存 hybrid channel 链表。
        # cached 模式复用原 handle，不替换其中的 metadata；本次只返回新 recv_x/recv_topk。

        # Repack SF
        # FP8 dispatch 输入是 (x_fp8, scale_factor)，C++ 返回也拆成两个 tensor；
        # Python API 为了保持调用习惯，在返回前重新打包成同样的 tuple 结构。
        recv_x = (recv_x, recv_sf) if recv_sf is not None else recv_x  # FP8 路径返回 (data,scale)，BF16 只返回 data。

        # Return
        # async_with_compute_stream=True 时 EventOverlap 需要由调用方显式 wait；否则事件通常已在当前流同步过。
        # 返回的 handle 与 recv_x 同步产生：即使异步返回，handle 中的张量 metadata 已经分配好，
        # 但其内容是否可读仍取决于调用方是否等待 EventOverlap。
        return recv_x, recv_topk_idx, recv_topk_weights, handle, EventOverlap(event)  # 返回接收数据、metadata handle 和异步事件。

    @staticmethod  # 这个 helper 不依赖实例状态。
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:  # 将 combine bias 拆成两个 C++ optional 参数；None 会返回 (None,None)。
        bias_0, bias_1 = None, None  # 默认没有 bias。
        if isinstance(bias, torch.Tensor):  # 单个 bias Tensor。
            # 单 bias 用于常见 epilogue：combined_x += bias。
            bias_0 = bias  # 放到第一个 bias 槽。
        elif isinstance(bias, tuple):  # 两段 bias 形式。
            # 双 bias 给部分 fused epilogue 使用，C++ 侧会按固定顺序处理。
            assert len(bias) == 2  # 必须恰好两个 Tensor。
            bias_0, bias_1 = bias  # 分别传给 C++ combine epilogue。
        return bias_0, bias_1  # 返回两个 optional bias。

    def combine(self,  # 执行 EP combine，把 expert 输出归约回原 token。
                x: torch.Tensor,  # expert 计算后的 token Tensor，通常 [num_recv_tokens,H]。
                handle: EPHandle,  # dispatch 返回的路由 metadata，combine 必须依赖它反向写回。
                topk_weights: Optional[torch.Tensor] = None,  # 原 token top-k 权重 [T,K]；expand 模式不使用。
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,  # 可选 combine epilogue bias。
                num_sms: int = 0, num_qps: int = 0,  # combine kernel 使用 SM/QP 数；0 表示复用/自动估算。
                previous_event: EventHandle = None,  # combine kernel 前需要等待的事件。
                previous_event_before_epilogue: Optional[EventHandle] = None,  # reduce epilogue 前等待事件。
                async_with_compute_stream: bool = False,  # True 时当前 compute stream 不等待 combine 完成。
                allocate_on_comm_stream: bool = False) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:  # 返回 combined_x、combined_topk_weights 和事件。
        """
        Combine (reduce) tokens from different ranks back to their original ranks.
        Supports both single-node and multi-node settings.

        Arguments:
            中文：combine 根据 dispatch 返回的 handle 把专家输出送回源 rank/源 token。
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the `dispatch` function.
            中文：handle 中的 prefix/source metadata 决定每行 x 要回到哪个原 token。
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to
                its original ranks. Not used in expand mode.
            bias: 0, 1 or 2 `[num_combined_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            中文：expanded 模式下权重已隐含在布局/epilogue 中，非 expanded 模式需要 topk_weights 做加权规约。
            num_sms: the number of SMs to use (0 to reuse the SM count from the dispatch handle).
            num_qps: the number of RDMA QPs to use (0 for automatic via `get_theoretical_num_qps`).
            中文：num_sms=0 会复用 dispatch handle 中记录的 SM 数，num_qps=0 会重新按拓扑估算。
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            previous_event_before_epilogue: the event to wait before actually executing the reduce epilogue.
            中文：reduce epilogue 是 combine 写回原 token 顺序前的本地规约/复制阶段。
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                中文：异步 combine 返回后 combined_x 需要等待 event 后才能安全消费。
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.
            中文：previous_event/async/allocate_on_comm_stream 控制 combine 与专家计算或后续计算的 overlap。

        Returns:
            中文：combined_x 是按原 token 顺序还原后的输出；event 表示异步通信完成点。
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            中文：num_combined_tokens 通常等于 dispatch 前的本地 token 数。
            combined_topk_weights: the reduced top-k weights, with shape `[num_combined_tokens, num_topk]` and type `torch.float`.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
        """
        check_torch_deterministic()  # 防止 deterministic 初始化 kernel 干扰通信 stream。

        # Automatic decide SM and QP count
        # combine 的路由规模必须和 dispatch handle 对齐，因此默认沿用 handle.num_sms。
        # 如果用户显式覆盖 num_sms，也会重新根据 SM 数估算 QP 数，但不会改变 handle 中的布局。
        num_sms = handle.num_sms if num_sms == 0 else num_sms  # 默认复用 dispatch 时的 SM 数。
        num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps  # QP 数随 SM 数估算。
        assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'  # 不能超过已分配 QP。

        # bias 支持 None / 单 tensor / 二元组。C++ combine epilogue 会在写回 combined_x 时
        # 统一处理 bias，因此 Python 这里只负责把用户输入拆成固定的两个 optional 参数。
        bias_0, bias_1 = ElasticBuffer._unpack_bias(bias)  # 将 None/Tensor/tuple bias 规范化。
        # combine 反向使用 dispatch handle 中的 metadata：recv_src_metadata 定位源 token，
        # topk_idx/topk_weights 决定归约权重，hybrid metadata 则恢复跨 scaleout 转发路径。
        combined_x, combined_topk_weights, event = \
            self.runtime.combine(x, topk_weights,  # 调用 C++ combine，将 expert 输出回传原 token rank。
                                 bias_0, bias_1,  # 可选 bias，在 epilogue 中加入。
                                 handle.recv_src_metadata,  # dispatch 保存的源 token metadata。
                                 handle.topk_idx,  # 原始路由矩阵 [T,K]。
                                 handle.psum_num_recv_tokens_per_scaleup_rank,  # scaleup 前缀和。
                                 handle.token_metadata_at_forward,  # hybrid forward metadata。
                                 handle.channel_linked_list,  # hybrid channel 链表。
                                 handle.num_experts,  # expert 总数。
                                 handle.num_max_tokens_per_rank,  # 源 token id 编码上界。
                                 num_sms, num_qps,  # kernel 使用 SM/QP 数。
                                 previous_event,  # combine 开始前等待事件。
                                 previous_event_before_epilogue,  # reduce epilogue 前等待事件。
                                 async_with_compute_stream,  # 是否异步返回。
                                 allocate_on_comm_stream,  # 输出 tensor 是否归属通信 stream。
                                 handle.do_expand)  # 是否使用 expand layout。
        return combined_x, combined_topk_weights, EventOverlap(event)  # 返回 combine 结果和事件。
