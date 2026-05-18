# legacy.py 是旧版 Buffer 的 Python 门面，负责参数组织、环境变量和 C++ runtime 生命周期。
import os  # 设置 NVSHMEM 相关环境变量，例如 QP depth、IBGDA、P2P 开关。
# torch 用于 Tensor/Stream 类型、CUDA stream 包装，以及和 C++ 扩展互传张量。
import torch  # 暴露 torch.Stream/Tensor 类型并构造 CUDA stream 包装。
# torch.distributed 路径用于普通训练/推理环境下同步 IPC handle 和 NVSHMEM unique id。
import torch.distributed as dist  # 使用 ProcessGroup rank/size/all_gather_object 同步 runtime metadata。
# Callable 标注 hook，List 标注 per-expert 计数，Tuple/Optional/Union 描述复杂返回值。
from typing import Callable, List, Tuple, Optional, Union  # 标注 dispatch/combine 的可选参数和返回 tuple。

# noinspection PyUnresolvedReferences
# _C 是 pybind 暴露的 native 扩展，Buffer 只是薄包装，真正通信逻辑在 C++/CUDA。
import deep_ep._C as _C  # C++/CUDA 扩展模块，legacy Buffer 的真正 runtime 实现在这里。
# noinspection PyUnresolvedReferences
# Config/EventHandle 也来自 native 扩展，分别对应 kernel 配置和 CUDA event 句柄。
from deep_ep._C import Config, EventHandle  # Config 是 kernel 调优配置；EventHandle 是 C++ CUDA event 包装。
# EventOverlap 在 Python 侧把 EventHandle 和保活 tensor 组合成 async API 的返回对象。
from ..utils.event import EventOverlap  # Python 侧异步事件/回调封装。
# envs helper 负责启动前安全检查，避免 deterministic 或 PCIe P2P 等配置破坏 kernel 假设。
from ..utils.envs import check_nvlink_connections, check_torch_deterministic  # 环境安全检查 helper。


class Buffer:  # legacy expert-parallel 通信 buffer，对应旧版 high-throughput/low-latency API。
    # 这个类同时覆盖 intranode、internode、low-latency 三套路径，因此方法签名会比较宽。
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
        中文：intranode 路径只要求同节点 GPU 通过 NVLink/NVSwitch 可达。
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
        中文：前两类 high-throughput 路径优先吞吐，通过批量 channel/ring 提高带宽。
        - low-latency all-to-all (dispatch and combine, using RDMA)
        中文：low-latency 路径优先时延，使用固定容量双缓冲和 IBGDA 控制面。

    Attributes:
        中文速览：这些字段描述 legacy Buffer 的通信域、buffer 容量和 C++ runtime。
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        中文：num_sms 是类级调优参数；rank/group_size 来自 ProcessGroup 或 MPI comm。
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        中文：num_nvl_bytes 主要承载节点内 ring buffer、head/tail 和 rank/expert metadata。
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        runtime: the C++ runtime.
        中文：num_nvl_bytes/num_rdma_bytes 控制两类 symmetric buffer 容量，runtime 持有真实 CUDA/NVSHMEM 资源。

    中文说明：
        这个类是 legacy API 的 Python 门面，负责把 PyTorch Tensor、ProcessGroup/MPI communicator、
        CUDA event 和调优 Config 组织好，然后调用 deep_ep._C 中的 C++/CUDA runtime。
        真正的数据搬运发生在 csrc/kernels/legacy 下的 CUDA kernel；这里的重点是参数检查、
        buffer 生命周期、异步事件保活，以及把 dispatch 返回的 handle 原样交给 combine。

    中文字段说明：
        num_sms 是 high-throughput legacy kernel 使用的 SM 数，dispatch/combine 默认配置表都会读取它；
        rank/group_size/group 描述当前 legacy EP 通信域，可以来自 torch.distributed 或 mpi4py；
        num_nvl_bytes 对应节点内 NVLink buffer，num_rdma_bytes 对应跨节点或 low-latency RDMA buffer；
        low_latency_mode 会切换到固定容量双缓冲路径，要求 QP 数、专家数和 C++ kernel 模板约束匹配；
        runtime 是 pybind C++ Buffer，持有 IPC/NVSHMEM/NVL/RDMA 指针表和通信 stream。
    """

    num_sms: int = 20  # high-throughput kernel 默认使用的 SM 数；必须为偶数，可由 set_num_sms 修改。
    # num_sms 是类变量而非实例变量；set_num_sms 会影响之后所有 Buffer 的 dispatch/combine。

    def __init__(self,  # 构造 legacy 通信 runtime。
                 group: Optional[dist.ProcessGroup],  # PyTorch ProcessGroup；如果使用 MPI 路径则可为 None。
                 num_nvl_bytes: int = 0,  # NVLink/intranode buffer 字节数。
                 num_rdma_bytes: int = 0,  # RDMA/NVSHMEM buffer 字节数。
                 low_latency_mode: bool = False,  # 是否启用低延迟 IBGDA 双缓冲路径。
                 num_qps_per_rank: int = 24,  # 每 peer RC QP 数，低延迟模式通常等于本地 expert 数。
                 allow_nvlink_for_low_latency_mode: bool = True,  # 低延迟模式是否允许 NVLink P2P。
                 allow_mnnvl: bool = False,  # 是否允许 multi-node NVLink。
                 explicitly_destroy: bool = False,  # 是否要求用户显式 destroy，避免析构期 hang。
                 enable_shrink: bool = False,  # 是否启用动态 rank shrink/mask buffer。
                 comm: Optional["mpi4py.MPI.Comm"] = None) -> None:  # noqa: F821  # MPI communicator fallback。
        # __init__ 的核心任务是把 Python 分布式环境转换成 C++ runtime 能直接使用的 peer 地址表。
        """
        Initialize the communication buffer.

        Arguments:
            中文：构造函数负责选择 torch.distributed 或 mpi4py 通信域，并初始化 NVL/RDMA buffer。
            group: the communication group.
            中文：group 缺省时可由 comm 兜底；二者最终都会转成 C++ runtime 所需的 rank/world 信息。
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            中文：num_nvl_bytes/num_rdma_bytes 分别对应节点内和跨节点 buffer；low_latency_mode 会切换固定容量 IBGDA 路径。
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            中文：num_qps_per_rank 与 C++ low-latency kernel 的 expert/QP 映射联动，不能随意小于 local expert 数。
            allow_nvlink_for_low_latency_mode: whether allow NVLink traffic for low-latency mode, you should notice
                this is somehow incompatible with the hook-based overlapping.
            中文：低延迟模式要求 QP 和 local expert 对齐，hook overlap 与 NVLink P2P 组合时需要特别谨慎。
                Warning: PCIe connections may lead to errors due to memory ordering issues,
                please make sure all connections are via NVLink.
            中文：如果机器是 PCIe 拓扑，建议先运行环境检查确认 P2P 内存序满足要求。
            allow_mnnvl: whether to allow MNNVL
            use_fabric: whether to use fabric API for memory buffers.
            中文：MNNVL/fabric 都属于底层内存注册策略，开启前需要与机器拓扑和驱动能力匹配。
            enable_shrink: whether to enable shrink mode. The enable mode allocates a mask buffer to support masking ranks dynamically.
            中文：enable_shrink 会额外分配 mask buffer，让超时 rank 后续被跳过。
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                中文：长生命周期服务建议显式 destroy，避免 Python 析构顺序不确定时卡在通信释放。
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
            comm: the `mpi4py.MPI.Comm` communicator to use in case the group parameter is absent.
            中文：explicitly_destroy/comm 主要服务长生命周期进程和非 torch.distributed benchmark 场景。

        中文说明：
            初始化阶段会收集所有 rank 的 device id、CUDA IPC handle 和 NVSHMEM unique id。
            这些 metadata 必须在所有 rank 间一致交换，否则 C++ runtime 无法把对端 buffer 映射成
            GPU 可访问地址。low_latency_mode 会额外调整 NVSHMEM 环境变量和 QP 深度。
        """
        # 该检查在构造时执行，而不是等到第一次 dispatch，便于更早发现不支持的拓扑。
        check_nvlink_connections(group)  # 对 PCIe GPU 做保守 NVLink 检查，避免 P2P ordering 问题。

        # Initialize the CPP runtime
        # Legacy Buffer 支持两种进程组织方式：
        # - torch.distributed.ProcessGroup：训练/推理中最常见；
        # - mpi4py.MPI.Comm：某些 benchmark 或 HPC 环境中可不依赖 torch distributed。
        # 后续同步 device id、CUDA IPC handle、NVSHMEM unique id 都依赖 all_gather_object，
        # 所以这里把两种 communicator 抽象成同一个小闭包。
        if group is not None:  # PyTorch distributed 路径。
            self.rank = group.rank()  # 当前 rank id。
            self.group = group  # 保存 ProcessGroup，后续测试/用户可能查询。
            self.group_size = group.size()  # 总 rank 数。
            # group_size 同时决定 num_ranks，后续 RDMA rank 数由 C++ runtime 按每节点 GPU 数拆分。

            def all_gather_object(obj):  # 用 torch.distributed 同步任意 Python 对象。
                # 这里用 object gather 是因为 IPC handle/NVSHMEM unique id 不是普通 tensor。
                object_list = [None] * self.group_size  # 接收列表长度等于 world size。
                dist.all_gather_object(object_list, obj, group)  # 所有 rank 交换 obj。
                return object_list  # 返回按 rank 排列的对象列表。
        elif comm is not None:  # MPI 路径。
            self.rank = comm.Get_rank()  # MPI rank id。
            self.group = comm  # 保存 MPI communicator。
            self.group_size = comm.Get_size()  # MPI world size。
            # MPI 路径主要用于 benchmark/HPC 环境，后续 allgather 仍返回 Python 对象列表。

            def all_gather_object(obj):  # 用 MPI allgather 同步 Python 对象。
                # mpi4py 可以直接 allgather pickle 后的 Python 对象，接口比 torch 更简单。
                return comm.allgather(obj)  # 返回按 rank 排列的对象列表。
        else:  # 两种通信上下文都没提供时无法初始化。
            raise ValueError("Either 'group' or 'comm' must be provided.")  # 提示调用方必须提供 group 或 comm。
        self.num_nvl_bytes = num_nvl_bytes  # NVLink buffer 字节数，intranode high-throughput 路径使用。
        self.num_rdma_bytes = num_rdma_bytes  # RDMA/NVSHMEM buffer 字节数，internode/low-latency 路径使用。
        self.low_latency_mode = low_latency_mode  # 是否启用低延迟 RDMA dispatch/combine。
        self.explicitly_destroy = explicitly_destroy  # 是否要求用户显式 destroy，避免析构期卡死。
        self.enable_shrink = enable_shrink  # 是否启用动态 rank mask/shrink buffer。
        self.runtime = _C.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode,  # 创建 C++ legacy runtime。
                                 explicitly_destroy, enable_shrink, allow_mnnvl)  # 传入析构策略、shrink、MNNVL 开关。

        # Synchronize device IDs
        # C++ runtime 构造时只知道本 rank 的 CUDA device。intranode NVLink IPC 需要打开同节点
        # 其他 rank 的 CUDA IPC handle，因此先 all-gather device id，再 all-gather IPC handle。
        local_device_id = self.runtime.get_local_device_id()  # C++ runtime 查询当前 CUDA device id。
        device_ids = all_gather_object(local_device_id)  # 所有 rank 交换 device id，列表长度为 group_size。

        # Synchronize IPC handles
        local_ipc_handle = self.runtime.get_local_ipc_handle()  # 当前 rank 本地 buffer 的 CUDA IPC handle。
        ipc_handles = all_gather_object(local_ipc_handle)  # 收集所有 rank 的 IPC handle，供 C++ 打开远端 buffer。

        # Synchronize NVSHMEM unique IDs
        # legacy internode/low-latency 路径使用 NVSHMEM。NVSHMEM 初始化需要一个 root unique id：
        # - low_latency_mode 下所有 rank 都在同一个 NVSHMEM team，rank 0 生成 root id；
        # - 高吞吐 internode 下按 RDMA rank 分组，rdma_rank==0 的 rank 生成 root id。
        root_unique_id = None  # 默认不需要 NVSHMEM root id；仅 RDMA/low-latency 路径需要。
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:  # 多 RDMA rank 或低延迟模式需要初始化 NVSHMEM。
            # Enable IBGDA
            assert num_qps_per_rank > 0  # NVSHMEM IBGDA 必须至少分配一个 RC QP。
            os.environ['NVSHMEM_DISABLE_P2P'] = '0' if allow_nvlink_for_low_latency_mode else '1'  # 控制 low-latency 是否允许 NVLink P2P。
            os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'  # 启用 device-side IBGDA 路径。
            os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'  # 每个 peer 的 RC QP 数。

            # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
            # low-latency kernel 会为每个 token/专家组合发起 RDMA work request。只要 QP depth 足够大，
            # device kernel 就能省掉昂贵的“发送前检查 WQ 是否有空槽”的逻辑，降低延迟。
            self.nvshmem_qp_depth = int(os.environ.get('NVSHMEM_QP_DEPTH', '1024'))  # 保存 QP depth，kernel 侧会据此假设 WQ 足够深。
            os.environ['NVSHMEM_QP_DEPTH'] = str(self.nvshmem_qp_depth)  # 回写环境变量，确保 NVSHMEM 初始化读取同一值。

            # Reduce gpu memory usage
            # 6 default teams + 1 extra team
            os.environ['NVSHMEM_MAX_TEAMS'] = '7'  # 限制 team 数，减少 NVSHMEM GPU 内存占用。
            # Disable NVLink SHARP
            os.environ['NVSHMEM_DISABLE_NVLS'] = '1'  # 禁用 NVLink SHARP，避免与当前通信路径冲突。
            # NOTES: NVSHMEM initialization requires at least 256 MiB
            os.environ['NVSHMEM_CUMEM_GRANULARITY'] = f'{2 ** 29}'  # 设置 cuMem 粒度为 512MiB，满足 NVSHMEM 初始化需求。

            if not allow_mnnvl:  # 默认关闭多节点 NVLink 检测。
                # Disable multi-node NVLink detection
                os.environ['NVSHMEM_DISABLE_MNNVL'] = '1'  # 避免 NVSHMEM 尝试使用未启用的 MNNVL 路径。

            # Synchronize using the root ID
            if (low_latency_mode and self.rank == 0) or (not low_latency_mode and self.runtime.get_rdma_rank() == 0):  # 选择负责生成 unique id 的 root rank。
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()  # C++ runtime 生成本地 NVSHMEM unique id。
            nvshmem_unique_ids = all_gather_object(root_unique_id)  # 所有 rank 收集 root id；非 root 贡献 None。
            root_unique_id = nvshmem_unique_ids[0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)]  # 取真正 root rank 的 id。

        # Make CPP runtime available
        # sync 会把 Python all-gather 到的 IPC/NVSHMEM 信息交给 csrc/legacy/buffer.hpp，
        # C++ 侧完成远端 IPC 映射、NVSHMEM 初始化、指针表复制到 GPU。只有 sync 之后，
        # CUDA kernel 才能通过 buffer_ptrs_gpu/barrier_signal_ptrs_gpu 访问其他 rank 的 buffer。
        self.runtime.sync(device_ids, ipc_handles, root_unique_id)  # 将 device/ipc/nvshmem metadata 交给 C++ runtime 完成映射。
        assert self.runtime.is_available()  # 确认 C++ runtime 初始化完成并可用。

    def destroy(self):  # 显式销毁 C++ runtime。
        """
        Destroy the cpp runtime and release resources.

        中文说明：
            只有 explicitly_destroy=True 时才允许调用。这样设计是为了避免 Python 解释器退出、
            异常栈展开或 torch distributed 清理期间自动析构 C++/NVSHMEM 资源导致卡住。
        """

        assert self.explicitly_destroy, '`explicitly_destroy` flag must be set'  # 只有显式销毁模式允许手动 destroy。

        self.runtime.destroy()  # 调用 C++ 释放 IPC/NVSHMEM/buffer 等资源。
        self.runtime = None  # 清空 Python 引用，避免重复使用已销毁 runtime。

    @staticmethod
    def is_sm90_compiled():  # 查询扩展是否包含 Hopper/SM90 专用 kernel。
        return _C.is_sm90_compiled()  # 返回 bool，测试可据此跳过不支持路径。

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:  # 设置 high-throughput kernel 使用的 SM 数。
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.

        中文说明：
            这个设置影响后续 high-throughput dispatch/combine 的 channel 并行度。
            旧版 kernel 通常用两个 SM 组成一个 channel pair，因此必须是偶数；
            修改的是类变量，已经创建的 Buffer 对象后续调用也会读取新值。
        """

        assert new_num_sms % 2 == 0, 'The SM count must be even'  # kernel 内按 pair/channel 分配 SM，要求偶数。
        Buffer.num_sms = new_num_sms  # 修改类变量，后续 dispatch/combine 使用新值。

    @staticmethod
    def capture() -> EventOverlap:  # 在当前 stream 上捕获一个 CUDA event。
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.

        中文说明：
            这个 helper 常用于把用户自定义计算 stream 与 DeepEP 通信 stream 串起来。
            例如先在当前 stream 上做预处理，再把 capture() 返回的 EventOverlap 作为
            previous_event 传给 dispatch/combine，让通信 kernel 等预处理完成后再开始。
        """
        return EventOverlap(EventHandle())  # C++ EventHandle 构造时记录当前 CUDA stream event。

    @staticmethod
    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int) -> int:  # 估算低延迟 RDMA buffer 最小字节数。
        """
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            中文：该 hint 面向 low_latency_dispatch/combine，按 BF16 payload 的最大容量估算。
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            中文：hidden 是每个 token 的特征维度，低延迟 kernel 只支持一组模板化 hidden 值。
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.

        中文说明：
            低延迟路径的 RDMA buffer 布局由 C++ kernel 固定定义，包含双缓冲 payload、
            source info、layout range、计数器和 mask 等控制区。这里直接调用 C++ 计算，
            避免 Python 侧手写公式和 device layout 不一致。
        """
        return _C.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)  # 交给 C++ 按 kernel layout 计算。

    def get_comm_stream(self) -> torch.Stream:  # 获取 legacy runtime 的通信 CUDA stream。
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.

        中文说明：
            DeepEP runtime 内部维护独立通信 stream。返回值重新包装成 torch.cuda.Stream，
            调用方可以用它做 stream wait、调试或把额外 kernel 排到同一通信流上。
        """
        ts: torch.Stream = self.runtime.get_comm_stream()  # C++ 返回一个 torch.Stream 兼容对象。
        return torch.cuda.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)  # 重新包装成 Python torch.cuda.Stream。

    def get_local_buffer_tensor(self,  # 以 Tensor 视图暴露本 rank 本地 buffer。
                                dtype: torch.dtype,  # 希望解释成的 PyTorch dtype。
                                size: Optional[torch.Size] = None,  # 可选输出 shape；None 表示返回剩余一维视图。
                                offset: int = 0,  # 以 dtype 元素为单位的起始偏移。
                                use_rdma_buffer: bool = False) -> torch.Tensor:  # True 返回 RDMA buffer，否则返回 NVLink buffer。
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            中文：该 helper 把底层 NVL/RDMA raw buffer 暴露成 torch.Tensor 视图，主要用于调试和测试。
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            中文：size 是 tensor 视图形状，不会改变底层 buffer 容量或布局。
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
            中文：offset 以 dtype 元素为单位，size 非空时会 reshape 成指定形状。
        """
        tensor = self.runtime.get_local_buffer_tensor(dtype, offset, use_rdma_buffer)  # C++ 返回从 offset 开始的一维 buffer 视图。
        if size is None:  # 未指定 size 时直接返回完整一维视图。
            return tensor  # 形状为 [remaining_elements]，dtype 为传入 dtype。

        assert tensor.numel() >= size.numel()  # 确保请求切片不会超过 buffer 剩余容量。
        return tensor[:size.numel()].view(size)  # 截取指定元素数并 reshape 成 size。

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):  # 将 combine bias 规范化成两个可选 Tensor。
        bias_0, bias_1 = None, None  # 默认无 bias。
        if isinstance(bias, torch.Tensor):  # 单个 bias Tensor。
            bias_0 = bias  # 放入 bias_0，bias_1 保持 None。
        elif isinstance(bias, tuple):  # 某些路径传入两个 bias 分量。
            assert len(bias) == 2  # tuple bias 必须恰好两个元素。
            bias_0, bias_1 = bias  # C++ combine 会分别处理两个 bias 输入。
        return bias_0, bias_1  # 返回二元组，元素可能为 None。

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:  # 按 EP rank 数返回 legacy dispatch 推荐配置。
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        # Config(num_sms, num_channels, num_warps_per_group, num_recv_warps, num_bytes_per_msg)
        # 是 legacy kernel 的手工调优表。dispatch 和 combine 使用不同表，因为 dispatch 更偏
        # scatter/write，combine 更偏 gather/reduce；rank 数变化会改变 NVLink/RDMA 的并发形态。
        # 这些参数最终传到 csrc/kernels/legacy/*.cu 的模板/launch 分支。
        config_map = {  # key 是 EP rank 数，value 是 C++ Config。
            2: Config(Buffer.num_sms, 24, 256, 6, 128),  # 2 rank 时增加 channel 数以填满 NVLink 带宽。
            4: Config(Buffer.num_sms, 6, 256, 6, 128),  # 4 rank 推荐配置。
            8: Config(Buffer.num_sms, 6, 256, 6, 128),  # 8 rank 推荐配置。
            16: Config(Buffer.num_sms, 36, 288, 20, 128),  # 16 rank internode/intranode 混合场景配置。
            24: Config(Buffer.num_sms, 32, 288, 8, 128),  # 24 rank 推荐配置。
            32: Config(Buffer.num_sms, 32, 288, 8, 128),  # 32 rank 推荐配置。
            48: Config(Buffer.num_sms, 32, 288, 8, 128),  # 48 rank 推荐配置。
            64: Config(Buffer.num_sms, 32, 288, 8, 128),  # 64 rank 推荐配置。
            96: Config(Buffer.num_sms, 20, 480, 12, 128),  # 96 rank 需要更大 warp/group 配置。
            128: Config(Buffer.num_sms, 20, 560, 12, 128),  # 128 rank 推荐配置。
            144: Config(Buffer.num_sms, 32, 720, 12, 128),  # 144 rank 推荐配置。
            160: Config(Buffer.num_sms, 28, 720, 12, 128),  # 160 rank 推荐配置。
        }  # config_map 目前是手工经验表。
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'  # 不支持的 rank 数直接报错。
        return config_map[num_ranks]  # 返回匹配 rank 数的配置。

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:  # 按 EP rank 数返回 legacy combine 推荐配置。
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {  # combine 配置表，参数含义同 dispatch Config。
            2: Config(Buffer.num_sms, 10, 256, 6, 128),  # 2 rank combine 推荐配置。
            4: Config(Buffer.num_sms, 9, 256, 6, 128),  # 4 rank combine 推荐配置。
            8: Config(Buffer.num_sms, 4, 256, 6, 128),  # 8 rank combine 推荐配置。
            16: Config(Buffer.num_sms, 4, 288, 12, 128),  # 16 rank combine 推荐配置。
            24: Config(Buffer.num_sms, 1, 288, 8, 128),  # 24 rank combine 推荐配置。
            32: Config(Buffer.num_sms, 1, 288, 8, 128),  # 32 rank combine 推荐配置。
            48: Config(Buffer.num_sms, 1, 288, 8, 128),  # 48 rank combine 推荐配置。
            64: Config(Buffer.num_sms, 1, 288, 8, 128),  # 64 rank combine 推荐配置。
            96: Config(Buffer.num_sms, 1, 480, 8, 128),  # 96 rank combine 推荐配置。
            128: Config(Buffer.num_sms, 1, 560, 8, 128),  # 128 rank combine 推荐配置。
            144: Config(Buffer.num_sms, 2, 720, 8, 128),  # 144 rank combine 推荐配置。
            160: Config(Buffer.num_sms, 2, 720, 8, 128),  # 160 rank combine 推荐配置。
        }  # combine 更偏 reduce/gather，因此 channel 数通常少于 dispatch。
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'  # 不支持 rank 数时报错。
        return config_map[num_ranks]  # 返回匹配配置。

    # noinspection PyTypeChecker
    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int,  # top-k 路由矩阵 [T,K] 和全局 expert 数。
                            previous_event: Optional[EventOverlap] = None, async_finish: bool = False,  # 前序事件与异步返回开关。
                            allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap]:  # 返回 rank/expert 布局和事件。
        """
        Calculate the layout required for later communication.

        Arguments:
            中文：layout 阶段只读取 topk_idx，不搬运 x payload，用于提前得到 rank/expert 计数。
            topk_idx: `[num_tokens, num_topk]`, dtype must be `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.
            num_experts: the number of experts.
            中文：num_experts 必须能按 rank 均分，否则 expert->rank 映射不成立。
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.
            中文：事件和 stream 归属设置影响 layout 与后续 dispatch 的 overlap。

        Returns:
            中文：这些返回值是 legacy dispatch 的控制面输入，可被 cached dispatch 复用。
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            中文：num_tokens_per_rdma_rank 只在跨节点路径存在，用于把全局 rank 压缩到同 GPU index 的 RDMA 维度。
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).

        中文说明：
            layout 阶段只统计“token 应该去哪里”，不搬运 hidden 数据。
            返回的 is_token_in_rank 是 [num_tokens, num_ranks] bool 矩阵，
            后续 dispatch 根据它生成 RDMA/NVL head、prefix 和实际发送列表。
        """
        (num_tokens_per_rank,  # 每个 rank 将接收的 token 数 [num_ranks]。
         num_tokens_per_rdma_rank,  # 每个 RDMA rank 将接收的 token 数；intranode 时可能为 None。
         num_tokens_per_expert,  # 每个 expert 将接收的 token 数 [num_experts]。
         is_token_in_rank,  # token-rank bool 矩阵 [num_tokens,num_ranks]。
         event) = self.runtime.get_dispatch_layout(  # 调用 C++ layout kernel。
            topk_idx, num_experts, getattr(previous_event, 'event', None),  # 传入路由、expert 数和可选前序事件。
            async_finish, allocate_on_comm_stream)  # 传入异步和 stream 归属开关。
        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, EventOverlap(event)  # 包装事件并返回 layout。

    # noinspection PyTypeChecker
    def dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # BF16 Tensor [T,H] 或 FP8 (data,scale) tuple。
                 handle: Optional[Tuple] = None,  # cached communication handle；提供时复用 layout。
                 num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,  # rank/RDMA rank token 计数。
                 is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,  # token-rank mask 与 expert 计数。
                 topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None,  # 原始路由矩阵和权重。
                 expert_alignment: int = 1, num_worst_tokens: int = 0,  # expert token 对齐与 CUDA graph 固定容量上界。
                 config: Optional[Config] = None,  # legacy kernel 调优配置；None 使用推荐表。
                 previous_event: Optional[EventOverlap] = None, async_finish: bool = False,  # 前序事件与异步返回开关。
                 allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],  # recv_x 和 recv_topk_idx。
            Optional[torch.Tensor], List[int], Tuple, EventOverlap]:  # recv_topk_weights、per-expert 计数、handle、event。
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        中文：dispatch 会根据 runtime.get_num_rdma_ranks() 自动选择 intranode 或 internode 实现。
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            中文：dispatch 输入可以是 BF16 tensor，也可以是 FP8 payload + scale 二元组。
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                中文：FP8 二元组的第一项仍按 token-major 排列，第二项按每 128 hidden 一组 scale。
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            中文：handle 非空时复用上一次 layout，适合路由不变的 cached dispatch。
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            中文：intranode 没有 RDMA 聚合维，所以该参数可以为 None。
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            中文：rank 级计数和 token-rank bool 矩阵由 layout kernel 生成，决定后续通信 buffer offset。
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            中文：num_tokens_per_expert 用于接收端按本地 expert 切分 recv_x。
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert indices
                selected by each token, `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            中文：expert 级计数、topk_idx/topk_weights 共同决定专家输入布局和 combine 加权回传。
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
            中文：num_worst_tokens 是 intranode CUDA graph 的容量上界，设置后 Python 不再同步真实接收数。
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            中文：previous_event 通常来自前一段 compute/communication，用于跨 stream 排依赖。
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.
            中文：num_worst_tokens/async_finish/allocate_on_comm_stream 影响 CUDA graph、异步 overlap 和 tensor 生命周期。

        Returns:
            中文：返回值包含专家输入、可选路由元数据、per-expert 计数、combine handle 和异步 event。
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            中文：recv_x 的第 0 维是本 rank 收到的专家输入 token 总数，不再等于原 num_tokens。
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            中文：recv_topk_idx/recv_topk_weights 的行数与 recv_x 对齐，描述每个接收 token 的原 top-k 槽信息。
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                中文：CUDA graph 模式下该列表为空，是为了避免把 device 计数同步回 CPU。
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            中文：handle 必须传给 combine；event 只在 async_finish=True 时表示通信完成点。

        中文说明：
            dispatch 会把输入 token 按 topk 路由发送到目标专家所在 rank。
            intranode 只走 NVLink buffer；internode 先走 RDMA 到目标节点中相同 GPU index，
            再在节点内通过 NVLink 分发到目标 NVL rank。返回的 handle 必须保存到 combine，
            因为 combine 需要按同一套 layout 把 expert 输出送回源 token。
            cached handle 模式只适合“路由不变、payload 改变”的重复调用；它不会重新统计
            topk_idx/topk_weights，因此用户不能在 cached 调用中传入新的路由张量。
        """
        check_torch_deterministic()  # 防止 PyTorch deterministic 模式插入额外填充 kernel。

        # Default config
        # Config 控制 block/warp/channel 等 kernel 参数；默认表按 group_size 选择经验配置。
        config = self.get_dispatch_config(self.group_size) if config is None else config  # 未传 Config 时按 rank 数取推荐 dispatch 配置。

        # Internode
        # runtime.get_num_rdma_ranks() > 1 表示 EP group 横跨多个 RDMA 域/节点。
        # internode 路径会先通过 RDMA 跨节点搬运，再在节点内通过 NVLink 分发；intranode 路径
        # 只需要 NVLink IPC。num_worst_tokens 依赖无 CPU sync 的固定容量布局，目前只支持 intranode。
        if self.runtime.get_num_rdma_ranks() > 1:
            assert num_worst_tokens == 0, 'Internode dispatch does not support `num_worst_tokens > 0`'  # internode 暂不支持固定最坏容量模式。
            # internode_dispatch 会继续区分 cached/non-cached，并调用 legacy internode CUDA kernel。
            return self.internode_dispatch(x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank,  # 转入 internode 两阶段 dispatch。
                                           num_tokens_per_expert, topk_idx, topk_weights, expert_alignment, config, previous_event,  # 透传 layout、路由、配置和事件。
                                           async_finish, allocate_on_comm_stream)  # 透传异步和 stream 归属策略。

        # Launch the kernel with cached or non-cached mode
        # 非 cached 模式需要调用 get_dispatch_layout 得到:
        #   num_tokens_per_rank: [num_ranks]
        #   is_token_in_rank: [num_tokens, num_ranks]
        #   num_tokens_per_expert: [num_experts]
        # cached 模式复用 handle 中的 prefix matrix 与 recv_src_idx，只重新搬运 x/x_scales。
        x, x_scales = x if isinstance(x, tuple) else (x, None)  # FP8 输入拆成 data/scale；BF16 scale 为 None。
        if handle is not None:  # cached intranode dispatch。
            # cached 模式复用上次路由，所以上层不能再传 topk_idx/topk_weights 覆盖 layout。
            assert topk_idx is None and topk_weights is None  # cached 模式不能传新路由。
            # handle 来自上一轮 non-cached dispatch；字段顺序必须与下面构造 handle 的位置严格一致。
            # rank_prefix_matrix/channel_prefix_matrix 描述发送侧分片，recv_channel_prefix_matrix/recv_src_idx
            # 描述接收侧输出布局；send_head 则给 combine 反向释放/寻址使用。
            rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle  # 解包 cached layout。
            num_recv_tokens = recv_src_idx.size(0)  # cached 接收 token 数来自源索引长度。
            # cached 调用把 topk/layout 相关参数传 None，只传已有 prefix 和 recv_src_idx。
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, event = self.runtime.intranode_dispatch(  # 只重新搬运 payload，不重新生成 metadata。
                x, x_scales, None, None, None, is_token_in_rank, None, num_recv_tokens, rank_prefix_matrix, channel_prefix_matrix,  # cached layout 参数。
                expert_alignment, num_worst_tokens, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)  # 运行配置和事件。
            # cached dispatch 不返回新的 topk/handle，因为 routing layout 未变。
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)  # cached 模式不返回新路由/handle。
        else:  # 非 cached intranode dispatch。
            # non-cached 模式必须先由 get_dispatch_layout 提供三个布局张量。
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None  # 必须先提供 layout。
            # 返回的 prefix matrix 和 send_head 会进入 handle，后续 cached dispatch/combine 都复用它们。
            recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, event = \
                self.runtime.intranode_dispatch(x, x_scales, topk_idx, topk_weights,  # 输入 payload、scale、路由和权重。
                                                num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, 0, None, None,  # layout 输入，cached 字段为空。
                                                expert_alignment, num_worst_tokens, config,  # 对齐、固定容量和调优配置。
                                                getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)  # 前序事件和异步策略。
            # handle 字段顺序和 cached 分支解包顺序强绑定；修改时必须两边一起改。
            # 这里保存的是 GPU tensor，而不是 Python 计算结果。后续 cached dispatch/combine 会把它们原样传回 C++，
            # 所以这些张量的生命周期必须至少覆盖到最后一次 combine 结束。
            handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)  # 保存 intranode cached handle。
            # 返回类型跟输入 x 保持同构：FP8 输入返回 (data,scale)，BF16 输入返回单 Tensor。
            return (
                recv_x, recv_x_scales  # FP8 路径返回 data/scale tuple。
            ) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(
                event)  # BF16 路径返回单 Tensor，并包装事件。

    # noinspection PyTypeChecker
    def combine(self, x: torch.Tensor, handle: Tuple,  # expert 输出 Tensor 和 dispatch 返回 handle。
                topk_weights: Optional[torch.Tensor] = None,  # 原 token top-k 权重 [T,K]。
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,  # 可选 combine bias。
                config: Optional[Config] = None,  # legacy combine 调优配置；None 使用推荐表。
                previous_event: Optional[EventOverlap] = None, async_finish: bool = False,  # 前序事件与异步返回开关。
                allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:  # 返回 combined_x、combined_topk_weights、event。
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        中文：combine 的“反向”方向与 dispatch 相反：expert 输出沿 handle 中记录的源位置回到原 token rank。
        Intranode kernels require all the ranks should be visible via NVLink.
        中文：这里的 addition without weights 指 C++ combine kernel 只做通信规约，权重语义由上层参数决定。
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            中文：combine 使用 dispatch 返回的 handle 反向寻址，将 expert 输出按源 token 聚合。
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            中文：legacy combine 的 topk_weights 行数与 expert 输出 x 行数一致。
            bias: 0, 1 or 2 `[num_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            中文：topk_weights/bias 只影响数学规约结果，不改变通信 slot metadata。
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            中文：previous_event 会被传到底层 stream wait，保证专家计算完成后再开始 combine 通信。
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.
            中文：config 和 async 参数决定 combine kernel 形状以及调用方是否立刻等待。

        Returns:
            中文：返回原 token 顺序的 combined_x、可选 combined_topk_weights 和事件。
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).

        中文说明：
            combine 是 dispatch 的反向通信：dispatch 把原 token 发给目标 expert，expert 计算后，
            combine 再按照 handle 中保存的 src_idx/head/prefix 把结果送回原 token 所属 rank。
            intranode handle 的关键字段是 rank_prefix_matrix、接收侧 channel_prefix_matrix、
            recv_src_idx 和 send_head；internode handle 还会多出 RDMA/NVL 两级 prefix。
        """
        check_torch_deterministic()  # 防止 deterministic 模式额外 kernel 干扰通信顺序。

        # Default config
        # combine 的推荐配置和 dispatch 分开维护，因为反向规约的访存/同步模式不同。
        config = self.get_combine_config(self.group_size) if config is None else config  # 未传 Config 时按 rank 数取推荐 combine 配置。

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            # 多 RDMA rank 时必须走 internode_combine；intranode_combine 只理解单节点 NVLink layout。
            return self.internode_combine(x, handle, topk_weights, bias, config, previous_event, async_finish, allocate_on_comm_stream)  # 多 RDMA rank 转入 internode combine。

        # NOTES: the second `_` is for the sending side, so we should use the third one
        # intranode dispatch 返回的 handle 中同时有发送侧与接收侧的 channel_prefix_matrix。
        # combine 是 dispatch 的反向路径，要用“接收侧视角”的 prefix matrix 和 src_idx，
        # 所以这里跳过第二个发送侧矩阵，取第三个矩阵作为 combine 的 channel 布局。
        # is_recv_token_in_rank 描述 combine 返回时哪些原 token 需要接收当前 rank 的 expert 输出。
        rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
        # _unpack_bias 将 None/Tensor/(Tensor,Tensor) 统一成 bias_0/bias_1 两个可空参数，匹配 C++ 签名。
        bias_0, bias_1 = Buffer._unpack_bias(bias)  # bias 可以是单 Tensor 或双 Tensor，C++ 侧分别加到规约结果上。

        # Launch the kernel
        # intranode_combine 会按 src_idx 读取 expert 输出 x，并根据 prefix/head 写回原 token 所在 rank。
        recv_x, recv_topk_weights, event = self.runtime.intranode_combine(x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix,  # payload/权重/bias/源索引/rank prefix。
                                                                          channel_prefix_matrix, send_head, config,  # 接收侧 channel prefix、发送 head 和调优配置。
                                                                          getattr(previous_event, 'event',
                                                                                  None), async_finish, allocate_on_comm_stream)  # 前序事件、异步和 stream 分配策略。
        # recv_topk_weights 只有 topk_weights 非空时才有意义；event 对应通信 stream 上的最后一个 combine kernel。
        return recv_x, recv_topk_weights, EventOverlap(event)  # 包装 C++ event，保持和 dispatch API 对称。

    # noinspection PyTypeChecker
    def internode_dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # BF16 Tensor 或 FP8 (data,scale)。
                           handle: Optional[Tuple] = None,  # cached internode handle，包含 RDMA/NVLink 两阶段 layout。
                           num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,  # rank 和 RDMA rank token 计数。
                           is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,  # token-rank mask 与 expert token 计数。
                           topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None, expert_alignment: int = 1,  # 路由、权重和 expert 对齐。
                           config: Optional[Config] = None,  # legacy internode dispatch 调优配置，必须由上层传入。
                           previous_event: Optional[EventOverlap] = None, async_finish: bool = False,  # 前序事件和异步返回开关。
                           allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],  # recv_x 与 recv_topk_idx。
            Optional[torch.Tensor], List[int], Tuple, EventOverlap]:  # recv_topk_weights、per-expert 计数、handle、event。
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.

        中文说明：
            internode_dispatch 将通信拆成 RDMA 和 NVLink 两层：先按 RDMA rank 聚合跨节点发送，
            再在目标节点内按 NVL rank 分发。返回的 handle 同时保存两层 prefix 和 head，
            因此只应由同一批 token 的 internode_combine 复用。
            与 intranode handle 相比，internode handle 多了 recv_rdma_rank_prefix_sum、
            recv_gbl_rank_prefix_sum、send_rdma_head 和 send_nvl_head；这些字段分别描述跨节点
            和节点内两个 ring 的消费位置。
        """
        assert config is not None  # internode_dispatch 需要调用方已选择好 Config。

        # Launch the kernel with cached or non-cached mode
        # FP8 输入以 (data,scale) 形式传入，C++ 内部按 hidden_int4 和 scale stride 分别搬运。
        x, x_scales = x if isinstance(x, tuple) else (x, None)  # FP8 输入拆成数据和 scale；BF16 scale=None。
        if handle is not None:  # cached internode layout 模式。
            # internode cached handle 比 intranode 更复杂：它同时保存 RDMA channel、global(NVLink)
            # channel、接收侧 prefix、源 token metadata、RDMA/NVL send head。二阶段通信必须复用
            # 完全一致的布局，否则 combine 会无法从 src_meta 找回原 token。
            assert topk_idx is None and topk_weights is None  # 复用 handle 时不能重新提供路由。
            (is_token_in_rank,
             # 中文：发送侧 prefix 描述 payload 如何分配到 RDMA/global channel。
             rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
             # 中文：接收侧 prefix 描述本 rank 收到的 RDMA/global rank 累积位置。
             recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
             recv_src_meta, send_rdma_head, send_nvl_head) = handle
            # recv_src_meta 形状为 [num_recv_tokens]，每个元素是 C++ SourceMeta，用于 combine 反向寻址。
            # cached internode dispatch 只重新搬运 x/x_scales。recv_src_meta/send_*_head 仍来自原 handle，
            # 因而 topk 路由、rank 数、num_max_tokens_per_rank 等 layout 条件都必须和原 dispatch 一致。
            num_recv_tokens = recv_src_meta.size(0)  # 本 rank 最终接收 token 数。
            # send_nvl_head 的行数等于 RDMA 阶段转入节点内 NVLink 的 token 数。
            num_rdma_recv_tokens = send_nvl_head.size(0)  # RDMA 阶段接收/转发 token 数。
            # cached 模式不会重新生成 recv_src_meta/send_head，因此 layout 相关返回值全部丢弃。
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, _, _, _, _, event = self.runtime.internode_dispatch(  # 只重新搬运 payload，丢弃新 layout 输出。
                x, x_scales, topk_idx, topk_weights, None, None, is_token_in_rank, None, num_recv_tokens, num_rdma_recv_tokens,  # cached 模式传入固定容量。
                rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,  # 复用 RDMA/global prefix。
                expert_alignment, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)  # 传入对齐、配置、事件和异步开关。
            # cached internode dispatch 只返回 payload 和 event；路由/handle 由调用方继续持有。
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)  # cached 模式不返回新 metadata。
        else:  # 非 cached 模式，需要完整计算 layout。
            # num_tokens_per_rdma_rank 是 internode 特有输入，描述每个 RDMA rank 的聚合 token 数。
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None  # layout 输入必须齐全。
            # C++ 返回两套 prefix：发送侧 rdma/gbl prefix，以及接收侧 recv_* prefix，combine 都会复用。
            # 中文：recv_* prefix 描述本 rank 作为接收端看到的 RDMA 与全局 rank 累积布局。
            (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list,
             # 中文：以下 prefix/head/metadata 会被打包成 internode combine handle。
             rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
             # 中文：recv_rdma/gbl prefix 的行列与 C++ internode_combine 解包顺序严格对应。
             recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
             recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
             recv_src_meta, send_rdma_head, send_nvl_head, event) = self.runtime.internode_dispatch(
                x, x_scales, topk_idx, topk_weights,  # 输入 token、scale、路由和权重。
                num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert,  # 预先计算的 dispatch layout。
                0, 0, None, None, None, None,  # 非 cached 模式不传 cached token 数和 prefix。
                expert_alignment, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)  # C++ kernel 运行参数。
            # handle 字段顺序与 internode_combine 解包顺序强绑定；RDMA/NVL 两级 head 缺一不可。
            # is_token_in_rank 放在 handle 第一项，是因为 combine 需要知道原 token 是否期待某个 rank 的贡献；
            # 后续 prefix/head 则分别服务 RDMA receiver、NVL forwarder 和最终输出规约。
            handle = (is_token_in_rank, rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, recv_rdma_channel_prefix_matrix,  # 保存发送/接收侧 prefix。
                      recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, recv_src_meta, send_rdma_head,  # 保存 src metadata 和 send heads。
                      send_nvl_head)  # NVLink 阶段 send head。
            # 返回类型同样与输入 x 同构，FP8 返回 (recv_x, recv_x_scales)，BF16 返回 recv_x。
            return (
                recv_x, recv_x_scales
            ) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(  # 返回 payload、路由、计数、handle。
                event)  # 包装异步事件。

    # noinspection PyTypeChecker
    def internode_combine(self, x: torch.Tensor, handle: Union[tuple, list],  # expert 输出 Tensor 和 internode dispatch handle。
                          topk_weights: Optional[torch.Tensor] = None,  # 原 token top-k 权重。
                          bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,  # 可选 epilogue bias。
                          config: Optional[Config] = None,  # legacy internode combine 调优配置。
                          previous_event: Optional[EventOverlap] = None, async_finish: bool = False,  # 前序事件与异步返回。
                          allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:  # 返回 combined_x、combined_topk_weights、event。
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.

        中文说明：
            这是 public combine 在 internode 场景下调用的内部实现。它要求 handle 来自
            internode_dispatch，handle 中保存了 RDMA/NVL 两级 prefix、源 token metadata、
            以及 send_rdma_head/send_nvl_head。缺少这些字段时，combine 无法知道每个 expert
            输出应沿哪条 RDMA/NVLink 路径回到原 token。
        """
        assert config is not None  # combine 也必须有明确 Config。

        # Unpack handle and bias
        # internode combine 复用 internode_dispatch 的 handle。注意 handle 前三个字段中，
        # 第一个 is_combined_token_in_rank 用于判断原 token 是否需要来自某 rank 的贡献；
        # 中间两个是 dispatch 侧矩阵，combine 不直接使用，所以用 `_` 丢弃。
        is_combined_token_in_rank, \
            _, _, \
            rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, gbl_rank_prefix_sum, \
            src_meta, send_rdma_head, send_nvl_head = handle  # 解包后的矩阵维度与 internode_dispatch 返回 handle 完全对应。
        bias_0, bias_1 = Buffer._unpack_bias(bias)  # 将 None/Tensor/tuple bias 拆成 C++ 可选参数。

        # Launch the kernel
        combined_x, combined_topk_weights, event = self.runtime.internode_combine(x, topk_weights, bias_0, bias_1, src_meta,  # 调用 C++ 两阶段 combine。
                                                                                  is_combined_token_in_rank, rdma_channel_prefix_matrix,  # 原 token/rank mask 与 RDMA prefix。
                                                                                  rdma_rank_prefix_sum, gbl_channel_prefix_matrix,  # RDMA rank 和 global channel prefix。
                                                                                  send_rdma_head, send_nvl_head, config,  # 发送 head 与调优配置。
                                                                                  getattr(previous_event, 'event',  # 可选前序事件。
                                                                                          None), async_finish, allocate_on_comm_stream)  # 异步和分配策略。
        return combined_x, combined_topk_weights, EventOverlap(event)  # 返回 combine 结果与事件。

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:  # 清空低延迟双缓冲区；三个参数定义固定容量 layout。
        """
        As low-latency kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.
        中文：这里的 dirty 指控制面区域残留旧计数/flag，而不是 payload 数值本身。
        For example, after running the normal dispatch/combine, you must run this function before executing any
            low-latency kernel.

        Arguments:
            中文：三个参数必须和后续 low_latency_dispatch 使用的固定 layout 一致。
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.

        中文说明：
            低延迟路径复用 RDMA buffer 中的控制面字段，例如 recv_count、layout_range 和双缓冲状态。
            如果前面跑过普通 dispatch/combine，buffer 中可能残留非零控制值；清理后才能安全进入
            low_latency_dispatch/low_latency_combine。
        """
        self.runtime.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)  # C++ 按固定容量 layout 将必要区域置零。

    # noinspection PyTypeChecker
    def low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,  # BF16 输入 [T,H] 与路由矩阵 [T,K]。
                             num_max_dispatch_tokens_per_rank: int, num_experts: int,  # 固定容量 token 上界和全局 expert 数。
                             cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,  # 可选累计 per-expert 接收统计。
                             dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,  # 可选等待接收耗时统计 [num_ranks,num_ranks]。
                             use_fp8: bool = True, round_scale: bool = False, use_ue8m0: bool = False,  # FP8 与 scale-factor 格式开关。
                             async_finish: bool = False, return_recv_hook: bool = False) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:  # 返回接收 buffer/count/handle/event/hook。
        """
        A low-latency implementation for dispatching with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            中文：低延迟 dispatch 依赖 IBGDA 直接写远端内存，因此要求跨 rank 都有 RDMA 可达性。
            (specifically, IBGDA must be enabled).
        中文：低延迟路径不走 NCCL all-to-all，而是直接用 IBGDA 写远端 buffer 和计数器。
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            中文：这里的 two buffers 是设备端双缓冲 phase，不是 Python 返回 tuple 的元素数量。
            low-latency kernels' result tensors at a single moment.
        中文：双缓冲限制意味着调用方不能长期保存超过两轮的返回 tensor 引用。

        Arguments:
            中文：低延迟 dispatch 固定输出容量，不像 high-throughput dispatch 那样返回紧凑 recv_x。
            x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
                中文：hidden 只支持若干编译期模板特化值；不匹配会在 C++ wrapper 里 assert。
                supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
            topk_idx: `torch.Tensor` with `deep_ep.topk_idx_t` (typically `torch.int64`), shaped as `[num_tokens, num_topk]`,
                中文：topk_idx 中的 -1 会被当作空路由槽，不会产生发送计数。
                only several top-k shapes are supported. `-1` indices (not selecting any expert) are supported.
            中文：x 的 token 维是当前 rank 本地 token 数，topk_idx 的第二维 K 决定每个 token 最多发往多少 expert。
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            num_experts: the number of all experts.
            中文：num_max_dispatch_tokens_per_rank 是固定槽位容量，所有 rank 必须一致。
            cumulative_local_expert_recv_stats: a cumulative expert count tensor for statistics, which should have shape
                `[num_local_experts]` and be typed as `torch.int`. This is useful for online service EP load balance
                中文：该统计可帮助在线服务观测每个本地 expert 的负载倾斜。
                monitoring.
            中文：该统计张量是累加型监控数据，kernel 会原地累加，不参与 dispatch 的地址计算。
            dispatch_wait_recv_cost_stats: a cumulative time spent waiting to receive each token tensor for statistics,
                which should have shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
                This is useful for detecting and precisely localizing slow anomalies.
            中文：等待耗时矩阵的行列通常对应源/目标 rank，可用来定位某一对 rank 的 RDMA 慢点。
            use_fp8: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            round_scale: whether round the scaling factors into power of 2.
            中文：round_scale=True 时 scale 可进入 UE8M0 打包路径，降低 scale 带宽。
            use_ue8m0: whether use UE8M0 as scaling factor format (available only with `round_scale=True`).
            中文：FP8/UE8M0 只改变 payload 与 scale 格式，不改变 src_info/count 的控制面含义。
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                中文：hook 模式适合把接收等待延后到专家计算真正需要输入之前。
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            中文：return_recv_hook=True 时 dispatch 只提交请求，调用者稍后执行 hook 才等待数据真正到达。

        Returns:
            中文：返回的 recv_x 是固定容量专家 buffer，recv_count/layout_range 标记哪些槽有效。
            recv_x: a tensor or tuple with received tokens for each expert.
                With `use_fp8=True`: the first element is a `torch.Tensor` shaped as
                中文：固定容量第二维等于每 rank 容量乘 world size，便于低延迟路径避免 CPU 计数同步。
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.float8_e4m3fn`.
                中文：FP8 路径的第一项是固定容量专家 token buffer，只有 recv_count 标记的前缀有效。
                The second tensor is the corresponding scales for the first element with shape
                中文：scale tensor 的第二维与 data tensor 对齐，第三维按 128-channel 分组。
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `torch.float`,
                if `use_ue8m0=False`. With `use_ue8m0=True`, the second one is packed and shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 512]` with type `torch.int`.
                中文：UE8M0 会把 4 个 128-channel scale 打包到 int，scale tensor 最后一维变成 hidden//512。
                Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
                With `use_fp8=False`, the result would be a tensor shaped as
                中文：BF16 路径没有独立 scale tensor，因此 recv_x 直接是三维专家 buffer。
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`.
                Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are,
                中文：无效槽位不能直接参与 GEMM，必须按 recv_count/layout_range 截取有效前缀。
                as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph if synced).
                中文：BF16 路径和 FP8 路径共享同一固定容量 slot 语义，区别只在 payload dtype/scale。
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receives. As mentioned before, not all tokens are valid in `recv_x`.
            中文：调用方应只处理每个 expert 前 recv_count[e] 个 slot，其余固定容量槽可能是旧值。
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
            中文：handle 内包含 src_info/layout_range/双缓冲 phase，combine 必须使用同一份 handle。

        中文说明：
            低延迟 dispatch 使用固定双缓冲和 IBGDA，目标是减少 CPU 同步和 launch 前等待。
            返回 buffer 会复用 runtime 内部 RDMA symmetric memory，生命周期与 Buffer 对象绑定。
            因为输出 buffer 是固定容量，所以 recv_x 里有无效槽位，真实有效数量由 packed_recv_count
            和 handle 里的 layout_range 描述。return_recv_hook=True 时，调用方必须稍后执行 hook 等待接收完成。
            该 API 与 high-throughput dispatch 最大区别是“不返回精确 num_recv_tokens 形状”：
            它固定返回 [num_local_experts, num_ranks*Tmax, hidden] 容量，靠 count/range 指示有效区间。

        中文参数速查：
            x: BF16 输入 token，形状 [T, H]，T 必须小于 num_max_dispatch_tokens_per_rank。
            topk_idx: 路由矩阵，形状 [T, K]，每个元素是全局 expert id，-1 表示该 top-k 槽无效。
            num_max_dispatch_tokens_per_rank: 固定容量 Tmax，所有 rank 必须一致，因为 buffer layout 由它静态决定。
            num_experts: 全局 expert 总数，C++ 会用它推导本地 expert 数和 expert 分片。
            cumulative_local_expert_recv_stats: 可选 [num_local_experts] int 累计计数，不参与正确性，只用于监控。
            dispatch_wait_recv_cost_stats: 可选 [num_ranks, num_ranks] int64 等待耗时矩阵，用于定位慢 rank。
            use_fp8/round_scale/use_ue8m0: 控制接收 buffer 的数据格式；FP8 时会额外返回 scale tensor。
            async_finish: True 表示 Python 当前 stream 不等待通信完成，调用方需要用返回 event 建立依赖。
            return_recv_hook: True 表示 kernel 只发起接收相关请求，真正等待数据到达由 hook 延后完成。

        中文返回速查：
            recv_x: 固定容量接收 buffer；use_fp8=True 时为 (data, scale)，否则为 BF16 tensor。
            recv_count: [num_local_experts] int，记录每个本地 expert 实际有效 token 数。
            handle: (src_info, layout_range, Tmax, H, num_experts)，必须原样传给 low_latency_combine。
            event: 通信 stream 上的 CUDA event，async_finish=True 时用于保活并建立跨 stream 依赖。
            hook: return_recv_hook=True 时非空，调用后才保证数据真正到达接收 buffer。
        """
        check_torch_deterministic()  # 防止 PyTorch deterministic 模式插入填充 kernel。

        # 低延迟 dispatch 采用固定容量布局，不等待 CPU 精确统计接收 token 数：
        # recv_x 容量为 [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]。
        # packed_recv_count: [num_local_experts] 才表示每个 expert 实际收到多少 token。
        # 这也是它能进入 CUDA graph/低 CPU 开销路径的原因，但调用方必须只消费 count 范围内的数据。
        # QP depth 的 +1 是为了覆盖控制/结束类 WR，*2 对应低延迟路径双缓冲轮转。
        assert self.nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2  # QP depth 必须覆盖每 peer 最坏 WR 数。
        # 返回值里 packed_recv_x_scales 在 use_fp8=False 时仍由 C++ 返回占位值，Python 侧通过 use_fp8 决定是否暴露。
        # C++ runtime 会在通信 stream 上发 IBGDA kernel，并把 event 记录在同一 stream 上。
        packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, hook = \
            self.runtime.low_latency_dispatch(x, topk_idx,  # 调用 C++ 低延迟 dispatch。
                                              cumulative_local_expert_recv_stats,  # 可选 per-expert 累计统计 [num_local_experts]。
                                              dispatch_wait_recv_cost_stats,  # 可选等待统计 [num_ranks,num_ranks]。
                                              num_max_dispatch_tokens_per_rank, num_experts,  # 固定容量上界与 expert 总数。
                                              use_fp8, round_scale, use_ue8m0,  # FP8/scale 格式配置。
                                              async_finish, return_recv_hook)  # 异步事件和分离 recv hook 开关。
        # handle 中保存 combine 所需的源信息和布局范围：
        # - packed_recv_src_info：每个接收 slot 对应的原 rank/token/top-k 信息；
        # - packed_recv_layout_range：每个 expert 的有效 slot 范围；
        # - num_max_dispatch_tokens_per_rank/hidden/num_experts：用于还原固定容量 buffer 维度。
        # 注意 handle 不是 Python 对象类而是 tuple，是为了让调用方能轻量传给 low_latency_combine，
        # 同时保持 CUDA graph 捕获期间没有额外 Python 状态变更。
        handle = (packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, x.size(1), num_experts)  # combine 所需 handle：[src_info,layout_range,Tmax,H,E]。
        # async_finish=True 时，event 可能还没完成；这些 tensor 必须被 EventOverlap 持有到等待完成。
        tensors_to_record = (x, topk_idx, packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info,  # 异步模式下保活输入/输出。
                             packed_recv_layout_range, cumulative_local_expert_recv_stats)  # 保活 layout 与统计 tensor。
        # use_fp8=True 返回 (data, scales)，use_fp8=False 只返回 BF16 data，保持上层训练/推理代码接口直观。
        return (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x, packed_recv_count, handle, \
            EventOverlap(event, tensors_to_record if async_finish else None), hook  # 返回接收 buffer、count、handle、事件和 hook。

    # noinspection PyTypeChecker
    def low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,  # expert 输出、原路由和权重。
                            handle: tuple, use_logfmt: bool = False, zero_copy: bool = False, async_finish: bool = False,  # dispatch handle、格式、zero-copy、异步开关。
                            return_recv_hook: bool = False, out: Optional[torch.Tensor] = None,  # 是否返回接收 hook，以及可选原地输出。
                            combine_wait_recv_cost_stats: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, EventOverlap, Callable]:  # 返回 combined_x、event、hook。
        """
        A low-latency implementation for combining tokens (reduce **with weights**) with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            中文：这里要求所有 rank 都能通过 IBGDA 互达，因此单机 NVLink-only 环境不适合该路径。
            (specifically, IBGDA must be enabled).
        中文：combine 与 dispatch 共享同一套双缓冲约束，返回 tensor 可能引用内部 RDMA buffer。
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            中文：low_latency_combine 根据 dispatch handle 把专家输出返回源 token，并按 topk_weights 加权求和。
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`,
                the local calculated tokens to be sent to this original rank and reduced.
            中文：x 必须和 low_latency_dispatch 返回的固定容量专家 buffer layout 一致。
            topk_idx: `[num_combined_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                中文：topk_idx 的 token 维是原始 token 数，不是固定容量专家 buffer 的 slot 数。
                indices selected by the dispatched tokens. `-1` indices (not selecting any expert) are supported. Note that,
                `num_combined_tokens` equals to the number of dispatched tokens.
            中文：num_combined_tokens 是 combine 后原 token 数，通常等于 dispatch 输入 token 数。
            topk_weights: `[num_combined_tokens, num_topk]` with `torch.float`, the expert weights selected by the dispatched
                tokens. The received tokens will be reduced with the weights in this tensor.
            handle: the communication handle given by the `dispatch` function.
            中文：topk_idx/topk_weights 必须对应 dispatch 时的原始 token 顺序，handle 提供 slot 到源 token 的映射。
            use_logfmt: whether to use an internal "LogFMT with dynamic per-64-channel cast" format (10 bits).
            zero_copy: whether the tensor is already copied into the RDMA buffer, should be cooperative
                with `get_next_low_latency_combine_buffer`.
            中文：LogFMT 是内部压缩格式；zero_copy 需要和 get_next_low_latency_combine_buffer 配套使用。
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                中文：hook 模式下本函数返回后，combined_x 还不能立即被当前 stream 安全读取。
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            中文：hook 模式把“发起通信”和“等待数据到达”拆开，便于与专家计算或后续操作 overlap。
            out: the in-place output tensor, if set, the kernel will write the result to this tensor and return it directly.
            combine_wait_recv_cost_stats: a cumulative time spent waiting to receive each token tensor for statistics,
                which should have shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
                中文：统计矩阵按 rank 对累计等待时间，方便定位某个通信方向异常。
                This is useful for detecting and precisely localizing slow anomalies.
            中文：out 支持原地输出；combine_wait_recv_cost_stats 用于定位哪个 rank 的返回等待慢。

        Returns:
            中文：返回 combined_x 和异步完成信息；hook 仅在 return_recv_hook=True 时有效。
            combined_x 的行顺序与原始 dispatch 输入 token 顺序一致，可直接接回 MoE 主干。
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).

        中文说明：
            low_latency_combine 是 low_latency_dispatch 的反向路径：每个 expert 的输出按 src_info
            找回原 token/rank，并用 topk_weights 做加权规约。zero_copy=True 时，x 必须已经写在
            get_next_low_latency_combine_buffer 返回的 RDMA buffer 中，否则 runtime 会先执行一次本地复制。
            use_logfmt 会启用内部 10-bit LogFMT 压缩通道，减少 RDMA 搬运量；这属于低延迟专用格式，
            对外返回的 combined_x 仍是 BF16。

        中文参数速查：
            x: expert 计算后的固定容量输出，形状 [num_local_experts, num_ranks*Tmax, H]。
            topk_idx/topk_weights: 原始 dispatch token 的路由和权重，形状均为 [num_combined_tokens, K]。
            handle: low_latency_dispatch 返回的五元组，提供 src_info、layout_range 和静态容量参数。
            use_logfmt: 启用内部 LogFMT 压缩；C++ 会在发送端压缩、接收端恢复并累加。
            zero_copy: True 时 x 已经位于下一块 RDMA buffer 中，runtime 不再做本地复制。
            async_finish: True 时只返回 event，不让当前 stream 同步等待通信完成。
            return_recv_hook: True 时延后等待接收完成，适合与专家计算或其他通信重叠。
            out: 可选原地输出 tensor，提供后 combined_x 会直接引用它。
            combine_wait_recv_cost_stats: 可选 [num_ranks, num_ranks] int64，用于累计接收等待耗时。
            中文：该统计矩阵只做性能诊断，不参与低延迟 combine 的正确性判断。

        中文返回速查：
            combined_x: [num_combined_tokens, H] BF16，已经按 topk_weights 对多路 expert 输出加权规约。
            event: 通信 stream 事件，异步模式下 EventOverlap 会保活输入、handle 和输出 tensor。
            hook: return_recv_hook=True 时由调用方执行，用于完成被延后的接收侧等待。
        """
        check_torch_deterministic()  # 防止额外填充 kernel 干扰低延迟通信。

        src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle  # 解包 dispatch handle 中的固定容量 layout。
        # layout_range 的维度由 C++ low-latency kernel 固定，按 local expert 记录该 expert 在 x 中的有效 token 范围；
        # src_info 则保存每个有效 token 的原始 rank/token/top-k 信息，用来把结果发回原 rank。
        # src_info/layout_range 是 dispatch 写出的 GPU tensor，不需要 CPU 同步即可被 combine kernel 直接消费。
        # low_latency_combine 的输入 x 是 expert 计算后的固定容量布局。
        # 如果 zero_copy=True，调用方已经把数据写入 get_next_low_latency_combine_buffer 返回的 RDMA buffer，
        # kernel 可以跳过一次本地复制；否则 runtime 会先把 x 复制到当前低延迟双缓冲区。
        # 这里再次检查 QP depth，是因为 combine 也会向每个 rank 发起最多 Tmax 级别的 IBGDA 写/读请求。
        assert self.nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2  # 与 dispatch 相同，保证 WR 不会超出 QP 深度。
        # topk_idx/topk_weights 的形状是 [num_combined_tokens,num_topk]，combine kernel 用它把多路 expert 输出加权求和。
        combined_x, event, hook = self.runtime.low_latency_combine(x, topk_idx, topk_weights, src_info, layout_range,  # 调用 C++ 低延迟 combine。
                                                                   combine_wait_recv_cost_stats, num_max_dispatch_tokens_per_rank,  # 等待统计和容量上界。
                                                                   num_experts, use_logfmt, zero_copy, async_finish, return_recv_hook, out)  # expert/格式/zero-copy/输出参数。
        # combined_x 可能是 out 本身，也可能是 runtime 分配的双缓冲输出；异步模式都需要保活。
        tensors_to_record = (x, topk_idx, topk_weights, src_info, layout_range, combined_x)  # 异步模式保活所有相关 tensor。
        # hook 只在 return_recv_hook=True 时有效；调用方必须执行它才能真正完成接收侧等待。
        return combined_x, EventOverlap(event, tensors_to_record if async_finish else None), hook  # 返回 combine 输出、事件、hook。

    def low_latency_update_mask_buffer(self, rank_to_mask: int, mask: bool = False):  # 更新低延迟通信的 rank mask。
        # mask buffer 在 C++ runtime 中常驻，后续 low-latency clean/dispatch/combine 都会读取它。
        """
        Mask (unmask) a rank during communication (dispatch, combine, and clean)

        Arguments:
            rank_to_mask: the rank to mask (unmask).
            mask: if True, will mask the rank (do not recv from/send to the rank), otherwise will unmask the rank.

        中文说明：
            mask buffer 是低延迟通信的动态故障/缩容开关。mask=True 时，后续 dispatch/combine/clean
            都会跳过 rank_to_mask 对应的对端 rank；mask=False 则恢复通信。
        """
        # rank_to_mask 是 EP group 内全局 rank，而不是节点内 NVL rank 或 RDMA rank。
        self.runtime.low_latency_update_mask_buffer(rank_to_mask, mask)  # C++ 修改 mask buffer 中对应 rank 的状态。

    def low_latency_query_mask_buffer(self, mask_status: torch.Tensor):  # 查询所有 rank 当前 mask 状态。
        # 查询结果写入调用方提供的 CUDA tensor，避免返回 Python list 导致 CPU 同步。
        """
        Query the mask status of all ranks

        Arguments:
            mask_status: `[num_ranks]` with `torch.int`, the mask status of each rank. `1` means mask and `0` means unmasked.

        中文说明：
            调用方需要提前传入形状 [num_ranks]、dtype=torch.int 的 CUDA tensor。
            C++ runtime 会把当前 mask buffer 拷贝到这个 tensor 中，1 表示被屏蔽，0 表示可通信。
        """
        # mask_status 的生命周期由调用方管理；runtime 只负责在当前/通信 stream 上写入。
        self.runtime.low_latency_query_mask_buffer(mask_status)  # C++ 将状态写入 mask_status，形状 [num_ranks]。

    def low_latency_clean_mask_buffer(self):  # 清空低延迟 mask buffer。
        # clean 会把所有 rank 设为 unmasked，通常用于恢复完整 EP 拓扑。
        """
        Clean the mask buffer

        中文说明：
            将所有 rank 的 mask 状态清零，恢复完整通信拓扑。通常在动态屏蔽测试结束、
            或新一轮服务状态确认后调用。
        """
        # 这个操作影响后续低延迟 API，不会修改 high-throughput legacy dispatch/combine 的路由。
        self.runtime.low_latency_clean_mask_buffer()  # 恢复所有 rank 为可通信状态。

    def get_next_low_latency_combine_buffer(self, handle: object):  # 获取下一次 low_latency_combine 可用的 zero-copy RDMA buffer。
        # zero-copy combine 的正确顺序是：dispatch -> get_next_low_latency_combine_buffer -> 写入 buffer -> combine(zero_copy=True)。
        """
        Get the raw registered RDMA buffer tensor for next low-latency combine, so that the next combine kernel can skip the copying.

        Arguments:
            中文：handle 提供下一块双缓冲区的 layout，调用方需要按该 layout 写入 expert 输出。
            handle: the communication handle given by the `dispatch` function.

        Returns:
            buffer: the raw RDMA low-latency buffer as a BF16 PyTorch tensor with shape
                中文：该 buffer 的形状与 low_latency_dispatch BF16 recv_x 一致，可直接作为 zero_copy combine 输入。
                `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`, you should fill this buffer
                by yourself.

        中文说明：
            这个接口服务于 zero-copy combine。调用方拿到的是下一块低延迟 RDMA 双缓冲区，
            形状与 low_latency_combine 输入 x 一致：[本地 expert 数, rank 数*Tmax, hidden]。
            填完后再以 zero_copy=True 调 combine，kernel 就直接从该 RDMA buffer 发数据。
        """
        # handle 中的 src_info/layout_range 不参与取 buffer，但保持解包一致可以验证这是 dispatch 返回的 handle。
        src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle  # 解包固定容量 layout 参数。
        # C++ runtime 会根据双缓冲状态返回“下一块”可写 RDMA buffer，调用方不能长期持有超过两轮结果。
        return self.runtime.get_next_low_latency_combine_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)  # 返回 [local_experts,R*Tmax,H] BF16 buffer。
