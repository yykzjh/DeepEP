# comm.py 负责 Elastic 路径的 NCCL communicator 获取与缓存。
import os  # 读取 EP_REUSE_NCCL_COMM 等环境变量。

import torch                    # 获取 CUDA backend 和 torch.device。
import torch.distributed as dist # 使用 ProcessGroup、all_gather_object 等分布式 API。

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # C++ 扩展，提供 NCCL unique id 和 communicator 创建/销毁函数。


class NCCLCommHandle:  # Python 侧 NCCL communicator 句柄包装。
    """
    A wrapper around a raw NCCL communicator. Manages the lifecycle of the communicator if created by DeepEP,
    or simply wraps an existing one if obtained from PyTorch.

    Attributes:
        nccl_comm: the raw NCCL communicator.
        managed: whether the communicator was created by DeepEP and should be destroyed when this handle is dropped.

    中文说明：
        Elastic kernel 的 Gin backend 需要裸 ncclComm_t。新 PyTorch 可以复用 ProcessGroup
        内部 communicator；旧 PyTorch 需要 DeepEP 自己创建一个 communicator。managed 字段
        正是用来区分“谁负责销毁这个 ncclComm_t”。
    """

    def __init__(self, nccl_comm: int, managed: bool):  # 保存裸 ncclComm_t 指针整数和生命周期标志。
        # nccl_comm 是 C++ 侧传回来的裸 ncclComm_t 指针整数。managed=True 表示这个 communicator
        # 是 DeepEP 自己创建的，Python 对象析构时要调用 _C.destroy_nccl_comm；managed=False 表示
        # 复用了 PyTorch backend 内部 communicator，生命周期归 PyTorch 管，DeepEP 不能销毁。
        self.nccl_comm = nccl_comm          # C++ 侧 reinterpret_cast<int64_t> 返回的 ncclComm_t。
        self.managed = managed              # True 表示 DeepEP 创建并负责销毁该 communicator。
        self.destroy = _C.destroy_nccl_comm # 缓存销毁函数，析构时避免重复属性查找。

    def __del__(self):                      # Python 对象析构时尝试释放托管 communicator。
        # 注意：__del__ 发生在 Python GC 时，进程退出阶段模块属性可能已被清理，所以 destroy 在 __init__ 中缓存。
        if self.managed:                    # 只有 DeepEP 自建 communicator 才能销毁。
            self.destroy(self.nccl_comm)    # 调用 C++ ncclCommAbort。

    def get(self) -> int:
        """
        Get the raw NCCL communicator.

        Returns:
            nccl_comm: the raw NCCL communicator.
        """
        return self.nccl_comm  # 返回裸指针整数，供 C++ ElasticBuffer 构造函数使用。

_storage = dict()  # 以 ProcessGroup 为 key 的 NCCLCommHandle cache。
# cache 的 key 是 ProcessGroup 对象本身；同一个 group 多次构造 ElasticBuffer 时会复用同一个 handle。


def get_nccl_comm_handle(group: dist.ProcessGroup) -> NCCLCommHandle:
    """
    Get or create an NCCL communicator handle for the given process group.
    中文：该 handle 是 Python 到 C++ runtime 的 NCCL communicator 桥接对象。
    Results are cached, so subsequent calls with the same group return the same handle.

    Arguments:
        group: the communication group.

    Returns:
        handle: the NCCL communicator handle.

    中文说明：
        这个函数优先复用 PyTorch 已有 NCCL communicator，避免重复建连；如果无法复用，
        就 all-gather rank0 的 NCCL unique id 并在 C++ 中创建 DeepEP 自己托管的 communicator。
    """
    # Check cache hit
    global _storage        # 使用模块级 cache，避免同一 group 重复创建 NCCL communicator。
    if group in _storage:  # 如果该 ProcessGroup 已经有句柄，直接复用。
        # cache 命中时不重新 all_gather unique id，也不重新 ncclCommInitRank。
        return _storage[group]

    # New PyTorch has such API
    # 较新的 PyTorch NCCL backend 可以暴露 _comm_ptr。设置 EP_REUSE_NCCL_COMM=1 时，
    # DeepEP 直接复用 PyTorch 的 communicator，这样 Elastic NCCL Gin backend 与模型训练
    # 使用同一通信域，避免额外创建 NCCL communicator 的开销。
    backend = group._get_backend(torch.device('cuda'))  # 获取该 group 的 CUDA/NCCL backend。
    if hasattr(backend, '_comm_ptr') and int(os.getenv('EP_REUSE_NCCL_COMM', '0')):  # 新 PyTorch 可直接取内部 communicator。
        # managed=False：这个指针归 PyTorch ProcessGroup 所有，DeepEP 只借用。
        _storage[group] = NCCLCommHandle(backend._comm_ptr(), False)                # 包装但不托管 PyTorch communicator。
        return _storage[group]                                                      # 返回复用句柄。

    # For old PyTorch, we have to recreate a NCCL comm
    # 老版本 PyTorch 拿不到内部 ncclComm_t，只能通过 all_gather local unique id 后在 C++ 侧
    # 重新 ncclCommInitRank。root_unique_id 来自 rank0，所有 rank 必须传入相同 id。
    nccl_unique_ids = [None, ] * group.size()                              # 为每个 rank 准备一个 unique id 槽位。
    dist.all_gather_object(nccl_unique_ids, _C.get_local_nccl_unique_id(), group)  # 收集所有 rank 本地生成的 id。
    root_unique_id = nccl_unique_ids[0]                                     # 约定 rank0 的 id 作为 ncclCommInitRank root id。
    # 其它 rank 本地生成的 id 只用于 all_gather 占位；真正 init 时所有 rank 都使用 root_unique_id。

    # Create a new communicator
    _storage[group] = NCCLCommHandle(                                      # 创建并缓存 DeepEP 自建 communicator。
        _C.create_nccl_comm(root_unique_id, group.size(), group.rank()), True)  # C++ 侧调用 ncclCommInitRank。
    return _storage[group]                                                 # 返回新建句柄。


def destroy_all_managed_nccl_comm() -> None:
    """
    Destroy all cached NCCL communicator handles and clear the cache.

    """
    # clear 会释放 cache 对 handle 的强引用；若外部无其它引用，托管 communicator 会在 __del__ 中销毁。
    _storage.clear()  # 清空 cache；托管句柄会随引用释放触发 __del__。
