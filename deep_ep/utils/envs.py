import functools  # 缓存硬件带宽、RDMA 能力等探测结果，避免每次测试重复执行系统命令。
import inspect  # 运行时检查 torch.distributed.init_process_group 是否支持 device_id 参数。
import os  # 读取 MASTER_ADDR、WORLD_SIZE、CUDA_VISIBLE_DEVICES 等分布式/硬件环境变量。
import random  # 同步设置 Python 标准库随机种子，保证测试数据可复现。
import re  # 解析 nvidia-smi/ibstat 文本输出中的带宽和网卡型号。
import subprocess  # 调用 nvidia-smi、ibstat 等系统工具探测硬件能力。
import torch  # 设置 CUDA 设备、默认 dtype/device，并访问 PyTorch deterministic 状态。
import torch.distributed as dist  # 初始化 NCCL 进程组、barrier、all_gather 等分布式操作。
from typing import Tuple  # 标注返回值中包含多个整数/进程组的元组结构。

# noinspection PyUnresolvedReferences
import deep_ep._C as _C  # C++/CUDA 扩展模块，提供拓扑探测等底层能力。

from .comm import get_nccl_comm_handle  # 把 torch ProcessGroup 转成 C++ 可使用的 NCCL communicator handle。

_local_rank = None  # 当前进程在单节点内的 rank；dist_print 用它决定是否只打印一次。
_local_seed = 0  # 当前 rank 的随机种子，等于 global_seed + global_rank。
_global_seed = 0  # 所有 rank 共享的基础随机种子。

# Default NIC name for RDMA operations, configurable via environment variable
_DEFAULT_NIC_NAME = os.getenv('EP_NIC_NAME', 'mlx5_0')  # 默认 IB/RDMA 网卡名，可通过 EP_NIC_NAME 覆盖。


def init_seed(global_seed: int) -> None:  # 初始化当前 rank 的随机种子。
    # DeepEP 测试通常在多 GPU 上同时构造输入；如果所有 rank 用同一个 seed，
    # dispatch/combine 可能只覆盖“完全同分布”数据，难以暴露跨 rank 路由差异。
    """
    Initialize the random seed for reproducibility. The local seed is derived from the global seed plus rank.

    Arguments:
        global_seed: the global random seed.
    """
    global _local_seed, _global_seed  # 修改模块级 seed，便于测试脚本后续查询。
    _local_seed = global_seed + dist.get_rank()  # 每个 rank 使用不同 seed，避免所有 GPU 生成完全相同数据。
    _global_seed = global_seed  # 保存原始全局 seed，便于日志或复现实验配置。
    torch.manual_seed(_local_seed)  # 设置 PyTorch CPU/CUDA 随机源。
    random.seed(_local_seed)  # 设置 Python random 随机源。


def get_local_seed() -> int:  # 返回当前 rank 实际使用的随机种子。
    # local seed = global seed + global rank；例如 global_seed=123、rank=5，则本 rank seed=128。
    """
    Get the local random seed.

    Returns:
        seed: the local random seed.
    """
    return _local_seed  # 在 init_seed 之前调用会返回默认值 0。


def get_global_seed() -> int:  # 返回所有 rank 共享的基础 seed。
    # global seed 用于记录实验配置，重跑时只要 world/rank 拓扑相同即可复现每个 rank 的 local seed。
    """
    Get the global random seed.

    Returns:
        seed: the global random seed.
    """
    return _global_seed  # 用于测试日志中记录可复现实验的基础 seed。


def dist_print(s: str = '', once_in_node: bool = False) -> None:  # 分布式安全打印，并在打印后同步所有 rank。
    # 这个 helper 把“打印”和“barrier”绑定，牺牲一点等待时间换取日志可读性；
    # 否则多进程 benchmark 输出很容易和后续阶段交错。
    """
    Print a message from all ranks, or only from rank 0 of each node, followed by a barrier.

    Arguments:
        中文：这个函数适合 benchmark 日志，不适合性能关键路径。
        s: the message to print.
        once_in_node: if `True`, only the first local rank in each node prints.
    """
    global _local_rank  # 使用 init_dist 设置的本地 rank。
    assert _local_rank is not None  # 必须先调用 init_dist，否则不知道本地 rank。
    if not once_in_node or _local_rank == 0:  # once_in_node=True 时每个节点只让 local_rank0 打印。
        print(s, flush=True)  # flush=True 避免多进程日志缓冲导致顺序错乱。
    dist.barrier()  # 打印后全局同步，让后续日志不会提前穿插。


def init_dist(local_rank: int, num_local_ranks: int, seed: int = 0) -> Tuple[int, int, dist.ProcessGroup]:  # 初始化 DeepEP 测试用 NCCL 环境。
    """
    Initialize the distributed environment with NCCL backend.

    Arguments:
        中文：local_rank/num_local_ranks 用于把节点号和本地卡号拼成全局 rank。
        local_rank: the local rank index.
        num_local_ranks: the number of local ranks.
        seed: the global random seed.
        中文：seed 会同时设置 torch/cuda 随机种子，保证测试输入可复现。

    Returns:
        中文：返回全局 rank、world size 和 NCCL ProcessGroup。
        rank: the global rank index.
        world_size: the total number of ranks.
        group: the communication group.
        中文：group 的 rank 顺序与 global rank 一致，后续 Buffer 会按这个顺序交换 peer 地址。
    """
    # NOTES: you may rewrite this function with your own cluster settings
    # 这个 helper 假设每个节点启动 num_local_ranks 个进程，环境变量 WORLD_SIZE/RANK 表示节点数/节点号。
    # 最终 global rank = node_rank * num_local_ranks + local_rank。tests 目录里的用例都按这个约定
    # 构造 EP group，因此如果接入 Slurm/K8s，需要保持返回的 group 覆盖同一组 EP rank。
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')  # 主节点地址，单机测试默认 localhost。
    port = int(os.getenv('MASTER_PORT', '8361'))  # rendezvous 端口，所有节点必须一致。
    num_nodes = int(os.getenv('WORLD_SIZE', 1))  # 这里 WORLD_SIZE 表示节点数，而不是总 GPU rank 数。
    node_rank = int(os.getenv('RANK', 0))  # 当前节点编号，配合 local_rank 计算全局 rank。

    # Set local rank
    global _local_rank  # 保存本地 rank 给 dist_print/check 等 helper 使用。
    _local_rank = local_rank  # 当前进程绑定的本地 GPU 序号。

    sig = inspect.signature(dist.init_process_group)  # 不同 PyTorch 版本的 init_process_group 参数略有差异。
    params = {  # 组装 NCCL 初始化参数，后面会按版本追加 device_id。
        'backend': 'nccl',  # DeepEP kernel 全部运行在 CUDA/NCCL 通信语境下。
        'init_method': f'tcp://{ip}:{port}',  # 用 TCP rendezvous 建立 NCCL 进程组。
        'world_size': num_nodes * num_local_ranks,  # 总 rank 数 = 节点数 * 每节点 GPU 数。
        'rank': node_rank * num_local_ranks + local_rank,  # 全局 rank 编号，范围 [0, world_size)。
    }  # params 会被解包传给 dist.init_process_group。
    if 'device_id' in sig.parameters:  # 新版本 PyTorch 支持 device_id，可提前绑定 communicator 到指定 CUDA 设备。
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')  # 避免 NCCL 初始化时猜错当前设备。
    dist.init_process_group(**params)  # 真正建立 NCCL 进程组。
    torch.set_default_dtype(torch.bfloat16)  # 测试默认使用 BF16，贴近 MoE 激活/权重常见精度。
    torch.set_default_device('cuda')  # 未显式指定 device 的新 Tensor 默认放到 CUDA。
    torch.cuda.set_device(local_rank)  # 当前 Python 进程绑定到本地对应 GPU。

    init_seed(seed)  # 初始化当前 rank 的随机源。
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))  # 返回全局 rank、world size 和覆盖全部 rank 的 EP group。


def get_physical_domain_size(group: dist.ProcessGroup) -> Tuple[int, int]:  # 查询物理 RDMA/NVLink 域大小。
    # 物理域来自真实拓扑：例如 2 节点 * 每节点 8 GPU 时，常见结果是 (2, 8)。
    # 它不一定等同于 kernel 最终使用的逻辑域，后者还受 hybrid/direct 策略影响。
    """
    Get the physical domain sizes (RDMA ranks and NVLink ranks).

    Arguments:
        中文：group 应该是 DeepEP 使用的完整 EP 通信组，而不是单节点子组。
        group: the communication group.

    Returns:
        中文：返回二元组用于判断物理拓扑，例如多节点 8 卡通常是 (num_nodes, 8)。
        num_rdma_ranks: the number of physical RDMA ranks.
        num_nvlink_ranks: the number of physical NVLink ranks.
    """
    # physical domain 由 NCCL topology 判断：
    # - num_rdma_ranks：跨 RDMA 域/节点的 rank 组数；
    # - num_nvlink_ranks：单个 NVLink 域内 rank 数。
    # 这个结果用于性能估计和 legacy/elastic 路径选择。
    return _C.get_physical_domain_size(get_nccl_comm_handle(group).get())  # C++ 返回 (num_rdma_ranks, num_nvlink_ranks)。


def get_logical_domain_size(group: dist.ProcessGroup, allow_hybrid_mode: bool = True) -> Tuple[int, int]:  # 查询 DeepEP kernel 规划出的逻辑通信域。
    """
    Get the logical domain sizes (scaleout ranks and scaleup ranks).

    Arguments:
        group: the communication group.
        中文：allow_hybrid_mode=False 时逻辑域会退化为更保守的非 hybrid 规划。
        allow_hybrid_mode: whether to enable hybrid mode.

    Returns:
        中文：scaleout/scaleup 是 DeepEP kernel 的逻辑域，可能与物理 RDMA/NVLink 域不同。
        num_scaleout_ranks: the number of logical scaleout ranks.
        num_scaleup_ranks: the number of logical scaleup ranks.
    """
    # logical domain 是 DeepEP 为 kernel 规划出的二层通信结构：
    # scaleout 通常对应 RDMA 维度，scaleup 通常对应 NVLink 维度。allow_hybrid_mode=False 时，
    # C++ 会尽量退化为 direct 模式，减少 hybrid 转发 metadata。
    return _C.get_logical_domain_size(get_nccl_comm_handle(group).get(), allow_hybrid_mode)  # 返回 (num_scaleout_ranks, num_scaleup_ranks)。


def check_nvlink_connections(group: dist.ProcessGroup) -> None:  # 检查 PCIe A100 等设备是否满足测试所需 NVLink 连接。
    # 这个检查只覆盖最容易出错的 PCIe GPU 情况：同节点 GPU 之间可能并非全互联，
    # 但 legacy/elastic 的 scaleup 路径默认假设节点内 peer 可以高效互通。
    """
    Check NVLink connection between every pair of GPUs.

    Arguments:
        group: the communication group.
        中文：group.size() 会被用来判断 PCIe GPU 是否只允许 EP2 这类成对 NVLink 拓扑。
    """
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    # 中文：这里的检查偏保守，目的是避免 PCIe 拓扑下错误地假设任意两卡都能 NVLink P2P。
    # TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
    if 'PCIE' in torch.cuda.get_device_name():  # 只对 PCIe 型号走保守检查；SXM 通常有更完整 NVSwitch/NVLink。
        assert group.size() <= 2, 'PCIe GPUs only have pairwise NVLink connections'  # PCIe A100 常见只有成对 NVLink，EP>2 可能不成立。

        # noinspection PyUnresolvedReferences
        import pynvml  # 延迟导入，只有需要 NVML 检查时才要求安装。
        pynvml.nvmlInit()  # 初始化 NVML，后续才能查询 P2P/NVLink 状态。

        # noinspection PyTypeChecker
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').strip(',').split(',')  # 将逻辑 GPU id 映射回物理 GPU id。
        physical_device_idx = int(devices[torch.cuda.current_device()])  # 当前 rank 使用的物理 GPU 编号。
        physical_device_indices = [0, ] * group.size()  # 收集所有 rank 的物理 GPU id，长度为 group.size()。
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)  # 跨 rank 汇总物理 GPU id。

        # Check whether they are all connected via NVLink
        # Reference: https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices]  # 为每张物理 GPU 获取 NVML handle。
        for i, handle in enumerate(handles):  # 遍历本组 GPU 对的左端点。
            for j, peer_handle in enumerate(handles):  # 遍历本组 GPU 对的右端点。
                if i >= j:  # 只检查上三角，避免自连接和重复检查。
                    continue  # 跳过 i==j 或已经检查过的 pair。
                status = pynvml.nvmlDeviceGetP2PStatus(handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)  # 查询两卡是否支持 NVLink P2P。
                assert status == pynvml.NVML_P2P_STATUS_OK, (  # 断言该 GPU pair 具备 NVLink 连接。
                    f'GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]} are not connected via NVLink'  # 报错时打印物理 GPU 编号。
                )  # 用括号续行，避免反斜杠后追加注释破坏 Python 语法。

        # Close NVML
        pynvml.nvmlShutdown()  # 关闭 NVML，释放底层资源。


def check_torch_deterministic() -> None:  # 防止 PyTorch deterministic 模式插入额外初始化 kernel 干扰通信。
    """
    Ensure PyTorch deterministic algorithms and fill_uninitialized_memory are not both enabled.
    When both are on, `torch.empty()` calls an initialization kernel that may overlap with communication streams,
    中文：该检查在 Buffer 初始化前执行，避免后续才出现难定位的 stream 数据竞争。
    causing errors.
    中文：DeepEP 依赖通信 kernel 写满输出 tensor，额外初始化 kernel 会破坏 stream 时序假设。
    """
    # PyTorch deterministic + fill_uninitialized_memory 会把 torch.empty() 变成“分配后立刻填充”的 kernel。
    # DeepEP 很多输出 tensor 在通信 stream 上分配并由通信 kernel 写入；如果 PyTorch 额外插入填充 kernel，
    # 就可能和通信 stream 发生未预期的 overlap，所以这里显式禁止这种组合。
    assert not (torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory)  # 两个开关同时开启时直接失败。


@functools.lru_cache()  # 同一进程内 NVLink 理论带宽不会变化，缓存系统命令结果。
def get_nvlink_gbs(factor: float = 0.9) -> float:  # 估算单 GPU NVLink 总带宽，单位 GB/s。
    """
    Get the total NVLink bandwidth in GB/s, cached.

    Arguments:
        中文：factor 是经验折扣系数，用来把理论链路速率折成可达到带宽。
        factor: the bandwidth efficiency factor.

    Returns:
        中文：失败返回 0，调用方应把它当作“未知带宽”而不是实际 0 GB/s。
        gbs: the total NVLink bandwidth in GB/s (0 if detection fails).
    """
    # noinspection PyBroadException
    # 中文：带宽探测属于 best-effort，任何异常都会回退到 0 而不是中断测试。
    try:
        result = subprocess.run(['nvidia-smi', 'nvlink', '-s'],  # 查询每条 NVLink 的速率。
                                capture_output=True, text=True, check=True)  # 捕获 stdout，命令失败时抛异常进入 fallback。
        # 中文：只解析 stdout，不依赖 stderr，避免不同驱动版本的提示信息干扰。
        output = result.stdout  # nvidia-smi 输出的原始文本。
        pattern = r'GPU \d+:.*?(?=^GPU \d+:|^$)'  # 提取第一个 GPU 的 NVLink 状态块。
        match = re.search(pattern, output, re.MULTILINE | re.DOTALL)  # DOTALL 允许跨多行匹配 GPU 块。
        assert match  # 没有匹配说明 nvidia-smi 输出格式不符合预期。

        gpu_block = match.group(0)  # 单张 GPU 的所有 Link 行。
        link_pattern = r'Link \d+:\s*([\d\.]+) GB/s'  # 抓取每条 Link 的数值带宽。
        link_matches = re.findall(link_pattern, gpu_block)  # 得到字符串列表，例如 ['50.0','50.0',...]。
        assert link_matches  # 没有 Link 速率时说明当前机器不支持该查询或无 NVLink。
        return sum(float(bw) for bw in link_matches) * factor  # 总带宽乘以经验效率系数，作为 benchmark 参考线。
    except Exception as e:  # 任何命令缺失、格式变化、无 NVLink 的情况都走 fallback。
        print(f'Failed to get NVLink connection speed: {e}')  # 打印原因，但不阻断测试。
        return 0  # 返回 0 表示不可用，调用方可隐藏或跳过带宽对比。


@functools.lru_cache()  # 网卡能力在进程生命周期内固定，缓存 ibstat 解析结果。
def check_fast_rdma_atomic_support(nic_name: str = _DEFAULT_NIC_NAME) -> bool:  # 判断网卡是否支持 DeepEP 期望的快速 RDMA atomic。
    # 快速 RDMA atomic 会影响部分控制面算法的可用性/性能；探测失败时保守返回 False，
    # 让上层选择更兼容的路径，而不是在不确定硬件上冒险启用优化。
    """
    Check whether the NIC supports fast RDMA atomic operations (MT4131 or newer).

    Arguments:
        中文：nic_name 为空时通常会检查默认或第一块 IB 设备。
        nic_name: the NIC device name.

    Returns:
        中文：该结果用于选择是否启用更激进的 IBGDA atomic 优化。
        supported: `True` if fast RDMA atomics are supported.
    """
    # noinspection PyBroadException
    # 中文：ibstat 不存在或输出格式变化时，函数会保守返回 False。
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)  # 查询 IB/RDMA 设备信息。
        output = result.stdout  # ibstat 的文本输出。
        # 中文：正则限制在指定 CA 块内查找 CA type，避免多网卡输出串扰。
        pattern = rf"CA '{nic_name}'.*?CA type:\s*(\S+)"  # 在指定网卡块中提取 CA type。
        match = re.search(pattern, output, re.DOTALL)  # 跨多行匹配该网卡信息。
        assert match  # 未匹配到网卡时进入 except，返回 False。
        return match.group(1) == 'MT4131'  # 当前项目把 MT4131 视作支持快速 atomic 的型号。
    except Exception:  # ibstat 不存在、网卡名错误、格式变化时都保守返回不支持。
        return False  # 调用方据此选择兼容路径或禁用相关优化。


@functools.lru_cache()  # RDMA 端口速率固定，缓存 ibstat 查询结果。
def get_rdma_gbs(nic_name: str = _DEFAULT_NIC_NAME) -> float:  # 估算指定 RDMA 网卡端口带宽，单位 GB/s。
    # 这个值只作为性能日志参考线，不参与 correctness；ibstat 不存在时返回 0，
    # 调用方仍可继续跑功能测试，只是不展示 RDMA 理论带宽对比。
    """
    Get the RDMA bandwidth in GB/s, cached.

    Arguments:
        中文：返回值用于 benchmark 打印和 sanity check，不参与 kernel 配置。
        nic_name: the NIC device name.

    Returns:
        gbs: the RDMA bandwidth in GB/s (0 if detection fails).
        中文：返回单位是十进制 GB/s，主要用于日志对比，不参与容量计算。
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)  # 查询 IB 端口状态和速率。
        output = result.stdout  # ibstat 的原始文本输出。

        pattern = rf"CA '{nic_name}'.*?Port \d+:\s*.*?Rate:\s*(\d+)"  # 提取第一个端口的 Rate 数值，ibstat 通常以 Gb/sec 展示。
        match = re.search(pattern, output, re.DOTALL)  # 在目标网卡块中跨行查找端口速率。
        assert match  # 未找到速率时进入 fallback。
        rate = int(match.group(1))  # rate 单位是 Gb/s。
        return rate / 8  # 转换为 GB/s，便于和 tensor 字节数/耗时对齐。
    except Exception as e:  # 无 RDMA 设备或命令不可用时不阻断测试。
        print(f'Failed to get RDMA connection speed: {e}')  # 打印探测失败原因。
        return 0  # 返回 0 表示不可用。
