import filecmp  # 用二进制比较确认运行时加载的 NCCL 和构建时链接的 NCCL 是同一个文件。
import functools  # 缓存 CUDA_HOME 查找，避免导入过程中重复执行 which nvcc。
import glob  # 查找 NCCL root 下的 libnccl.so* 候选文件。
import subprocess  # 调用 which nvcc 探测 CUDA 安装路径。
import torch  # 导入包入口时确保 PyTorch 已加载；后续 C++ 扩展也依赖 torch runtime。
import os  # 读取/写入环境变量，并访问 /proc/self/maps。

from .utils.find_pkgs import find_nccl_root  # 查找 NCCL root，供 NCCL 检查和 JIT 初始化使用。

# Set some default environment provided at setup
# setup.py 在 build_py 阶段会把当前集群相关的持久化环境变量写入 deep_ep/envs.py。
# 这里导入后只填充“用户没有手动设置”的变量，避免覆盖运行时显式传入的 EP_JIT_CACHE_DIR、
# EP_NCCL_ROOT_DIR 等配置。这个逻辑和 setup.py 中的 persistent_env_names 联动。
try:  # 安装包中可能存在 deep_ep/envs.py；源码开发环境中可能不存在。
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs  # setup.py 生成的持久化环境变量字典。
    for key, value in persistent_envs.items():  # 遍历安装时记录的环境变量。
        if key not in os.environ:  # 只填充用户未显式设置的变量，保留运行时覆盖能力。
            os.environ[key] = value  # 写入当前进程环境，供后续 JIT/编译查找路径。
except ImportError:  # 开发模式或尚未生成 envs.py 时会进入这里。
    pass  # 没有持久化环境变量也可以继续，后续会走默认查找逻辑。

# Initialize
@functools.lru_cache()  # CUDA_HOME 一次进程内不应变化，因此缓存。
def find_cuda_home() -> str:  # 查找 CUDA Toolkit 根目录。
    """
    Find the CUDA installation directory, cached.

    Returns:
        中文：返回路径会写入 JIT 编译器，后续用于定位 CUDA include/lib。
        cuda_home: the CUDA installation path.
    """
    # TODO: reuse PyTorch API later
    # 这里没有直接使用 PyTorch 的 CUDA_HOME 探测函数，是因为部分 PyTorch 版本会在探测过程中
    # 初始化 CUDA runtime。DeepEP 常在多进程 fork 前导入，如果过早初始化 CUDA，子进程可能继承
    # 不完整的 CUDA 状态，因此优先走环境变量和 `which nvcc` 这种不会触发 CUDA runtime 的路径。
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')  # 优先使用用户显式配置。
    if cuda_home is None:  # 未配置时尝试从 nvcc 路径反推。
        # noinspection PyBroadException
        try:
            with open(os.devnull, 'w') as devnull:  # 屏蔽 which nvcc 的错误输出，避免导入包时污染日志。
                nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')  # 获取 nvcc 绝对路径。
                cuda_home = os.path.dirname(os.path.dirname(nvcc))  # nvcc 通常在 CUDA_HOME/bin/nvcc。
        except Exception:  # which nvcc 失败时使用常见默认路径兜底。
            cuda_home = '/usr/local/cuda'  # Linux CUDA Toolkit 默认安装目录。
            if not os.path.exists(cuda_home):  # 默认路径不存在说明当前环境没有可见 CUDA Toolkit。
                cuda_home = None  # 交给后面的断言给出明确失败。
    assert cuda_home is not None  # DeepEP JIT 编译必须有 CUDA Toolkit。
    return cuda_home  # 返回 CUDA root，例如 /usr/local/cuda 或 conda/env 下的 cuda。


def check_nccl_so():  # 检查运行时 NCCL 与 DeepEP 链接 NCCL 是否一致。
    """
    Verify that the NCCL library loaded at runtime matches the linked version.
    中文：该检查用于避免系统 NCCL、PyTorch NCCL、DeepEP 链接 NCCL 混用导致 ABI 问题。
    Aborts if duplicate NCCL libraries are found or if versions mismatch.
    """
    if int(os.environ.get('EP_SUPPRESS_NCCL_CHECK', 0)):  # 用户显式关闭检查时直接跳过。
        return  # 适用于特殊容器/系统路径，但可能隐藏 NCCL ABI 不一致风险。

    # PyTorch 可能已经通过自己的 wheel 或系统路径加载了一个 NCCL；DeepEP 扩展链接的 NCCL
    # 又可能来自 nvidia-nccl-cuXX 包或 EP_NCCL_ROOT_DIR。两份 NCCL 同时存在时，NCCL communicator
    # 和 device-side Gin 结构体布局可能不一致，轻则初始化失败，重则 kernel 侧读错 handle。
    # 因此这里先扫描 /proc/self/maps，要求当前进程内出现的 NCCL so 路径唯一。
    with open('/proc/self/maps', 'r') as f:  # /proc/self/maps 列出当前进程已映射的动态库。
        loaded_nccl_so = None  # 记录第一个看到的 NCCL so 路径。
        for so in [line.strip().split(' ')[-1] for line in f if 'nccl' in line]:  # 筛选包含 nccl 的映射行并取路径列。
            loaded_nccl_so = so if loaded_nccl_so is None else loaded_nccl_so  # 第一条 NCCL 记录作为基准。
            assert so == loaded_nccl_so, f'Duplicate NCCL runtime found in the current system: {so} and {loaded_nccl_so}'  # 多个路径说明运行时混用 NCCL。
    linked_nccl_so_candidates = sorted(glob.glob(f'{find_nccl_root()}/lib/libnccl.so*'))  # 查找构建/链接期预期的 NCCL so。
    assert linked_nccl_so_candidates, f'No libnccl.so found in {find_nccl_root()}/lib/'  # 找不到 NCCL 库时给出明确错误。
    linked_nccl_so = linked_nccl_so_candidates[0]  # 选择排序后的第一个候选，通常是 libnccl.so 或最基础软链接。

    # 仅检查文件名或版本号不够：某些环境会把 libnccl.so 软链接到不同 build 的二进制。
    # DeepEP 需要确保“运行时已加载的 NCCL”和“JIT/扩展构建时找到的 NCCL”是同一个二进制，
    # 这样 csrc/kernels/backend/nccl.cu 与 deep_ep/include/deep_ep/common/handle.cuh 中使用的
    # NCCL device-side 类型才能保持一致。
    # noinspection PyTypeChecker
    assert filecmp.cmp(loaded_nccl_so, linked_nccl_so, shallow=False), (  # 深度比较两个 so 的文件内容而非仅比较路径。
        f'Invalid NCCL versions: {loaded_nccl_so} (loaded) v.s. {linked_nccl_so} (expected), '  # 报告当前已加载与期望库路径。
        f'please contact Chenggang or Shangyan to upgrade PyTorch NCCL version'  # 保留原始维护者提示。
    )  # 使用括号续行，便于逐行注释且不破坏语法。


def init_jit():  # 初始化 C++ JIT 编译器所需的路径信息。
    """
    Initialize the JIT compilation runtime. Sets up CUDA and NCCL root paths for the JIT compiler.
    中文：调用后 C++ 扩展会记录 library_root/include_root/cuda_home/nccl_home，供运行时编译 kernel。
    """
    # noinspection PyUnresolvedReferences
    import deep_ep._C as _C  # 延迟导入 C++ 扩展，确保前面的环境变量已经填好。
    # library_root_path 指向安装后的 deep_ep 包目录。C++ JIT 编译器会基于它定位
    # deep_ep/include，因此 JIT 生成的代码可以使用 `#include <deep_ep/impls/...>`。
    library_root_path = os.path.dirname(os.path.abspath(__file__))  # 当前 deep_ep 包目录。
    _C.init_jit(library_root_path,  # Library root directory path；JIT 用它定位 include/deep_ep。
                find_cuda_home(),   # CUDA home；JIT 编译器用它定位 nvcc/include/lib。
                find_nccl_root())   # NCCL root；JIT 编译器用它定位 NCCL 头文件和库。

# Run initialization
# 导入 deep_ep 时立即做 NCCL 检查和 JIT 初始化。注意：Buffer/ElasticBuffer 的 Python 类必须在
# init_jit 之后再导入，因为它们会 import deep_ep._C 并依赖 C++ 侧已经知道 CUDA/NCCL/include 路径。
check_nccl_so()  # 导入时立即检查 NCCL 一致性，尽早暴露环境问题。
init_jit()  # 初始化 JIT 路径，保证后续 Buffer/ElasticBuffer 可以按需编译 kernel。


# Import APIs after initialization
from .buffers.legacy import Buffer  # legacy DeepEP buffer API。
from .buffers.elastic import ElasticBuffer, EPHandle  # elastic buffer API 与可复用句柄类型。
# noinspection PyUnresolvedReferences
from .utils.event import EventOverlap, EventHandle  # 暴露异步事件包装，方便用户从 deep_ep 顶层导入。
from .utils.envs import get_physical_domain_size, get_logical_domain_size  # 暴露拓扑查询 helper。

# noinspection PyUnresolvedReferences
from deep_ep._C import Config, topk_idx_t  # 暴露 C++ 配置类型和 topk 索引 dtype 常量。

__version__ = '2.0.0'  # Python 包版本号。
