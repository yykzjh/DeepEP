import functools  # 使用 lru_cache 缓存包路径查找结果，避免 setup/import 阶段重复扫描 metadata。
import os  # 读取环境变量和判断 wheel 内部目录是否存在。
import sys  # 使用 sys.path 顺序模拟 Python import 的包解析优先级。
from importlib.metadata import distributions  # 枚举当前 Python 环境中已安装的 wheel/包元数据。
from typing import Optional  # 标注 lib_name 可以为空，表示不通过具体动态库文件定位 root。


def find_pkg_root(name: str, lib_name: Optional[str] = None, optional: bool = False):  # 返回 NCCL/NVSHMEM 等 NVIDIA 包的根目录。
    """
    Find the root directory of an installed NVIDIA package by inspecting Python package metadata.
    中文：优先使用环境变量覆盖，找不到时再扫描 Python wheel 中携带的库文件。
    Checks environment variables `EP_{NAME}_ROOT_DIR` and `{NAME}_DIR` first.

    Arguments:
        中文：name/lib_name 分别对应 Python 包名和需要在包内查找的动态库文件名。
        name: the package name (e.g., `'nccl'`, `'nvshmem'`).
        lib_name: the library filename to search for within the package files.
        optional: if ``False``, raises an assertion error when the package is not found.

    Returns:
        root: the package root directory, or `None` if not found and optional.
    """
    upper = name.upper()  # 环境变量约定使用大写包名，例如 nccl -> NCCL。
    # 用户显式指定的路径优先级最高，便于在多版本 NCCL/NVSHMEM 环境中覆盖
    # Python wheel 自带路径。支持 EP_NCCL_ROOT_DIR/NCCL_DIR 两种命名。
    for env_name in (f'EP_{upper}_ROOT_DIR', f'{upper}_DIR'):  # 按项目专用变量、通用变量的顺序尝试覆盖路径。
        if env_name in os.environ:  # 用户显式配置时直接信任该路径，不再扫描 wheel。
            return os.environ[env_name]  # 例如 EP_NCCL_ROOT_DIR=/opt/nccl 会让编译链接到该安装。

    # 若环境里安装了多个 nvidia-* wheel，选择 sys.path 中优先级最高的那个，
    # 保持和 Python import 解析顺序一致。
    path_priority = {p: i for i, p in enumerate(sys.path)}  # 映射 site-packages 路径到 import 优先级，数值越小越优先。
    best, best_priority = None, len(sys.path)  # best 保存当前最优 root；初始优先级放到最差。

    for dist in distributions():  # 遍历所有已安装 distribution，例如 nvidia-nccl-cu12。
        dist_name = (dist.metadata['Name'] or '').lower()  # 读取 wheel 名称并转小写，兼容大小写差异。
        if f'nvidia-{name}' not in dist_name and f'nvidia_{name}' not in dist_name:  # 同时兼容横线和下划线命名。
            continue  # 不是目标 NVIDIA 包时跳过，避免误把其他 wheel 当成依赖根目录。

        dist_site = str(dist._path.parent)  # dist-info 的父目录通常就是 site-packages 根路径。
        priority = path_priority.get(dist_site, len(sys.path))  # 不在 sys.path 中时给最低优先级。
        if priority > best_priority:  # 已经找到更靠前的同名包时，不再用低优先级结果覆盖。
            continue  # 这能处理多个虚拟环境/路径叠加时的重复 NVIDIA wheel。

        if lib_name is not None:  # 需要通过具体动态库定位 root，例如 NCCL 要找 libnccl.so。
            # NCCL 需要找到 libnccl.so 所在目录，再向上一层作为 root；有些 wheel
            # 布局是 nvidia/nccl/lib/libnccl.so，有些直接把库放在包目录下。
            for f in (dist.files or []):  # 遍历 wheel 记录的所有文件路径；可能为空，因此用空列表兜底。
                if lib_name in str(f):  # 找到目标动态库文件，比如 nvidia/nccl/lib/libnccl.so。
                    lib_dir = os.path.dirname(str(f.locate()))  # locate() 得到真实文件路径，dirname 得到库所在目录。
                    root = os.path.dirname(lib_dir) if os.path.basename(lib_dir) == 'lib' else lib_dir  # 若在 lib/ 下，则 root 取 lib 的父目录。
                    best, best_priority = root, priority  # 更新当前最优 root 及其 sys.path 优先级。
                    break  # 一个 distribution 找到一次即可，继续扫描其他 distribution 只为比较优先级。
        else:
            # NVSHMEM 这类包不一定通过库名定位，直接寻找 nvidia/<name> 包目录。
            pkg_dir = os.path.join(dist_site, 'nvidia', name)  # 典型 wheel 包目录，例如 site-packages/nvidia/nvshmem。
            if os.path.isdir(pkg_dir):  # 确认该目录真实存在，避免只有 metadata 没有 payload 的异常安装。
                best, best_priority = pkg_dir, priority  # 记录包目录作为根目录，供后续 include/lib 查找使用。

    # Raise error if not optional
    if not optional:  # 强依赖场景下，如果没找到包应立即失败，避免编译阶段报更晦涩的链接错误。
        assert best is not None, f'Cannot find package: {name}'  # optional=False 时把依赖缺失暴露为清晰的断言信息。
    return best  # optional=True 且未找到时返回 None；调用方可按功能开关决定是否继续。


@functools.lru_cache()  # NCCL 路径在一次进程生命周期内不应变化，因此直接缓存。
def find_nccl_root(optional: bool = False):  # 查找 NCCL root，供 setup.py/CMake 传入 include 和 library 路径。
    """
    Find the NCCL installation root directory, cached.

    Arguments:
        optional: if `False`, raises an assertion error when NCCL is not found.

    Returns:
        root: the NCCL root directory.
    """
    return find_pkg_root('nccl', lib_name='libnccl.so', optional=optional)  # NCCL 通过 libnccl.so 精确反推出安装根目录。


@functools.lru_cache()  # NVSHMEM 路径同样缓存，减少 repeated import/setup 的 metadata 扫描成本。
def find_nvshmem_root(optional: bool = False):  # 查找 NVSHMEM root，启用低延迟 RDMA/NVSHMEM 路径时会用到。
    # NVSHMEM 负责 GPU 侧 one-sided 通信语义；DeepEP 的低延迟/elastic 路径会在 JIT 编译时需要它的 include/lib。
    """
    Find the NVSHMEM installation root directory, cached.

    Arguments:
        optional: if `False`, raises an assertion error when NVSHMEM is not found.

    Returns:
        root: the NVSHMEM root directory.
    """
    # 与 NCCL 不同，这里不传 lib_name：部分 NVSHMEM Python wheel 的库文件布局会随版本变化，
    # 直接按 package root 定位更适合后续拼接 include/、lib/ 等子目录。
    # optional=True 常用于“探测是否安装”；optional=False 则表示当前路径必须启用 NVSHMEM，找不到就立刻报错。
    return find_pkg_root('nvshmem', optional=optional)  # NVSHMEM 这里不指定库名，按 nvidia/nvshmem 包目录定位。
