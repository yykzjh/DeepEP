import ast  # 安全解析 deep_ep/__init__.py 中的 __version__ 字符串字面量。
import re  # 用正则从 __init__.py 中提取版本号定义。
import os  # 读取环境变量、拼接路径、设置 TORCH_CUDA_ARCH_LIST。
import subprocess  # 调用 git 命令，把 revision 附加到 package version。
import setuptools  # 执行 Python package 构建与 CUDAExtension 注册。
import importlib  # 动态加载 find_pkgs.py，避免提前 import deep_ep 触发扩展初始化。

from pathlib import Path  # 以跨平台方式处理 __init__.py、NVSHMEM lib 路径。
from setuptools.command.build_py import build_py  # 自定义 build_py，把持久环境变量写入包内 envs.py。
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # PyTorch 提供的 C++/CUDA 扩展构建入口。

current_dir = os.path.dirname(os.path.realpath(__file__))  # 项目根目录绝对路径，setup.py 后续所有相对路径都基于它。
persistent_env_names = ('EP_JIT_CACHE_DIR', 'EP_JIT_PRINT_COMPILER_COMMAND', 'EP_NUM_TOPK_IDX_BITS', 'EP_NCCL_ROOT_DIR')  # 构建时需要固化到 wheel 的环境变量白名单。

# Load discover module without triggering `deep_ep.__init__`
# 不能直接 `from deep_ep.utils import find_pkgs`，否则 deep_ep/__init__.py 会尝试加载尚未编译的 _C 扩展。
find_pkgs_spec = importlib.util.spec_from_file_location('find_pkgs', os.path.join(current_dir, 'deep_ep', 'utils', 'find_pkgs.py'))  # 创建从源码文件加载模块的 spec。
find_pkgs = importlib.util.module_from_spec(find_pkgs_spec)  # 基于 spec 创建一个未执行的临时模块对象。
find_pkgs_spec.loader.exec_module(find_pkgs)  # 执行 find_pkgs.py，把 find_nvshmem_root/find_nccl_root 加载进模块。


# Wheel specific: the wheels only include the SO name of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')  # NVSHMEM 安装目录下的 lib 子目录。
    for file in path.rglob('libnvshmem_host.so.*'):  # wheel 场景下 host so 通常带 ABI 版本后缀。
        return file.name  # 返回实际文件名，例如 libnvshmem_host.so.3，而不是无版本软链名。
    raise ModuleNotFoundError('libnvshmem_host.so not found')  # 没找到说明 NVSHMEM 安装不完整或路径探测错误。


def get_package_version():
    with open(Path(current_dir) / 'deep_ep' / '__init__.py', 'r') as f:  # 直接读取源码版本文件，避免 import deep_ep。
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)  # 匹配形如 __version__ = 'x.y.z' 的行。
    public_version = ast.literal_eval(version_match.group(1))  # literal_eval 只解析字符串字面量，比 eval 安全。

    # noinspection PyBroadException
    try:
        status_cmd = ['git', 'status', '--porcelain']  # porcelain 输出稳定，适合脚本判断工作区是否干净。
        status_output = subprocess.check_output(status_cmd).decode('ascii').strip()  # 非空表示有未提交改动。
        if status_output:  # 官方构建希望版本精确对应 commit，脏工作区会退回 local 后缀。
            print(f'Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}')  # 打印未提交文件，方便定位。
            assert False, 'Git working directory is not clean'  # 进入 except，把版本标为 +local。

        cmd = ['git', 'rev-parse', '--short', 'HEAD']  # 获取短 commit id，用于 wheel 版本后缀。
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()  # PEP440 local version 形如 1.0.0+abc123。
    except:
        revision = '+local'  # 没有 git、工作区脏或命令失败时统一使用 local 标识。
    return f'{public_version}{revision}'  # 返回完整 package version。


class CustomBuildPy(build_py):
    def run(self):
        # Make clusters' cache setting default into `envs.py`
        self.generate_default_envs()  # 在复制 Python 包前生成 deep_ep/envs.py。

        # Finally, run the regular build
        build_py.run(self)  # 调用 setuptools 原始 build_py，继续复制 Python 文件和 package_data。

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'  # 生成文件头，说明这些值来自构建环境。
        code += 'persistent_envs = dict()\n'  # deep_ep/utils/envs.py 会读取这个字典并设置默认值。
        # noinspection PyShadowingNames
        for name in persistent_env_names:  # 遍历允许固化的变量，避免把任意敏感环境变量写入 wheel。
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''  # 只写构建时存在的变量。

        # Create temporary build directory
        build_include_dir = os.path.join(self.build_lib, 'deep_ep')  # build_lib/deep_ep 是构建产物包目录。
        os.makedirs(build_include_dir, exist_ok=True)  # 确保目录存在，避免写 envs.py 失败。
        with open(os.path.join(self.build_lib, 'deep_ep', 'envs.py'), 'w') as f:  # 在构建产物里写入 envs.py。
            f.write(code)  # 写入可被 Python import 的默认环境变量字典。


if __name__ == '__main__':
    # TODO: make NVSHMEM and legacy optional
    nvshmem_root_dir = find_pkgs.find_nvshmem_root()  # 探测 NVSHMEM 根目录，legacy internode kernel 依赖它。
    nccl_root_dir = find_pkgs.find_nccl_root()  # 探测 NCCL 根目录，JIT/legacy 后端都需要 include 和 lib。

    # `128,2417` is used to suppress warnings of `fmt`
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']  # host 编译 flags，关闭第三方/模板常见噪声。
    nvcc_flags = ['-O3', '-Xcompiler', '-O3', '--extended-lambda', '--diag-suppress=128,2417']  # device 编译 flags，extended-lambda 支持 CUDA lambda。
    sources = ['csrc/python_api.cpp', 'csrc/kernels/legacy/layout.cu', 'csrc/kernels/legacy/intranode.cu']  # 基础扩展源文件：绑定、layout、intranode legacy。
    # include/ 放在最前面，方便编译器和 IDE 从一个根目录解析所有项目内头文件。
    # deep_ep/include 仍然保留在 include path 中，因为它是 wheel 打包时的公开头文件目录。
    include_dirs = [f'{current_dir}/include',  # 新增根 include，里面软链到 csrc/deep_ep include，便于编译器和 IDE 跳转声明。
                    f'{current_dir}/deep_ep/include',  # wheel 公开 JIT 头文件目录，运行时 JIT 也依赖它。
                    f'{current_dir}/third-party/fmt/include',  # fmt 头文件，用于 csrc/utils/format.hpp。
                    '/usr/local/cuda/include/cccl']  # CUDA CCCL 头目录，部分环境下需要显式补充。
    library_dirs = []  # 由 NVSHMEM/NCCL 探测结果逐步追加。
    nvcc_dlink = []  # NVSHMEM device 静态库需要 nvcc device link 参数。
    extra_link_args = ['-lcuda']  # CUDA driver API 来自 libcuda.so。

    # NVSHMEM flags
    sources.extend(['csrc/kernels/legacy/internode.cu', 'csrc/kernels/legacy/internode_ll.cu', 'csrc/kernels/backend/nvshmem.cu'])  # 加入跨节点 legacy kernel 与 NVSHMEM backend。
    include_dirs.extend([f'{nvshmem_root_dir}/include'])  # NVSHMEM host/device 头文件。
    library_dirs.extend([f'{nvshmem_root_dir}/lib'])  # NVSHMEM 动态/静态库目录。
    nvcc_dlink.extend(['-dlink', f'-L{nvshmem_root_dir}/lib', '-lnvshmem_device'])  # device 符号需要单独 dlink。
    extra_link_args.extend([f'-l:libnvshmem_host.so', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_root_dir}/lib'])  # host/device 库和运行时 rpath。

    # NCCL flags
    sources.extend(['csrc/kernels/backend/nccl.cu'])  # 加入 NCCL communicator helper backend。
    include_dirs.extend([f'{nccl_root_dir}/include'])  # NCCL 头文件，C++ 后端和 JIT 都会用到。
    extra_link_args.extend([f'-l:libnccl.so', f'-Wl,-rpath,{nccl_root_dir}/lib'])  # 链接 NCCL 并写入 rpath。

    # CUDA driver sources
    sources.extend(['csrc/kernels/backend/cuda_driver.cu'])  # 加入 cuStreamBatchMemOp 封装，用于 GPU-side wait/write。

    # TODO: remove these
    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')  # 禁用 SM90 时默认编译 A100/SM80。

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')  # host 代码通过宏关闭 SM90-only 分支。
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')  # device 代码通过宏关闭 FP8/TMA 等指令路径。

        # Disable internode and low-latency kernels
        assert False, 'Not implemented'  # 当前 setup 仍强依赖 legacy internode，SM80 兼容构建尚未完成。
    else:
        # Prefer H800 series
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')  # 默认编译 Hopper/H800/SM90。

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])  # NVSHMEM device link 需要 rdc，ptxas flag 打开寄存器使用控制。

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1  # 非 SM90 架构必须关闭激进 PTX 指令。
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'  # 强制传递给 Python/JIT 侧，保持编译期行为一致。

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')  # host 编译期也需要知道是否禁用这些 inline PTX。
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')  # device 侧宏会切换 load/store helper 实现。

    # Legacy environment name
    if 'TOPK_IDX_BITS' in os.environ:
        assert 'EP_NUM_TOPK_IDX_BITS' not in os.environ  # 避免新旧两个变量同时设置造成歧义。
        os.environ['EP_NUM_TOPK_IDX_BITS'] = os.environ['TOPK_IDX_BITS']  # 兼容旧变量名，统一转成 EP_ 前缀。

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if 'EP_NUM_TOPK_IDX_BITS' in os.environ:
        num_topk_idx_bits = int(os.environ['EP_NUM_TOPK_IDX_BITS'])  # topk_idx_t 位宽，影响 Python dtype 和 C++ ABI。
        cxx_flags.append(f'-DEP_NUM_TOPK_IDX_BITS={num_topk_idx_bits}')  # host 绑定暴露 dtype 时需要同一宏。
        nvcc_flags.append(f'-DEP_NUM_TOPK_IDX_BITS={num_topk_idx_bits}')  # CUDA kernel 读写 topk_idx 时需要同一类型。

    # Put them together
    extra_compile_args = {  # PyTorch CUDAExtension 按编译器类型拆分 flags。
        'cxx': cxx_flags,  # 传给 C++ host compiler。
        'nvcc': nvcc_flags,  # 传给 NVCC。
    }
    if len(nvcc_dlink) > 0:  # 只有依赖 device 静态库时才需要 dlink 参数。
        extra_compile_args['nvcc_dlink'] = nvcc_dlink  # PyTorch BuildExtension 会执行 device link 步骤。

    # Summary
    print('Build summary:')  # 打印完整构建摘要，便于复现用户机器上的编译环境。
    print(f' > Sources: {sources}')  # 本次扩展编译的源文件列表。
    print(f' > Includes: {include_dirs}')  # include 搜索路径，根 include 应排在最前面。
    print(f' > Libraries: {library_dirs}')  # link 搜索路径。
    print(f' > Compilation flags: {extra_compile_args}')  # host/device/dlink flags。
    print(f' > Link flags: {extra_link_args}')  # 链接库和 rpath。
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')  # 实际 CUDA 架构列表。
    print(f' > NVSHMEM path: {nvshmem_root_dir}')  # NVSHMEM 探测路径。
    print(f' > NCCL path: {nccl_root_dir}')  # NCCL 探测路径。
    # Print persistent env variables
    persistent_envs = []  # 收集实际会写入 deep_ep/envs.py 的构建环境变量。
    for name in persistent_env_names:  # 只遍历白名单变量。
        if name in os.environ:  # 变量存在才固化。
            persistent_envs.append((name, os.environ[name]))  # 保存变量名和值，用于摘要输出。
    if len(persistent_envs) > 0:  # 没有持久变量时不打印空段。
        print(f' > Persistent envs:')  # 输出持久变量标题。
        for k, v in persistent_envs:  # 逐项打印变量。
            print(f'   > {k}: {v}')  # 这些值会被 CustomBuildPy 写进 envs.py。
    print()  # 摘要结束空行。

    setuptools.setup(  # 标准 setuptools 入口，构建 Python 包和 CUDA 扩展。
        name='deep_ep',  # 发布包名。
        version=get_package_version(),  # 版本号来自 __init__.py + git/local 后缀。
        packages=setuptools.find_packages(include=['deep_ep', 'deep_ep.*']),  # 打包 deep_ep 及其子包。
        package_data={  # 额外打包非 .py 文件。
            'deep_ep': [  # deep_ep 包下的公开 JIT 头文件。
                'include/deep_ep/**/*',  # 运行时 JIT 编译会从 wheel 中读取这些 .cuh 头。
            ]
        },
        ext_modules=[
            CUDAExtension(name='deep_ep._C',  # 生成 Python 可 import 的 deep_ep._C 扩展。
                          include_dirs=include_dirs,  # C++/CUDA include 路径。
                          library_dirs=library_dirs,  # 链接库搜索路径。
                          sources=sources,  # 扩展源文件列表。
                          extra_compile_args=extra_compile_args,  # 编译 flags。
                          extra_link_args=extra_link_args)  # 链接 flags。
        ],
        cmdclass={  # 覆盖 setuptools 命令实现。
            'build_ext': BuildExtension,  # 使用 PyTorch 的 CUDA aware build_ext。
            'build_py': CustomBuildPy  # 在 build_py 阶段生成默认环境变量文件。
        }
    )
