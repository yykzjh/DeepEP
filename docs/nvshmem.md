# Install NVSHMEM

> 中文学习注释：本页说明 DeepEP V1/legacy 路径依赖的 NVSHMEM 安装方式。V2 的主路径已经转向 NCCL Gin，但 legacy internode、low-latency kernel 仍会用 NVSHMEM/IBGDA，因此学习 `csrc/kernels/legacy/*` 时需要理解这里的环境前置条件。

## Important notices

中文注释：这一节是许可与支持边界，意思是 NVSHMEM 本身由 NVIDIA 提供，DeepEP 项目只是在 legacy 后端中调用它。

**This project is neither sponsored nor supported by NVIDIA.**

**Use of NVIDIA NVSHMEM is governed by the terms at [NVSHMEM Software License Agreement](https://docs.nvidia.com/nvshmem/api/sla.html).**

中文注释：如果把 DeepEP 部署到生产环境，DeepEP 的 MIT License 和 NVSHMEM 的软件许可需要分别遵守；本文件只提供安装指引，不改变上游组件许可。

## Prerequisites

中文注释：硬件要求对应代码里的两类通信路径：节点内走 NVLink/P2P，节点间走 RDMA/IBGDA；如果缺少 IBGDA，`internode_ll.cu` 里的设备端 RDMA put/amo 路径就不能正常工作。

Hardware requirements:
   - GPUs inside one node needs to be connected by NVLink
     - 中文注释：节点内 NVLink 对应 legacy intranode 和部分 P2P 快速路径。
   - GPUs across different nodes needs to be connected by RDMA devices, see [GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
     - 中文注释：跨节点 RDMA 是 `internode.cu` / `internode_ll.cu` 能达到高性能的前提。
   - InfiniBand GPUDirect Async (IBGDA) support, see [IBGDA Overview](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)
     - 中文注释：IBGDA 允许 GPU kernel 直接提交网络操作，减少 CPU 参与和同步延迟。
   - For more detailed requirements, see [NVSHMEM Hardware Specifications](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html#hardware-requirements)
     - 中文注释：不同 NVSHMEM 版本对驱动、OFED、CUDA 和 NIC 固件有细节要求，应以官方规格为准。

Software requirements:
   - NVSHMEM v3.3.9 or later
     - 中文注释：DeepEP 文档按 3.3.9+ 编写，低版本可能缺少所需 IBGDA 或 API 能力。

## Installation procedure

中文注释：安装流程分为“安装 NVSHMEM 二进制”和“启用 IBGDA”。前者提供库和头文件，后者让 GPU 能直接向网卡提交异步通信操作。

### 1. Install NVSHMEM binaries

中文注释：DeepEP legacy 编译时会通过 `NVSHMEM_DIR` 找到 include/lib；如果只使用 V2 Elastic NCCL Gin 主路径，NVSHMEM 主要用于保留 V1 兼容能力。

NVSHMEM 3.3.9 binaries are available in several formats:
   - Tarballs for  [x86_64](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.3.9_cuda12-archive.tar.xz) and [aarch64](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-sbsa/libnvshmem-linux-sbsa-3.3.9_cuda12-archive.tar.xz)
     - 中文注释：tarball 适合无 root 权限或手动指定 `NVSHMEM_DIR` 的环境。
   - RPM and deb packages: instructions can be found on the [NVSHMEM installer page](https://developer.nvidia.com/nvshmem-downloads?target_os=Linux)
     - 中文注释：系统包适合集群统一安装，但需要管理员权限和匹配发行版。
   - Conda packages through conda-forge
     - 中文注释：Conda 方式方便隔离环境，但要确认动态库路径能被运行时找到。
   - pip wheels through PyPI: `pip install nvidia-nvshmem-cu12`
     - 中文注释：pip wheel 适合 Python 环境，但仍要检查 CUDA/NCCL/驱动兼容性。
DeepEP is compatible with upstream NVSHMEM 3.3.9 and later.
中文注释：兼容“upstream”表示不需要 DeepEP 私有修改版 NVSHMEM。


### 2. Enable NVSHMEM IBGDA support

中文注释：IBGDA 是 low-latency kernel 的关键，它允许 kernel 内直接发 RDMA 操作，减少 CPU 参与和 launch 间隔。

NVSHMEM Supports two modes with different requirements. Either of the following methods can be used to enable IBGDA support.

中文注释：下面两种方式二选一即可。传统驱动参数方式更直接，GDRCopy 方式在无法修改驱动配置时更实用。

#### 2.1 Configure NVIDIA driver

中文注释：这是传统 IBGDA 开启方式，通过驱动参数允许 stream memory operations 和 peer mapping。

This configuration enables traditional IBGDA support.

Modify `/etc/modprobe.d/nvidia.conf`:

中文注释：该配置会影响 NVIDIA 内核模块加载参数，通常需要管理员权限。

```bash
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
```

Update kernel configuration:

中文注释：更新 initramfs 和重启是为了让新的驱动参数真正生效。

```bash
sudo update-initramfs -u
sudo reboot
```

#### 2.2 Install GDRCopy and load the gdrdrv kernel module

中文注释：这是 CPU-assisted IBGDA 方式，性能略有损失，但在无法修改驱动参数的集群中更容易落地。

This configuration enables IBGDA through asynchronous post-send operations assisted by the CPU. More information about CPU-assisted IBGDA can be found in [this blog](https://developer.nvidia.com/blog/enhancing-application-portability-and-compatibility-across-new-platforms-using-nvidia-magnum-io-nvshmem-3-0/#cpu-assisted_infiniband_gpu_direct_async%C2%A0).
It comes with a small performance penalty, but can be used when modifying the driver regkeys is not an option.

Download GDRCopy
GDRCopy is available as prebuilt deb and rpm packages [here](https://developer.download.nvidia.com/compute/redist/gdrcopy/). or as source code on the [GDRCopy github repository](https://github.com/NVIDIA/gdrcopy).

中文注释：GDRCopy 提供用户态/内核模块辅助路径，让 CPU 帮助 post-send，从而在不改驱动参数时启用部分 IBGDA 能力。

Install GDRCopy following the instructions on the [GDRCopy github repository](https://github.com/NVIDIA/gdrcopy?tab=readme-ov-file#build-and-installation).

中文注释：安装后需要确认 `gdrdrv` 模块已加载，否则 CPU-assisted IBGDA 仍不可用。

## Post-installation configuration

中文注释：这些环境变量影响 `setup.py` 查找 NVSHMEM，以及运行时动态链接 `libnvshmem.so`。如果编译或 import 报找不到 NVSHMEM，优先检查这里。

When not installing NVSHMEM from RPM or deb packages, set the following environment variables in your shell configuration:

```bash
export NVSHMEM_DIR=/path/to/your/dir/to/install  # Use for DeepEP installation
# 中文注释：`NVSHMEM_DIR` 会被 DeepEP 构建逻辑读取，用来定位 include/lib。
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
# 中文注释：`LD_LIBRARY_PATH` 确保运行时能找到 `libnvshmem.so`。
export PATH="${NVSHMEM_DIR}/bin:$PATH"
# 中文注释：`PATH` 让 `nvshmem-info` 等工具可以直接执行。
```

## Verification

中文注释：`nvshmem-info -a` 是最小验证项，只能说明库可见；真正验证 DeepEP legacy internode 还需要运行对应多节点测试。

```bash
nvshmem-info -a # Should display details of nvshmem
```

中文注释：如果该命令失败，先修复 NVSHMEM 安装和动态库路径，再继续构建或运行 DeepEP legacy 测试。
