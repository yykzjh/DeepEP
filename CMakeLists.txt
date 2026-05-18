# NOTES: this CMake is only for debugging; for setup, please use Torch extension
# 这个 CMakeLists 主要服务本地调试、IDE 索引和手工构建；正式 pip/wheel 构建走 setup.py + Torch CUDAExtension。
cmake_minimum_required(VERSION 3.10)  # 选择较低 CMake 版本门槛，便于集群老环境打开工程。
project(deep_ep LANGUAGES CUDA CXX)  # 声明项目同时包含 CUDA 与 C++ 编译单元。
set(CMAKE_VERBOSE_MAKEFILE ON)  # 打开详细编译命令，调试 include/lib/flags 时更直观。

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")  # C 源若被间接编译，默认使用 O3 和位置无关代码。
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")  # C++ 扩展最终会链接成 .so，因此需要 -fPIC。
set(CUDA_SEPARABLE_COMPILATION ON)  # NVSHMEM/device 符号需要 separable compilation 和 device link。
list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")  # 调试构建启用 fast debug 宏，供部分 CUDA 路径条件编译。
list(APPEND CUDA_NVCC_FLAGS "-O3")  # CUDA kernel 保持优化级别，避免调试构建性能/寄存器形态差异过大。
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")  # 打印 ptxas 资源并警告 local memory，方便定位寄存器溢出。
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-rdynamic")  # 让 host 符号更容易被 profiler/调试器解析。
list(APPEND CUDA_NVCC_FLAGS "-lineinfo")  # 在 cubin 中保留源码行号，Nsight/SASS 分析可回到源码。
# Suppress warnings for the `fmt` library
list(APPEND CUDA_NVCC_FLAGS "--diag-suppress=128,2417")  # 抑制 fmt 在 NVCC 下常见但无害的诊断。

set(USE_SYSTEM_NVTX on)  # 让 PyTorch/CUDA 扩展使用系统 NVTX，避免重复携带 NVTX 组件。
set(CUDA_ARCH_LIST "9.0" CACHE STRING "List of CUDA architectures to compile")  # 默认面向 Hopper/SM90，与 DeepEP 高性能路径一致。
set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")  # PyTorch CMake helper 读取 TORCH_CUDA_ARCH_LIST 生成 nvcc arch flags。

find_package(CUDAToolkit REQUIRED)  # 查找 CUDA Toolkit，提供 include/lib/nvcc 相关变量。
find_package(pybind11 REQUIRED)  # 查找 pybind11，用于构建 Python 扩展模块 _C。
find_package(Torch REQUIRED)  # 查找 LibTorch/PyTorch C++ 扩展依赖。

# NVSHMEM
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT_DIR}/lib/cmake/nvshmem)  # legacy internode/LL 路径依赖 NVSHMEM host/device 库。
add_library(nvshmem ALIAS nvshmem::nvshmem)  # 给下层 CMake 保留简短 nvshmem target 名。
add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)  # host runtime target 别名。
add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)  # device 静态库 target 别名。

# NCCL
# TODO: use `find_package` instead of manual checks
if (NOT NCCL_ROOT_DIR)  # 当前 CMake 调试构建要求用户显式传入 NCCL_ROOT_DIR。
    message(FATAL_ERROR "NCCL_ROOT_DIR is not set.")  # 缺少 NCCL 时无法编译 backend/nccl.cu 和 JIT include。
endif()  # 结束 NCCL_ROOT_DIR 检查。

set(CMAKE_CXX_STANDARD 20)  # host 代码使用 C++20，和 setup.py 的默认 JIT/扩展标准一致。
set(CMAKE_CUDA_STANDARD 20)  # CUDA 编译单元同样使用 C++20。

# include/ 是给 IDE 与 CMake 调试构建使用的统一头文件入口：
# - include/deep_ep -> deep_ep/include/deep_ep，承接 <deep_ep/...> 公开 CUDA 头；
# - include/csrc    -> csrc，承接扩展内部头文件（jit、elastic、legacy、utils）。
# 使用符号链接而不是复制文件，可以保证跳转到的声明/实现始终是项目中的真实源文件。
include_directories(include deep_ep/include third-party/fmt/include)  # 项目内头文件：根 include、JIT 公开头、fmt。
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/include/cccl ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NVSHMEM_INCLUDE_DIR} ${NCCL_ROOT_DIR}/include .)  # 外部依赖 include：CUDA/CCCL/Torch/Python/NVSHMEM/NCCL/项目根。
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib ${NVSHMEM_LIB_DIR} ${NCCL_ROOT_DIR}/lib)  # 外部依赖库目录，供 _C 和 CUDA 静态库链接。

# Add kernels
add_subdirectory(csrc)  # 进入 csrc，构建 backend、legacy、elastic 等 CUDA/C++ target。

# Link CPP and CUDA together
pybind11_add_module(_C csrc/python_api.cpp)  # 构建 Python 扩展模块 _C，绑定入口在 csrc/python_api.cpp。
target_link_libraries(_C PRIVATE nccl ${RUNTIME_CUDA_LIBRARIES} ${LEGACY_CUDA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)  # 链接 NCCL、运行时 CUDA 库、legacy CUDA 库和 PyTorch Python 库。

# Enable kernel code indexing with CMake-based IDEs
cuda_add_library(deep_ep_indexing_cuda STATIC csrc/indexing/main.cu)  # 编译索引用翻译单元，让 IDE 能解析 deep_ep/include 下的 JIT kernel 模板。
