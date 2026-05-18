# 所有预编译 CUDA 静态库都走这个 helper。
# Elastic/V2 的核心 kernel 主要在运行时 JIT，因此这里主要服务两类代码：
# 1. legacy 路径的固定 CUDA kernel，例如 intranode/internode/layout；
# 2. backend 路径的 NCCL/NVSHMEM/CUDA-driver 封装代码，它们给 legacy 和 elastic 共同使用。
function(add_deep_ep_library target_name source_file)
    add_library(${target_name} STATIC ${source_file})  # 每个 .cu 编译成一个静态库，最后链接进 deep_ep._C 扩展。
    set_target_properties(${target_name} PROPERTIES  # 给这个静态库目标设置编译属性。
            POSITION_INDEPENDENT_CODE ON             # Python 扩展是共享库，静态库也必须可重定位。
            CXX_STANDARD_REQUIRED ON                 # 要求 C++ 标准配置必须被编译器接受。
            CUDA_STANDARD_REQUIRED ON                # 要求 CUDA C++ 标准配置必须被 nvcc 接受。
            CXX_STANDARD 17                          # 这里保留 upstream 的 C++17；顶层 CMake 的 C++20 主要给宿主构建。
            CUDA_STANDARD 17                         # 预编译 kernel 使用 CUDA C++17，和 setup.py 的 torch extension 构建保持兼容。
            CUDA_SEPARABLE_COMPILATION ON            # 允许不同 CUDA 编译单元之间解析 device symbol，例如 NVSHMEM device 侧函数。
    )                                                # 完成目标属性设置。
    # nvshmem_host/nvshmem_device 支撑 legacy 的跨节点路径；cudadevrt 是 separable compilation 的 device runtime；
    # mlx5 用于 IB/RDMA 相关后端符号。Elastic 的 JIT kernel 不直接在这里编译，但会复用这些后端对象。
    target_link_libraries(${target_name} PUBLIC nvshmem_host nvshmem_device cudart cudadevrt mlx5)
endfunction()

add_subdirectory(legacy)   # legacy/V1 预编译通信 kernel。
add_subdirectory(backend)  # NCCL、NVSHMEM、CUDA driver runtime helper。
