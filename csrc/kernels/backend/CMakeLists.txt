# backend 目录不直接实现 token dispatch/combine 算法，而是封装通信基础设施：
# - nccl.cu：Elastic/V2 主要使用的 NCCL communicator、NCCL window、Gin device communicator；
# - nvshmem.cu：legacy/V1 跨节点路径使用的 NVSHMEM 初始化、对称内存分配、barrier；
# - cuda_driver.cu：批量写 flag / 等 flag 的 CUDA driver helper，用于 host 侧编排同步。
add_deep_ep_library(runtime_nccl_cuda nccl.cu)          # 编译 NCCL/Gin symmetric memory 后端静态库。
add_deep_ep_library(runtime_nvshmem_cuda nvshmem.cu)    # 编译 NVSHMEM 初始化和对称内存后端静态库。
add_deep_ep_library(runtime_cuda_driver cuda_driver.cu) # 编译 CUDA driver 批量 flag 操作后端静态库。

# 这些库不会在本 CMakeLists 里生成最终可执行目标，而是通过 CACHE INTERNAL 暴露给顶层扩展链接。
# csrc/kernels/legacy/CMakeLists.txt 也用同样模式暴露 LEGACY_CUDA_LIBRARIES。
set(RUNTIME_CUDA_LIBRARIES runtime_nccl_cuda runtime_nvshmem_cuda runtime_cuda_driver CACHE INTERNAL "Runtime kernels")  # 把 runtime 后端库列表暴露给上层链接逻辑。
