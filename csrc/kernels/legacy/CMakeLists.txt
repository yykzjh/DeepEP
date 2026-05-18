# legacy/V1 路径使用预编译 CUDA kernel，而不是 Elastic/V2 那样按 hidden/topk/rank 配置 JIT。
# 这些库会链接进 deep_ep._C，Python 的 deep_ep.buffers.legacy.Buffer 最终会通过
# csrc/legacy/buffer.hpp -> csrc/kernels/legacy/api.cuh 调到这里的 kernel。
add_deep_ep_library(legacy_layout_cuda layout.cu)         # 计算 dispatch layout：token/rank/expert 计数与前缀和。
add_deep_ep_library(legacy_intranode_cuda intranode.cu)   # 单节点 NVLink/PCIe 内的普通 dispatch/combine。
add_deep_ep_library(legacy_internode_cuda internode.cu)   # 多节点普通精度跨节点路径。
add_deep_ep_library(legacy_internode_ll_cuda internode_ll.cu)  # 多节点 low-latency 路径，偏推理场景。

# 与 backend 一样，先把静态库列表写入内部 cache，最终由扩展构建脚本统一链接。
set(LEGACY_CUDA_LIBRARIES legacy_layout_cuda legacy_intranode_cuda legacy_internode_cuda legacy_internode_ll_cuda CACHE INTERNAL "Legacy kernels")  # 把 legacy kernel 库列表暴露给上层链接逻辑。
