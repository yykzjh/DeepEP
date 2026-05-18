"""DeepEP buffer 实现子包。

elastic.py 是当前 NCCL/Gin/JIT 路径，legacy.py 保留 NVSHMEM/旧 kernel 路径。
该 __init__ 不主动导入具体 Buffer，避免 import deep_ep 时过早加载 CUDA 扩展。
"""
# 本文件故意没有额外 import：Buffer 类由 deep_ep/__init__.py 选择性导出，
# 这样可以集中控制 CUDA 扩展初始化顺序。
