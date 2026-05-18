# csrc 是 Python 扩展里的 C++/CUDA 实现根目录。
# 目前单独下钻到 kernels 子目录，是因为实际需要编译成静态库的 CUDA 代码都集中在
# csrc/kernels/{legacy,backend}；上层的 csrc/elastic、csrc/jit、csrc/utils 多数是 header-only
# runtime 逻辑，会被 csrc/python_api.cpp 通过 include 方式拉入扩展编译单元。
add_subdirectory(kernels)
