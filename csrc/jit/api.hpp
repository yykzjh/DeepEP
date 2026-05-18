#pragma once  // 防止 JIT API 绑定声明重复包含。

#include "compiler.hpp"        // Compiler 负责 NVCC/NVRTC 编译和磁盘 cache。
#include "include_parser.hpp"  // IncludeParser 负责递归计算项目内 include hash。
#include "kernel_runtime.hpp"  // KernelRuntime 负责加载 cubin 并解析 kernel symbol。

namespace deep_ep::jit {  // DeepEP JIT 子系统命名空间。

// JIT API 的 Python 绑定入口。deep_ep/__init__.py 会调用 _C.init_jit，把 Python 探测到的
// deep_ep 包路径、CUDA_HOME 和 NCCL 根目录传给 C++。后续 Elastic kernel 第一次 launch 时，
// Compiler/KernelRuntime/IncludeParser 都依赖这些路径查找头文件、编译 cubin、解析符号。
static void init(const std::string& library_root_path,
                 const std::string& cuda_home_path_by_python, const std::string& nccl_root_path_by_python) {
    Compiler::prepare_init(library_root_path, cuda_home_path_by_python, nccl_root_path_by_python);  // 初始化编译器查找路径。
    KernelRuntime::prepare_init(cuda_home_path_by_python);                                         // 初始化 cubin 加载所需 CUDA 路径。
    IncludeParser::prepare_init(library_root_path);                                                // 初始化 include hash 根路径。
}

static void register_apis(pybind11::module_& m) {  // 向 pybind11 模块注册 JIT 相关函数。
    m.def("init_jit", &init);                      // 暴露为 _C.init_jit(...)，由 deep_ep/__init__.py 调用。
}

}  // namespace deep_ep::jit  // 结束 DeepEP JIT 子系统命名空间。
