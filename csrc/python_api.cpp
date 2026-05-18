#include <pybind11/pybind11.h>  // 使用 PYBIND11_MODULE 和 pybind11::module_ 注册 Python 扩展。
#include <torch/python.h>       // 使用 PyTorch C++/Python 互操作类型和 dtype 映射。

#include <deep_ep/common/compiled.cuh>  // 引入 topk_idx_t、kEnableSM90Features 等全局编译配置。

#include "jit/api.hpp"        // 注册 _C.init_jit。
#include "elastic/buffer.hpp" // 注册 ElasticBuffer 与 elastic 相关函数。
#include "legacy/buffer.hpp"  // 注册 legacy Buffer 与旧路径函数。

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C  // setup.py 未注入扩展名时，默认模块名为 _C。
#endif                           // 结束 TORCH_EXTENSION_NAME 默认值保护。

bool is_sm90_compiled() {  // 旧接口：返回当前扩展是否启用了 SM90/Hopper 特性。
#ifndef DISABLE_SM90_FEATURES
    return true;           // 未禁用 SM90 特性，说明编译进了 FP8/TMA 等路径。
#else
    return false;          // 禁用了 SM90 特性，通常用于 Ampere 或兼容构建。
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // 定义 Python 扩展模块初始化函数。
    m.doc() = "DeepEP: an efficient expert-parallel communication library";  // 设置 Python 模块文档字符串。

    // 这个文件是 Python 和 C++/CUDA 的唯一公开绑定入口。
    // Python 侧 `import deep_ep._C as _C` 后，deep_ep/buffers/*.py 中的所有 runtime 调用
    // 都会进入这里注册的函数或类。注册顺序也建议作为学习顺序：
    //   1. JIT API：先让 C++ 知道 CUDA/NCCL/include/cache 路径；
    //   2. legacy Buffer：注册 V1/NVSHMEM 预编译 kernel 路径；
    //   3. elastic Buffer：注册 V2/JIT kernel 路径。
    // 注意：张量形状校验不在这里做，而是在 Python wrapper 和各 C++ runtime 方法中完成。

    // Whether support FP8 and TMA features
    m.def("is_sm90_compiled", []() { return deep_ep::kEnableSM90Features; });  // 暴露 SM90 特性是否启用。

    // The integer type of top-k indices
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value);  // 暴露 topk_idx_t 对应 PyTorch dtype。

    // JIT API
    deep_ep::jit::register_apis(m);      // 注册 JIT 初始化 API。

    // Register legacy buffer APIs
    deep_ep::legacy::register_apis(m);   // 注册 legacy Buffer API。

    // Register elastic buffer (DeepEP V2) APIs
    deep_ep::elastic::register_apis(m);  // 注册 ElasticBuffer API。
}
