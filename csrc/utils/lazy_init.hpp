#pragma once  // 防止 LazyInit 定义被重复包含。

#include <functional>  // 使用 std::function 保存延迟构造 factory。
#include <memory>      // 使用 std::shared_ptr 持有延迟构造出的对象。

#define EP_DECLARE_STATIC_VAR_IN_CLASS(cls, name) decltype(cls::name) cls::name  // 在头文件中定义类静态成员的便捷宏。

namespace deep_ep {  // DeepEP C++ 工具命名空间。

template <typename T>  // T 是需要延迟构造的对象类型，例如 Compiler 或 DeviceRuntime。
class LazyInit {       // LazyInit 把“首次使用时构造”封装成类似指针的对象。
public:
    explicit LazyInit(std::function<std::shared_ptr<T>()> factory)  // 构造时只保存 factory，不立即创建 T。
        : factory(std::move(factory)) {}                            // 移动 factory，避免复制捕获状态。

    T* operator -> () {
        // 第一次访问时才构造对象，之后复用 shared_ptr。JIT compiler/device runtime
        // 这类全局对象依赖 CUDA/NCCL 环境，延迟初始化可以避免 import 阶段过早触发。
        if (ptr == nullptr)  // 第一次访问才真正构造对象。
            ptr = factory(); // 调用用户传入的 factory，返回 shared_ptr<T>。
        return ptr.get();    // 返回裸指针，让调用方可以使用 lazy->method() 语法。
    }

private:
    // factory 保存构造逻辑，ptr 保存已初始化实例；本类不做线程同步，调用方默认在
    // Python GIL 或单线程初始化路径下访问。
    std::shared_ptr<T> ptr;                         // 保存已构造对象；为空表示尚未初始化。
    std::function<std::shared_ptr<T>()> factory;    // 保存构造逻辑；通常捕获环境变量或路径。
};

} // namespace deep_ep  // 结束 DeepEP C++ 工具命名空间。
