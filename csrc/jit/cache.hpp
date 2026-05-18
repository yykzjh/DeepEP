#pragma once  // KernelRuntimeCache 是 header-only 小工具，防止重复定义。

#include <filesystem>   // cache key 使用文件系统路径，指向 JIT 编译产物目录。
#include <memory>       // 使用 shared_ptr 共享 KernelRuntime module 句柄。
#include <unordered_map> // 进程内用 hash map 从目录路径映射到已加载 KernelRuntime。

#include "kernel_runtime.hpp"  // KernelRuntime 负责检查 cache 目录并加载 cubin。

namespace deep_ep::jit {  // DeepEP JIT 子系统命名空间。

// 进程内 KernelRuntime cache。文件系统 cache 负责跨 rank 复用 cubin；这里负责避免同一进程
// 重复 cuModuleLoad/cudaLibraryLoad。同一个 dir_path 对应一个已经加载好的 kernel module。
class KernelRuntimeCache {
    std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> cache;  // key 是 cache 目录字符串，value 是已加载 module。

public:
    KernelRuntimeCache() = default;  // 默认构造空 cache。

    void clear() {
        // 测试或调试时可清空进程内已加载 module。磁盘 cache 不受影响，下次 get 会
        // 重新从 kernel.cubin 加载。
        cache.clear();  // 清空 shared_ptr 后，未被外部持有的 KernelRuntime 会析构并 unload CUDA module。
    }

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        // Hit the runtime cache。这里 key 使用 path string，避免 std::filesystem::path
        // 在不同标准库实现中 hash 行为差异。
        if (const auto iterator = cache.find(dir_path); iterator != cache.end())  // 先查进程内 cache，避免重复加载 cubin。
            return iterator->second;                                             // 命中时直接返回已有 KernelRuntime。

        // KernelRuntime::check_validity 会检查 kernel.cu/kernel.cubin 是否都存在。
        // 如果另一个 rank 还在编译，最终目录尚未通过 rename 形成，这里返回 nullptr，
        // 调用方会继续走 build 流程或等待已有目录出现。
        if (KernelRuntime::check_validity(dir_path))  // 磁盘 cache 目录完整时才允许加载。
            // 构造 KernelRuntime 会解析 cubin symbol 并加载 CUDA module；放入 cache 后，
            // 同一进程再次启动同形状 kernel 时只需复用 handle。
            return cache[dir_path] = std::make_shared<KernelRuntime>(dir_path);  // 加载并保存到进程内 cache。
        return nullptr;  // 目录不存在或尚未形成完整 cache 时返回空，让调用方继续编译/等待。
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();  // 全局进程内 JIT module cache。

} // namespace deep_ep::jit  // 结束 DeepEP JIT 子系统命名空间。
