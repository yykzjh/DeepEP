#pragma once  // KernelRuntime 是 header-only JIT runtime wrapper，防止重复定义。

#include <ATen/cuda/CUDAContext.h>  // 使用 ATen/CUDA 上下文，保证和 PyTorch 当前 CUDA 设备/stream 一致。

#include <deep_ep/common/exception.cuh>  // EP_HOST_ASSERT 和 CUDA 错误检查宏。

#include "../utils/format.hpp"      // fmt::format 构造 cuobjdump 命令和错误信息。
#include "../utils/lazy_init.hpp"   // EP_DECLARE_STATIC_VAR_IN_CLASS 定义静态成员。
#include "handle.hpp"               // LibraryHandle/KernelHandle/load_kernel/unload_library 抽象。

namespace deep_ep::jit {  // DeepEP JIT 子系统命名空间。

class KernelRuntime final {
public:
    static std::filesystem::path cuda_home;  // Python init_jit 传入的 CUDA_HOME；用于定位 bin/cuobjdump。

    LibraryHandle library;  // 已加载 cubin module/library 句柄，析构时需要 unload。
    KernelHandle kernel;    // module 中真正可 launch 的 __global__ kernel 句柄。

    explicit KernelRuntime(const std::filesystem::path& dir_path) {
        EP_HOST_ASSERT(not cuda_home.empty());  // prepare_init 必须先设置 CUDA_HOME，否则无法定位 cuobjdump。

        // NOLINT(*-pro-type-member-init)
        const auto cuobjdump_path = cuda_home / "bin" / "cuobjdump";  // cuobjdump 用来从 cubin 中枚举 kernel symbol。
        const auto cubin_path = dir_path / "kernel.cubin";            // JIT 编译得到的二进制 cubin 路径。
        // dir_path 是 build() 计算出的 cache 目录，里面至少包含 kernel.cu 和 kernel.cubin。
        // KernelRuntime 不重新编译，只负责解析 cubin symbol 并加载 module。
        if (get_env<int>("EP_JIT_DEBUG"))                 // debug 模式下打印正在加载的 cubin。
            printf("Loading CUBIN: %s\n", cubin_path.c_str());  // 输出 cubin 绝对/相对路径。

        // Find the only symbol
        // JIT 生成的翻译单元里会有一个 __instantiate_kernel 辅助函数和一个真正的 __global__ kernel。
        // 这里用 cuobjdump -symbols 找唯一的 entry symbol，并过滤 vprintf/assert/internal 等符号。
        // 如果一个 cache 目录里出现 0 个或多个 kernel symbol，说明编译产物不完整或 cache 被污染。
        // TODO: use kernel enumeration for newer drivers
        const std::vector<std::string> illegal_names = {"vprintf", "__instantiate_kernel", "__internal", "__assertfail"};  // 需要过滤掉的非业务 kernel 符号片段。
        const auto [exit_code, symbols] = call_external_command(fmt::format("{} -symbols {}", cuobjdump_path.c_str(), cubin_path.c_str()));  // 执行 cuobjdump 并拿到符号表文本。
        EP_HOST_ASSERT(exit_code == 0);           // cuobjdump 失败通常说明 CUDA_HOME 不对或 cubin 损坏。
        std::istringstream iss(symbols);          // 把多行符号表包装成输入流逐行解析。
        std::vector<std::string> symbol_names;    // 保存候选 kernel entry 名称；最终必须恰好一个。
        for (std::string line; std::getline(iss, line); ) {  // 遍历 cuobjdump 输出的每一行。
            // cuobjdump 每行 symbol 信息包含类型和名字；STT_FUNC + STO_ENTRY 表示
            // 可启动 kernel。过滤 illegal_names 是为了排除辅助函数和设备断言。
            if (line.find("STT_FUNC") == 0 and line.find("STO_ENTRY") != std::string::npos and
                std::none_of(illegal_names.begin(), illegal_names.end(),
                [&](const auto name) { return line.find(name) != std::string::npos; })) {  // 排除辅助函数、assert、vprintf 等符号。
                const auto last_space = line.rfind(' ');                                   // cuobjdump 行尾通常是符号名。
                symbol_names.push_back(line.substr(last_space + 1));                       // 截取最后一个空格后的 kernel 名。
            }
        }

        // Print symbols
        if (symbol_names.size() != 1) {
            printf("Corrupted JIT cache directory (expected 1 kernel symbol, found %zu): %s, "  // 报告符号数不符合预期。
                   "please run `rm -rf %s` and restart your task.\n",
                   symbol_names.size(), dir_path.c_str(), dir_path.c_str());  // 打印 cache 目录，便于用户删除重建。
            printf("Symbol names: ");  // 打印候选符号名前缀。
            for (const auto& symbol: symbol_names)  // 遍历所有候选符号。
                printf("%s, ", symbol.c_str());     // 输出单个符号名，帮助判断污染来源。
            printf("\n");  // 符号列表换行。
            EP_HOST_ASSERT(false and "Corrupted JIT cache directory");  // 中止加载损坏 cache。
        }

        // Load from the library
        // load_kernel 会根据 handle.hpp 选择 runtime/driver API，把 cubin 加载成可 launch 的
        // KernelHandle；library 保存在对象里，析构时必须 unload，否则长时间 JIT 会积累 module。
        kernel = load_kernel(cubin_path, symbol_names[0], &library);  // 加载 cubin 并取出唯一 kernel symbol。
    }

    static void prepare_init(const std::string& cuda_home_path_by_python) {
        // deep_ep/__init__.py 在导入扩展后调用 prepare_init，把 Python 检测到的
        // CUDA_HOME 传给 C++，避免 C++ 自己猜测路径。
        cuda_home = cuda_home_path_by_python;  // 保存 Python 探测到的 CUDA_HOME。
    }

    static bool check_validity(const std::filesystem::path& dir_path) {
        // cache 目录采用“先编译到 tmp，再原子 rename 到最终目录”的方式创建。
        // 因此最终目录如果存在，至少应同时包含 kernel.cu 和 kernel.cubin；缺任意一个都说明
        // 可能是并发进程或共享文件系统留下的损坏目录，需要用户删除重建。
        if (not std::filesystem::exists(dir_path))  // 最终 cache 目录尚不存在，说明需要编译或等待其它进程编译。
            return false;                          // 不报错，调用方会继续走编译路径。
        // NOTES: if the directory exists, kernel.cu and kernel.cubin must both exist,
        // because the directory is created atomically via rename
        if (not std::filesystem::exists(dir_path / "kernel.cu") or        // 源码文件缺失说明 cache 不完整。
            not std::filesystem::exists(dir_path / "kernel.cubin")) {     // cubin 文件缺失说明编译产物不完整。
            printf("Corrupted JIT cache directory (missing kernel.cu or kernel.cubin): %s, "  // 报告损坏目录。
                   "please run `rm -rf %s` and restart your task.\n",
                   dir_path.c_str(), dir_path.c_str());  // 同时给出可删除路径。
            EP_HOST_ASSERT(false and "Corrupted JIT cache directory");  // 已存在但不完整的目录不能静默复用。
        }
        return true;  // 目录存在且关键文件齐全，可以加载。
    }

    ~KernelRuntime() noexcept(false) {
        // library 必须晚于 kernel handle 卸载。KernelHandle 只是 module 内符号句柄，
        // 真正资源由 LibraryHandle 持有。
        unload_library(library);  // 释放 CUDA module/library，避免进程长时间运行时 module 泄漏。
    }
};

EP_DECLARE_STATIC_VAR_IN_CLASS(KernelRuntime, cuda_home);  // 在头文件中定义 KernelRuntime::cuda_home 静态成员。

} // namespace deep_ep::jit  // 结束 DeepEP JIT 子系统命名空间。
