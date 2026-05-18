#pragma once  // 防止异常与断言宏被重复定义。

// std::exception 是 host 侧自定义异常的基类。
#include <exception>
// std::string 保存格式化后的错误信息。
#include <string>
// std::stringstream 用于拼接错误码、文件名和行号。
#include <sstream>

// EP_STATIC_ASSERT 包一层 static_assert，便于项目里统一命名并允许外部覆盖。
#ifndef EP_STATIC_ASSERT
// cond 必须是编译期常量；reason 保持字符串字面量，方便编译器直接输出。
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

// EPException 是所有 host 侧 DeepEP 错误的统一异常类型。
class EPException : public std::exception {  // DeepEP host 侧统一异常对象。
private:
    // message 保存完整错误文本，what() 返回它的 c_str。
    std::string message = {};

public:  // 对外只暴露构造函数和 what()，避免调用方修改内部 message。
    // host 侧异常统一携带异常类别、文件和行号，Python binding 捕获后会把信息作为
    // RuntimeError 抛出，便于用户定位到 C++/CUDA 入口。
    explicit EPException(const char* name, const char* file, const int line, const std::string& error) {
        // 使用 stringstream 拼接，兼容错误码、字符串和条件表达式。
        std::stringstream ss;
        // 格式示例：CUDA runtime exception (foo.cu:12): 2 (cudaErrorMemoryAllocation, ...)
        ss << name << " exception (" << file << ":" << line << "): " << error;
        // 保存到成员变量，保证 what() 返回指针在异常对象生命周期内有效。
        message = ss.str();
    }

    // noexcept 符合 std::exception::what 的接口约束。
    const char* what() const noexcept override { return message.c_str(); }
};

// 构造带当前文件/行号的 EPException，便于手工抛出自定义错误信息。
#define EPExceptionWithLineInfo(name, message) EPException(name, __FILE__, __LINE__, message)  // 自动填入当前调用位置。

// EP_HOST_ASSERT 用于 host 侧不变量检查，失败时抛 EPException。
#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                           \
    do {                                                               \
        /* not(cond) 与项目 C++ 风格一致，失败时携带原条件字符串。 */ \
        /* 中文：host assert 抛异常，pybind 会把它转换成 Python 可见错误。 */ \
        if (not(cond)) {                                               \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                              \
        /* 中文：do-while(0) 让宏在 if/else 语句中表现得像单条语句。 */ \
    } while (0)
#endif

// EP_HOST_UNREACHABLE 标记 host 侧理论不可达分支。
#ifndef EP_HOST_UNREACHABLE
// 用表达式形式定义，方便在 switch/default 或三元表达式中直接使用。
#define EP_HOST_UNREACHABLE(reason) (throw EPException("Assertion", __FILE__, __LINE__, reason))
#endif

// EP_DEVICE_ASSERT 用于 device 侧不变量检查。
#ifndef EP_DEVICE_ASSERT
// device assert 不能抛 C++ 异常，因此打印失败条件后执行 trap，让 CUDA runtime 在
// 同步点报告 illegal instruction/assertion。这个宏只用于真正不该恢复的内部不变量。
#define EP_DEVICE_ASSERT(cond)                                                             \
    do {                                                                                   \
        /* device 侧使用 printf 输出文件、行号和条件，然后 trap 终止 kernel。 */           \
        /* 中文：device assert 不能安全抛异常，只能通过 trap 让 CUDA runtime 报错。 */     \
        if (not(cond)) {                                                                   \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
            /* 中文：trap 会让当前 kernel 立即异常退出，调用方在同步 CUDA stream 时看到错误。 */       \
            asm("trap;");                                                                  \
        }                                                                                  \
        /* 中文：device 宏同样用 do-while(0) 保持调用语句语法一致。 */                     \
    } while (0)
#endif

// EP_UNIFIED_ASSERT 在 host/device 两端选择对应 assert 实现。
#ifndef EP_UNIFIED_ASSERT
#ifdef __CUDA_ARCH__
// device 编译路径下使用 trap 版本。
// __CUDA_ARCH__ 只在 device pass 定义，因此同一 header 可同时服务 host/device 函数。
#define EP_UNIFIED_ASSERT(cond) EP_DEVICE_ASSERT(cond)
#else
// host 编译路径下使用异常版本。
#define EP_UNIFIED_ASSERT(cond) EP_HOST_ASSERT(cond)
#endif
#endif

// CUDA_RUNTIME_CHECK 把 CUDA runtime API 返回码转换成 EPException。
#ifndef CUDA_RUNTIME_CHECK
// CUDA runtime/driver/NCCL 的检查宏都转换为 EPException，避免不同库返回码在上层
// 形成多套错误处理逻辑。
#define CUDA_RUNTIME_CHECK(cmd) \
do { /* 宏体使用 do-while(0)，让调用点能像普通语句一样安全加分号。 */ \
    /* 立即执行 cmd 并保存返回码，避免宏参数被求值多次。 */ \
    const auto e = (cmd); /* e 保存 cudaError_t 返回码。 */ \
    /* cudaSuccess 以外都视为失败。 */ \
    if (e != cudaSuccess) { /* CUDA runtime API 调用失败，转换为 EPException。 */ \
        std::stringstream ss; /* 临时拼接错误码、错误名和错误描述。 */ \
        /* 同时输出错误整数、错误名和错误描述。 */ \
        ss << static_cast<int>(e) << " (" << cudaGetErrorName(e) << ", " << cudaGetErrorString(e) << ")"; \
        throw EPException("CUDA runtime", __FILE__, __LINE__, ss.str()); /* 抛出带调用位置的 runtime 异常。 */ \
    } /* 失败分支结束。 */ \
} while (0) /* 宏语句块结束。 */
#endif

// CUDA_DRIVER_CHECK 把 CUDA driver API 返回码转换成 EPException。
#ifndef CUDA_DRIVER_CHECK
#define CUDA_DRIVER_CHECK(cmd) \
do { /* driver 检查同样包装成单条语句，便于 if/else 调用。 */ \
    /* 执行 driver API 并保存 CUresult。 */ \
    const auto e = (cmd); /* e 保存 CUresult。 */ \
    /* CUDA_SUCCESS 以外都视为失败。 */ \
    if (e != CUDA_SUCCESS) { /* CUDA driver API 调用失败。 */ \
        std::stringstream ss; /* 拼接 driver 错误文本。 */ \
        /* driver 错误名/描述通过 lazy loader 解析，避免直接链接 driver 符号。 */ \
        const char *name, *info; /* lazy driver 返回的错误名和错误说明指针。 */ \
        lazy_cuGetErrorName(e, &name), lazy_cuGetErrorString(e, &info); /* 通过懒加载 driver 符号获取错误文本。 */ \
        ss << static_cast<int>(e) << " (" << name << ", " << info << ")"; /* 统一错误输出格式。 */ \
        throw EPException("CUDA driver", __FILE__, __LINE__, ss.str()); /* 抛出带文件行号的 driver 异常。 */ \
    } /* 失败分支结束。 */ \
} while (0) /* 宏语句块结束。 */
#endif

// NCCL_CHECK 把 NCCL 返回码转换成 EPException。
#ifndef NCCL_CHECK
#define NCCL_CHECK(cmd) \
do { /* NCCL 检查宏保持与 CUDA 检查宏相同的单语句形态。 */ \
    /* 执行 NCCL API 并保存 ncclResult_t。 */ \
    const auto e = (cmd); /* e 保存 NCCL 返回码。 */ \
    /* ncclSuccess 以外都抛出异常。 */ \
    if (e != ncclSuccess) { /* NCCL 调用失败。 */ \
        std::stringstream ss; /* 拼接 NCCL 错误文本。 */ \
        /* ncclGetLastError(nullptr) 返回最近 NCCL 错误文本。 */ \
        ss << static_cast<int>(e) << " (" << ncclGetLastError(nullptr) << ")"; /* NCCL 没有像 CUDA 一样拆分 name/string。 */ \
        throw EPException("NCCL", __FILE__, __LINE__, ss.str()); /* 抛出带调用位置的 NCCL 异常。 */ \
    } /* 失败分支结束。 */ \
} while (0) /* 宏语句块结束。 */
#endif
