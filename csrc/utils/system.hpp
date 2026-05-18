#pragma once  // 防止系统工具函数在同一个翻译单元里被重复定义。

#include <algorithm>    // 提供通用算法；本文件保留该 include 以兼容后续路径处理逻辑。
#include <array>        // call_external_command 用固定大小 std::array 作为读取缓冲区。
#include <filesystem>  // make_dirs/safe_remove_all 使用 std::filesystem 操作目录和文件。
#include <functional>  // 保留函数对象相关类型；和 lazy/system 工具头的依赖风格一致。
#include <random>      // get_uuid 用随机数生成临时文件/目录后缀。
#include <string>      // 环境变量、命令输出和 UUID 都使用 std::string。
#include <memory>      // call_external_command 用 unique_ptr 管理 popen 返回的 FILE*。
#include <unistd.h>    // get_uuid 使用 getpid() 把进程号放进临时名称。

#include <deep_ep/common/exception.cuh>  // EP_HOST_ASSERT/EP_HOST_UNREACHABLE 用于 host 侧错误检查。

#include "format.hpp"  // fmt::format 的 header-only wrapper，用于构造错误信息。

namespace deep_ep {  // DeepEP C++ 公共工具命名空间。

// ReSharper disable once CppNotAllPathsReturnValue
template <typename dtype_t>
static dtype_t get_env(const std::string& name, const dtype_t& default_value = dtype_t()) {
    // JIT 和通信路径大量通过环境变量调试：EP_JIT_DEBUG、EP_JIT_CACHE_DIR、
    // EP_NUM_TOPK_IDX_BITS 等。模板化 get_env 统一完成字符串到目标类型的转换。
    const auto c_str = std::getenv(name.c_str());  // 从当前进程环境中读取 name 对应的 C 字符串指针。
    if (c_str == nullptr)                          // 环境变量不存在时返回调用方给的默认值。
        return default_value;                      // 例如没有设置 EP_JIT_DEBUG 时默认返回 0。

    // Read the env and convert to the desired type
    if constexpr (std::is_same_v<dtype_t, std::string>) {  // 字符串类型不需要解析，直接复制。
        return std::string(c_str);                         // 返回 std::string，避免悬挂到 getenv 的内部指针。
    } else if constexpr (std::is_same_v<dtype_t, int>) {   // int 类型用于 EP_JIT_DEBUG 等开关/计数。
        int value;                                         // 保存 sscanf 解析出来的整数。
        std::sscanf(c_str, "%d", &value);                  // 用 C 格式解析，输入非法时行为沿用原实现。
        return value;                                      // 返回解析后的整数值。
    } else {
        EP_HOST_ASSERT(false and "Unexpected type");       // 其它 dtype_t 没有实现转换，直接报错提醒维护者补分支。
    }
}

static std::tuple<int, std::string> call_external_command(std::string command) {
    // 用于 NVCC/cuobjdump 等外部命令。stderr 被合并到 stdout，JIT 编译失败时调用方
    // 可以把完整命令输出打印出来，便于定位环境和编译参数问题。
    command = command + " 2>&1";  // 合并 stderr，保证 NVCC 报错也能进入 output。
    const auto deleter = [](FILE* f) { if (f) pclose(f); };  // unique_ptr 析构时兜底关闭 pipe，避免异常路径泄漏。
    std::unique_ptr<FILE, decltype(deleter)> pipe(popen(command.c_str(), "r"), deleter);  // 用 popen 启动命令并读取 stdout。
    EP_HOST_ASSERT(pipe != nullptr);  // popen 失败说明 shell/命令环境不可用，JIT 不能继续。

    std::array<char, 512> buffer;  // 每次从 pipe 读取最多 511 个字符，最后一个位置留给 C 字符串结尾。
    std::string output;            // 累积完整命令输出，失败时会进入异常/日志。
    while (fgets(buffer.data(), buffer.size(), pipe.get()))  // 持续读取直到 EOF。
        output += buffer.data();                             // 把当前 chunk 追加到完整输出。
    const auto status = pclose(pipe.release());              // 主动关闭 pipe 并拿到子进程状态，同时 release 避免 deleter 重复 pclose。
    // NOTES: if the child was killed by a signal (e.g., SIGINT from Ctrl+C),
    // WEXITSTATUS would incorrectly return 0. Treat signal death as failure.
    const auto exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);  // 正常退出取退出码，被信号杀死则映射成 shell 风格 128+signal。
    return {exit_code, output};  // 返回二元组：退出码和完整输出。
}

static std::filesystem::path make_dirs(const std::filesystem::path& path) {
    // OK if existed
    std::error_code capture;  // filesystem API 的非异常错误输出，避免目录创建失败直接抛异常。
    const bool created = std::filesystem::create_directories(path, capture);  // 递归创建目录；已存在时返回 false 且 error_code 为 0。
    if (not (created or capture.value() == 0)) {  // 既没创建成功，也不是“已存在/无错误”，说明真实失败。
        EP_HOST_UNREACHABLE(fmt::format("Failed to make directory: {}, created: {}, value: {}",
                                        path.c_str(), created, capture.value()));  // 报告路径、created 标志和系统错误码。
    }
    if (created and get_env<int>("EP_JIT_DEBUG"))  // JIT debug 打开时打印新建目录，方便追踪 cache/临时目录位置。
        printf("Create directory: %s\n", path.c_str());  // 输出创建的目录路径。
    return path;  // 返回原 path，便于调用方链式使用 make_dirs(cache_dir)。
}

static std::string get_uuid() {
    static std::random_device rd;  // 非确定性随机源，用来给伪随机引擎播种。
    static std::mt19937 gen([]() {  // 进程内共享的 Mersenne Twister 引擎，避免每次调用重新构造。
        return rd() ^ std::chrono::steady_clock::now().time_since_epoch().count();  // 混合随机源和当前时间，降低多进程同种子的概率。
    }());  // 立即执行 lambda 完成静态初始化。
    static std::uniform_int_distribution<uint32_t> dist;  // 生成 32-bit 随机段，用于临时文件名后缀。

    std::stringstream ss;  // 用流拼接 pid 和随机十六进制段。
    ss << getpid() << "-"  // 先放进程号；同一用户多进程并发 JIT 时能降低名称冲突概率。
       << std::hex << std::setfill('0')  // 后续随机段用十六进制，并用 0 补足宽度。
       << std::setw(8) << dist(gen) << "-"  // 第一段 8 位十六进制随机数。
       << std::setw(8) << dist(gen) << "-"  // 第二段 8 位十六进制随机数。
       << std::setw(8) << dist(gen);        // 第三段 8 位十六进制随机数。
    return ss.str();  // 返回形如 pid-xxxxxxxx-xxxxxxxx-xxxxxxxx 的临时 ID。
}

static void safe_remove_all(const std::filesystem::path& path) {
    std::error_code ec;  // 所有 filesystem 操作用 error_code 接收错误，保证清理函数尽力而为、不抛异常。
    if (not std::filesystem::exists(path, ec) or ec)  // 路径不存在或查询失败时直接返回。
        return;  // 清理临时目录/cache 时，缺失路径不算错误。

    // A single file
    if (not std::filesystem::is_directory(path, ec) or ec) {  // 不是目录，或者无法判断为目录时按单文件处理。
        std::filesystem::remove(path, ec);                    // 删除单个文件；失败也只记录在 ec 中。
        return;                                               // 单文件删除后结束递归。
    }

    // Remove directory
    auto it = std::filesystem::directory_iterator(path,  // 构造目录迭代器，逐项清理子文件/子目录。
        std::filesystem::directory_options::skip_permission_denied, ec);  // 遇到权限不足条目时跳过，避免中断整个清理。
    for (auto end = std::filesystem::directory_iterator(); it != end and not ec;) {  // 遍历目录直到结束或迭代器出错。
        const auto entry_path = it->path();  // 保存当前条目路径；后面先递增迭代器再递归删除它。

        // Increase firstly to avoid failures
        it.increment(ec);  // 先移动迭代器，避免递归删除当前 entry 后迭代器失效。
        if (ec)            // 迭代器移动失败时停止遍历，保持清理函数的容错语义。
            break;         // 跳出循环后仍尝试删除当前目录。

        // Recursively clean
        safe_remove_all(entry_path);  // 递归删除子路径；子路径可以是文件或目录。
    }
    std::filesystem::remove(path, ec);  // 子项清理后删除空目录；若残留条目导致失败也不抛异常。
}

} // deep_ep  // 结束 DeepEP C++ 公共工具命名空间。
