#pragma once  // IncludeParser 是 header-only JIT 依赖分析工具，防止重复定义。

#include <filesystem>  // 递归读取 include 文件路径。
#include <regex>       // 用正则从源码中解析 #include 语句。
#include <string>      // 源码文本、文件名、hash 字符串。
#include <vector>      // 保存当前源码直接 include 的文件列表。

#include "../utils/format.hpp"  // fmt::format 构造错误信息。
#include "../utils/system.hpp"  // get_hex_digest/EP_HOST_UNREACHABLE 等工具间接可用。

namespace deep_ep::jit {  // DeepEP JIT 子系统命名空间。
// DeepEP JIT 子系统命名空间。

class IncludeParser {  // 递归解析 JIT 源码 include 并计算依赖 hash。
    // IncludeParser 用来把 deep_ep/include 下的头文件内容纳入 JIT cache key。
    // 这样改了 device 实现头后，即使 Python 传入的模板参数不变，也会重新编译 cubin。
    std::unordered_map<std::string, std::optional<std::string>> cache;
    // cache[path] 保存某个头文件的递归 hash；nullopt 表示该文件正在解析，用来检测循环 include。

    static std::vector<std::string> get_includes(const std::string& code, const std::filesystem::path& file_path = "") {  // 提取源码中的 deep_ep include。
        // 从一段源码中找出标准格式的 #include <deep_ep/...> 列表。
        std::vector<std::string> includes;  // 当前源码直接依赖的项目内头文件列表。
        // 保存当前源码直接 include 的 deep_ep 头文件名。
        const std::regex pattern(R"(#\s*include\s*[<"][^>"]+[>"])");  // 捕获尖括号或引号形式的 include。
        // 匹配 #include <...> 或 #include "..."，后续再做格式约束。
        std::sregex_iterator iter(code.begin(), code.end(), pattern);  // include 匹配迭代器。
        // 正则迭代器起点。
        const std::sregex_iterator end;  // regex 迭代终止哨兵。
        // 正则迭代器终点。

        // TODO: parse relative paths as well
        // JIT 生成代码要求 include 写成标准的 `#include <deep_ep/...>`。
        // 只解析 deep_ep 前缀，是因为这些头文件属于包内 device 实现，变化后必须让 JIT cache 失效；
        // CUDA/NCCL 系统头由 compiler flags / 外部版本控制，不递归 hash。
        for (; iter != end; ++ iter) {
            // 遍历所有 include 语句。
            const auto include_str = iter->str();  // 当前 include 语句原文。
            // 当前匹配到的完整 include 字符串，例如 "#include <deep_ep/common/math.cuh>"。
            const int len = include_str.length();  // 用于截取文件名和检查末尾字符。
            // include 字符串长度，用于检查首尾字符。
            if (include_str.substr(0, 10) == "#include <" and include_str[len - 1] == '>' and include_str[10] != ' ' and include_str[len - 2] != ' ') {
                // JIT 只接受严格的尖括号 include，避免不同写法产生歧义。
                std::string filename = include_str.substr(10, len - 11);  // 去除 #include < 和 > 后的文件名。
                // 去掉 "#include <" 和末尾 ">" 得到文件名。
                if (filename.substr(0, 7) == "deep_ep")  // We only parse `<deep_ep/*>`
                    // 只把项目内 deep_ep 头文件计入 hash；系统头由编译器版本/flags 管理。
                    includes.push_back(filename);
            } else {
                // 非标准 include 直接报错，要求 JIT 生成代码保持可预测。
                std::string error_info = fmt::format("Non-standard include: {}", include_str);  // 报错信息主体。
                // 构造错误信息。
                if (file_path != "")
                    // 如果来自某个具体文件，把文件路径附上。
                    error_info += fmt::format(" ({})", file_path.string());
                EP_HOST_UNREACHABLE(error_info);
                // 触发 host 侧不可达错误。
            }
        }
        return includes;
        // 返回当前源码直接依赖的 deep_ep 头文件列表。
    }

public:
    static std::filesystem::path library_include_path;  // deep_ep/include 根目录。
    // deep_ep/include 根目录，由 init_jit 设置。

    static void prepare_init(const std::string& library_root_path) {  // 初始化 include 根目录。
        // Python 传入的是 deep_ep 包根目录；JIT include 统一从 <root>/include 解析。
        library_include_path = std::filesystem::path(library_root_path) / "include";
        // 保存 include 根路径，后续 deep_ep/foo.cuh 会拼成 <root>/include/deep_ep/foo.cuh。
    }

    std::string get_hash_value(const std::string& code, const bool& exclude_code = true) {  // 计算源码及其 include 递归 hash。
        // exclude_code=true 用于 LaunchRuntime 的 include hash：只关心依赖头文件内容；
        // exclude_code=false 用于递归解析头文件本身，把当前文件内容也计入 hash。
        std::stringstream ss;  // 拼接依赖 hash 的临时流。
        // 用流拼接所有直接 include 的递归 hash。
        for (const auto& i: get_includes(code))
            // 对当前源码每个直接 include 递归计算文件 hash。
            ss << get_hash_value_by_path(library_include_path / i) << "$";
            // 用 "$" 分隔，避免 hash 串简单拼接产生歧义。
        if (not exclude_code)
            // 递归解析头文件自身时，需要把当前文件内容也纳入 hash。
            ss << "#" << get_hex_digest(code);
            // "#" 分隔 include hash 和当前源码 hash。
        return get_hex_digest(ss.str());
        // 返回最终摘要，作为 JIT 源码头部注释的一部分进入 compiler cache key。
    }

    std::string get_hash_value_by_path(const std::filesystem::path& path) {  // 按文件路径递归读取并计算 hash。
        // Check whether hit in cache
        // ReSharper disable once CppUseAssociativeContains
        if (cache.count(path) > 0) {
            // 已访问过该 path，可能是缓存命中，也可能是递归中再次遇到。
            const auto opt = cache[path];  // 读取缓存项，可能是 nullopt 循环哨兵。
            // opt.has_value=false 表示该文件正在解析。
            if (not opt.has_value())
                // 先写入 nullopt 再递归解析 include。若递归过程中再次访问同一 path，
                // 说明出现循环 include；这里直接报错，避免 hash 计算无限递归。
                EP_HOST_UNREACHABLE(fmt::format("Circular include may occur: {}", path.string()));
            return opt.value();
            // 正常缓存命中，返回已计算的递归 hash。
        }

        // 读取文件后先把 cache[path] 标成 nullopt，表示“正在解析”；递归完成后再写入
        // 真实 hash。这个状态机能检测 A include B、B 又 include A 的循环依赖。
        std::ifstream in(path);  // 打开项目内 include 文件。
        // 打开 include 文件。
        if (not in.is_open())
            // 找不到文件说明 JIT include 路径或生成源码中的 include 写错。
            EP_HOST_UNREACHABLE(fmt::format("Failed to open: {}", path.string()));
        std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());  // 一次性读完整文件内容。
        // 读取完整文件内容到字符串。
        cache[path] = std::nullopt;
        // 标记为“正在解析”，用于检测循环 include。
        return (cache[path] = get_hash_value(code, false)).value();
        // 递归计算 hash，写入 cache 后返回。
    }
};

EP_DECLARE_STATIC_VAR_IN_CLASS(IncludeParser, library_include_path);
// 定义 IncludeParser::library_include_path 静态成员。

static auto include_parser = std::make_shared<IncludeParser>();
// 全局 include parser 实例；LaunchRuntime::generate 通过它计算 include hash。

}  // namespace deep_ep::jit
// DeepEP JIT 子系统命名空间结束。
