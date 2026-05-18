#pragma once  // 防止 hash 工具函数重复定义。

#include <string>   // 使用 std::string 作为输入和十六进制输出。
#include <cstdint>  // 使用 uint64_t/uint8_t 做稳定宽度整数运算。

namespace deep_ep {  // DeepEP C++ 工具命名空间。

static uint64_t fnv1a(const std::vector<char>& data, const uint64_t& seed) {  // 对字节数组计算带 seed 的 FNV-1a 64-bit hash。
    // JIT cache key 用的轻量 hash。FNV-1a 足够快，配合两个不同 seed 与 splitmix
    // 输出 128-bit hex，降低不同 kernel 源码碰撞概率。
    uint64_t h = seed;                              // h 保存滚动 hash 状态，初始值由 seed 决定。
    constexpr uint64_t prime = 0x100000001b3ull;    // FNV-1a 64-bit 标准乘数。
    for (const char& c: data) {                     // 逐字节扫描输入数据。
        h ^= static_cast<uint8_t>(c);               // 先把当前字节 xor 到低位。
        h *= prime;                                 // 再乘 prime 扩散状态。
    }
    return h;                                       // 返回 64-bit FNV-1a 状态。
}

static std::string get_hex_digest(const std::vector<char>& data) {             // 把字节数组转换成 128-bit 风格十六进制摘要。
    const auto state_0 = fnv1a(data, 0xc6a4a7935bd1e995ull);                   // 用第一个 seed 得到第一段 64-bit 状态。
    const auto state_1 = fnv1a(data, 0x9e3779b97f4a7c15ull);                   // 用第二个 seed 得到第二段 64-bit 状态。

    // SplitMix64 再混洗一次 FNV 状态，改善高位/低位分布，最终输出 32 位十六进制字符。
    const auto split_mix = [](uint64_t z) {                                    // 对 64-bit 状态做 SplitMix64 后处理。
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;                           // 第一轮 xor-shift + 乘法混洗。
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;                           // 第二轮 xor-shift + 乘法混洗。
        return z ^ (z >> 31);                                                  // 最后一轮 xor-shift，返回混洗结果。
    };

    std::ostringstream oss;                                                    // 用 stringstream 组装固定宽度十六进制字符串。
    oss << std::hex << std::setfill('0')                                       // 设置十六进制输出，并用 0 补齐宽度。
        << std::setw(16) << split_mix(state_0)                                 // 输出第一段 16 个十六进制字符。
        << std::setw(16) << split_mix(state_1);                                // 输出第二段 16 个十六进制字符。
    return oss.str();                                                          // 返回 32 字符摘要，供 JIT cache 路径使用。
}

static std::string get_hex_digest(const std::string& data) {                   // 字符串输入的便捷重载。
    return get_hex_digest(std::vector<char>{data.begin(), data.end()});        // 转成字节数组后复用主实现。
}

} // namespace deep_ep  // 结束 DeepEP C++ 工具命名空间。
