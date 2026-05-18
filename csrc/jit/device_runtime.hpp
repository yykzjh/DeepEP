#pragma once  // DeviceRuntime 是 header-only JIT helper，防止重复定义。

#include <deep_ep/common/exception.cuh>  // CUDA_RUNTIME_CHECK 用于 CUDA runtime API 错误检查。

#include "../utils/lazy_init.hpp"  // LazyInit 用于首次访问时再构造全局 DeviceRuntime。

namespace deep_ep::jit {  // DeepEP JIT 子系统命名空间。

// 当前 CUDA device 的能力缓存。Elastic host wrapper 会读取 SM 数、opt-in shared memory 上限、
// clock rate 和 arch 字符串，用来选择 launch 形状、计算 timeout cycles、拼接 NVCC arch flags。
class DeviceRuntime {
    int64_t cached_clock_rate = 0;                // 缓存当前 device 时钟频率，单位 Hz；0 表示尚未查询。
    std::shared_ptr<cudaDeviceProp> cached_prop;  // 缓存 cudaDeviceProp，避免每次 launch 前重复 cudaGetDeviceProperties。

    std::shared_ptr<cudaDeviceProp> get_prop() {
        if (cached_prop == nullptr) {  // 首次读取属性时才调用 CUDA runtime。
            int device_idx;            // 当前线程绑定的 CUDA device id。
            cudaDeviceProp prop;       // CUDA runtime 填充的设备属性结构体。
            CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));                  // 查询当前 device。
            CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&prop, device_idx));  // 查询当前 device 的 SM 数、共享内存、架构等属性。
            cached_prop = std::make_shared<cudaDeviceProp>(prop);            // 复制到 shared_ptr，后续返回同一份缓存。
        }
        return cached_prop;  // 返回缓存属性；调用方只读其中字段。
    }

public:
    int64_t get_clock_rate() {
        if (cached_clock_rate == 0) {  // 时钟频率尚未缓存时查询一次。
            // NOTES: we should convert kHz into Hz
            // cudaDevAttrClockRate 返回 kHz。dispatch/combine 的 GPU timeout 使用 clock64()，
            // 所以这里转成 Hz 后乘以 Python 传入的秒数。
            int device_idx, rate;  // device_idx 是当前 GPU，rate 是 kHz 单位的 clock rate。
            CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));  // 查询当前 GPU。
            CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(&rate, cudaDevAttrClockRate, device_idx));  // 查询 SM clock rate，CUDA 返回 kHz。
            cached_clock_rate = static_cast<int64_t>(rate) * 1000ll;  // 转成 Hz，后续 timeout_seconds * Hz 得到 clock64 周期数。
        }
        return cached_clock_rate;  // 返回 Hz 单位频率。
    }

    int get_num_smem_bytes() {
        return static_cast<int>(get_prop()->sharedMemPerBlockOptin);  // 返回每个 block 可 opt-in 的动态 shared memory 上限。
    }

    int get_num_sms() {
        return get_prop()->multiProcessorCount;  // 返回 SM 数，Elastic launch 会据此决定 block 数和通信通道分配。
    }

    std::pair<int, int> get_arch_pair() {
        const auto prop = get_prop();       // 读取缓存设备属性。
        return {prop->major, prop->minor};  // 返回 CUDA compute capability，例如 H100 为 {9, 0}。
    }

    std::string get_arch(const bool& number_only = false,
                         const bool& support_arch_family = false) {
        // SM100 有 100a/100f 的架构后缀差异。support_arch_family=true 时可生成 family arch，
        // 否则使用具体 arch，避免 NVCC 对某些新 ISA 生成不兼容代码。
        const auto [major, minor] = get_arch_pair();  // 拆出 compute capability 主版本和次版本。
        if (major == 10 and minor != 1) {             // Blackwell/SM100 系列存在 a/f family 后缀差异。
            if (number_only)                          // 调用方只需要数字时不带 a/f 后缀。
                return "100";                         // 返回 NVCC arch 数字部分。
            return support_arch_family ? "100f" : "100a";  // 根据调用方是否支持 family arch 返回 100f 或具体 100a。
        }
        return std::to_string(major * 10 + minor) + (number_only ? "" : "a");  // 其它架构用 smXYa 风格字符串，例如 90a。
    }

    int get_arch_major() {
        return get_arch_pair().first;  // 返回 compute capability 主版本，便于快速判断架构代际。
    }
};

static auto device_runtime = LazyInit<DeviceRuntime>([](){ return std::make_shared<DeviceRuntime>(); });  // 全局懒初始化入口，使用 device_runtime->get_num_sms() 风格访问。

} // namespace deep_ep::jit  // 结束 DeepEP JIT 子系统命名空间。
