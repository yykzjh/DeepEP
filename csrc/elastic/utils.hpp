#pragma once  // 防止 elastic 工具函数重复定义。

#include <torch/python.h>                 // 使用 torch::Tensor、CUDAStream 与 PyTorch C++ API。
#include <deep_ep/common/exception.cuh>   // 使用 EP_HOST_ASSERT 做 host 侧参数检查。

namespace deep_ep::elastic {  // Elastic Buffer C++ runtime 命名空间。

static at::cuda::CUDAStream get_global_comm_stream() {
    // Elastic runtime 默认共享一个 high-priority communication stream。
    // Python 的 async_with_compute_stream=True 只是不让 compute stream 立即等待它；实际同步
    // 通过 EventOverlap 完成。这样通信 kernel 可与 expert GEMM 在不同 stream 上重叠。
    static std::optional<at::cuda::CUDAStream> comm_stream = std::nullopt;  // 进程内懒加载的全局通信 stream。
    if (not comm_stream.has_value())                                        // 第一次调用时才向 PyTorch stream pool 申请。
        comm_stream = at::cuda::getStreamFromPool(true);                    // true 表示 high-priority stream pool。
    return comm_stream.value();                                             // 返回同一个通信 stream，供所有 ElasticBuffer 复用。
}

template <int kNumDims>                         // kNumDims 是期望张量维度数，例如 1/2。
static auto get_shape(const torch::Tensor& t) {  // 检查维度并返回 int tuple。
    // 统一做维度检查并把 torch::SymInt/ssize 转成 int。多数 kernel 模板参数要求 int 维度，
    // 例如 hidden、num_tokens、num_topk，提前检查可以让错误在 host 侧暴露。
    EP_HOST_ASSERT(t.dim() == kNumDims);                                      // 维度不匹配时立即在 host 侧报错。
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {                 // 用 index_sequence 展开每一维。
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);           // 把每个 size 转成 int 后组成 tuple。
    }(std::make_index_sequence<kNumDims>());                                  // 生成 0..kNumDims-1 的编译期索引序列。
}

template <typename dtype_t = void>                         // dtype_t 指定返回裸指针类型，默认 void。
static dtype_t* get_data_ptr(const std::optional<torch::Tensor>& t) {  // 从 optional tensor 安全取得 data_ptr。
    // optional tensor 到裸指针的桥接 helper。C++ runtime 里 topk_weights、bias、scale factor
    // 都可能为空，device kernel 用 nullptr 判断是否启用对应功能。
    return t.has_value() ? t->data_ptr<dtype_t>() : nullptr;           // 空 optional 对应 nullptr，供 kernel 分支判断。
}

}  // deep_ep::elastic  // 结束 Elastic Buffer C++ runtime 命名空间。
