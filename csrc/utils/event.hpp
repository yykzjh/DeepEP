#pragma once  // 防止 EventHandle 工具重复定义。

#include <ATen/cuda/CUDAContext.h>  // 获取当前 CUDA stream 和 ATen CUDA stream wrapper。
#include <memory>                   // 使用 std::shared_ptr 持有 torch::Event。

#include <deep_ep/common/exception.cuh>  // 使用 EP_HOST_ASSERT 做 stream 检查。

namespace deep_ep {  // DeepEP C++ 公共命名空间。

struct EventHandle {
    // C++ 侧 CUDA event wrapper，会通过 pybind 暴露给 Python EventOverlap。
    // tensors_to_record 用来持有异步 kernel 仍会访问的张量，避免 Python 对象释放后底层缓存被复用。
    std::shared_ptr<torch::Event> event;                         // 实际 CUDA event 对象。
    std::vector<std::optional<torch::Tensor>> tensors_to_record;  // 需要延长生命周期的异步输出/输入张量。

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);     // 创建 CUDA event；torch::Event 会管理底层 cudaEvent_t 生命周期。
        event->record(at::cuda::getCurrentCUDAStream());          // 在当前 PyTorch CUDA stream 上记录 event，表示“当前已提交工作完成点”。
    }

    explicit EventHandle(const at::cuda::CUDAStream& stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);     // 创建 CUDA event；显式 stream 构造用于非当前 stream 的 overlap 场景。
        event->record(stream);                                    // 在指定 stream 上记录 event，后续其它 stream 可 wait 它。
    }

    EventHandle(const EventHandle& other) = default;              // 复制时共享底层 event 与张量引用。

    void current_stream_wait() const { at::cuda::getCurrentCUDAStream().unwrap().wait(*event); }  // 让当前 PyTorch stream 等待该 event，保证异步输出可安全使用。
};

static torch::Event create_event(const at::cuda::CUDAStream& s) {
    auto event = torch::Event(torch::kCUDA);  // 创建一个临时 CUDA event；它只用于建立 stream 间依赖。
    event.record(s);                         // 在输入 stream 上记录当前进度，也就是 s 之前排队的 kernel 完成点。
    return event;                            // 返回 event，调用方通常立即让另一个 stream wait。
}

static void stream_wait(const at::cuda::CUDAStream& s_0, const at::cuda::CUDAStream& s_1) {
    // 用事件把 s_1 上已提交的工作串到 s_0，而不是 cudaDeviceSynchronize。
    // DeepEP 的 overlap 设计大量依赖这种细粒度 stream 依赖。
    EP_HOST_ASSERT(s_0.id() != s_1.id());  // 同一 stream 不需要也不应该插入自等待；自等待可能暴露 CUDA API 未定义行为。
    s_0.unwrap().wait(create_event(s_1));  // 在 s_1 上记 event，并让 s_0 等它，从而形成 s_1 -> s_0 的依赖。
}

static void stream_wait(const at::cuda::CUDAStream& s, const EventHandle& event) {
    s.unwrap().wait(*event.event);  // 让指定 stream 等待 EventHandle 中保存的 CUDA event；Python EventOverlap 会走这条路径。
}

}  // namespace deep_ep  // 结束 DeepEP C++ 公共命名空间。
