#pragma once  // legacy 配置头是 header-only，被 Python 绑定和 kernel launcher 多处包含。

#include <deep_ep/common/exception.cuh>  // 使用 EP_HOST_ASSERT 做 host 侧参数和容量检查。

#include "../kernels/legacy/api.cuh"  // 读取 LEGACY_NUM_MAX_NVL_PEERS、topk_idx_t、internode metadata 字节数等常量。

namespace deep_ep::legacy {  // legacy 预编译 kernel 的 host 配置命名空间。

// legacy runtime 的整数工具，和 device 端 math.cuh 的 align/ceil_div 语义一致。
// 这里运行在 host 侧，用于估算 NVLink/RDMA buffer 字节数，要求与 kernel 的布局假设一致。
template <typename dtype_t>
dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;  // 向上整除，例如 ceil_div(17, 8)=3。
}

template <typename dtype_t>
dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;  // 向上对齐到 b 的倍数，例如 align_up(17,8)=24。
}

template <typename dtype_t>
dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;  // 向下对齐到 b 的倍数，例如 align_down(17,8)=16。
}

struct Config {
    // legacy 高吞吐 dispatch/combine 的手工调优参数。
    // num_sms 决定通信 kernel 使用多少 SM；每个 NVLink channel 使用 send/recv 两个 block，
    // 所以 Python set_num_sms 要求偶数。chunked_send/recv_tokens 控制 pipeline 中每个 channel
    // 一次搬运多少 token，过小会增加同步开销，过大则降低 overlap。
    int num_sms;  // 本次 legacy EP kernel 占用的 SM 数，internode 通常要求偶数以拆分 send/recv channel。
    int num_max_nvl_chunked_send_tokens;  // 单个 NVLink channel 每次最多发送 token 数。
    int num_max_nvl_chunked_recv_tokens;  // 单个 NVLink channel 接收侧缓冲 token 容量，必须大于 send chunk。
    int num_max_rdma_chunked_send_tokens;  // 单个 RDMA channel 每次最多发送 token 数。
    int num_max_rdma_chunked_recv_tokens;  // 单个 RDMA channel 接收侧缓冲 token 容量，会向上对齐到 send chunk 倍数。

    Config(int num_sms,
           // 中文：下面四个 chunk 参数分别控制 NVL/RDMA 发送批量与接收 ring 容量。
           int num_max_nvl_chunked_send_tokens,
           int num_max_nvl_chunked_recv_tokens,
           int num_max_rdma_chunked_send_tokens,
           int num_max_rdma_chunked_recv_tokens)  // 构造函数只保存和校验调参参数，不分配任何 buffer。
        : num_sms(num_sms),  // 初始化 SM 数。
          num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),  // 初始化 NVL send chunk。
          num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),  // 初始化 NVL recv chunk。
          num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),  // 初始化 RDMA send chunk。
          num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {  // 初始化 RDMA recv chunk，随后会修正对齐。
        EP_HOST_ASSERT(num_sms >= 0);  // 0 表示某些路径不占 SM，正常 benchmark 会传正数。
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens > 0 and num_max_nvl_chunked_recv_tokens > 0);  // NVL chunk 必须为正。
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens);  // 接收缓冲要比发送 chunk 大，给流水线留余量。
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens > 0 and num_max_rdma_chunked_recv_tokens > 0);  // RDMA chunk 必须为正。

        // Ceil up RDMA buffer size
        this->num_max_rdma_chunked_recv_tokens = align_up<int>(num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);  // 让接收容量是发送 chunk 的整数倍，便于 lazy head 更新。
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens < num_max_rdma_chunked_recv_tokens);  // 至少要能容纳超过一个发送 chunk。
        // NOTES: this assertion is related to RDMA lazy head update, we must ensure senders always have space to push
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens / 2);  // 接收缓冲至少两倍 send chunk，防止 head 未及时更新时写满。
    }

    size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;  // legacy 元数据按最多 128 个 top-k 槽预留，覆盖所有测试/模型配置。
        constexpr int kNumMaxScales = 128;  // FP8 scale 以 hidden/128 为单位，legacy 这里按最多 128 组预留。
        EP_HOST_ASSERT(num_ranks < LEGACY_NUM_MAX_NVL_PEERS or num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0);  // 跨节点时 rank 数必须是每节点 8 卡的倍数。
        EP_HOST_ASSERT(num_ranks <= LEGACY_NUM_MAX_NVL_PEERS or num_sms % 2 == 0);  // internode 时一个 channel 对应 send/recv 两个 SM，因此 SM 数要为偶数。
        const auto num_rdma_ranks = std::max(num_ranks / LEGACY_NUM_MAX_NVL_PEERS, 1);  // RDMA rank 数等于节点数，单节点时至少为 1。
        const auto num_nvl_ranks = std::min(num_ranks, LEGACY_NUM_MAX_NVL_PEERS);  // 每个节点内最多 8 个 NVLink peer。
        const int num_channels = num_sms / 2;  // legacy internode 把两个 SM 组成一个 channel，分别处理收发。

        size_t num_bytes = 0;  // 累计 NVL workspace 字节数。
        num_bytes += num_channels * num_nvl_ranks * (2 * num_rdma_ranks + 3) * sizeof(int);  // 每 channel/peer 的控制计数与 head/tail 状态。
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * hidden_bytes;  // token payload，形状近似 [C,NVL,chunk,H_bytes]。
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * internode::get_source_meta_bytes();  // source metadata，记录源 rank/token/top-k 等回写信息。
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t);  // 可选 topk_idx metadata。
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(float);  // 可选 topk_weights metadata。
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxScales * sizeof(float);  // FP8 scale metadata。
        num_bytes = ((num_bytes + 127) / 128) * 128;  // 整体对齐到 128 字节，匹配 CUDA 向量化/通信控制结构对齐。
        return num_bytes;  // 返回调用方应至少分配的 NVLink buffer 字节数。
    }

    size_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const {
        // Legacy mode
        if (num_ranks <= LEGACY_NUM_MAX_NVL_PEERS)  // 单节点只走 NVLink，不需要 RDMA buffer。
            return 0;  // 返回 0 表示调用方无需注册 RDMA workspace。

        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;  // RDMA 侧同样为 top-k metadata 预留最大槽数。
        constexpr int kNumMaxScales = 128;  // RDMA 侧同样为 FP8 scale 预留最大组数。
        EP_HOST_ASSERT(num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0);  // internode rank 数必须能整除节点内 8 卡。
        EP_HOST_ASSERT(num_sms % 2 == 0);  // RDMA channel 需要成对 SM。
        const int num_rdma_ranks = num_ranks / LEGACY_NUM_MAX_NVL_PEERS;  // 节点数，也就是 RDMA peer 数。
        const int num_channels = num_sms / 2;  // 每两个 SM 形成一个 RDMA channel。

        size_t num_bytes = 0;  // 累计 RDMA workspace 字节数。
        num_bytes += num_channels * num_rdma_ranks * (LEGACY_NUM_MAX_NVL_PEERS * 2 + 2) * 2 * sizeof(int);  // 双缓冲控制区：每节点内 peer 的 head/tail 加额外状态。
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * hidden_bytes * 2;  // payload 双缓冲，形状近似 [2,C,RDMA,chunk,H_bytes]。
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * internode::get_source_meta_bytes() * 2;  // source metadata 双缓冲。
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t) * 2;  // topk_idx 双缓冲。
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(float) * 2;  // topk_weights 双缓冲。
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxScales * sizeof(float) * 2;  // FP8 scale 双缓冲。
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * sizeof(int4) * 2;  // int4 控制消息双缓冲，用于跨节点同步/slot 状态。
        num_bytes = ((num_bytes + 127) / 128) * 128;  // 与 NVL 一样整体 128 字节对齐。
        return num_bytes;  // 返回调用方应至少分配的 RDMA buffer 字节数。
    }
};

struct LowLatencyBuffer {
    int num_clean_int = 0;  // clean_meta 需要清零的 int 数量，覆盖 recv count/flag 区。

    void* dispatch_rdma_send_buffer = nullptr;  // low-latency dispatch 发送消息区。
    void* dispatch_rdma_recv_data_buffer = nullptr;  // low-latency dispatch 接收数据区。
    int* dispatch_rdma_recv_count_buffer = nullptr;  // dispatch 接收计数/信号区。

    void* combine_rdma_send_buffer = nullptr;  // low-latency combine 发送消息区。
    void* combine_rdma_recv_data_buffer = nullptr;  // low-latency combine 接收数据区。
    int* combine_rdma_recv_flag_buffer = nullptr;  // combine 接收 flag 区，和 dispatch count 区复用。

    void* combine_rdma_send_buffer_data_start = nullptr;  // combine 发送区中真正 payload 起点，跳过可能的控制头。
    size_t num_bytes_per_combine_msg = 0;  // 单条 combine 消息字节数，kernel 用它计算 token slot 偏移。

    std::pair<int*, int> clean_meta() {
        EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer == combine_rdma_recv_flag_buffer);  // 两条路径必须复用同一信号区，才能一次清零。
        return {dispatch_rdma_recv_count_buffer, num_clean_int};  // 返回待清零指针和 int 数量。
    }
};

struct LowLatencyLayout {
    size_t total_bytes = 0;  // 整个 low-latency RDMA workspace 需要的字节数。
    LowLatencyBuffer buffers[2];  // 奇偶双缓冲，kernel 通过 buffer_idx 在两组 layout 间切换，避免读写冲突。

    template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*, typename in_ptr_t = void*>
    out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
        return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);  // 在字节指针上前进 count，再转回目标指针类型。
    }

    LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
        const int num_scales = hidden / 128;  // FP8/scale 相关路径按每 128 hidden 一组 scale 计算。

        // Dispatch and combine layout:
        //  - 2 symmetric odd/even send buffer
        // 中文：odd/even 双缓冲对应 low-latency dispatch/combine 的 phase 切换。
        //  - 2 symmetric odd/even receive buffers
        //  - 2 symmetric odd/even signaling buffers

        // Message sizes
        // NOTES: you should add a control `int4` for combine messages if you want to do data transformation
        // NOTES: `num_scales * sizeof(nv_bfloat162)` means the per-128-channel min/max
        EP_HOST_ASSERT(num_scales * sizeof(float) <= hidden);  // scale 区不能比原 hidden 字节还大，否则 FP8 消息没有意义。
        size_t num_bytes_per_dispatch_msg = sizeof(int4) + std::max(hidden * sizeof(nv_bfloat16), hidden + num_scales * sizeof(float));  // dispatch 消息 = 控制 int4 + max(BF16 payload, FP8 payload+scale)。
        size_t num_bytes_per_combine_msg = num_scales * sizeof(nv_bfloat162) + hidden * sizeof(nv_bfloat16);  // combine 消息 = scale/minmax 辅助区 + BF16 payload。

        // Send buffer
        size_t dispatch_send_buffer_bytes = num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;  // dispatch send 形状近似 [tokens_per_rank,msg_bytes]。
        size_t combine_send_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;  // combine send 按 [expert,tokens_per_rank,msg_bytes] 预留。
        size_t send_buffer_bytes = std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);  // 同一发送区给 dispatch/combine 复用，取较大者。
        EP_HOST_ASSERT(send_buffer_bytes % sizeof(int4) == 0);  // 消息区按 int4 对齐，匹配 device 端向量化控制头。
        total_bytes += send_buffer_bytes * 2;  // 奇偶双缓冲各一份 send buffer。

        // Symmetric receive buffers
        // TODO: optimize memory usages
        size_t dispatch_recv_data_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;  // dispatch recv 按每 expert 最坏 token 数预留。
        size_t combine_recv_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;  // combine recv 同样按 expert 维度预留。
        size_t recv_buffer_bytes = std::max(dispatch_recv_data_buffer_bytes, combine_recv_buffer_bytes);  // 接收区复用，取 dispatch/combine 较大值。
        EP_HOST_ASSERT(recv_buffer_bytes % sizeof(int4) == 0);  // 保证接收消息边界可按 int4 对齐访问。
        total_bytes += recv_buffer_bytes * 2;  // 奇偶双缓冲各一份 recv buffer。

        // Symmetric signaling buffers
        size_t dispatch_recv_count_buffer_bytes = num_experts * sizeof(int);  // dispatch 每 expert 一个接收计数。
        size_t combine_recv_flag_buffer_bytes = dispatch_recv_count_buffer_bytes;  // combine 每 expert 一个接收 flag，大小与 dispatch count 一致。
        size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes, combine_recv_flag_buffer_bytes);  // 信号区复用，取较大值。
        size_t signaling_buffer_bytes_aligned = align_up<size_t>(signaling_buffer_bytes, 128);  // 信号区按 128 字节对齐，便于 RDMA/atomic 访问。
        total_bytes += signaling_buffer_bytes_aligned * 2;  // 奇偶双缓冲各一份信号区。

        // Assign pointers
        // NOTES: we still leave some space for distinguishing dispatch/combine buffer,
        // so you may see some parameters are duplicated
        for (int i = 0; i < 2; ++i) {  // 为奇偶两个 buffer 计算相同结构、不同偏移的指针集合。
            buffers[i] = {static_cast<int>(signaling_buffer_bytes / sizeof(int)),  // clean_meta 需要清理的 int 数，不含对齐 padding。
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),  // dispatch send 指向第 i 份 send 区。
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 + recv_buffer_bytes * i),  // dispatch recv 指向第 i 份 recv 区。
                          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),  // dispatch count 指向第 i 份信号区。
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),  // combine send 复用第 i 份 send 区。
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 + recv_buffer_bytes * i),  // combine recv 复用第 i 份 recv 区。
                          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),  // combine flag 复用第 i 份信号区。
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),  // combine payload 起点当前等于 send 区起点。
                          num_bytes_per_combine_msg};  // 保存单条 combine 消息字节数给 kernel 使用。
        }
    }
};

size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    auto num_bytes = LowLatencyLayout(nullptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts).total_bytes;  // 用空指针构造 layout，只计算 total_bytes。
    return ((num_bytes + LEGACY_NUM_BUFFER_ALIGNMENT_BYTES) / LEGACY_NUM_BUFFER_ALIGNMENT_BYTES) * LEGACY_NUM_BUFFER_ALIGNMENT_BYTES;  // 对齐到 legacy buffer 统一粒度。
}

}  // namespace deep_ep::legacy
