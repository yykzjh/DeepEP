#pragma once  // legacy device buffer 切片工具是 header-only，多个 .cu 会包含。

#include <deep_ep/common/exception.cuh>  // 使用 EP_STATIC_ASSERT 做编译期分支约束。

#include "compiled.cuh"  // 引入 legacy 编译常量和 CUDA 类型配置。

namespace deep_ep::legacy {  // legacy CUDA kernel 的辅助 buffer 命名空间。

template <typename dtype_t>
struct Buffer {  // 从线性 workspace 中连续切出一段 dtype_t 数组的最简单封装。
private:
    uint8_t* ptr;  // 实际保存字节指针，便于按 byte offset 做切片。

public:
    int64_t total_bytes;  // 当前 Buffer 占用的总字节数，advance_also 会用它推进其它指针。

    __device__ __forceinline__ Buffer() : ptr(nullptr), total_bytes(0) {}  // 默认构造空 buffer，供条件路径占位。

    __device__ __forceinline__ Buffer(void*& gbl_ptr, int num_elems, int offset = 0) {
        // 从一整块线性 workspace 中切出 num_elems 个 dtype_t。构造函数会推进
        // gbl_ptr，后续 Buffer 继续从新位置分配，类似 device 端 bump allocator。
        // offset 只影响当前 Buffer 的起始访问点，不影响 gbl_ptr 推进长度。
        total_bytes = num_elems * sizeof(dtype_t);  // 记录当前切片大小，单位是字节。
        ptr = static_cast<uint8_t*>(gbl_ptr) + offset * sizeof(dtype_t);  // 起点允许额外偏移 offset 个元素。
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;  // 推进全局 bump pointer，给下一个 Buffer 使用。
    }

    __device__ __forceinline__ Buffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;  // 用当前 Buffer 的大小推进另一条线性指针。
        return *this;  // 返回自身，方便链式初始化。
    }

    __device__ __forceinline__ dtype_t* buffer() { return reinterpret_cast<dtype_t*>(ptr); }  // 以 dtype_t* 形式访问切片起点。

    __device__ __forceinline__ dtype_t& operator[](int idx) { return buffer()[idx]; }  // 直接按元素下标访问，等价于 buffer()[idx]。
};

template <typename dtype_t, int kNumRanks = 1>
struct AsymBuffer {  // 非对称 buffer：每个 peer/rank 有独立指针或独立区域。
private:
    uint8_t* ptrs[kNumRanks];  // kNumRanks=1 时只有本地线性指针；>1 时每个 NVLink peer 一个映射指针。
    int64_t num_bytes;  // 单个 rank/peer 的一个 slot 占用字节数。

public:
    int64_t total_bytes;  // 整个 AsymBuffer 在每条底层指针上需要推进的字节数。

    __device__ __forceinline__ AsymBuffer(void*& gbl_ptr, int num_elems, int num_ranks, int sm_id = 0, int num_sms = 1, int offset = 0) {
        // 单指针版本用于 RDMA buffer：同一块内存按 [num_sms, num_ranks, num_elems]
        // 切片，每个 SM/channel 拥有独立发送区域，减少 atomic 冲突。
        // 逻辑地址：base + sm_id * (num_ranks*num_bytes) + offset * num_bytes。
        EP_STATIC_ASSERT(kNumRanks == 1, "");
        num_bytes = num_elems * sizeof(dtype_t);  // 一个目标 rank 的数据区大小。

        int64_t per_channel_bytes = num_bytes * num_ranks;  // 一个 SM/channel 包含所有目标 rank 的区域。
        total_bytes = per_channel_bytes * num_sms;  // 所有 SM/channel 的总区域。
        ptrs[0] = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id + num_bytes * offset;  // 当前 SM 的第 offset 个 rank 区域。
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;  // 推进 RDMA 线性 workspace。
    }

    __device__ __forceinline__ AsymBuffer(void** gbl_ptrs, int num_elems, int num_ranks, int sm_id = 0, int num_sms = 1, int offset = 0) {
        // 多指针版本用于 NVLink peer buffer：gbl_ptrs[i] 是第 i 个 peer 的映射地址，
        // 每个 peer 的指针都以相同 total_bytes 推进，保持各 rank 布局一致。
        // kNumRanks 是编译期 peer 数，num_ranks 是运行期逻辑 rank/slot 数，两者语义不要混淆。
        EP_STATIC_ASSERT(kNumRanks > 1, "");
        num_bytes = num_elems * sizeof(dtype_t);  // 每个 peer/rank 的一个 slot 字节数。

        int64_t per_channel_bytes = num_bytes * num_ranks;  // 每个 SM/channel 里按 rank 连续排布。
        total_bytes = per_channel_bytes * num_sms;  // 每个 peer 映射指针都要保留相同大小。
        for (int i = 0; i < kNumRanks; ++i) {  // 遍历所有 peer 映射指针。
            ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + per_channel_bytes * sm_id + num_bytes * offset;  // 指向 peer i 当前 SM/offset 区域。
            gbl_ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;  // 推进 peer i 的 bump pointer。
        }
    }

    __device__ __forceinline__ void advance(int shift) {
        // 在现有指针基础上整体前移 shift 个 dtype_t 元素，常用于跳过若干 rank/slot。
        #pragma unroll
        for (int i = 0; i < kNumRanks; ++i)  // 所有 peer 指针保持相同偏移。
            ptrs[i] = ptrs[i] + shift * sizeof(dtype_t);  // 字节指针按元素数转换成字节数推进。
    }

    __device__ __forceinline__ AsymBuffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;  // 用当前总大小推进另一条单指针 workspace。
        return *this;  // 返回自身方便链式表达。
    }

    template <int kNumAlsoRanks>
    __device__ __forceinline__ AsymBuffer advance_also(void** gbl_ptrs) {
        for (int i = 0; i < kNumAlsoRanks; ++i)  // 额外推进一组 peer 指针。
            gbl_ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;  // 每个 peer 都推进相同布局大小。
        return *this;  // 返回自身。
    }

    __device__ __forceinline__ dtype_t* buffer(int idx = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "`buffer` is only available for single rank case");  // 单指针 RDMA 情况才允许按 idx 取 buffer。
        return reinterpret_cast<dtype_t*>(ptrs[0] + num_bytes * idx);  // 返回第 idx 个 rank/slot 的 dtype_t*。
    }

    __device__ __forceinline__ dtype_t* buffer_by(int rank_idx, int idx = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "`buffer` is only available for single rank case");  // 多 peer NVLink 情况才允许选择 rank_idx。
        // rank_idx 选择 peer 映射指针，idx 选择该 peer 内的 rank/slot 子区域。
        return reinterpret_cast<dtype_t*>(ptrs[rank_idx] + num_bytes * idx);  // 返回 peer rank_idx 的第 idx 个 slot。
    }
};

template <typename dtype_t, bool kDecoupled = true>
struct SymBuffer {  // 对称 buffer：同一 workspace 内按 send/recv 或共享区域对称切分。
private:
    // NOTES: for non-decoupled case, `recv_ptr` is not used
    uint8_t* send_ptr;  // send 区起点；kDecoupled=false 时也作为共享 buffer 起点。
    uint8_t* recv_ptr;  // recv 区起点；kDecoupled=false 时不使用。
    int64_t num_bytes;  // 单个 rank/slot 的字节数。

public:
    int64_t total_bytes;  // 当前 SymBuffer 在底层 workspace 中占用的总字节数。

    __device__ __forceinline__ SymBuffer(void*& gbl_ptr, int num_elems, int num_ranks, int sm_id = 0, int num_sms = 1) {
        // SymBuffer 为每个 SM/channel 分配 send/recv 两段对称区域，逻辑形状为
        // [2, num_sms, num_ranks, num_elems]。kDecoupled=false 时只保留一段共享区域。
        num_bytes = num_elems * sizeof(dtype_t);  // 单个 rank/slot 的数据区大小。

        int64_t per_channel_bytes = num_bytes * num_ranks;  // 一个 SM/channel 内按 rank 连续排布。
        total_bytes = per_channel_bytes * num_sms * (static_cast<int>(kDecoupled) + 1);  // decoupled 时 send/recv 两份，否则一份。
        // decoupled=true 逻辑形状为 send[num_sms,num_ranks] + recv[num_sms,num_ranks]。
        send_ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id;  // 当前 SM/channel 的 send 区。
        recv_ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * (sm_id + num_sms);  // 当前 SM/channel 的 recv 区，位于所有 send 区之后。
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;  // 推进线性 workspace。
    }

    __device__ __forceinline__ dtype_t* send_buffer(int idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`send_buffer` is only available for non-decoupled case");  // 只有 decoupled=true 才有独立 send 区。
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);  // 返回第 idx 个 rank/slot 的 send 指针。
    }

    __device__ __forceinline__ dtype_t* recv_buffer(int idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`recv_buffer` is only available for non-decoupled case");  // 只有 decoupled=true 才有独立 recv 区。
        return reinterpret_cast<dtype_t*>(recv_ptr + num_bytes * idx);  // 返回第 idx 个 rank/slot 的 recv 指针。
    }

    __device__ __forceinline__ dtype_t* buffer(int idx = 0) {
        EP_STATIC_ASSERT(not kDecoupled, "`buffer` is only available for decoupled case");  // kDecoupled=false 时只有一份共享区。
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);  // 返回共享区第 idx 个 rank/slot。
    }
};

}  // namespace deep_ep::legacy
