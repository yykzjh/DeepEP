#pragma once  // Elastic layout 公共头会被所有 JIT device 实现 include，防止重复定义。

#include <deep_ep/common/compiled.cuh>   // sf_pack_t 等全局类型和配置。
#include <deep_ep/common/exception.cuh>  // EP_UNIFIED_ASSERT/EP_STATIC_ASSERT。
#include <deep_ep/common/math.cuh>       // align/advance_ptr/ceil_div 等地址计算工具。
#include <deep_ep/common/ptx.cuh>        // TMA 对齐常量和 mbarrier 类型。

namespace deep_ep::elastic::layout {  // elastic 运行时所有 workspace/buffer/token 地址计算工具。
// Elastic/V2 的 workspace/token/buffer 布局工具命名空间。

struct WorkspaceLayout {  // 将一整块 symmetric workspace 切成固定控制面区域。
    // WorkspaceLayout 把一整块 workspace 指针解释成固定控制面区域。
    void* workspace;  // workspace 原始基址。
    // workspace 基址，来自 ElasticBuffer 的 symmetric window 切片。

    int num_ranks;  // 逻辑 rank 总数。
    // 当前逻辑总 rank 数 = scaleout * scaleup。
    int num_scaleout_ranks, num_scaleup_ranks;  // 跨节点维度和节点内维度。
    // 逻辑二维拓扑维度。
    int num_experts, num_experts_per_rank;  // expert 总数和每 rank expert 数。
    // 全局 expert 数和每 rank expert 数。

    // We want to fix the layout position for all settings,
    // so that one buffer can be reused for all cases
    // WorkspaceLayout 是 Elastic/V2 的“控制面内存地图”。为了让同一个 symmetric window
    // 可以被 dispatch/combine/barrier/PP/AGRS 复用，这里按最大 rank/expert/channel 容量
    // 固定每个区域的偏移，而不是按实际运行参数动态排布。这样 JIT kernel 只需要知道基址
    // 和当前 num_ranks/num_experts，就能在所有配置下定位同一类元数据。
    static constexpr int kNumMaxRanks = 1024;  // workspace 中 rank 相关数组的固定最大长度。
    // workspace 固定预留的最大 rank 数。
    static constexpr int kNumMaxExperts = 2048;  // workspace 中 expert 相关数组的固定最大长度。
    // workspace 固定预留的最大 expert 数。
    static constexpr int kNumMaxExpertsPerRank = 256;  // 单 rank expert 数固定上限。
    // 每 rank 最大 expert 数。
    static constexpr int kNumMaxInflightAGRS = 32;  // AGRS pipeline 同时在途 slot 上限。
    // AGRS 同时在途 session/slot 上限。

    static constexpr int64_t kNumBarrierSignalBytes = 16;  // barrier counter/signal 区固定 16B。
    // NVLink barrier counter/signal 固定占 16B，满足 128-bit/LDG.256 对齐设计。

    __forceinline__ __device__ __host__
    WorkspaceLayout(void* workspace,  // symmetric workspace 基址。
                    const int& num_scaleout_ranks,  // scaleout 逻辑 rank 数。
                    const int& num_scaleup_ranks,  // 每个 scaleout 域内的 scaleup rank 数。
                    const int& num_experts):  // 全局 expert 总数。
        workspace(workspace),  // 初始化 workspace 基址。
        // 保存 workspace 基址。
        num_ranks(num_scaleout_ranks * num_scaleup_ranks),  // 二维拓扑展平成一维总 rank。
        // 计算逻辑总 rank 数。
        num_scaleout_ranks(num_scaleout_ranks),  // 保存跨节点维度。
        // 保存 scaleout 维度。
        num_scaleup_ranks(num_scaleup_ranks),  // 保存节点内维度。
        // 保存 scaleup 维度。
        num_experts(num_experts) {  // 保存 expert 总数并进入构造器校验。
        // 保存全局 expert 数。
        num_experts_per_rank = num_experts / num_ranks;
        // 计算每 rank expert 数，后续 expert count 指针会用它做 stride。
        EP_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        // expert 必须能均匀分配到 rank。
        EP_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        // 当前 rank 数不能超过固定 workspace 容量。
        EP_UNIFIED_ASSERT(num_experts <= kNumMaxExperts);
        // 当前 expert 数不能超过固定 workspace 容量。
        EP_UNIFIED_ASSERT(num_experts_per_rank <= kNumMaxExpertsPerRank);
        // 每 rank expert 数也不能超过固定容量。
    }

    static int64_t get_num_bytes() {  // 计算 WorkspaceLayout 固定控制区总字节数。
        // 返回 workspace 固定控制区大小。注意这里不包含真正的 token payload buffer；
        // csrc/elastic/buffer.hpp 会把 symmetric window 切成
        // [WorkspaceLayout::get_num_bytes() | num_buffer_bytes] 两段。
        // Pure NVLink scaleup barrier signals
        int64_t num_bytes = 0;  // 从 0 开始顺序累计各段偏移。
        // 从 workspace 起点开始累计各区域大小。
        num_bytes += kNumBarrierSignalBytes;
        // NVLink barrier counter/signal 区。

        // Notify reduction workspace
        num_bytes += (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t);
        // notify 规约临时区，rank 计数和 expert 计数共用。

        // Scaleup notify threads
        // Rank send/recv count
        num_bytes += kNumMaxRanks * sizeof(int64_t) * 2;
        // scaleup rank send/recv 计数，int64_t 便于 prefix/累加。
        // Expert send/recv count
        num_bytes += kNumMaxExperts * sizeof(int64_t) * 2;
        // scaleup expert send/recv 计数。

        // Scaleup atomic sender count
        num_bytes += kNumMaxRanks * sizeof(int);
        // scaleup 发送方 atomic counter。

        // Scaleout notify threads
        // Rank send/recv count
        num_bytes += kNumMaxRanks * sizeof(int) * 2;
        // scaleout rank send/recv 计数，int 足够。
        // Expert send/recv count
        num_bytes += kNumMaxExperts * sizeof(int) * 2;
        // scaleout expert send/recv 计数。

        // Scaleout channel metadata (finish flag and tails)
        num_bytes += kNumMaxRanks * kNumMaxChannels * sizeof(int64_t);
        // 每个 channel/rank 的 signaled tail 或完成标记。

        // Channel aggregated into the scaleup domains
        // Also reused for channel scaleup tail
        num_bytes += kNumMaxRanks * kNumMaxChannels * sizeof(int);
        // scaleup 域 channel tail，按 max ranks/channels 固定容量。

        // Rank send/recv count, for PP prev/next ranks
        num_bytes += 2 * 2 * sizeof(int64_t);
        // PP send/recv 只需要两个方向槽，每个方向 send/recv 各一个计数。

        // AGRS signals
        num_bytes += (kNumMaxInflightAGRS + 1) * kNumMaxRanks * sizeof(int);
        // AGRS recv signal 和 session signal。

        // Ensure LDG.256 work
        return math::align<int64_t>(num_bytes, 32);
        // 整体按 32B 对齐，保证后续 256-bit load/store 安全高效。
    }

    __forceinline__ __device__ __host__ unsigned long long* get_nvl_barrier_counter_ptr() const {  // 返回 NVLink barrier counter 地址。
        // workspace 起点存 NVLink barrier counter。
        return static_cast<unsigned long long*>(workspace);
    }

    __forceinline__ __device__ __host__ int* get_nvl_barrier_signal_ptr(const int& phase) const {  // 返回指定 phase 的 barrier signal 地址。
        // phase 0/1 使用不同 signal 槽，避免相邻 barrier 轮次信号混淆。
        return math::advance_ptr<int>(workspace, (2 + phase) * sizeof(int));
    }

    __forceinline__ __device__ __host__ int64_t* get_notify_reduction_workspace_ptr() const {  // 返回 notify 规约临时区地址。
        // notify reduction 临时区紧跟 barrier signal 区。
        return math::advance_ptr<int64_t>(workspace, kNumBarrierSignalBytes);
    }

    template <bool kIsSendBuffer>  // true 取 send count 区，false 取 recv count 区。
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_expert_count_ptr() const {  // 返回 scaleup rank+expert 计数区。
        // scaleup rank/expert send/recv 计数区位于 notify reduction workspace 之后。
        const auto base_ptr =  // scaleup 计数区起点。
            math::advance_ptr<int64_t>(get_notify_reduction_workspace_ptr(), (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
        // base_ptr 指向 scaleup send count 区起点。
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
        // send 区在前，recv 区紧随其后。
    }

    template <bool kIsSendBuffer>  // true/false 区分发送和接收计数。
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_count_ptr() const {  // 返回 scaleup rank count 起点。
        // scaleup rank count 位于 rank_expert_count 区前 num_scaleup_ranks 个槽。
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>();
    }

    template <bool kIsSendBuffer>  // true/false 区分发送和接收计数。
    __forceinline__ __device__ __host__ int64_t* get_scaleup_expert_count_ptr() const {  // 返回 scaleup expert count 起点。
        // scaleup expert count 紧跟 scaleup rank count 之后。
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>() + num_scaleup_ranks;
    }

    __forceinline__ __device__ __host__ int* get_scaleup_atomic_sender_counter() const {  // 返回 scaleup 发送方 atomic counter 区。
        // scaleup atomic sender counter 位于两组 int64_t scaleup count 之后。
        return math::advance_ptr<int>(
            get_scaleup_rank_expert_count_ptr<true>(), 2 * (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
    }

    template <bool kIsSendBuffer>  // true 取 scaleout send count，false 取 recv count。
    __forceinline__ __device__ __host__ int* get_scaleout_rank_expert_count_ptr() const {  // 返回 scaleout rank+expert 计数区。
        // scaleout rank/expert count 位于 scaleup atomic sender counter 之后。
        const auto base_ptr =  // scaleout 计数区起点。
            math::advance_ptr<int>(get_scaleup_atomic_sender_counter(), kNumMaxRanks * sizeof(int));
        // base_ptr 指向 scaleout send count 区起点。
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
        // send/recv 各占 kNumMaxRanks+kNumMaxExperts 个 int。
    }

    template <bool kIsSendBuffer>  // true/false 区分 send/recv。
    __forceinline__ __device__ __host__ int* get_scaleout_rank_count_ptr(
        const int& scaleout_rank_idx = 0, const int& scaleup_rank_idx = 0) const {  // 返回指定二维 rank 的 scaleout rank count。
        // scaleout rank count 按二维坐标 [scaleout_rank, scaleup_rank] 展平。
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>();  // scaleout rank/expert count 区起点。
        // scaleout rank count 起点。
        return base_ptr + scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx;
        // 返回指定 scaleout/scaleup rank 的计数槽。
    }

    template <bool kIsSendBuffer>  // true/false 区分 send/recv。
    __forceinline__ __device__ __host__ int* get_scaleout_expert_count_ptr(
        const int& scaleout_rank_idx = 0, const int& expert_idx = 0) const {  // 返回指定 scaleout 域和 expert 的计数槽。
        // scaleout expert count 紧跟 scaleout rank count 后，按 [scaleout_rank, local_expert] 展平。
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>() + num_ranks;  // 跳过展平 rank count。
        // 跳过 rank count 区。
        return base_ptr + scaleout_rank_idx * (num_scaleup_ranks * num_experts_per_rank) + expert_idx;
        // expert_idx 是 scaleout 域内 local expert 编号。
    }

    __forceinline__ __device__ __host__ int64_t* get_scaleout_channel_signaled_tail_ptr(
        const int& channel_idx, const int& scaleout_rank_idx) const {  // 返回 [channel,scaleout_rank] 的 signaled tail。
        // 每个 scaleout channel/rank 保存 signaled tail，用于 hybrid channel 推进。
        const auto base_ptr = math::advance_ptr<int64_t>(  // scaleout channel tail 区起点。
            get_scaleout_rank_expert_count_ptr<true>(),
            (kNumMaxRanks + kNumMaxExperts) * sizeof(int) * 2);
        // 跳过 scaleout send/recv count 区。
        return base_ptr + (channel_idx * num_scaleout_ranks + scaleout_rank_idx);
        // 按 [channel, scaleout_rank] 展平。
    }

    __forceinline__ __device__ __host__ int* get_channel_scaleup_tail_ptr(
        const int& channel_idx, const int& scaleup_rank_idx) const {  // 返回 [channel,scaleup_rank] 的 scaleup tail。
        // channel scaleup tail 位于 scaleout signaled tail 区之后。
        const auto base_ptr = math::advance_ptr<int>(  // channel scaleup tail 区起点。
            get_scaleout_channel_signaled_tail_ptr(0, 0),
            kNumMaxRanks * kNumMaxChannels * sizeof(int64_t));
        // 跳过所有 channel/rank 的 int64_t tail。
        return base_ptr + (channel_idx * num_scaleup_ranks + scaleup_rank_idx);
        // 按 [channel, scaleup_rank] 展平。
    }

    __forceinline__ __device__ __host__ int64_t* get_pp_send_count_ptr(const int& offset) const {  // 返回 PP 指定方向 send count。
        // PP send count 区位于 channel scaleup tail 后，offset 为方向槽 0/1。
        const auto base_ptr = math::advance_ptr<int64_t>(  // PP send count 区起点。
            get_channel_scaleup_tail_ptr(0, 0),
            kNumMaxRanks * kNumMaxChannels * sizeof(int));
        // 跳过 channel scaleup tail 区。
        return base_ptr + offset;
        // 返回指定方向的 send count。
    }

    __forceinline__ __device__ __host__ int64_t* get_pp_recv_count_ptr(const int& offset) const {  // 返回 PP 指定方向 recv count。
        // PP recv count 紧跟两个 PP send count 后。
        const auto base_ptr = math::advance_ptr<int64_t>(  // PP recv count 区起点。
            get_pp_send_count_ptr(0), 2 * sizeof(int64_t));
        // 跳过两个方向的 send count。
        return base_ptr + offset;
        // 返回指定方向的 recv count。
    }

    __forceinline__ __device__ __host__ int* get_agrs_recv_signal_ptr(const int& slot, const int& rank_idx) const {  // 返回 AGRS [slot,rank] recv signal。
        // AGRS recv signal 区位于 PP recv count 后，按 [slot, rank] 展平。
        const auto base_ptr = math::advance_ptr<int>(  // AGRS recv signal 区起点。
            get_pp_recv_count_ptr(0), 2 * sizeof(int64_t));
        // 跳过两个方向的 recv count。
        return base_ptr + slot * kNumMaxRanks + rank_idx;
        // 返回指定 inflight slot/rank 的 signal 地址。
    }

    __forceinline__ __device__ __host__ int* get_agrs_session_signal_ptr(const int& rank_idx) const {  // 返回 AGRS session signal。
        // AGRS session signal 位于所有 recv signal 后，用于整体 session 同步。
        const auto base_ptr = math::advance_ptr<int>(  // AGRS session signal 区起点。
            get_agrs_recv_signal_ptr(0, 0), kNumMaxInflightAGRS * kNumMaxRanks * sizeof(int));
        // 跳过 kNumMaxInflightAGRS 个 recv slot。
        return base_ptr + rank_idx;
        // 返回指定 rank 的 session signal。
    }
};

struct TokenLayout {  // 描述单个 token slot 的内部字段偏移。
    // TokenLayout 描述单个 token slot 在通信 buffer 中的内部结构：
    // [hidden payload | scale factor | metadata | optional mbarrier]
    // hidden payload: num_hidden_bytes，通常 hidden * sizeof(BF16/FP8)；
    // scale factor: FP8 dispatch 时存在，维度对应 Python 的 [num_tokens, hidden/128]；
    // metadata: top-k idx/weight/source 信息，dispatch_copy_epilogue/combine 会读取。
    int num_hidden_bytes, num_sf_bytes;  // hidden payload 和 scale-factor 字节数。
    // hidden payload 字节数和 scale factor 字节数。
    // 两者都以字节为单位，避免模板里反复依赖 dtype 大小。
    // NOTES: the top-k index is always 32-bit
    bool with_metadata;  // 是否附带 dispatch/combine 需要的 source metadata。
    // 是否包含 source metadata；dispatch 需要 true，combine 通常 false。
    // with_metadata=false 时仍保留 topk_idx/topk_weight 区，但不包含 source/linked-list 扩展字段。
    int num_topk, num_metadata_bytes;  // top-k 路数和预计算 metadata 总字节数。
    // top-k 数和 metadata 总字节数。
    // num_metadata_bytes 是构造时预计算结果，避免 device 端多次重复算复杂表达式。
    void* base;  // 当前 token slot 基址。
    // 当前 token slot 的基址。

    __forceinline__ __device__ __host__
    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,  // hidden 与 scale factor 字节数。
                const int& num_topk, const bool& with_metadata, void* base = nullptr) :  // top-k、metadata 开关和可选基址。
        num_hidden_bytes(num_hidden_bytes),  // 保存 hidden 字节数。
        // 保存 hidden payload 字节数。
        num_sf_bytes(num_sf_bytes),  // 保存 scale-factor 字节数。
        // 保存 scale factor 字节数。
        // Metadata includes: top-k indices, weight and source rank/token index
        with_metadata(with_metadata),  // 保存 metadata 开关。
        // 保存 metadata 开关。
        num_topk(num_topk),  // 保存 top-k 路数。
        // 保存 top-k 数。
        num_metadata_bytes(num_topk * (sizeof(int) + sizeof(float)) +
                           (with_metadata ? (1 + num_topk) * sizeof(int) : 0)),  // 计算完整 metadata 字节数。
        // metadata = topk_idx[int] + topk_weight[float] + 可选 src token 与 linked list。
        // 可选部分中 1 个 int 是 src_token_global_idx，num_topk 个 int 是 linked-list slot。
        base(base) {  // 保存 token 基址并校验布局约束。
        // 保存 token 基址。
        EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");
        // metadata 布局假设 int/float 都是 4B。
        EP_UNIFIED_ASSERT(num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);
        // hidden payload 必须满足 TMA 对齐。
        // scale/metadata 即使不是天然对齐，也会在 get_num_bytes 中向上 align 到 TMA 粒度。
    }

    template <bool kWithMBarrier, typename dtype_t = int>  // 是否包含 mbarrier，以及返回整数类型。
    __forceinline__ __device__ __host__ dtype_t get_num_bytes() const {  // 返回单 token slot 字节数。
        // 计算一个 token slot 的总字节数，各段都按 TMA 对齐。
        const auto num_bytes = math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes) +  // hidden 段按 TMA 对齐。
                               math::align(num_sf_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes) +
                               math::align<int>(kWithMBarrier ? sizeof(ptx::mbarrier) : 0, ptx::kNumTMAAlignBytes);
        // 可选 mbarrier 放在 token slot 末尾，dispatch/forward staging 会用到。
        // 对齐后的 slot 大小是后续 BufferLayout 的 token stride，必须 host/device 计算一致。
        return static_cast<dtype_t>(num_bytes);
        // dtype_t 允许 host 侧用 int64_t 防止大 buffer 溢出。
    }

    __forceinline__ __device__ __host__ void* get_base_ptr() const {  // 获取当前 token slot 基址。
        // 返回 token slot 基址。
        return base;
    }

    __forceinline__ __device__ __host__ void set_base_ptr(void* ptr) {  // 更新当前 token slot 基址。
        // 修改 token slot 基址，便于复用同一个 TokenLayout 描述不同 token。
        // 只改 base，不改 hidden/sf/metadata 字节数，因此所有字段偏移保持不变。
        base = ptr;
    }

    __forceinline__ __device__ __host__ void* get_hidden_ptr() const {  // 获取 hidden payload 起点。
        // hidden payload 位于 token slot 开头。
        return get_base_ptr();
    }

    __forceinline__ __device__ __host__ sf_pack_t* get_sf_ptr() const {  // 获取 scale-factor 起点。
        // scale factor 区紧跟 hidden payload 对齐后的末尾。
        // 例：hidden payload 为 4096B 且 TMA 对齐为 16B 时，scale 区从 base+4096 开始；
        // 若 hidden 不是对齐倍数，则先向上补齐，保证后续 TMA load/store 地址合法。
        return math::advance_ptr<sf_pack_t>(base, math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__ int* get_metadata_ptr() const {  // 获取 metadata 起点。
        // metadata 区紧跟 scale factor 区对齐后的末尾。
        // metadata 统一按 int* 暴露，因为 topk_idx、source index、linked-list index 都是 32-bit。
        return math::advance_ptr<int>(get_sf_ptr(), math::align(num_sf_bytes, ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__ int* get_topk_idx_ptr() const {  // 获取 topk_idx 数组起点。
        // metadata 第一段是 topk_idx[num_topk]，每个元素 32-bit。
        // dispatch epilogue 会把接收 token 的 expert id 写到这里；combine layout 通常不读取这一段。
        return get_metadata_ptr();
    }

    __forceinline__ __device__ __host__ float* get_topk_weights_ptr() const {  // 获取 topk_weights 数组起点。
        // topk_weights 紧跟 topk_idx，长度 num_topk。
        // 与 topk_idx 共享 metadata 区，偏移为 num_topk*sizeof(int)，因此要求 sizeof(int)==sizeof(float)。
        return math::advance_ptr<float>(get_metadata_ptr(), num_topk * sizeof(int));
    }

    __forceinline__ __device__ __host__ int* get_src_token_global_idx_ptr() const {  // 获取源 token 全局索引字段。
        // source token global index 紧跟 topk_weights，仅 with_metadata=true 时有意义。
        // 这个字段记录“该接收 token 原本来自哪个全局 token”，combine 需要依赖它把结果写回原位置。
        return math::advance_ptr<int>(get_topk_weights_ptr(), num_topk * sizeof(float));
    }

    __forceinline__ __device__ __host__ int* get_linked_list_idx_ptr() const {  // 获取 linked-list index 数组。
        // linked list index 紧跟 source token index，长度通常为 num_topk。
        // hybrid dispatch 使用它把同一 channel 内的 token 串成链，后续 forward/combine 可按链遍历。
        // 每一路 top-k 一个 linked-list 槽，值通常是下一个同 expert/token 链节点的索引或 -1。
        return get_src_token_global_idx_ptr() + 1;
    }

    __forceinline__ __device__ ptx::mbarrier* get_mbarrier_ptr() const {  // 获取 token slot 末尾 mbarrier。
        // mbarrier 位于 metadata 对齐后的末尾，只在 kWithMBarrier layout 中有效。
        // 带 mbarrier 的 token slot 常用于 producer/consumer TMA pipeline：payload 写完后通过 barrier 通知 reader。
        return math::advance_ptr<ptx::mbarrier>(get_metadata_ptr(), math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes));
    }
};

template <bool kWithMBarrier>  // 是否在每个 token slot 末尾包含 mbarrier。
struct BufferLayout {  // 描述一整块按 rank/token 二维组织的通信 buffer。
    // BufferLayout 描述一组 rank/slot 组成的通信 buffer。
    TokenLayout token_layout;  // 单 token 内部布局。
    // 单个 token slot 的内部布局。
    int num_ranks;  // 第一维槽位数。
    // 第一维槽位数，可表示 rank/topk/channel 等逻辑。
    // 命名为 num_ranks 是历史原因，hybrid_combine 中也可能把它当 top-k rank_buffer 维使用。
    int num_max_tokens_per_rank;  // 每个第一维槽位的 token 容量。
    // 每个第一维槽位可容纳的最大 token 数。
    // 逻辑张量形状可理解为 [num_ranks, num_max_tokens_per_rank, token_layout]。

    void* base;  // buffer 基址。
    // buffer 基址。

    __forceinline__ __device__ __host__
    BufferLayout(const TokenLayout& token_layout,  // 单 token layout 描述。
                 const int& num_ranks,  // 第一维槽位数。
                 const int& max_num_tokens_per_rank,  // 每槽最大 token 数。
                 void* base = nullptr) :  // 可选 buffer 基址。
        token_layout(token_layout),  // 保存 token layout。
        // 保存 token layout。
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),  // 保存二维 buffer 形状。
        // 保存第一维和 token 容量。
        // max_num_tokens_per_rank 决定 rank stride，后续 channel 子 layout 也刻意保持该 stride。
        base(base) {}  // 保存 buffer 基址。
        // 保存 buffer 基址。

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_token() const {  // 返回 token stride。
        // 返回一个 token slot 的字节数。
        // 该值是 BufferLayout 的最小 stride；所有 token_idx 都按这个 stride 定位。
        return token_layout.get_num_bytes<kWithMBarrier, int64_t>();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_rank() const {  // 返回第一维槽位 stride。
        // 返回一个第一维槽位/rank 的总字节数。
        // 逻辑形状为 [num_ranks, num_max_tokens_per_rank, token_bytes]，
        // 因此 rank stride = num_max_tokens_per_rank * token_bytes。
        return num_max_tokens_per_rank * get_num_bytes_per_token();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes() const {  // 返回整个二维 buffer 字节数。
        // 返回整个 buffer 总字节数。
        // 总大小 = 第一维槽位数 * rank stride；host 侧用它串接 dispatch/combine buffer。
        return get_num_bytes_per_rank() * num_ranks;
    }

    __forceinline__ __device__ __host__
    void* get_buffer_end_ptr() const {  // 返回 buffer 末尾地址。
        // 返回 buffer 末尾指针，用于在同一大 buffer 内串接多个 layout。
        return math::advance_ptr(base, get_num_bytes());
    }

    __forceinline__ __device__ __host__
    BufferLayout get_rank_buffer(const int& rank_idx) const {  // 切出指定第一维槽位的子 layout。
        // 返回第 rank_idx 个第一维槽位的子 layout。
        // 例：dispatch send buffer 的第一维是目标 rank，get_rank_buffer(3) 就返回目标 rank=3 的 token 队列。
        // 子 layout 的 base = 原 base + rank_idx * rank_stride。
        return BufferLayout(token_layout,
                            1, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_rank() * rank_idx);
        // 子 layout 的 num_ranks=1，token 容量保持不变。
    }

    template <int kNumTokensPerChannel>  // 每个 channel 固定切出的 token 数。
    __forceinline__ __device__ __host__
    BufferLayout get_channel_buffer(const int& channel_idx) const {  // 切出指定 channel 的起始 token 视图。
        // 按 channel 切分 token 维，每个 channel 负责 kNumTokensPerChannel 个 token stride。
        // 注意这里不是创建紧凑子数组，而是移动起始 token offset；rank stride 仍保持原始容量，
        // 这样不同 channel 下 `rank_idx` 的地址公式一致，避免跨 channel/rank 组合时算错地址。
        // 例：channel_idx=2 且每 channel 16 token，则 base 移动 32 个 token stride。
        EP_UNIFIED_ASSERT(num_max_tokens_per_rank % kNumTokensPerChannel == 0);
        // token 容量必须能被 channel token 数整除。
        return BufferLayout(token_layout,
                            // Do not use `num_max_tokens_per_rank / kNumTokensPerChannel` as the false stride
                            // 中文：这里保留原 rank stride，只移动 channel 起点，确保 rank 维地址不被压缩。
                            num_ranks, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_token() * kNumTokensPerChannel * channel_idx);
        // 保持原 stride 不变，只移动起点，避免不同 channel 的 rank stride 不一致。
    }

    __forceinline__ __device__ __host__
    TokenLayout get_token_buffer(const int& token_idx, const bool& global = false) const {  // 获取指定 token slot layout。
        // 返回当前 layout 下第 token_idx 个 token slot。
        // 当 num_ranks==1 时 token_idx 是 rank 内局部 token；global=true 时调用方承诺 token_idx
        // 已经包含 rank/channel 偏移，可在多 rank layout 上直接按全局 token stride 定位。
        // 常见安全用法是先 get_rank_buffer(rank).get_token_buffer(local_token)。
        EP_UNIFIED_ASSERT(num_ranks == 1 or global);
        // 多 rank layout 下默认不允许直接按 token 取，除非调用方确认 token_idx 是 global token offset。
        return TokenLayout(token_layout.num_hidden_bytes, token_layout.num_sf_bytes, token_layout.num_topk, token_layout.with_metadata,
                           static_cast<int8_t*>(base) + token_layout.get_num_bytes<kWithMBarrier, int64_t>() * token_idx);
        // 构造同样内部布局但基址移动到目标 token。
    }
};

}  // namespace deep_ep::elastic
// Elastic layout 命名空间结束。
