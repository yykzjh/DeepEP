#pragma once  // legacy kernel 的 host API 声明头，供 csrc/legacy/buffer.hpp 调用。

#include "compiled.cuh"  // 引入 topk_idx_t、LEGACY_NUM_MAX_NVL_PEERS 等 legacy 编译常量。

namespace deep_ep::legacy {  // legacy 预编译通信 kernel 命名空间。

// legacy kernel API 的声明层。legacy 路径分成三组：
// 1. layout：根据 topk_idx 统计 token->rank/expert 的路由矩阵；
// 2. intranode：只在 NVLink 域内做 dispatch/combine；
// 3. internode：先 RDMA 跨节点，再 NVLink 节点内分发。
// 这里的函数都由 csrc/legacy/buffer.hpp 调用，张量维度约定在对应实现文件内注释。

// Layout kernels
namespace layout {

void get_dispatch_layout(const topk_idx_t* topk_idx,  // 输入路由矩阵 [num_tokens,num_topk]，元素是全局 expert id。
                         int* num_tokens_per_rank,  // 输出每个全局 rank 需要接收的去重 token 数 [num_ranks]。
                         int* num_tokens_per_rdma_rank,  // 输出每个 RDMA/node rank 接收 token 数 [num_rdma_ranks]。
                         int* num_tokens_per_expert,  // 输出每个 expert 的 token 命中次数 [num_experts]。
                         bool* is_token_in_rank,  // 输出 token->rank 布尔矩阵 [num_tokens,num_ranks]。
                         int num_tokens,  // 当前 rank 输入 token 数 T。
                         int num_topk,  // 每个 token 选择的 expert 数 K。
                         int num_ranks,  // 全局 rank 数 R。
                         int num_experts,  // 全局 expert 数 E。
                         cudaStream_t stream);  // 执行 layout kernel 的 CUDA stream。

}  // namespace layout

// Intranode kernels
namespace intranode {

// intra-node barrier 使用每个 peer 暴露的 barrier_signal_ptrs 做 GPU 侧同步，
// rank/num_ranks 均指 NVLink 域内 rank。
void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream);  // 单节点 peer 间 GPU barrier：信号指针数组、当前 rank、rank 数、stream。

// notify_dispatch 在真正搬运 token 前先写入接收端计数器：
// num_tokens_per_rank [num_ranks]、num_tokens_per_expert [num_experts]；
// channel_prefix_matrix [num_ranks, num_sms] 用于把每个 channel 的写入区间切开。
void notify_dispatch(const int* num_tokens_per_rank,  // 输入当前 rank 发往各目标 rank 的 token 数 [num_ranks]。
                     int* moe_recv_counter_mapped,  // 输出/映射到本 rank 的接收计数器，peer 会写入。
                     int num_ranks,  // NVLink 域内 rank 数。
                     const int* num_tokens_per_expert,  // 输入当前 rank 发往各 expert 的 token 数 [num_experts]。
                     int* moe_recv_expert_counter_mapped,  // 输出/映射到本 rank 的 per-expert 接收计数器。
                     int num_experts,  // 全局 expert 数或本路径可见 expert 数。
                     int num_tokens,  // 当前 rank 输入 token 数。
                     const bool* is_token_in_rank,  // token->rank 布尔矩阵 [num_tokens,num_ranks]。
                     int* channel_prefix_matrix,  // 输出每个 rank/channel 的写入前缀 [num_ranks,num_sms]。
                     int* rank_prefix_matrix_copy,  // 输出 rank prefix 副本，cached dispatch 会复用它。
                     int num_memset_int,  // notify 前需要清零的 int 数量。
                     int expert_alignment,  // 每个 expert 接收 token 数对齐粒度。
                     void** buffer_ptrs,  // 节点内所有 peer 暴露的 NVLink buffer 指针数组。
                     int** barrier_signal_ptrs,  // 节点内 barrier 信号指针数组。
                     int rank,  // 当前 NVLink rank。
                     cudaStream_t stream,  // CUDA stream。
                     int num_sms);  // 通信 kernel 使用的 SM/channel 数。

// cached_notify_dispatch 跳过 layout 统计，复用上一次 dispatch 生成的 rank_prefix_matrix。
// cached 模式适用于 MoE routing 不变的多次前后向调用。
void cached_notify_dispatch(const int* rank_prefix_matrix,  // 上次 dispatch 生成的 rank prefix 矩阵。
                            int num_memset_int,  // 本轮需要清零的 int 数量。
                            void** buffer_ptrs,  // peer NVLink buffer 指针数组。
                            int** barrier_signal_ptrs,  // peer barrier 信号数组。
                            int rank,  // 当前 rank。
                            int num_ranks,  // NVLink 域 rank 数。
                            cudaStream_t stream);  // CUDA stream。

// intranode dispatch 把 x [num_tokens, hidden] 和可选 x_scales
// 搬到目标 rank 的 recv_x，recv_src_idx 记录原始 token id，供 combine 回写。
void dispatch(void* recv_x,  // 输出接收 token payload，形状 [num_recv_tokens,hidden] 或 worst capacity。
              float* recv_x_scales,  // 输出 FP8 scale，形状 [num_recv_tokens,num_scales]，BF16 时可为空。
              int* recv_src_idx,  // 输出每个接收 token 的原始 token id [num_recv_tokens]。
              topk_idx_t* recv_topk_idx,  // 输出接收 token 对应的本地 expert lane [num_recv_tokens,num_topk]。
              float* recv_topk_weights,  // 输出接收 token 的 top-k 权重 [num_recv_tokens,num_topk]。
              int* recv_channel_offset,  // 输出每个 channel 的接收偏移，用于 combine/cached 路径。
              int* send_head,  // send ring/channel 的 head 指针数组。
              const void* x,  // 输入 token payload [num_tokens,hidden]，BF16 或 FP8 data。
              const float* x_scales,  // 输入 FP8 scale，BF16 时为空。
              const topk_idx_t* topk_idx,  // 输入 top-k expert id [num_tokens,num_topk]。
              const float* topk_weights,  // 输入 top-k 权重 [num_tokens,num_topk]，可为空。
              const bool* is_token_in_rank,  // 输入 token->rank 布尔矩阵 [num_tokens,num_ranks]。
              const int* channel_prefix_matrix,  // notify 生成的 channel 前缀矩阵。
              int num_tokens,  // 当前 rank token 数。
              int num_worst_tokens,  // worst capacity 输出行数，0 表示使用真实接收数。
              int hidden_int4,  // hidden 按 int4 向量化后的长度。
              int num_topk,  // top-k 槽数。
              int num_experts,  // expert 总数。
              int num_scales,  // FP8 scale 组数。
              int scale_token_stride,  // scale 张量 token 维 stride。
              int scale_hidden_stride,  // scale 张量 hidden/scale 维 stride。
              void** buffer_ptrs,  // peer NVLink buffer 指针数组。
              int rank,  // 当前 rank。
              int num_ranks,  // NVLink 域 rank 数。
              cudaStream_t stream,  // CUDA stream。
              int num_sms,  // 使用 SM 数。
              int num_max_send_tokens,  // 每 channel send buffer token 容量。
              int num_recv_buffer_tokens);  // 每 channel recv buffer token 容量。

// combine 前的 cached notify 会把本 rank 本次要发送的 token 数写给 peer，并清零
// send_head 等临时队列头。
void cached_notify_combine(void** buffer_ptrs,  // peer NVLink buffer 指针数组。
                           int* send_head,  // combine 发送队列 head 数组。
                           int num_channels,  // combine channel 数。
                           int num_recv_tokens,  // 本 rank expert 输出 token 数。
                           int num_memset_int,  // 需要清零的 int 数量。
                           int** barrier_signal_ptrs,  // peer barrier 信号数组。
                           int rank,  // 当前 rank。
                           int num_ranks,  // NVLink 域 rank 数。
                           cudaStream_t stream);  // CUDA stream。

// intranode combine 根据 src_idx/rank_prefix_matrix 把专家输出归并回原 token，
// recv_x 输出形状 [num_tokens, hidden]，recv_topk_weights 可选为 [num_tokens, num_topk]。
void combine(cudaDataType_t type,  // payload 数据类型，目前 legacy combine 主要支持 BF16。
             void* recv_x,  // 输出回原 token 顺序的 combined payload [num_tokens,hidden]。
             float* recv_topk_weights,  // 输出回原 token 顺序的权重 [num_tokens,num_topk]，可为空。
             const void* x,  // 输入专家侧输出 token [num_recv_tokens,hidden]。
             const float* topk_weights,  // 输入专家侧权重 [num_recv_tokens,num_topk]，可为空。
             const void* bias_0,  // 可选 bias0 [num_tokens,hidden]，测试中用于验证归并。
             const void* bias_1,  // 可选 bias1 [num_tokens,hidden]。
             const int* src_idx,  // dispatch 记录的原始 token id [num_recv_tokens]。
             const int* rank_prefix_matrix,  // rank 级 prefix，定位每个源 rank 的 token 段。
             const int* channel_prefix_matrix,  // channel 级 prefix，定位 channel 内写回区间。
             int* send_head,  // combine send 队列 head。
             int num_tokens,  // 原始 token 数。
             int num_recv_tokens,  // 当前 rank expert 输出 token 数。
             int hidden,  // hidden 维度。
             int num_topk,  // top-k 槽数。
             void** buffer_ptrs,  // peer NVLink buffer 指针。
             int rank,  // 当前 rank。
             int num_ranks,  // NVLink 域 rank 数。
             cudaStream_t stream,  // CUDA stream。
             int num_sms,  // 使用 SM 数。
             int num_max_send_tokens,  // 每 channel send buffer token 容量。
             int num_recv_buffer_tokens);  // 每 channel recv buffer token 容量。

}  // namespace intranode

// Internode kernels
namespace internode {

// src_meta 是 internode dispatch 写给 combine 的压缩元数据结构；具体字节数由实现
// 决定，Python/C++ 只按 get_source_meta_bytes 分配。
int get_source_meta_bytes();  // 返回单条 internode source metadata 的字节数，host 用它分配 recv_src_meta。

// internode notify 同时计算 RDMA 与 NVLink 两级 prefix：
// rdma_channel_prefix_matrix 负责跨节点分块，gbl_channel_prefix_matrix 负责全局 rank
// 的最终 token 顺序，recv_*_prefix_sum 是接收端根据 peer 计数得到的前缀和。
void notify_dispatch(const int* num_tokens_per_rank,  // 输入发往每个全局 rank 的 token 数 [num_ranks]。
                     int* moe_recv_counter_mapped,  // 接收端全局 rank 计数器映射地址。
                     int num_ranks,  // 全局 rank 数。
                     const int* num_tokens_per_rdma_rank,  // 输入发往每个 RDMA/node rank 的 token 数 [num_rdma_ranks]。
                     int* moe_recv_rdma_counter_mapped,  // 接收端 RDMA rank 计数器映射地址。
                     const int* num_tokens_per_expert,  // 输入发往每个 expert 的 token 数 [num_experts]。
                     int* moe_recv_expert_counter_mapped,  // 接收端 expert 计数器映射地址。
                     int num_experts,  // 全局 expert 数。
                     const bool* is_token_in_rank,  // token->rank 布尔矩阵 [num_tokens,num_ranks]。
                     int num_tokens,  // 当前 rank token 数。
                     int num_worst_tokens,  // worst capacity 输出 token 数，0 表示真实容量。
                     int num_channels,  // RDMA/NVL channel 数，通常等于 num_sms/2。
                     int hidden_int4,  // hidden 按 int4 向量化后的长度。
                     int num_scales,  // FP8 scale 组数。
                     int num_topk,  // top-k 槽数。
                     int expert_alignment,  // expert 接收 token 对齐粒度。
                     int* rdma_channel_prefix_matrix,  // 输出 RDMA channel 前缀矩阵。
                     int* recv_rdma_rank_prefix_sum,  // 输出接收端 RDMA rank 前缀和 [num_rdma_ranks]。
                     int* gbl_channel_prefix_matrix,  // 输出全局 rank/channel 前缀矩阵。
                     int* recv_gbl_rank_prefix_sum,  // 输出接收端全局 rank 前缀和 [num_ranks]。
                     void* rdma_buffer_ptr,  // 本 rank RDMA workspace 起点。
                     int num_max_rdma_chunked_recv_tokens,  // 每 RDMA channel 接收 chunk 容量。
                     void** buffer_ptrs,  // 节点内 NVLink peer buffer 指针。
                     int num_max_nvl_chunked_recv_tokens,  // 每 NVL channel 接收 chunk 容量。
                     int** barrier_signal_ptrs,  // 节点内 barrier 信号指针。
                     int rank,  // 当前全局 rank。
                     cudaStream_t stream,  // CUDA stream。
                     int64_t num_rdma_bytes,  // RDMA workspace 总字节数，用于清零/边界保护。
                     int64_t num_nvl_bytes,  // NVL workspace 总字节数。
                     bool low_latency_mode);  // 是否与 low-latency buffer 复用/兼容。

// internode dispatch 的输出 recv_x/recv_topk_* 是专家侧看到的 token 集合；
// recv_src_meta 记录 RDMA peer、NVLink peer、原始 token/topk lane，用于 combine 反向传回。
void dispatch(void* recv_x,  // 输出专家侧接收 payload [num_recv_tokens,hidden]。
              float* recv_x_scales,  // 输出 FP8 scale [num_recv_tokens,num_scales]，BF16 时为空。
              topk_idx_t* recv_topk_idx,  // 输出本地 expert id/lane [num_recv_tokens,num_topk]。
              float* recv_topk_weights,  // 输出 top-k 权重 [num_recv_tokens,num_topk]。
              void* recv_src_meta,  // 输出 source metadata [num_recv_tokens,meta_bytes]，combine 反向使用。
              const void* x,  // 输入 token payload [num_tokens,hidden]。
              const float* x_scales,  // 输入 FP8 scale，可为空。
              const topk_idx_t* topk_idx,  // 输入 top-k expert id [num_tokens,num_topk]。
              const float* topk_weights,  // 输入 top-k 权重 [num_tokens,num_topk]，可为空。
              int* send_rdma_head,  // RDMA 发送 ring head 数组。
              int* send_nvl_head,  // NVLink 发送 ring head 数组。
              int* recv_rdma_channel_prefix_matrix,  // 接收端 RDMA channel 前缀矩阵副本。
              int* recv_gbl_channel_prefix_matrix,  // 接收端全局 channel 前缀矩阵副本。
              const int* rdma_channel_prefix_matrix,  // notify 生成的 RDMA channel 前缀。
              const int* recv_rdma_rank_prefix_sum,  // 接收端 RDMA rank 前缀和。
              const int* gbl_channel_prefix_matrix,  // notify 生成的全局 rank/channel 前缀。
              const int* recv_gbl_rank_prefix_sum,  // 接收端全局 rank 前缀和。
              const bool* is_token_in_rank,  // token->rank 布尔矩阵 [num_tokens,num_ranks]。
              int num_tokens,  // 当前 rank token 数。
              int num_worst_tokens,  // worst capacity 输出行数，0 表示真实容量。
              int hidden_int4,  // hidden / sizeof(int4) 后的向量长度。
              int num_scales,  // FP8 scale 组数。
              int num_topk,  // top-k 槽数。
              int num_experts,  // 全局 expert 数。
              int scale_token_stride,  // scale token stride。
              int scale_hidden_stride,  // scale hidden/scale stride。
              void* rdma_buffer_ptr,  // RDMA workspace 起点。
              int num_max_rdma_chunked_send_tokens,  // RDMA send chunk token 数。
              int num_max_rdma_chunked_recv_tokens,  // RDMA recv chunk token 数。
              void** buffer_ptrs,  // 节点内 NVLink peer buffer 指针。
              int num_max_nvl_chunked_send_tokens,  // NVL send chunk token 数。
              int num_max_nvl_chunked_recv_tokens,  // NVL recv chunk token 数。
              int rank,  // 当前全局 rank。
              int num_ranks,  // 全局 rank 数。
              bool is_cached_dispatch,  // 是否复用已有 prefix/handle 的 cached dispatch。
              cudaStream_t stream,  // CUDA stream。
              int num_channels,  // RDMA/NVL channel 数。
              bool low_latency_mode);  // 是否兼容 low-latency buffer 资源。

// cached_notify 为 cached dispatch/combine 复用已有 prefix 矩阵，只更新本轮 token 数、
// ring buffer head 与 barrier signal。
void cached_notify(int hidden_int4,  // hidden 向量化长度。
                   int num_scales,  // FP8 scale 组数。
                   int num_topk_idx,  // 是否存在 topk_idx metadata 或其数量。
                   int num_topk_weights,  // 是否存在 topk_weights metadata 或其数量。
                   int num_ranks,  // 全局 rank 数。
                   int num_channels,  // channel 数。
                   int num_combined_tokens,  // combine 侧要发送/归并的 token 数。
                   int* combined_rdma_head,  // combine RDMA head 数组。
                   const int* rdma_channel_prefix_matrix,  // RDMA channel 前缀矩阵。
                   const int* rdma_rank_prefix_sum,  // RDMA rank 前缀和。
                   int* combined_nvl_head,  // combine NVL head 数组。
                   void* rdma_buffer_ptr,  // RDMA workspace 起点。
                   int num_max_rdma_chunked_recv_tokens,  // RDMA recv chunk 容量。
                   void** buffer_ptrs,  // NVLink peer buffer 指针。
                   int num_max_nvl_chunked_recv_tokens,  // NVL recv chunk 容量。
                   int** barrier_signal_ptrs,  // barrier 信号指针。
                   int rank,  // 当前 rank。
                   cudaStream_t stream,  // CUDA stream。
                   int64_t num_rdma_bytes,  // RDMA workspace 字节数。
                   int64_t num_nvl_bytes,  // NVL workspace 字节数。
                   bool is_cached_dispatch,  // 当前是否为 cached dispatch 路径。
                   bool low_latency_mode);  // 是否兼容 low-latency 资源。

// internode combine 是 dispatch 的反向两级路径：先把本地专家结果写入 NVLink/RDMA
// staging buffer，再按 src_meta 归并到 combined_x [num_combined_tokens, hidden]。
void combine(cudaDataType_t type,  // payload 数据类型。
             void* combined_x,  // 输出回原 token 顺序的 payload [num_tokens,hidden]。
             float* combined_topk_weights,  // 输出回原 token 顺序的权重 [num_tokens,num_topk]，可为空。
             const bool* is_combined_token_in_rank,  // 原 token 是否来自某 rank 的布尔矩阵 [num_tokens,num_ranks]。
             const void* x,  // 输入专家侧输出 token [num_combined_tokens,hidden]。
             const float* topk_weights,  // 输入专家侧 top-k 权重，可为空。
             const void* bias_0,  // 可选 bias0 [num_tokens,hidden]。
             const void* bias_1,  // 可选 bias1 [num_tokens,hidden]。
             const int* combined_rdma_head,  // combine RDMA head 数组。
             const int* combined_nvl_head,  // combine NVL head 数组。
             const void* src_meta,  // dispatch 保存的 source metadata。
             const int* rdma_channel_prefix_matrix,  // RDMA channel 前缀矩阵。
             const int* rdma_rank_prefix_sum,  // RDMA rank 前缀和。
             const int* gbl_channel_prefix_matrix,  // 全局 rank/channel 前缀矩阵。
             int num_tokens,  // 原始 token 数。
             int num_combined_tokens,  // 专家侧 token 数。
             int hidden,  // hidden 维度。
             int num_topk,  // top-k 槽数。
             void* rdma_buffer_ptr,  // RDMA workspace 起点。
             int num_max_rdma_chunked_send_tokens,  // RDMA send chunk 容量。
             int num_max_rdma_chunked_recv_tokens,  // RDMA recv chunk 容量。
             void** buffer_ptrs,  // NVLink peer buffer 指针。
             int num_max_nvl_chunked_send_tokens,  // NVL send chunk 容量。
             int num_max_nvl_chunked_recv_tokens,  // NVL recv chunk 容量。
             int rank,  // 当前 rank。
             int num_ranks,  // 全局 rank 数。
             cudaStream_t stream,  // CUDA stream。
             int num_channels,  // channel 数。
             bool low_latency_mode);  // 是否兼容 low-latency buffer。

}  // namespace internode

// Internode low-latency kernels
namespace internode_ll {

void clean_low_latency_buffer(int* clean_0,  // 第一段待清零 int buffer，通常是 dispatch/combine 信号区。
                              int num_clean_int_0,  // 第一段需要清零的 int 数量。
                              int* clean_1,  // 第二段待清零 int buffer，可为另一组奇偶缓冲。
                              int num_clean_int_1,  // 第二段需要清零的 int 数量。
                              int rank,  // 当前全局 rank。
                              int num_ranks,  // 全局 rank 数。
                              int* mask_buffer,  // rank mask buffer，用于屏蔽故障/禁用 rank。
                              int* sync_buffer,  // 清理过程的跨 rank 同步 buffer。
                              cudaStream_t stream);  // CUDA stream。

void dispatch(void* packed_recv_x,  // 输出 packed 接收 token，形状由 layout_range 压缩描述。
              void* packed_recv_x_scales,  // 输出 FP8 scale packed buffer，BF16 时可为空。
              int* packed_recv_src_info,  // 输出源 token/rank/top-k 信息 packed 数组。
              int64_t* packed_recv_layout_range,  // 输出每个 expert/rank 的 packed layout 范围。
              int* packed_recv_count,  // 输出接收 token 计数。
              int* mask_buffer,  // rank mask buffer，masked rank 不参与通信。
              int* cumulative_local_expert_recv_stats,  // 输出本地 expert 累计接收统计。
              int64_t* dispatch_wait_recv_cost_stats,  // 输出等待接收的 profiling 统计。
              void* rdma_recv_x,  // RDMA 接收数据区。
              int* rdma_recv_count,  // RDMA 接收计数区。
              void* rdma_x,  // RDMA 发送/中转数据区。
              const void* x,  // 输入 token payload [num_tokens,hidden]。
              const topk_idx_t* topk_idx,  // 输入 top-k expert id [num_tokens,num_topk]。
              int* next_clean,  // 下一轮需要清零的信号区指针。
              int num_next_clean_int,  // 下一轮需要清零的 int 数量。
              int num_tokens,  // 当前 rank token 数。
              int hidden,  // hidden 维度。
              int num_max_dispatch_tokens_per_rank,  // 每 rank dispatch token 容量上限。
              int num_topk,  // top-k 槽数。
              int num_experts,  // expert 总数。
              int rank,  // 当前全局 rank。
              int num_ranks,  // 全局 rank 数。
              bool use_fp8,  // 是否使用 FP8 dispatch 数据格式。
              bool round_scale,  // 是否对 scale 做舍入处理。
              bool use_ue8m0,  // 是否使用 UE8M0 scale 格式。
              void* workspace,  // kernel 临时 workspace。
              int num_device_sms,  // 使用的 device SM 数。
              cudaStream_t stream,  // CUDA stream。
              int phases);  // low-latency 协议阶段数，用于分阶段发送/等待。

void combine(void* combined_x,  // 输出回原 token 顺序的 combined payload [num_combined_tokens,hidden]。
             void* rdma_recv_x,  // RDMA 接收数据区。
             int* rdma_recv_flag,  // RDMA 接收 flag 区。
             void* rdma_send_x,  // RDMA 发送数据区。
             const void* x,  // 输入专家侧输出 token。
             const topk_idx_t* topk_idx,  // 输入原始 top-k expert id [num_tokens,num_topk]。
             const float* topk_weights,  // 输入 top-k 权重 [num_tokens,num_topk]。
             const int* src_info,  // dispatch 输出的源信息 packed 数组。
             const int64_t* layout_range,  // dispatch 输出的 layout 范围数组。
             int* mask_buffer,  // rank mask buffer。
             int64_t* combine_wait_recv_cost_stats,  // 输出 combine 等待接收 profiling 统计。
             int* next_clean,  // 下一轮需要清零的信号区。
             int num_next_clean_int,  // 下一轮需要清零的 int 数量。
             int num_combined_tokens,  // combine 输出 token 数。
             int hidden,  // hidden 维度。
             int num_max_dispatch_tokens_per_rank,  // 每 rank token 容量上限。
             int num_topk,  // top-k 槽数。
             int num_experts,  // expert 总数。
             int rank,  // 当前 rank。
             int num_ranks,  // 全局 rank 数。
             bool use_logfmt,  // 是否使用 log 格式缩放/量化路径。
             void* workspace,  // kernel 临时 workspace。
             int num_device_sms,  // 使用 SM 数。
             cudaStream_t stream,  // CUDA stream。
             int phases,  // low-latency 协议阶段数。
             bool zero_copy);  // 是否使用 zero-copy combine 路径。

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* output_mask_tensor, cudaStream_t stream);  // 读取 mask buffer 到输出 tensor [num_ranks]。

void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask, cudaStream_t stream);  // 设置某个 rank 的 mask 状态。

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream);  // 清空全部 rank mask 状态。

}  // namespace internode_ll

}  // namespace deep_ep::legacy
