// Portions derived from NVSHMEM (https://developer.nvidia.com/nvshmem)
// 中文：以下版权段说明本文件派生自 NVSHMEM，后续实现用于 DeepEP 的 IBGDA device WQE 封装。
// Copyright (c) NVIDIA Corporation.
// Licensed under the NVSHMEM Software License Agreement (version: September 3, 2019).
// 中文：许可证行必须原样保留，表示该文件的底层 IBGDA 逻辑来源。
// See full license at: https://docs.nvidia.com/nvshmem/api/sla.html
//
// 中文：修改源码时需要保留原始许可证信息，这也是本段注释不改写为纯中文的原因。
// Modified from original source:
//  - nvshmem/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh
// 中文：本文件保留 NVSHMEM 派生声明，并在其基础上裁剪出 DeepEP 低延迟路径需要的 device 端 IBGDA helper。
#pragma once


#include <nvshmem.h>  // 使用 NVSHMEM device state、heap base、PE 等基础定义。
#include <device_host_transport/nvshmem_common_ibgda.h>  // 使用 IBGDA QP/CQ/WQE 管理结构。
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>  // 使用 device 端 NVSHMEM inline helper/全局状态。

#include <deep_ep/common/exception.cuh>  // 使用 EP_STATIC_ASSERT/EP_DEVICE_ASSERT 做协议假设检查。

#include "utils.cuh"  // 使用 st_na_relaxed、ld_na_relaxed、memory_fence 等低层 PTX helper。

namespace deep_ep::legacy {  // legacy internode kernel 内嵌的 IBGDA device helper 命名空间。
// 这个文件把 NVSHMEM/MLX5 的 IBGDA device API 缩成 DeepEP low-latency 所需的几类操作：
// - 4B inline RDMA write：用于写远端 flag/count；
// - warp 协作 RDMA put：用于发送 token payload；
// - non-fetch atomic add：用于远端计数器累加；
// - quiet/poll CQ：用于清理 buffer 或 barrier 前确认 WQE 已完成。
// 由于这里直接写 WQE/doorbell，字节序、lkey/rkey、QP producer index 和 CQ consumer index
// 都必须严格匹配 MLX5/NVSHMEM 协议；上层 internode_ll.cu 只调用这些封装，不再手写 WQE。

EP_STATIC_ASSERT(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64, "Invalid QP minimum depth");  // DeepEP 批量 post WQE 假设 QP 深度至少 64。

__device__ static __forceinline__ uint64_t HtoBE64(uint64_t x) {
    // Mellanox WQE 字段采用 big-endian，GPU 端需要手写字节序转换后再写 doorbell/WQE。
    uint64_t ret;  // 输出 big-endian 64-bit 值。
    // 这里用 prmt.b32 对 64-bit 的高/低 32-bit 半字分别做字节翻转，再交换位置。
    // 结果可直接写入 MLX5 WQE 的 big-endian 字段。
    asm("{\n\t"  // 汇编作用域开始。
        ".reg .b32 ign;\n\t"  // prmt 的无关输入寄存器。
        ".reg .b32 lo;\n\t"  // 原始低 32 位。
        ".reg .b32 hi;\n\t"  // 原始高 32 位。
        ".reg .b32 new_lo;\n\t"  // 翻转后的新低 32 位。
        ".reg .b32 new_hi;\n\t"  // 翻转后的新高 32 位。
        "mov.b64 {lo,hi}, %1;\n\t"  // 拆分 64-bit 输入。
        "prmt.b32 new_hi, lo, ign, 0x0123;\n\t"  // 低 32 位字节翻转后成为新高位。
        "prmt.b32 new_lo, hi, ign, 0x0123;\n\t"  // 高 32 位字节翻转后成为新低位。
        "mov.b64 %0, {new_lo,new_hi};\n\t"  // 重新拼成 big-endian 64-bit。
        "}"  // 汇编作用域结束。
        : "=l"(ret)
        : "l"(x));
    return ret;  // 返回字节序转换后的 64-bit 值。
}

__device__ static __forceinline__ uint32_t HtoBE32(uint32_t x) {
    // 32-bit 版本只需一次 prmt.b32，把 host/little-endian 寄存器顺序翻成 big-endian。
    uint32_t ret;  // 输出 big-endian 32-bit 值。
    asm("{\n\t"
        ".reg .b32 ign;\n\t"  // prmt 的无关输入寄存器。
        "prmt.b32 %0, %1, ign, 0x0123;\n\t"  // 选择字节 0/1/2/3 的反序，得到 big-endian 字段。
        "}"  // 汇编作用域结束。
        : "=r"(ret)
        : "r"(x));
    return ret;  // 返回字节序转换后的 32-bit 值。
}

__device__ static __forceinline__ uint16_t HtoBE16(uint16_t x) {
    // TODO: simplify PTX using 16-bit instructions
    auto a = static_cast<uint32_t>(x);  // prmt 指令按 32-bit 寄存器工作，先扩展到 uint32。
    uint32_t d;  // 暂存 permute 后的 32-bit 结果。
    // 16-bit 版本仍借助 32-bit prmt，通过 mask 只选择低 16-bit 的两个字节并翻转。
    asm volatile(
        "{\n\t"  // 汇编作用域开始。
        ".reg .b32 mask;\n\t"  // prmt 选择 mask。
        ".reg .b32 ign;\n\t"  // prmt 的无关输入寄存器。
        "mov.b32 mask, 0x4401;\n\t"  // 选择并翻转低 16-bit 两个字节。
        "mov.b32 ign, 0x0;\n\t"  // 无关输入置零。
        "prmt.b32 %0, %1, ign, mask;\n\t"  // 生成 big-endian 16-bit 位形。
        "}"  // 汇编作用域结束。
        : "=r"(d)
        : "r"(a));
    return static_cast<uint16_t>(d);  // 低 16 位即 big-endian 16-bit 结果。
}

typedef struct mlx5_wqe_ctrl_seg __attribute__((__aligned__(8))) ibgda_ctrl_seg_t;  // MLX5 控制段要求 8 字节对齐。

typedef struct {
    uint32_t add_data;  // masked fetch-add 的加数，写入 WQE atomic segment。
    uint32_t field_boundary;  // 字段边界/掩码，DeepEP 这里使用 0 表示 32-bit 加法。
    uint64_t reserved;  // 保留字段，保持 segment 大小与 MLX5 atomic segment 对齐。
} __attribute__((__packed__)) ibgda_atomic_32_masked_fa_seg_t;  // packed 避免编译器插入 padding。

__device__ static __forceinline__ nvshmemi_ibgda_device_state_t* ibgda_get_state() {
    return &nvshmemi_ibgda_device_state_d;  // 返回 NVSHMEM 放在 device constant/global 中的 IBGDA 状态。
}

__device__ static __forceinline__ nvshmemi_ibgda_device_qp_t* ibgda_get_rc(int pe, int id) {
    // 从 NVSHMEM device state 中取目标 PE 的 RC QP。id 会按当前设备初始化的 QP
    // 数取模，允许多个 channel 在不同 QP 上分摊发送。
    auto state = ibgda_get_state();  // 读取 IBGDA 全局 device state。
    const auto num_rc_per_pe = ibgda_get_state()->num_rc_per_pe;  // 每个目标 PE 可用的 RC QP 数。
    return &state->globalmem
                .rcs[pe * num_rc_per_pe * state->num_devices_initialized + id % (num_rc_per_pe * state->num_devices_initialized)];  // 按 PE 和 QP id 选择 RC QP。
}

__device__ static __forceinline__ void ibgda_lock_acquire(int* lock) {
    // 多个 CTA 可能同时向同一个 QP post WQE，使用轻量自旋锁保护 producer index
    // 和 doorbell record 更新。
    while (atomicCAS(lock, 0, 1) == 1)  // CAS 成功把 0 改 1 才拿到锁。
        ;  // 自旋等待其它 CTA 释放同一个 QP 的 post_send 锁。

    // Prevent reordering before the lock is acquired
    memory_fence_cta();  // 拿锁后做 CTA scope fence，防止临界区前后内存操作重排。
}

__device__ static __forceinline__ void ibgda_lock_release(int* lock) {
    memory_fence_cta();  // 释放锁前确保临界区内写入对同 CTA/SM 可见。

    // Prevent reordering before lock is released
    st_na_relaxed(lock, 0);  // 写 0 释放锁；no-allocate 避免污染 cache。
}

__device__ static __forceinline__ void ibgda_update_dbr(nvshmemi_ibgda_device_qp_t* qp, uint32_t dbrec_head) {
    // DBREC 保存下一个空 WQEBB 的 index，NIC 会据此知道哪些 WQE 已经准备好。
    __be32 dbrec_val;  // DB record 中实际写入的 big-endian 值。
    __be32* dbrec_ptr = qp->tx_wq.dbrec;  // NIC 可见的 doorbell record 地址。

    // This is equivalent to `WRITE_ONCE(dbrec_ptr, HtoBE32(dbrec_head & 0xffff))`
    asm("{\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"  // 只保留 DB record 协议需要的低 16 位 producer index。
        ".reg .b32 ign;\n\t"  // prmt 的无关输入寄存器。
        "and.b32 dbrec_head_16b, %1, 0xffff;\n\t"  // ring index 低 16 位写入 NIC 可见 DBREC。
        "prmt.b32 %0, dbrec_head_16b, ign, 0x123;\n\t"  // 生成 big-endian 32-bit DBREC 值。
        "}"  // 汇编作用域结束。
        : "=r"(dbrec_val)
        : "r"(dbrec_head));
    st_na_release(dbrec_ptr, dbrec_val);  // release store 发布 DBREC，保证 WQE 内容先可见。
}

__device__ static __forceinline__ void ibgda_ring_db(nvshmemi_ibgda_device_qp_t* qp, uint16_t prod_idx) {
    auto bf_ptr = reinterpret_cast<uint64_t*>(qp->tx_wq.bf);  // BlueFlame doorbell MMIO 地址。
    ibgda_ctrl_seg_t ctrl_seg = {.opmod_idx_opcode = HtoBE32(prod_idx << 8), .qpn_ds = HtoBE32(qp->qpn << 8)};  // doorbell 只需要 prod_idx 和 QPN 关键字段。

    EP_STATIC_ASSERT(sizeof(decltype(&ctrl_seg)) == sizeof(uint64_t), "");  // 保持原 NVSHMEM 的大小假设。
    st_na_release(bf_ptr, *(reinterpret_cast<uint64_t*>(&ctrl_seg)));  // 写 BlueFlame，通知 NIC 拉取新 WQE。
}

__device__ static __forceinline__ void ibgda_post_send(nvshmemi_ibgda_device_qp_t* qp, uint64_t new_prod_idx) {
    nvshmemi_ibgda_device_qp_management_t* mvars = &qp->mvars;  // QP 的 device 管理变量，包括 prod_idx 和锁。
    uint64_t old_prod_idx;  // atomicMax 返回的旧 producer index。

    // 先更新 prod_idx 再敲门铃，quiet/fence 才知道需要等待到哪个 WQE 序号。
    ibgda_lock_acquire(&mvars->post_send_lock);

    old_prod_idx = atomicMax(reinterpret_cast<unsigned long long int*>(&mvars->tx_wq.prod_idx), new_prod_idx);  // 只允许 producer index 单调增加。
    if (new_prod_idx > old_prod_idx) {  // 当前 CTA 推进了 producer index，才需要通知 NIC。
        ibgda_update_dbr(qp, new_prod_idx);  // 更新 doorbell record。
        ibgda_ring_db(qp, new_prod_idx);  // 敲 BlueFlame doorbell。
    }
    ibgda_lock_release(&mvars->post_send_lock);  // 释放 QP post_send 锁。
}

template <bool kAlwaysDoPostSend>
// 中文：submit_requests 负责把已经写好的 WQE 批量提交给 QP，并按配置决定是否敲门铃。
__device__ static __forceinline__ void ibgda_submit_requests(nvshmemi_ibgda_device_qp_t* qp,
                                                             uint64_t base_wqe_idx,
                                                             // 中文：base_wqe_idx/num_wqes 描述本次要提交的连续 WQE 区间。
                                                             uint32_t num_wqes,
                                                             int message_idx = 0) {
    auto state = ibgda_get_state();  // 读取 IBGDA runtime 开关，例如 async postsend。
    nvshmemi_ibgda_device_qp_management_t* mvars = &qp->mvars;  // 当前 QP 管理变量。
    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;  // 当前批次写完后的下一 WQE index。

    // 所有 WQE 内容必须先写完，再推进 ready_head/prod_idx；否则 NIC 可能读到半写入
    // 的 ctrl/raddr/data segment。
    __threadfence();  // WQE 写入位于 global/系统可见内存，提交前必须 fence。

    unsigned long long int* ready_idx =
        (unsigned long long int*)(state->use_async_postsend ? qp->tx_wq.prod_idx : &mvars->tx_wq.ready_head);  // async 模式直接推进 prod_idx，否则推进 ready_head。

    // Wait for prior WQE slots to be filled first
    while (atomicCAS(ready_idx, base_wqe_idx, new_wqe_idx) != base_wqe_idx)  // 保证 WQE 批次按 index 顺序提交。
        ;  // 等待前一批 WQE 填充完成。

    // use_async_postsend=false 时需要本 CTA 负责敲门铃；为了减少 doorbell 开销，
    // 默认每 4 个 message 批量 post 一次，kAlwaysDoPostSend 可强制立即 post。
    if (!state->use_async_postsend) {  // 非 async 模式由当前线程负责实际 post_send。
        constexpr int kNumRequestInBatch = 4;  // 默认每 4 条消息敲一次门铃，降低 MMIO 开销。
        if (kAlwaysDoPostSend or (message_idx + 1) % kNumRequestInBatch == 0)  // 强制提交或达到批量边界时 post。
            ibgda_post_send(qp, new_wqe_idx);  // 更新 DBREC 并 ring doorbell。
    }
}

__device__ static __forceinline__ void ibgda_write_rdma_write_inl_wqe(
    nvshmemi_ibgda_device_qp_t* qp, const uint32_t* val, uint64_t raddr, __be32 rkey, uint16_t wqe_idx, void** out_wqes, uint32_t imm) {
    ibgda_ctrl_seg_t ctrl_seg;  // WQE 控制段，包含 opcode、WQE index、QPN 等。
    struct mlx5_wqe_raddr_seg raddr_seg;  // 远端地址段，包含 raddr/rkey。
    struct mlx5_wqe_inl_data_seg inl_seg;  // inline data 段，4 字节 payload 直接写进 WQE。

    auto* ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);  // WQE 第一个 segment 是 ctrl。
    auto* raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));  // ctrl 后紧跟 remote address。
    auto* inl_seg_ptr = reinterpret_cast<mlx5_wqe_inl_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));  // raddr 后紧跟 inline header。
    auto* wqe_data_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(inl_seg_ptr) + sizeof(*inl_seg_ptr));  // inline header 后是真正 4 字节数据。

    raddr_seg.raddr = HtoBE64(raddr);  // NIC 读取 big-endian 远端虚拟地址。
    raddr_seg.rkey = rkey;  // 远端 memory key 已经是 __be32。
    raddr_seg.reserved = 0;  // 保留字段清零。

    inl_seg.byte_count = HtoBE32(4 | MLX5_INLINE_SEG);  // inline 写 4 字节，并设置 MLX5_INLINE_SEG 标志。

    // imm 为 uint32_t max 表示普通 RDMA_WRITE；否则使用 RDMA_WRITE_IMM，把 imm
    // 作为完成通知附带给远端 CQ。
    ctrl_seg = {0};  // 先清零控制段，避免保留位携带脏值。
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);  // QPN 和 DS(segment count=3) 合并写入。
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;  // 请求 CQ 更新，后续 quiet 可轮询完成。
    ctrl_seg.opmod_idx_opcode =
        HtoBE32((wqe_idx << 8) | (imm != std::numeric_limits<uint32_t>::max() ? MLX5_OPCODE_RDMA_WRITE_IMM : MLX5_OPCODE_RDMA_WRITE));  // 根据 imm 是否有效选择普通写或带立即数写。
    if (imm != std::numeric_limits<uint32_t>::max())  // uint32 max 是本文件约定的“无 imm”哨兵。
        ctrl_seg.imm = HtoBE32(imm);  // 带 imm 时把完成通知值写入控制段。

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");  // ctrl segment 必须可用 int4 一次写。
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");  // raddr segment 必须可用 int4 一次写。
    EP_STATIC_ASSERT(sizeof(*inl_seg_ptr) == 4, "sizeof(*inl_seg_ptr) == 4");  // inline header 是 4 字节。
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));  // 写 ctrl segment。
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));  // 写 remote address segment。
    st_na_relaxed(reinterpret_cast<uint32_t*>(inl_seg_ptr), *reinterpret_cast<const uint32_t*>(&inl_seg));  // 写 inline data header。
    st_na_relaxed(reinterpret_cast<uint32_t*>(wqe_data_ptr), *reinterpret_cast<const uint32_t*>(val));  // 写 4 字节 inline payload。
}

__device__ static __forceinline__ uint64_t
ibgda_get_lkey_and_rkey(uint64_t laddr, __be32* lkey, uint64_t raddr, int dst_pe, uint64_t* out_raddr, __be32* out_rkey, uint32_t dev_idx) {
    auto state = ibgda_get_state();  // 读取 IBGDA device state。
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);  // 本 PE NVSHMEM heap 起点。
    auto log2_cumem_granularity = state->log2_cumem_granularity;  // memory key 粒度的 log2。

    // Local key
    uint64_t idx = ((laddr - heap_start) >> log2_cumem_granularity) * state->num_devices_initialized + dev_idx;  // 本地地址所在 chunk 的 lkey 索引。
    auto device_key = state->constmem.lkeys[idx];  // lkey 通常放在 constmem，device 端快速读取。
    auto lchunk_size = device_key.next_addr - laddr;  // 当前 lkey 从 laddr 起最多覆盖多少字节。
    *lkey = device_key.key;  // 输出本地 key。

    // Remote key
    uint64_t roffset = raddr - heap_start;  // 远端地址相对 heap 起点的偏移。

    idx = ((roffset >> log2_cumem_granularity) * nvshmemi_device_state_d.npes) * state->num_devices_initialized +
        dst_pe * state->num_devices_initialized + dev_idx;  // 远端 rkey 索引包含 chunk、目标 PE 和 device id。
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS) {  // 小索引 rkey 放在 constmem。
        device_key = state->constmem.rkeys[idx];  // 从 constmem 读取 rkey。
    } else {
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];  // 大索引 rkey 放在 global memory。
    }
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;  // 输出 NIC 可访问的远端地址。
    *out_rkey = device_key.key;  // 输出远端 key。

    // Return the minimum of local and remote chunk sizes
    auto rchunk_size = device_key.next_addr - roffset;  // 当前 rkey 从 roffset 起覆盖的字节数。
    return min(lchunk_size, rchunk_size);  // 单个 WQE 不能跨越 local/remote 任一 key chunk。
}

__device__ static __forceinline__ void ibgda_get_rkey(uint64_t addr, int dst_pe, uint64_t* out_raddr, __be32* out_rkey, uint32_t dev_idx) {
    auto state = ibgda_get_state();  // 读取 IBGDA device state。
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);  // 本地 NVSHMEM heap 基址。

    uint64_t roffset = addr - heap_start;  // 远端地址相对 heap 的偏移。
    uint64_t idx = ((roffset >> state->log2_cumem_granularity) * nvshmemi_device_state_d.npes * state->num_devices_initialized) +
        dst_pe * state->num_devices_initialized + dev_idx;  // rkey 索引由 chunk、PE、device id 共同决定。
    nvshmemi_ibgda_device_key_t device_key;  // 保存查到的远端 key。
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS)  // 小索引在 constmem。
        device_key = state->constmem.rkeys[idx];  // 从 constmem 读取。
    else
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];  // 大索引在 globalmem。
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;  // 输出 NIC 可访问远端地址。
    *out_rkey = device_key.key;  // 输出 rkey。
}

__device__ static __forceinline__ uint64_t ibgda_reserve_wqe_slots(nvshmemi_ibgda_device_qp_t* qp, uint32_t num_wqes) {
    auto mvars = &qp->mvars;  // QP 管理变量。
    return atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.resv_head), static_cast<unsigned long long>(num_wqes));  // 原子预留 num_wqes 个 WQE slot，返回起始 index。
}

__device__ static __forceinline__ void* ibgda_get_wqe_ptr(nvshmemi_ibgda_device_qp_t* qp, uint16_t wqe_idx) {
    uint16_t cnt = qp->tx_wq.nwqes;  // send queue 中 WQE slot 数，通常是 2 的幂。
    uint16_t idx = wqe_idx & (cnt - 1);  // 环形队列取模，依赖 cnt 为 2 的幂。
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(qp->tx_wq.wqe) + (idx << MLX5_SEND_WQE_SHIFT));  // 返回第 idx 个 WQE 的字节地址。
}

__device__ static __forceinline__ void nvshmemi_ibgda_rma_p(
    int* rptr, const int value, int dst_pe, int qp_id, uint32_t imm = std::numeric_limits<uint32_t>::max()) {
    // Get rkey
    // 中文：rma_p 是小整数 put，常用于写远端 flag/counter，不跨越多个远端 chunk。
    // NOTES: the `p` operation will not cross multiple remote chunks
    __be32 rkey;  // 远端 memory key。
    uint64_t raddr;  // NIC 可访问远端地址。
    auto qp = ibgda_get_rc(dst_pe, qp_id);  // 根据目标 PE 和 QP id 选择 RC QP。
    ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), dst_pe, &raddr, &rkey, qp->dev_idx);  // 查找远端地址和 rkey。

    // Write WQEs
    uint64_t base_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);  // 单个 4B put 只需要一个 inline WQE。
    void* wqe_ptrs;  // WQE 指针容器，和 NVSHMEM 原始 helper 保持接口一致。
    wqe_ptrs = ibgda_get_wqe_ptr(qp, base_wqe_idx);  // 获取环形队列中的 WQE 地址。
    ibgda_write_rdma_write_inl_wqe(qp, reinterpret_cast<const uint32_t*>(&value), raddr, rkey, base_wqe_idx, &wqe_ptrs, imm);  // 写 inline RDMA_WRITE WQE。

    // Submit requests
    ibgda_submit_requests<true>(qp, base_wqe_idx, 1);  // 强制立即 post，确保控制字尽快发出。
}

__device__ static __forceinline__ void ibgda_write_rdma_write_wqe(nvshmemi_ibgda_device_qp_t* qp,
                                                                  uint64_t laddr,  // 本地 payload 地址，必须位于已注册/可被 NIC 读取的内存。
                                                                  __be32 lkey,  // 本地内存注册 key，big-endian 形式由 NVSHMEM 提供。
                                                                  uint64_t raddr,  // 远端 NIC 虚拟地址。
                                                                  __be32 rkey,  // 远端 memory key。
                                                                  uint32_t bytes,  // 本次 RDMA_WRITE 传输字节数。
                                                                  uint16_t wqe_idx,  // 当前 WQE 在 QP SQ 中的 producer index。
                                                                  void** out_wqes) {  // 指向待写 WQE 槽位的指针。
    ibgda_ctrl_seg_t ctrl_seg;  // WQE 控制段。
    struct mlx5_wqe_raddr_seg raddr_seg;  // 远端地址段。
    struct mlx5_wqe_data_seg data_seg;  // 本地数据段。

    auto* ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);  // WQE 起点是 ctrl segment。
    void* av_seg_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));  // RC send 不需要 AV，位置直接作为 raddr 起点。
    struct mlx5_wqe_raddr_seg* raddr_seg_ptr;  // remote address segment 指针。
    struct mlx5_wqe_data_seg* data_seg_ptr;  // data segment 指针。

    raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(av_seg_ptr));  // raddr segment 紧跟 ctrl。
    data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));  // data segment 紧跟 raddr。

    raddr_seg.raddr = HtoBE64(raddr);  // 写 big-endian 远端地址。
    raddr_seg.rkey = rkey;  // 写远端 key。
    raddr_seg.reserved = 0;  // 保留字段清零。

    data_seg.byte_count = HtoBE32(bytes);  // 本次 RDMA_WRITE 字节数。
    data_seg.lkey = lkey;  // 本地 memory key。
    data_seg.addr = HtoBE64(laddr);  // 本地源地址 big-endian。

    ctrl_seg = {0};  // 清零控制段保留位。
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);  // QPN + 3 个 segment。
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;  // 请求 CQ 更新。
    ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);  // 写 WQE index 和 RDMA_WRITE opcode。

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");  // ctrl 段必须 16B。
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");  // raddr 段必须 16B。
    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == 16, "sizeof(*data_seg_ptr) == 16");  // data 段必须 16B。
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));  // 向 WQE 写 ctrl。
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));  // 向 WQE 写 raddr。
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));  // 向 WQE 写 data。
}

__device__ static __forceinline__ void ibgda_write_empty_recv_wqe(void* out_wqe) {
    auto* data_seg_ptr = reinterpret_cast<struct mlx5_wqe_data_seg*>(out_wqe);  // empty recv WQE 直接写 data segment。
    struct mlx5_wqe_data_seg data_seg;  // 临时 data segment。

    // Make the first segment in the WQE invalid, then the entire list will be invalid
    data_seg.byte_count = 0;  // 0 字节使该 WQE 无效。
    data_seg.lkey = HtoBE64(MLX5_INVALID_LKEY);  // invalid lkey 进一步确保 NIC 不会使用它。
    data_seg.addr = 0;  // 地址清零。

    EP_STATIC_ASSERT(sizeof(mlx5_wqe_data_seg) == sizeof(int4), "Invalid data type length");  // data segment 必须 16B。
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));  // 写入无效 data segment。
}

template <bool kAlwaysDoPostSend = false>
// 中文：warp 级 put 把连续字节拆成最多 32 个 lane 分工的 WQE，再由 QP 发送到 dst_pe。
__device__ static __forceinline__ void nvshmemi_ibgda_put_nbi_warp(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int qp_id, int lane_id, int message_idx) {
    // Get lkey and rkey, store them into lanes
    uint32_t num_wqes = 0;  // 本次 put 需要拆成多少个 WQE，最多一个 warp 32 个。
    __be32 my_lkey = 0;  // 当前 lane 负责 chunk 的本地 key。
    uint64_t my_laddr = 0;  // 当前 lane 负责 chunk 的本地地址。
    __be32 my_rkey = 0;  // 当前 lane 负责 chunk 的远端 key。
    uint64_t my_raddr = 0;  // 当前 lane 负责 chunk 的远端地址。
    uint64_t my_chunk_size = 0;  // 当前 lane 负责 chunk 的字节数。

    auto qp = ibgda_get_rc(dst_pe, qp_id);  // 获取目标 PE/QP。

    // Decide how many messages (theoretically 3 for maximum)
    auto remaining_bytes = bytes;  // 尚未覆盖的字节数。
    while (remaining_bytes > 0) {  // 按 lkey/rkey chunk 边界切分消息。
        if (lane_id == num_wqes) {  // 第 num_wqes 个 lane 负责计算第 num_wqes 个 chunk。
            my_chunk_size = min(remaining_bytes,
                                ibgda_get_lkey_and_rkey(my_laddr = req_lptr, &my_lkey, req_rptr, dst_pe, &my_raddr, &my_rkey, qp->dev_idx));  // chunk 不能跨 local/remote key 边界。
        }

        // Move one more message
        auto chunk_size = __shfl_sync(0xffffffff, my_chunk_size, static_cast<int>(num_wqes));  // 广播当前 chunk 大小给所有 lane。
        remaining_bytes -= chunk_size;  // 消耗该 chunk。
        req_lptr += chunk_size;  // 本地地址推进。
        req_rptr += chunk_size;  // 远端地址推进。
        ++num_wqes;  // WQE 数加一。
    }
    EP_DEVICE_ASSERT(num_wqes <= 32);  // 一个 warp 最多让 32 个 lane 各写一个 WQE。

    // Process WQE
    uint64_t base_wqe_idx = 0;  // 当前批次预留到的起始 WQE index。
    if (lane_id == 0)  // 只有 lane0 做原子预留。
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes);  // 预留连续 num_wqes 个 slot。
    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);  // 广播起始 index。
    if (lane_id < num_wqes) {  // 有效 lane 各写一个 WQE。
        auto wqe_idx = base_wqe_idx + lane_id;  // 当前 lane 对应的 WQE index。
        auto wqe_ptr = ibgda_get_wqe_ptr(qp, wqe_idx);  // 当前 WQE 地址。
        ibgda_write_rdma_write_wqe(qp, my_laddr, my_lkey, my_raddr, my_rkey, my_chunk_size, wqe_idx, &wqe_ptr);  // 写 RDMA_WRITE WQE。
    }
    __syncwarp();  // 确保所有 lane 写完 WQE 后再提交。

    // Submit
    if (lane_id == 0)  // lane0 负责提交该批 WQE。
        ibgda_submit_requests<kAlwaysDoPostSend>(qp, base_wqe_idx, num_wqes, message_idx);  // 根据批量策略 post_send。
    __syncwarp();  // 提交后同步 warp，避免调用方过早复用临时状态。
}

__device__ static __forceinline__ void ibgda_write_amo_add_wqe(nvshmemi_ibgda_device_qp_t* qp,
                                                               const int& value,  // 要累加到远端 32-bit 计数器的值。
                                                               uint64_t laddr,  // 本地 scratch buffer 地址，满足 atomic WQE data segment 格式。
                                                               __be32 lkey,  // 本地 scratch buffer 的 lkey。
                                                               uint64_t raddr,  // 远端计数器地址。
                                                               __be32 rkey,  // 远端计数器所在内存的 rkey。
                                                               uint16_t wqe_idx,  // 当前 atomic WQE 的 producer index。
                                                               void** out_wqes) {  // 待填写的 WQE 槽位。
    ibgda_ctrl_seg_t ctrl_seg = {0};  // atomic WQE 控制段。
    struct mlx5_wqe_raddr_seg raddr_seg;  // 远端地址段。
    struct mlx5_wqe_atomic_seg atomic_seg_1;  // atomic 操作段。
    struct mlx5_wqe_data_seg data_seg;  // 本地 scratch data 段。

    auto ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);  // WQE 起点。
    auto raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));  // raddr segment 地址。
    auto atomic_seg_ptr = reinterpret_cast<mlx5_wqe_atomic_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));  // atomic segment 地址。
    auto data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(atomic_seg_ptr) + sizeof(*atomic_seg_ptr));  // data segment 地址。

    raddr_seg.raddr = HtoBE64(raddr);  // big-endian 远端地址。
    raddr_seg.rkey = rkey;  // 远端 key。
    raddr_seg.reserved = 0;  // 保留字段。

    // NOTES: `0x08000000` means `IBGDA_4_BYTE_EXT_AMO_OPMOD`
    ctrl_seg.opmod_idx_opcode = HtoBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | 0x08000000);  // opcode=WQE masked fetch-add，opmod 指定 4B。
    auto atomic_32_masked_fa_seg = reinterpret_cast<ibgda_atomic_32_masked_fa_seg_t*>(&atomic_seg_1);  // 用自定义 packed 视图填写 atomic 段。
    atomic_32_masked_fa_seg->add_data = HtoBE32(value);  // atomic add 的加数。
    atomic_32_masked_fa_seg->field_boundary = 0;  // 不设置额外边界。

    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 4);  // QPN + 4 个 segment。
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;  // 请求 CQ 更新。

    data_seg.byte_count = HtoBE32(sizeof(int));  // 本地 scratch 数据大小为 4B。
    data_seg.lkey = lkey;  // 本地 key。
    data_seg.addr = HtoBE64(laddr);  // 本地 scratch 地址。

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == sizeof(int4), "Invalid vectorization");  // ctrl 可 int4 写。
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == sizeof(int4), "Invalid vectorization");  // raddr 可 int4 写。
    EP_STATIC_ASSERT(sizeof(*atomic_seg_ptr) == sizeof(int4), "Invalid vectorization");  // atomic 可 int4 写。
    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == sizeof(int4), "Invalid vectorization");  // data 可 int4 写。
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<int4*>(&ctrl_seg));  // 写 ctrl。
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<int4*>(&raddr_seg));  // 写 raddr。
    st_na_relaxed(reinterpret_cast<int4*>(atomic_seg_ptr), *reinterpret_cast<int4*>(&atomic_seg_1));  // 写 atomic。
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<int4*>(&data_seg));  // 写 data。
}

__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(
    void* rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
    if (is_local_copy) {  // 目标就是本地可直接访问地址。
        atomicAdd(static_cast<unsigned long long*>(rptr), value);  // 使用本地 atomicAdd，跳过 RDMA WQE。
    } else {
        nvshmemi_ibgda_device_qp_t* qp = ibgda_get_rc(pe, qp_id);  // 选择远端 PE 的 QP。

        __be32 rkey;  // 远端 key。
        uint64_t raddr;  // 远端 NIC 地址。
        ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), pe, &raddr, &rkey, qp->dev_idx);  // 查找远端 key/address。

        uint64_t my_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);  // atomic add 只需要一个 WQE。
        void* wqe_ptrs = ibgda_get_wqe_ptr(qp, my_wqe_idx);  // 获取 WQE 地址。

        ibgda_write_amo_add_wqe(qp, value, reinterpret_cast<uint64_t>(qp->ibuf.buf), qp->ibuf.lkey, raddr, rkey, my_wqe_idx, &wqe_ptrs);  // 写 masked fetch-add WQE。

        ibgda_submit_requests<true>(qp, my_wqe_idx, 1);  // 立即提交 atomic WQE。
    }
}

__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    // 该 helper 是 low-latency 路径的“先本地/同节点直写，跨节点才 RDMA”分叉点。
    // ptr 必须是当前 rank NVSHMEM heap 内地址；若目标 rank 有 P2P heap 映射，则使用相同
    // heap offset 映射到 peer 地址。返回 0 明确告诉调用方需要走 IBGDA WQE。
    // Local rank, no need for mapping
    if (rank == dst_rank)  // 目标 rank 是本 rank。
        return ptr;  // 本地地址无需映射。
    auto peer_base = __ldg(reinterpret_cast<uint64_t*>(nvshmemi_device_state_d.peer_heap_base_p2p) + dst_rank);  // 读取目标 rank 的 P2P heap base。

    // RDMA connected
    if (peer_base == 0)  // peer_base=0 表示没有 NVLink/PCIe P2P 映射，只能走 RDMA。
        return 0;  // 返回 0 让调用方选择非 P2P 路径。

    // NVLink P2P is enabled
    return peer_base + (ptr - reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base));  // 用相同 heap offset 映射到 peer 地址空间。
}

// This is a simplified version of NVSHMEM's `ibgda_poll_cq`.
// Note that this implementation does not guarantee thread safety,
// so we must ensure that no other threads are concurrently using the same QP.
// 中文：DeepEP 只在已知独占 QP 的清理/quiet 场景调用它，因此省掉 NVSHMEM 通用实现中的并发保护。
__device__ static __forceinline__ void ibgda_poll_cq(nvshmemi_ibgda_device_cq_t* cq, uint64_t idx) {
    const auto cqe64 = static_cast<mlx5_cqe64*>(cq->cqe);  // CQE 环形数组起点。
    const uint32_t ncqes = cq->ncqes;  // CQE 数量，通常为 2 的幂。
    memory_fence_cta();  // 读取 CQ 前做 CTA fence，避免和之前 WQE 写入重排。
    if (*cq->cons_idx >= idx)  // 软件 consumer index 已经覆盖目标 idx。
        return;  // 无需再轮询硬件 CQ。
    // NOTES: this while loop is part of do-while below.
    // `wqe_counter` is the HW consumer index. However, we always maintain `index + 1`.
    // 中文：硬件 CQ 只给完成到哪个 WQE，软件用 index+1 语义简化“是否完成”比较。
    // To be able to compare with the index, we need to use `wqe_counter + 1`.
    // Because `wqe_counter` is `uint16_t`, it may be overflow. Still, we know for
    // 中文：这里利用 uint16_t 自然回绕比较环形 CQ 的完成进度。
    // sure that if `idx - wqe_counter - 1 < ncqes`, `wqe_counter + 1 is less than
    // idx, and thus we need to wait. We don't need to wait when `idx == wqe_counter + 1`
    // That's why we use `- 2` here to make this case overflow.
    uint16_t wqe_counter;  // 硬件 CQE 中的 WQE consumer counter。
    do {
        wqe_counter = HtoBE16(ld_na_relaxed(&cqe64->wqe_counter));  // 读取并转换 CQE counter 字节序。
    } while ((static_cast<uint16_t>(static_cast<uint16_t>(idx) - wqe_counter - static_cast<uint16_t>(2)) < ncqes));  // 等待硬件完成到 idx-1。
    *cq->cons_idx = idx;  // 更新软件 consumer index，避免下次重复轮询。

    // Prevent reordering of this function and later instructions
    memory_fence_cta();  // quiet 后的操作不能被重排到 CQ 完成之前。
}

// Wait until wqe `idx - 1` is completed.
__device__ static __forceinline__ void nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
    auto qp = ibgda_get_rc(dst_pe, qp_id);  // 获取目标 PE/QP。
    auto state = ibgda_get_state();  // 读取 async postsend 开关。
    uint64_t prod_idx = state->use_async_postsend ? ld_na_relaxed(qp->tx_wq.prod_idx) : ld_na_relaxed(&qp->mvars.tx_wq.ready_head);  // 选择已经提交或 ready 的 producer index。
    ibgda_poll_cq(qp->tx_wq.cq, prod_idx);  // 等待 prod_idx 之前的 WQE 完成。
}

}  // namespace deep_ep::legacy
