#pragma once  // 防止 legacy 编译期常量头被重复包含。

#include <deep_ep/common/compiled.cuh>  // 复用公共 dtype、断言、rank switch 等基础定义。

// legacy 路径的静态上限。NVLink peer 最大 8，对应单节点 8 GPU；RDMA peer 最大 20，
// 对应最多 20 个节点/rail。workspace 固定 32 MiB，用于 prefix、barrier 和临时队列。
#define LEGACY_NUM_MAX_NVL_PEERS 8  // 节点内 NVLink peer 数上限；legacy intranode 模板按该上限展开。
#define LEGACY_NUM_MAX_RDMA_PEERS 20  // 节点间 RDMA peer 数上限；internode/low-latency buffer 按该上限预留控制面。
#define LEGACY_NUM_WORKSPACE_BYTES (32 * 1024 * 1024)  // 每个 rank 暴露给 legacy kernel 的 workspace 字节数。
#define LEGACY_NUM_MAX_LOCAL_EXPERTS 1024  // 单 rank 本地 expert 数上限，用于 expert 计数数组和 host 检查。
#define LEGACY_NUM_BUFFER_ALIGNMENT_BYTES 128  // buffer 起始地址对齐到 128B，匹配 cache line/TMA/向量化访问习惯。

// low-latency kernel 用 phase 区分双缓冲槽处于发送还是接收阶段，避免 producer
// 覆盖 consumer 尚未读取的数据。
#define LEGACY_LOW_LATENCY_SEND_PHASE 1  // 发送阶段标记，sender 写入 payload 后推进 phase。
#define LEGACY_LOW_LATENCY_RECV_PHASE 2  // 接收阶段标记，receiver 读完 payload 后把槽位交还给 sender。

#define LEGACY_FINISHED_SUM_TAG 1024  // low-latency 完成归约时使用的特殊 tag，和普通 token tag 区分。
#define LEGACY_NUM_WAIT_NANOSECONDS 500  // CPU 等待 mapped counter 时的短 sleep 间隔，避免忙等占满核心。

#define LEGACY_NUM_CPU_TIMEOUT_SECS 100  // host 侧等待 GPU/peer 信号的超时时间，单位秒。
#define LEGACY_NUM_TIMEOUT_CYCLES 200000000000ull  // device 侧超时阈值，200G cycles 约等于 100s。
