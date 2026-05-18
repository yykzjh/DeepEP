#pragma once  // 防止 elastic kernel 统一入口头重复包含。

#include <deep_ep/common/compiled.cuh>  // 引入公共编译期宏、dtype trait 和 host/device 断言。

// Elastic kernel 的统一入口头文件。C++ binding 层只需要 include 本文件，就能拿到
// barrier / dispatch / combine / engram / PP send-recv 的 launch_* host 函数；这些
// host 函数再按运行时形状生成 JIT CUDA 源码，并 include deep_ep/include 下的实现头。
#include "barrier.hpp"  // 声明 elastic barrier JIT launch 入口，用于跨 rank 同步。
#include "dispatch.hpp"  // 声明 dispatch JIT launch 入口，把 token 分发到目标 rank/expert。
#include "combine.hpp"  // 声明 combine JIT launch 入口，把 expert 输出按源 token 聚合回去。
#include "engram.hpp"  // 声明 engram 辅助 kernel 入口，用于 elastic 路径的元数据抓取/转换。
#include "pp_send_recv.hpp"  // 声明 pipeline-parallel send/recv 入口，用于相邻 PP stage 交换张量。
