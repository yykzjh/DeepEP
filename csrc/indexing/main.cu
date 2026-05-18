// 这个文件不是运行时入口，而是给 IDE/CUDA 索引用的“汇总翻译单元”：
// 把所有 JIT include 进来，CLion/clangd 才能解析模板实现并支持跳转。
// 真正执行时，kernel 会由 csrc/jit/compiler.hpp 按形状生成源码并单独编译。

// Utils  // 工具类 kernel 分组。
#include <deep_ep/impls/barrier.cuh>  // 引入 elastic barrier device 实现，供 IDE 建索引。

// EP  // Expert Parallel dispatch/combine kernel 分组。
#include <deep_ep/impls/dispatch.cuh>                         // 单层 dispatch 主 kernel 实现。
#include <deep_ep/impls/combine.cuh>                          // 单层 combine 主 kernel 实现。
#include <deep_ep/impls/dispatch_deterministic_prologue.cuh>   // deterministic dispatch 前置 slot 计算。
#include <deep_ep/impls/dispatch_copy_epilogue.cuh>            // dispatch 后处理拷贝/metadata kernel。
#include <deep_ep/impls/combine_reduce_epilogue.cuh>           // combine 归并后处理 kernel。
#include <deep_ep/impls/hybrid_dispatch.cuh>                   // hybrid RDMA+NVLink dispatch kernel。
#include <deep_ep/impls/hybrid_combine.cuh>                    // hybrid RDMA+NVLink combine kernel。

// Engram  // 远端 KV/embedding fetch kernel 分组。
#include <deep_ep/impls/engram_fetch.cuh>  // 引入 Engram RDMA get 与 wait kernel。

// PP  // Pipeline Parallel 点对点 send/recv kernel 分组。
#include <deep_ep/impls/pp_send_recv.cuh>  // 引入 PP 环形槽 send/recv kernel。

int main() {   // 索引用翻译单元的最小 main，避免链接器抱怨没有入口。
    return 0;  // 不执行任何逻辑；本文件只服务 IDE/clangd/CUDA 索引。
}
