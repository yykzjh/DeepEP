#include <cuda.h>              // 使用 CUDA driver API 的 CUstreamBatchMemOpParams 等类型。
#include <pybind11/pytypes.h>  // 该编译单元经由 pybind11 扩展链接，保持和其他 backend 文件一致的类型可见性。
#include <pybind11/stl.h>      // 允许 std::vector 等类型在同一 backend 头声明中可用。

#include <deep_ep/common/exception.cuh>  // CUDA_DRIVER_CHECK 宏会把 driver 返回值转成可读异常。

#include "api.cuh"                     // 声明 batched_write/batched_wait 等对外接口。
#include "../../utils/lazy_driver.hpp"  // lazy_cuStreamBatchMemOp 延迟解析 driver symbol，避免导入时强依赖特定 CUDA 版本。

namespace deep_ep::cuda_driver {  // CUDA driver 辅助函数命名空间。

static CUstreamBatchMemOpParams create_mem_op(
    void *ptr, const int& value,
    // 中文：type 决定这是 wait 还是 write，wait_flag 控制等待条件，默认等于 value。
    const CUstreamBatchMemOpType& type,
    const CUstreamWaitValue_flags& wait_flag = CU_STREAM_WAIT_VALUE_EQ) {
    // CUDA driver 的 stream batch mem op 可以把“写控制字/等待控制字”放入 GPU
    // stream 中执行，不需要 CPU 轮询。这里统一构造 32-bit 写或等待操作。
    CUstreamBatchMemOpParams params;  // driver API 要求传入 union-like 参数结构，按 operation 填不同字段。
    if (type == CU_STREAM_MEM_OP_WRITE_VALUE_32) {
        params.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;          // 表示本 op 是写 32-bit 控制字。
        params.writeValue.address = reinterpret_cast<CUdeviceptr>(ptr);  // flag 地址必须转成 CUdeviceptr。
        params.writeValue.value = value;                             // 写入的控制值，例如 handshake 轮次号。
        params.writeValue.flags = 0;                                 // 写操作通常不需要额外 flag。
    } else {
        params.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;           // 表示本 op 是等待 32-bit 控制字。
        params.waitValue.address = reinterpret_cast<CUdeviceptr>(ptr);   // 要等待的 flag 地址。
        params.waitValue.value = value;                              // 目标值；调用方可要求等于或大于等于。
        params.waitValue.flags = wait_flag;                          // 默认等于；DeepEP 同步轮次常用 GEQ 防止错过更大的值。
    }
    return params;  // 返回可直接放入 cuStreamBatchMemOp 数组的参数。
}

void batched_write(CUstream stream, const std::vector<void*>& ptrs, const int& value) {
    // 在同一个 stream 上批量写多个控制地址，常用于初始化跨 rank handshake flag。
    std::vector<CUstreamBatchMemOpParams> ops(ptrs.size());  // 每个 flag 地址对应一个 stream mem op。
    for (int i = 0; i < ptrs.size(); ++ i)                   // ptrs 通常是多个 rank/通道的状态字地址。
        ops[i] = create_mem_op(ptrs[i], value, CU_STREAM_MEM_OP_WRITE_VALUE_32);  // 构造写 value 的 op。
    CUDA_DRIVER_CHECK(lazy_cuStreamBatchMemOp(stream, ops.size(), ops.data(), 0)); // 一次性提交到 stream，保持和后续 kernel 的顺序关系。
}

void batched_wait(CUstream stream, const std::vector<void*>& ptrs, const int& value) {
    // 等待所有 ptr 的值 >= value；这比逐个 cudaMemcpy/CPU wait 更轻，因为等待发生
    // 在 GPU stream 内，后续 kernel 自动排在这些 wait 之后。
    std::vector<CUstreamBatchMemOpParams> ops(ptrs.size());  // 每个等待地址对应一个 wait op。
    for (int i = 0; i < ptrs.size(); ++ i)                   // 多个 wait 会按数组顺序排入同一个 stream。
        ops[i] = create_mem_op(ptrs[i], value, CU_STREAM_MEM_OP_WAIT_VALUE_32, CU_STREAM_WAIT_VALUE_GEQ);  // 等待 flag >= value。
    CUDA_DRIVER_CHECK(lazy_cuStreamBatchMemOp(stream, ops.size(), ops.data(), 0)); // wait 完成后，stream 后续 kernel 才会执行。
}

void batched_write_and_wait(CUstream stream, const std::vector<void*>& write_ptrs, const std::vector<void*>& wait_ptrs, const int& value) {
    // 把写入和等待合并到一个 driver 调用，减少 host API 开销；ops 顺序保持先写后等。
    std::vector<CUstreamBatchMemOpParams> ops(write_ptrs.size() + wait_ptrs.size());  // 前半段写，后半段等。
    for (int i = 0; i < write_ptrs.size(); ++ i)                                      // 先发布本 rank 已到达某阶段。
       ops[i] = create_mem_op(write_ptrs[i], value, CU_STREAM_MEM_OP_WRITE_VALUE_32);  // 写入 handshake 轮次。
    for (int i = 0; i < wait_ptrs.size(); ++ i)                                        // 再等待 peer rank 也到达同一阶段。
       ops[write_ptrs.size() + i] = create_mem_op(wait_ptrs[i], value, CU_STREAM_MEM_OP_WAIT_VALUE_32, CU_STREAM_WAIT_VALUE_GEQ);  // 等待轮次不小于 value。
    CUDA_DRIVER_CHECK(lazy_cuStreamBatchMemOp(stream, ops.size(), ops.data(), 0));      // 单次提交降低 host 调度开销。
}

}  // namespace deep_ep::cuda_driver
