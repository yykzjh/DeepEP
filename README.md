# DeepEP

DeepEP (DeepEveryParallel) is a high-performance communication library for modern machine learning training and inference. The library currently focuses on expert parallelism (EP) — providing high-throughput and low-latency all-to-all GPU kernels (MoE dispatch and combine) with low-precision support including FP8 — while also offering experimental primitives for pipeline parallelism (PP), context parallelism (CP), and remote memory access (Engram), all designed for zero or minimal SM occupation. All kernels are compiled at runtime via a lightweight Just-In-Time (JIT) module, requiring no CUDA compilation during installation.

> 中文学习注释：本 README 是 V2 主路径入口。代码上主要对应 `deep_ep/buffers/elastic.py`、`csrc/kernels/elastic/*.hpp`、`deep_ep/include/deep_ep/impls/*.cuh` 和 `csrc/jit/*`。核心思想是 Python 只负责组织 buffer/handle，真正的 dispatch/combine/PP/Engram kernel 由 JIT 在运行时按 hidden、top-k、rank 拓扑等模板参数生成。

Despite its lightweight design, DeepEP's performance matches or exceeds hardware bandwidth limits across various configurations.

## News

中文注释：这一节概括 V2 相对 V1 的变化，最重要的是从 NVSHMEM 后端迁移到 NCCL Gin，并把高吞吐和低延迟 EP 接口统一到 `ElasticBuffer`。

- **V2 release**: A complete refactoring of Expert Parallelism — achieving extreme performance with several times fewer SM resources compared to V1, while supporting significantly larger scale-up and scale-out domains. V2 has also switched from the NVSHMEM backend to the more lightweight **NCCL Gin backend**.

### New features

中文注释：下面这些 feature 对应代码模块：EPv2 是主 MoE 通信路径，Engram/PP/CP 是实验性远端访问或并行原语，JIT 是所有 CUDA device 实现的编译入口。

- **Fully JIT** (Just-In-Time compilation)
- **NCCL Gin backend**
  - Header-only & lightweight
  - Able to reuse existing NCCL communicators
- **EPv2**
  - High-throughput and low-latency APIs unified into a single `ElasticBuffer` interface, with a new GEMM layout
  - Larger scale-up & scale-out domain support (up to EP2048)
  - Analytical SM & QP count calculation — no more auto-tuning needed
  - Both hybrid & direct modes remain supported
  - For V3-like legacy training, SM usage reduced from 24 to 4 - 6 while maintaining equivalent or better performance
- **0 SM Engram** (with RDMA)
- **0 SM PP** (with RDMA)
- **0 SM CP** (with Copy Engine)

### Notes

中文注释：V2 buffer 更大，是因为它用更通用的对称窗口和 replay/metadata 布局换取更少 SM、更大规模和统一接口。

- Buffer size consumption is larger than V1
- 0 SM RDMA low-latency EP is no longer supported
- Engram, PP, and CP are experimental features

### Still on-going features

中文注释：这些项目说明代码仍在演进，学习时应把 `ElasticBuffer`、JIT runtime、layout helper 视为当前设计中心，而不是固定最终形态。

- **Elastic GPU & CPU buffers**: A contiguous virtual address space that maps to a hybrid of GPU and CPU physical memory under the hood, enabling fully automatic and transparent Engram or imbalanced EP
- Reducing intermediate buffer sizes by leveraging EP replay to handle load imbalance
- All-gather updates and reduce-scatter implementations for DP & TP

For the legacy V1 documentation (NVSHMEM-based), see [docs/legacy.md](docs/legacy.md).

## Performance

中文注释：性能表展示的是 V2 在不同拓扑下用较少 SM 达到接近硬件瓶颈的效果。`#SMs` 对应 Python 调用中的 `num_sms` 或理论 SM 估算。

Following V3's configuration, we tested with 8K tokens per batch, 7168 hidden dimensions, top 8 experts, FP8 dispatching, and BF16 combining, and obtained the following results:

| Arch | NIC type | Topo | Dispatch Bottleneck Bandwidth | Combine Bottleneck Bandwidth | #SMs |
|--|--|--|--|--|--|
| SM90 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |
| SM90 | CX7 | EP 8 x 4 | 61 GB/s (RDMA) | 61 GB/s (RDMA) | 6 |
| SM100 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 91 GB/s (RDMA) | 12 |
| SM100 | N/A | EP 8 | 726 GB/s (NVLink) | 740 GB/s (NVLink) | 64 (Max perf) |
| SM100 | N/A | EP 8 | 643 GB/s (NVLink) | 675 GB/s (NVLink) | 24 (Min #SM) |

Notes: the results are logical bandwidth. For example, under the `EP 8 x 2` case, 90 GB/s actually contains local rank traffic.

Comparing with V1, **V2 achieves up to 1.3x peak performance, while saving up to 4x SM count**.

We omit results for larger EP configurations for the time being, but encourage interested users to benchmark them directly. Based on our internal experience, we expect the kernel to continue saturating hardware bandwidth at scale.

For V1 performance data, see [docs/legacy.md](docs/legacy.md#performance).

## Quick start

中文注释：快速开始说明运行环境和安装方式。V2 的安装阶段不编译具体 CUDA kernel，只有运行时第一次调用某个形状时才 JIT 编译。

### Requirements

中文注释：V2 主路径依赖 SM90 PTX ISA、较新的 PyTorch、NCCL 2.30.4+ 和 RDMA/NVLink 拓扑；这些要求来自 NCCL Gin 与 TMA/PTX helper 的使用。

- Hopper (SM90) GPUs, or other architectures with SM90 PTX ISA support
- Python 3.8 and above
- CUDA version
  - CUDA 12.3 and above for SM90 GPUs
- PyTorch 2.10 and above
- NCCL 2.30.4 and above
- NVLink for intranode communication
- RDMA network for internode communication

### Install NCCL dependency

中文注释：推荐 pip 安装 NCCL 是为了让 `deep_ep/__init__.py` 和 JIT 初始化更容易定位 NCCL 头文件/库路径。

We recommend using pip to install NCCL so that DeepEP can automatically locate it within the Python environment. You can install it using the following command:

```bash
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

### Install NVSHMEM dependency

中文注释：NVSHMEM 主要服务 legacy V1 路径。如果只学习 V2 Elastic，可以先看 NCCL Gin 后端；但完整项目仍保留 NVSHMEM 安装文档。

DeepEP also depends on NVSHMEM to provide support for legacy methods. Please refer to our [NVSHMEM Installation Guide](docs/nvshmem.md) for instructions.

### Development

中文注释：开发模式会构建 Python/C++ 扩展壳层；真正的 EP CUDA kernel 仍由运行时 JIT 编译并缓存。

```bash
# Build and make symbolic links for SO files
python setup.py build
# You may modify the specific SO names according to your own platform
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so

# Run test cases
# NOTES: you may modify the `init_dist` function in `tests/utils/envs.py`
# according to your own cluster settings, and launch into multiple nodes
python tests/elastic/test_ep.py
python tests/elastic/test_agrs.py
python tests/elastic/test_engram.py
python tests/elastic/test_pp.py
```

### Installation

中文注释：安装后 import `deep_ep` 会初始化 JIT 路径、CUDA/NCCL 路径和扩展模块，后续 API 调用再触发具体 kernel build。

```bash
python setup.py install
```

Then, import `deep_ep` in your Python project, and enjoy!

## Interfaces and examples

中文注释：接口示例是学习 Python 调用链的入口。建议从 `ElasticBuffer.dispatch` 和 `ElasticBuffer.combine` 看起，再顺着 host wrapper 到 device `.cuh` 实现。

### Buffer initialization

中文注释：`ElasticBuffer` 初始化时会根据 MoE 形状估算通信 buffer 大小，`get_theoretical_num_sms` 则用解析公式给出通信 kernel 的 SM 数。

In V2, all EP operations — high-throughput and low-latency — are unified under a single `ElasticBuffer` interface. The buffer can be initialized by specifying MoE settings directly, and the optimal SM and QP counts are calculated analytically.

```python
import torch
import torch.distributed as dist
from typing import Optional

from deep_ep import ElasticBuffer

# Communication buffer (will allocate at runtime)
_buffer: Optional[ElasticBuffer] = None

# Number of SMs to use for communication kernels (will be set at buffer creation)
_num_comm_sms: int = 0


def get_buffer(group: dist.ProcessGroup,
               num_max_tokens_per_rank: int,
               hidden: int,
               num_topk: int,
               num_experts: int,
               use_fp8_dispatch: bool = False) -> ElasticBuffer:
    """Initialize or retrieve the ElasticBuffer for EP communication."""
    global _buffer, _num_comm_sms

    # Check if we can reuse the existing buffer
    required_bytes = ElasticBuffer.get_buffer_size_hint(
        group, num_max_tokens_per_rank, hidden,
        num_topk=num_topk, use_fp8_dispatch=use_fp8_dispatch,
    )
    if _buffer is not None and _buffer.group == group and _buffer.num_bytes >= required_bytes:
        return _buffer

    # Allocate a new buffer with MoE settings
    # NOTES: V2 buffer size consumption is larger than V1
    _buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    # V2 analytically calculates the optimal SM count — no more auto-tuning needed
    # You may also specify `num_sms` manually in dispatch/combine calls to override
    _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)

    return _buffer
```

### Example use in model training or inference prefilling

中文注释：训练/prefill 流程是 dispatch -> expert GEMM -> combine。`EPHandle` 保存 dispatch 产生的 prefix、metadata 和 replay 信息，是 combine 的关键输入。

V2 unifies the dispatch and combine APIs into a single `ElasticBuffer` interface. The example below shows how to use them for training (with backward passes) or inference prefilling.

```python
import torch
import torch.distributed as dist
from typing import Tuple, Union

from deep_ep import ElasticBuffer, EPHandle, EventOverlap


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int,
                     num_max_tokens_per_rank: int,
                     expert_alignment: int = 1) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
              torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
    """
    MoE dispatch: route tokens to the corresponding experts across all ranks.
    Supports both BF16 and FP8 (x as a tuple of [data, scale_factors]) inputs.
    """
    global _buffer, _num_comm_sms

    recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=expert_alignment,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    # `handle` contains routing metadata for the subsequent combine call
    # `handle.num_recv_tokens_per_expert_list` provides per-expert token counts for GEMM
    # Use `event.current_stream_wait()` to synchronize the compute stream before using results
    return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor,
                      grad_recv_topk_weights: torch.Tensor,
                      handle: EPHandle) -> Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    """The backward pass of MoE dispatch is actually a combine."""
    # 中文注释：dispatch 的反向传播在通信语义上等价于 combine，
    # 因为前向 dispatch 把 token 发给 expert，反向需要把 expert 输出梯度按原 token 聚合回来。
    global _buffer, _num_comm_sms

    combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
        # 中文注释：grad_recv_x 的布局与前向 recv_x 对齐，handle 记录反向路由所需 metadata。
        grad_recv_x,
        handle=handle,
        # 中文注释：topk 权重梯度或权重相关回传沿用前向 handle 中的 top-k 对应关系。
        topk_weights=grad_recv_topk_weights,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_grad_x, combined_grad_topk_weights, event


def combine_forward(x: torch.Tensor,
                    handle: EPHandle) -> Tuple[torch.Tensor, EventOverlap]:
    """MoE combine: reduce expert outputs back to their original ranks."""
    # 中文注释：combine_forward 是专家计算后的正常回传路径。
    # x 通常是按 dispatch 输出布局排列的 expert 结果，handle 决定它们回到哪些原始 token。
    global _buffer, _num_comm_sms

    combined_x, _, event = _buffer.combine(
        # 中文注释：combined_x 的逻辑形状恢复为 [num_original_tokens, hidden]。
        x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: EPHandle) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    """The backward pass of MoE combine is actually a dispatch."""
    # 中文注释：combine 的反向传播在通信语义上等价于 dispatch，
    # 因为 combined_x 的梯度需要重新按前向路由发送给每个 expert 输出位置。
    global _buffer, _num_comm_sms

    grad_x, _, _, _, event = _buffer.dispatch(
        # 中文注释：这里传入 handle 而不是重新传 topk_idx，表示复用前向 dispatch/combine 的布局。
        grad_combined_x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return grad_x, event
```

For communication-computation overlap, use the `EventOverlap` interface to manage dependencies between the communication stream and the compute stream:

中文注释：`EventOverlap` 是 Python 层暴露的轻量同步对象。通信 kernel 通常运行在通信 stream 上，计算 kernel 在当前/计算 stream 上；调用 `current_stream_wait()` 会让当前 stream 等待通信完成，而不是强制 CPU 同步。

```python
# After dispatch, overlap computation while communication is in-flight
recv_x, recv_topk_idx, recv_topk_weights, handle, event = dispatch_forward(...)
# 中文注释：dispatch_forward 返回时，如果 async 开启，通信可能仍在通信 stream 上执行。

# ... do some independent computation here ...
# 中文注释：这里可以执行不依赖 recv_x 的计算，用来隐藏通信延迟。

# Wait for communication to finish before using results
event.current_stream_wait()
# 中文注释：真正消费 recv_x 前再等待，避免读到尚未写完的输出。

# Now safe to use recv_x, recv_topk_idx, recv_topk_weights
```

### Example use in inference decoding

中文注释：decode 场景通常 batch 很小、迭代很多，减少 layout 计算和 CPU 同步尤其重要；cached handle 的目的就是复用稳定路由。

For inference decoding, the same `ElasticBuffer` is used. The handle-caching pattern allows reusing routing metadata across iterations when the gating decisions remain unchanged, avoiding redundant CPU synchronization.

```python
import torch
from typing import Tuple, Optional, Union

from deep_ep import ElasticBuffer, EPHandle, EventOverlap


def decode_dispatch(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                    topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                    num_experts: int,
                    num_max_tokens_per_rank: int,
                    cached_handle: Optional[EPHandle] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
              torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
    """
    MoE dispatch for inference decoding.
    If `cached_handle` is provided, the layout is reused without CPU synchronization.
    """
    # 中文注释：cached_handle 非空时，Python 不再重新统计 token->rank/expert 的布局，
    # 而是直接使用上一次 dispatch 生成的 slot/prefix metadata。
    global _buffer, _num_comm_sms

    if cached_handle is not None:
        # Reuse cached handle: skip layout recomputation and CPU sync
        # 中文注释：复用 handle 的路径只需要传入新的 x payload；
        # topk_idx/topk_weights 等路由元数据从 cached_handle 中继承。
        recv_x, _, _, handle, event = _buffer.dispatch(
            x,
            handle=cached_handle,
            num_sms=_num_comm_sms,
            async_with_compute_stream=True,
        )
        # 中文注释：这里返回 cached_handle.topk_idx 是为了保持函数返回结构与非 cached 路径一致。
        return recv_x, cached_handle.topk_idx, None, handle, event

    recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
        # 中文注释：首次 decode 或路由变化时仍需要完整 dispatch，生成新的 handle 供后续复用。
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def decode_combine(x: torch.Tensor,
                   handle: EPHandle) -> Tuple[torch.Tensor, EventOverlap]:
    """MoE combine for inference decoding."""
    # 中文注释：decode combine 与训练 combine 的核心语义相同，只是通常 token 数更少、延迟更敏感。
    global _buffer, _num_comm_sms

    combined_x, _, event = _buffer.combine(
        # 中文注释：handle 必须与本次 decode dispatch 的布局匹配，否则回填 token 会错位。
        x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_x, event
```

### Environment variables

中文注释：环境变量分为运行时调试、网络、JIT、构建几类。学习源码时可重点关注 `deep_ep/utils/envs.py`、`csrc/utils/system.hpp` 和 `csrc/jit/compiler.hpp`，它们负责把这些变量传入 Python/C++/JIT。

The library provides some environment variables, which may be useful:

- General
    - 中文注释：General 变量主要影响 Python buffer 行为、debug 输出和 tensor 生命周期管理。
    - `EP_BUFFER_DEBUG`: `0` or `1`, print buffer initialization, SM approximation, and backend debugging information, `0` by default
    - `EP_SUPPRESS_NCCL_CHECK`: `0` or `1`, suppress NCCL version mismatch checking, `0` by default
    - `EP_AVOID_RECORD_STREAM`: `0` or `1`, avoid `record_stream` on output tensors, `0` by default
    - `EP_NUM_TOPK_IDX_BITS`: integer, override the number of bits for top-k index encoding, `0` (auto) by default
- Networking
    - 中文注释：Networking 变量会影响 NCCL Gin/RDMA 选路、service level 和后端开关。
    - `EP_NIC_NAME`: string, the default NIC name used to query NIC properties, `mlx5_0` by default
    - `EP_OVERRIDE_RDMA_SL`: integer, override the RDMA service level index for traffic isolation
    - `EP_DISABLE_GIN`: `0` or `1`, disable the NCCL Gin backend (fall back to non-Gin path), `0` by default
- JIT
    - 中文注释：JIT 变量决定 runtime 编译日志、缓存目录、NVCC 路径、PTX/SASS dump 和 ptxas 检查。
    - `EP_JIT_DEBUG`: `0` or `1`, print JIT debugging information, `0` by default
    - `EP_JIT_CACHE_DIR`: string, cache directory for compiled kernels, `$HOME/.deep_ep` by default
    - `EP_JIT_NVCC_COMPILER`: string, NVCC compiler path; defaults to `torch.utils.cpp_extension.CUDA_HOME`
    - `EP_JIT_CPP_STANDARD`: integer, C++ standard version, `20` by default
    - `EP_JIT_PRINT_COMPILER_COMMAND`: `0` or `1`, print compilation commands, `0` by default
    - `EP_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS output, `0` by default
    - `EP_JIT_PTXAS_CHECK`: `0` or `1`, assert no local memory usage in compiled kernels, `0` by default
    - `EP_JIT_WITH_LINEINFO`: `0` or `1`, embed source line info for profiling tools, `0` by default
    - `EP_JIT_DUMP_ASM`: `0` or `1`, dump both PTX and SASS, `0` by default
    - `EP_JIT_DUMP_PTX`: `0` or `1`, dump PTX output, `0` by default
    - `EP_JIT_DUMP_SASS`: `0` or `1`, dump SASS output, `0` by default
- Debug and profiling
    - 中文注释：Debug/profiling 变量主要用于 NCCL Gin 诊断或外部 profiler 场景。
    - `EP_GIN_GDAKI_DEBUG`: `0` or `1`, enable NCCL Gin GDAKI debugging output, `0` by default
    - `EP_USE_NVIDIA_TOOLS`: `0` or `1`, skip internal profiling when running under external NVIDIA tools, `0` by default
    - `EP_DISABLE_BARRIER_PROFILING`: `0` or `1`, disable barrier-based communication profiling in benchmarks, `0` by default
- Build
    - 中文注释：Build 变量会在安装/构建阶段影响 C++ 扩展和 legacy kernel 编译宏。
    - `EP_NCCL_ROOT_DIR`: string, path to the NCCL installation directory; auto-detected from the Python environment if not set
    - `EP_NVSHMEM_ROOT_DIR`: string, path to the NVSHMEM installation directory; auto-detected from the Python environment if not set
    - `TORCH_CUDA_ARCH_LIST`: string, list of target CUDA architectures, e.g. `"9.0"`
    - `DISABLE_SM90_FEATURES`: `0` or `1`, disable SM90 features for legacy methods, `0` by default
    - `DISABLE_AGGRESSIVE_PTX_INSTRS`: `0` or `1`, disable aggressive load/store instructions in legacy methods, `0` by default

Some environment variables are **persistent**: they are captured at build time and baked into the installed package as default values. At import time, these defaults are applied automatically unless overridden by current environment variables. The persistent variables are: `EP_JIT_CACHE_DIR`, `EP_JIT_PRINT_COMPILER_COMMAND`, `EP_NUM_TOPK_IDX_BITS`, `EP_NCCL_ROOT_DIR`.

中文注释：persistent 变量会被构建时记录下来，后续 import 时作为默认值恢复；因此修改这些变量后如果没有重新安装，行为可能和当前 shell 期望不一致。

For additional details, please refer to [the test code](tests/elastic/test_ep.py) or review the corresponding Python documentation.

## Network configurations

中文注释：网络配置影响跨节点 EP 的实际吞吐和尾延迟。V2 使用 NCCL Gin 后端，但仍依赖 InfiniBand/RDMA fabric 的 SL、路由和 atomic 性能。

DeepEP is fully tested with InfiniBand networks. However, it is theoretically compatible with RDMA over Converged Ethernet (RoCE) as well.

### Traffic isolation

中文注释：Traffic isolation 通过不同 virtual lane/service level 把 EP 流量和其它业务隔离，减少互相干扰。

Traffic isolation is supported by InfiniBand through Virtual Lanes (VL).

To prevent interference between different types of traffic, we recommend segregating workloads across different virtual lanes as follows:

- expert-parallel workloads
- other workloads

For DeepEP V2, you can control the virtual lane assignment by setting the `sl_idx` argument or the `EP_OVERRIDE_RDMA_SL` environment variable.

### Adaptive routing

中文注释：Adaptive routing 能缓解多路径拥塞；即使会增加单包延迟，V2 文档仍建议默认开启，因为大规模 EP 更怕拥塞导致的带宽塌陷。

Adaptive routing is an advanced routing feature provided by InfiniBand switches that can evenly distribute traffic across multiple paths. Even though adaptive routing introduces additional latency, we still recommend enabling it under all network load conditions.

### Congestion control

中文注释：拥塞控制可能降低峰值带宽；文档建议把不可避免的拥塞流量放到低优先级 VL，而不是牺牲 EP 主路径。

Congestion control is disabled because it hurts maximum bandwidth. If congestion is unavoidable in some scenarios, we recommend assigning those workloads to low-priority virtual lanes.

### PCI atomic mode

中文注释：PCI atomic mode 影响 RDMA atomic 操作性能，和 Gin signal / barrier / control counter 的延迟相关。

If the hardware supports it, we recommend using the following command to set the NIC's `PCI_ATOMIC_MODE` to improve RDMA atomic operation performance:

```bash
sudo mlxconfig -y -d mlx5_$i set PCI_ATOMIC_MODE=4
```

## Experimental branches

中文注释：实验分支展示了社区和上游在 zero-copy、eager、hybrid EP、ROCm 等方向的探索。阅读主线代码时不必依赖这些分支，但它们能帮助理解设计取舍。

- [Zero-copy](https://github.com/deepseek-ai/DeepEP/pull/453)
    - Removing the copy between PyTorch tensors and communication buffers, which reduces the SM usages significantly for normal kernels
    - This PR is authored by **Tencent Network Platform Department**
- [Eager](https://github.com/deepseek-ai/DeepEP/pull/437)
    - Using a low-latency protocol removes the extra RTT latency introduced by RDMA atomic OPs
- [Hybrid-EP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
    - A new backend implementation using TMA instructions for minimal SM usage and larger NVLink domain support
    - Fine-grained communication-computation overlap for single-batch scenarios
    - PCIe kernel support for non-NVLink environments
    - NVFP4 data type support
- [AntGroup-Opt](https://github.com/deepseek-ai/DeepEP/tree/antgroup-opt)
    - This optimization series is authored by **AntGroup Network Platform Department**
    - [Normal-SMFree](https://github.com/deepseek-ai/DeepEP/pull/347) Eliminating SM from RDMA path by decoupling comm-kernel execution from NIC token transfer, freeing SMs for compute
    - [LL-SBO](https://github.com/deepseek-ai/DeepEP/pull/483) Overlapping Down GEMM computation with Combine Send communication via signaling mechanism to reduce end-to-end latency
    - [LL-Layered](https://github.com/deepseek-ai/DeepEP/pull/500) Optimizing cross-node LL operator communication using rail-optimized forwarding and data merging to reduce latency
- [Mori-EP](https://github.com/deepseek-ai/DeepEP/tree/mori-ep)
    - ROCm/AMD GPU support powered by [MORI](https://github.com/ROCm/mori) backend (low-latency mode)

## Community forks

中文注释：社区 fork 说明 DeepEP 的通信抽象可迁移到不同 GPU/NIC 后端，但本仓库当前主线仍以 NVIDIA/NCCL Gin 为中心。

- [uccl/uccl-ep](https://github.com/uccl-project/uccl/tree/main/ep) - Enables running DeepEP on heterogeneous GPUs (e.g., Nvidia, AMD) and NICs (e.g., EFA, Broadcom, CX7)
- [Infrawaves/DeepEP_ibrc_dual-ports_multiQP](https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP) - Adds multi-QP solution and dual-port NIC support in IBRC transport
- [antgroup/DeepXTrace](https://github.com/antgroup/DeepXTrace) - A diagnostic analyzer for efficient and precise localization of slow ranks
- [ROCm/mori](https://github.com/ROCm/mori) - AMD's next-generation communication library for performance-critical AI workloads (e.g., Wide EP, KVCache transfer, Collectives)

## Acknowledgement

中文注释：V2 的核心后端建立在 NCCL Gin 之上，因此阅读 `csrc/kernels/backend/nccl.cu` 和 `deep_ep/include/deep_ep/common/handle.cuh` 时可以把 NCCL Gin 看成底层通信原语。

DeepEP V2 is built on top of the [NCCL](https://github.com/nvidia/nccl) Gin backend. Thanks to @sjeaugey, @pakmarkthub, @sb17v, @xiaofanl-nvidia, and the NCCL team for their support!

## License

中文注释：项目以 MIT License 发布；依赖组件如 NCCL/NVSHMEM 仍各自受其上游许可证约束。

This code repository is released under [the MIT License](LICENSE).

## Citation

中文注释：论文/项目引用信息，和代码实现无直接联动。

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```
