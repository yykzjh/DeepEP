# DeepEP V1 (Legacy)

> **Note:** This is the archived documentation for DeepEP V1 (NVSHMEM-based). For the latest V2 documentation, see the [main README](../README.md).

> 中文学习注释：本页是 V1/legacy 路径文档，主要对应 `deep_ep/buffers/legacy.py`、`csrc/kernels/legacy/intranode.cu`、`csrc/kernels/legacy/internode.cu`、`csrc/kernels/legacy/internode_ll.cu`。V2 已经把主接口统一到 `ElasticBuffer`，但理解 V1 有助于对比 NVSHMEM 后端和 NCCL Gin 后端的设计差异。

---

DeepEP (DeepEveryParallel) V1 is the original high-performance communication library for modern machine learning, focused on expert parallelism (EP). It provides high-throughput and low-latency all-to-all GPU kernels, which are also known as MoE dispatch and combine. The library also supports low-precision operations, including FP8.

中文注释：这里的 dispatch 指“按 top-k 路由把 token 发给拥有 expert 的 rank”，combine 指“专家计算结束后按源 token 把结果发回并规约”。V1 同时支持训练/Prefill 的高吞吐路径和 Decode 的低延迟路径。

To align with the group-limited gating algorithm proposed in the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) paper, DeepEP V1 offers a set of kernels optimized for asymmetric-domain bandwidth forwarding, such as forwarding data from NVLink domain to RDMA domain. These kernels deliver high throughput, making them suitable for both training and inference prefilling tasks. Additionally, they support SM (Streaming Multiprocessors) number control.

中文注释：这段对应 legacy normal internode 的设计动机：节点内 NVLink 域带宽高，节点间 RDMA 域带宽低，因此代码会把转发、prefix、buffer layout 做成分层结构，尽量让 RDMA 成为可控瓶颈。

For latency-sensitive inference decoding, DeepEP V1 includes a set of low-latency kernels with pure RDMA to minimize delays. The library also introduces a hook-based communication-computation overlapping method that does not occupy any SM resource.

中文注释：low-latency 路径的 hook 不是 Python 回调计算，而是把接收阶段拆成可延迟触发的通信完成逻辑；这和 `internode_ll.cu` 里的 SEND/RECV phase 设计是联动的。

Notice: the implementation in this library may have some slight differences from the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) paper.

中文注释：论文算法和工程实现存在差异，尤其 buffer 组织、同步协议、容错 mask、FP8/LogFMT 等工程细节要以本仓库代码为准。

## Performance

中文注释：性能表中的带宽是逻辑带宽，通常会把本地流量和远端流量折算进同一口径；学习代码时不要把它直接等同于物理网卡单向吞吐。

### Normal kernels with NVLink and RDMA forwarding

中文注释：Normal kernels 对应 `intranode.cu` 与 `internode.cu`，重点是吞吐和大 batch，通常要先统计 layout/prefix，再搬运 payload。

We test normal kernels on H800 (~160 GB/s NVLink maximum bandwidth), with each connected to a CX7 InfiniBand 400 Gb/s RDMA network card (~50 GB/s maximum bandwidth). And we follow the DeepSeek-V3/R1 pretraining setting (4096 tokens per batch, 7168 hidden, top-4 groups, top-8 experts, FP8 dispatching and BF16 combining).

中文注释：这里的测试形状对应典型大 batch 训练/prefill：`num_tokens=4096`、`hidden=7168`、`num_topk=8`。源码中的 hidden specialization 和 FP8 scale layout 很多都围绕这类形状优化。

|   Type    | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
|:---------:|:------------:|:--------------------:|:-----------:|:--------------------:|
| Intranode |      8       |  153 GB/s (NVLink)   |      8      |  158 GB/s (NVLink)   |
| Internode |      16      |    43 GB/s (RDMA)    |     16      |    43 GB/s (RDMA)    |
| Internode |      32      |    58 GB/s (RDMA)    |     32      |    57 GB/s (RDMA)    |
| Internode |      64      |    51 GB/s (RDMA)    |     64      |    50 GB/s (RDMA)    |

### Low-latency kernels with pure RDMA

中文注释：Low-latency kernels 对应 `internode_ll.cu`，重点是 decode 小 batch。它用 RDMA message、flag、phase 和 hook 机制尽量减少 CPU 等待与 SM 占用。

We test low-latency kernels on H800 with each connected to a CX7 InfiniBand 400 Gb/s RDMA network card (~50 GB/s maximum bandwidth). And we follow a typical DeepSeek-V3/R1 production setting (128 tokens per batch, 7168 hidden, top-8 experts, FP8 dispatching and BF16 combining).

中文注释：low-latency 表格中的 `128 tokens` 是 decode 场景的小 batch，和 normal kernel 的目标不同；因此 `internode_ll.cu` 更关注每次通信的轮次、flag、atomic 和 hook，而不是只追求大吞吐。

| Dispatch #EP | Latency | RDMA bandwidth | Combine #EP | Latency | RDMA bandwidth |
|:------------:|:-------:|:--------------:|:-----------:|:-------:|:--------------:|
|      8       |  77 us  |    98 GB/s     |      8      | 114 us  |    127 GB/s    |
|      16      | 118 us  |    63 GB/s     |     16      | 195 us  |    74 GB/s     |
|      32      | 155 us  |    48 GB/s     |     32      | 273 us  |    53 GB/s     |
|      64      | 173 us  |    43 GB/s     |     64      | 314 us  |    46 GB/s     |
|     128      | 192 us  |    39 GB/s     |     128     | 369 us  |    39 GB/s     |
|     256      | 194 us  |    39 GB/s     |     256     | 360 us  |    40 GB/s     |

## Quick Start

中文注释：这一节描述 V1 的构建和运行方式。当前项目新增了 V2/JIT 后，V1 仍可作为兼容路径存在。

### Requirements

中文注释：SM80/SM90、CUDA、PyTorch、NVLink、RDMA 要求直接决定哪些 legacy kernel 能编译和运行；跨节点路径还额外依赖 NVSHMEM。

- Ampere (SM80), Hopper (SM90) GPUs, or other architectures with SM90 PTX ISA support
  - 中文注释：SM80 主要覆盖较老 intranode 路径；SM90 及以上才能充分使用更激进的 PTX/TMA 优化。
- Python 3.8 and above
  - 中文注释：Python 层主要提供 Buffer API、测试脚本和分布式初始化。
- CUDA version
    - CUDA 11.0 and above for SM80 GPUs
    - CUDA 12.3 and above for SM90 GPUs
    - 中文注释：CUDA 版本决定 NVCC/PTX ISA 和某些底层 intrinsic 是否可用。
- PyTorch 2.1 and above
  - 中文注释：PyTorch 提供 tensor、stream、distributed group 和 C++ extension 构建能力。
- NVLink for intranode communication
  - 中文注释：节点内路径依赖 GPU P2P/NVLink，否则会退化或不可用。
- RDMA network for internode communication
  - 中文注释：跨节点路径依赖 RDMA/NVSHMEM/IBGDA，和普通 TCP 后端不是一个性能级别。

### Download and install NVSHMEM dependency

中文注释：V1 internode 和 low-latency 均依赖 NVSHMEM；没有 `NVSHMEM_DIR` 时项目会禁用部分 legacy 跨节点能力。

DeepEP V1 depends on NVSHMEM. Please refer to the NVSHMEM Installation Guide for instructions.

### Development

中文注释：开发模式会编译 C++/CUDA 扩展并把 `.so` 链接到源码包目录，方便直接从工作树 import。

```bash
# Build and make symbolic links for SO files
# 中文注释：下面命令会触发 setup.py 构建 legacy C++/CUDA 扩展。
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py build
# You may modify the specific SO names according to your own platform
# 中文注释：`.so` 文件名与 Python ABI、平台、编译器有关，需要按实际 build 输出调整。
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so

# Run test cases
# NOTES: you may modify the `init_dist` function in `tests/utils.py`
# according to your own cluster settings, and launch into multiple nodes
# 中文注释：这些测试需要按集群实际 launcher、rank、节点数配置启动，单机直接运行可能只覆盖部分路径。
python tests/test_intranode.py
python tests/test_internode.py
python tests/test_low_latency.py
```

### Installation

中文注释：安装模式会把 legacy 扩展和 Python 包安装到当前环境，适合非源码调试使用。

```bash
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py install
```

#### Installation environment variables

中文注释：这些变量会影响 `setup.py` 编译宏和目标架构，尤其 `DISABLE_SM90_FEATURES`、`DISABLE_AGGRESSIVE_PTX_INSTRS` 会改变底层 PTX helper 路径。

- `NVSHMEM_DIR`: the path to the NVSHMEM directory, disable all internode and low-latency features if not specified
  - 中文注释：没有 NVSHMEM 路径时，legacy 跨节点和低延迟路径会在构建阶段被禁用。
- `DISABLE_SM90_FEATURES`: 0 or 1, whether to disable SM90 features, it is required for SM90 devices or CUDA 11
  - 中文注释：该变量会影响是否使用 SM90 特有 PTX 指令，兼容旧 CUDA 时可能需要关闭。
- `TORCH_CUDA_ARCH_LIST`: the list of target architectures, e.g. `TORCH_CUDA_ARCH_LIST="9.0"`
  - 中文注释：控制 C++ extension 编译哪些 GPU 架构的 cubin/PTX。
- `DISABLE_AGGRESSIVE_PTX_INSTRS`: 0 or 1, whether to disable aggressive load/store instructions, see [Undefined-behavior PTX usage](#undefined-behavior-ptx-usage) for more details
  - 中文注释：关闭更激进的 PTX load/store，可提高兼容性但可能损失性能。

## Network Configurations

中文注释：网络配置影响 legacy internode 的稳定性和尾延迟，尤其 low-latency decode 对拥塞、路由和超时非常敏感。

DeepEP is fully tested with InfiniBand networks. However, it is theoretically compatible with RDMA over Converged Ethernet (RoCE) as well.

### Traffic isolation

中文注释：Virtual Lane/Service Level 隔离可以减少训练大流量与低延迟 decode 小流量互相干扰。

Traffic isolation is supported by InfiniBand through Virtual Lanes (VL).

To prevent interference between different types of traffic, we recommend segregating workloads across different virtual lanes as follows:

- workloads using normal kernels
- workloads using low-latency kernels
- other workloads

For DeepEP V1, you can control the virtual lane assignment by setting the `NVSHMEM_IB_SL` environment variable.

中文注释：`NVSHMEM_IB_SL` 会进入 NVSHMEM/IB 网络 service level 选择，适合把 normal、low-latency 和其它业务流量分开。

### Adaptive routing

中文注释：自适应路由能缓解拥塞，但可能增加尾延迟；训练吞吐和推理 decode 对它的偏好不完全相同。

Adaptive routing is an advanced routing feature provided by InfiniBand switches that can evenly distribute traffic across multiple paths. Enabling adaptive routing can completely eliminate network congestion caused by routing conflicts, but it also introduces additional latency. We recommend the following configuration for optimal performance:

- enable adaptive routing in environments with heavy network loads
- use static routing in environments with light network loads

### Congestion control

中文注释：这里说明生产环境未启用拥塞控制；如果你的集群拓扑不同，应结合实际网络监控重新评估。

Congestion control is disabled as we have not observed significant congestion in our production environment.

中文注释：如果你的网络环境拥塞明显，不能机械照搬这里的结论；需要结合交换机计数器和端到端延迟重新调参。

## Interfaces and Examples

中文注释：这一节给出 V1 Python API 使用方式，核心对象是 `Buffer`，与 V2 的 `ElasticBuffer` 形成对比。

### Example use in model training or inference prefilling

中文注释：普通 dispatch/combine 适合训练或 prefill。流程是先 `get_dispatch_layout` 得到 token 分布，再 `dispatch` 搬运，最后用 `handle` 调 `combine`。

The normal kernels can be used in model training or the inference prefilling phase (without the backward part) as the below example code shows.

```python
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Union

from deep_ep import Buffer, EventOverlap

# Communication buffer (will allocate at runtime)
# 中文注释：`_buffer` 是全局缓存，避免每次 forward 都重新分配通信 buffer。
_buffer: Optional[Buffer] = None

# Set the number of SMs to use
# NOTES: this is a static variable
# 中文注释：V1 normal kernel 需要手动设置通信 kernel 使用的 SM 数，过多会抢计算资源，过少会影响通信吞吐。
Buffer.set_num_sms(24)


# You may call this function at the framework initialization
def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    # 中文注释：V1 依赖配置对象估算 NVLink/RDMA buffer 大小；也可以通过测试调优后写死配置。
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    # 中文注释：只有 group 或容量不足时才重新分配，避免频繁 cuda/nvshmem allocation。
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    # 中文注释：FP8 输入可能是 (data, scale) tuple，因此取 x[0] 作为 payload tensor。
    t = x[0] if isinstance(x, tuple) else x
    # 中文注释：max(element_size,2) 让 FP8 payload 按 BF16 等效字节估算通信 buffer。
    return t.size(1) * max(t.element_size(), 2)


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
    # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency
    # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
    # refer to the docs of `Buffer.dispatch`
    global _buffer

    # Calculate layout before actual dispatch
    # 中文注释：layout 阶段先统计每个 rank/expert 要收多少 token，供真正 dispatch 分配输出 buffer。
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
        _buffer.get_dispatch_layout(topk_idx, num_experts,
                                    previous_event=previous_event, async_finish=True,
                                    allocate_on_comm_stream=previous_event is not None)
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
    # Unless you specify `num_worst_tokens`, but this flag is for intranode only
    # For more advanced usages, please refer to the docs of the `dispatch` function
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        # 中文注释：dispatch 返回的 handle 会在 combine_forward/backward 中复用，不能随意丢弃。
        _buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                         num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                         is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                         previous_event=previous_event, async_finish=True,
                         allocate_on_comm_stream=True)
    # For event management, please refer to the docs of the `EventOverlap` class
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor, grad_recv_topk_weights: torch.Tensor, handle: Tuple) -> \
        Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    global _buffer

    # The backward process of MoE dispatch is actually a combine
    # 中文注释：dispatch backward 把 expert 侧梯度按源 token 聚合回去，所以调用 combine。
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_grad_x, combined_grad_recv_topk_weights, event = \
        _buffer.combine(grad_recv_x, handle, topk_weights=grad_recv_topk_weights, async_finish=True)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_grad_x, combined_grad_recv_topk_weights, event


def combine_forward(x: torch.Tensor, handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[torch.Tensor, EventOverlap]:
    global _buffer

    # Do MoE combine
    # 中文注释：combine_forward 把 expert 输出回传到原始 token 顺序。
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_x, _, event = _buffer.combine(x, handle, async_finish=True, previous_event=previous_event,
                                           allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    global _buffer

    # The backward process of MoE combine is actually a dispatch
    # 中文注释：combine backward 需要把 combined 输出梯度重新发往 expert 侧，因此调用 dispatch。
    # For more advanced usages, please refer to the docs of the `dispatch` function
    grad_x, _, _, _, _, event = _buffer.dispatch(grad_combined_x, handle=handle, async_finish=True,
                                                 previous_event=previous_event,
                                                 allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return grad_x, event
```

Moreover, inside the dispatch function, we may not know how many tokens to receive for the current rank. So an implicit CPU wait for GPU received count signal will be involved, as the following figure shows.

中文注释：这解释了 V1 normal dispatch 对 CUDA graph 不友好的原因：接收 token 数在 GPU 上统计，CPU 需要等待该计数才能创建正确形状的输出 tensor。

![normal](../figures/normal.png)

### Example use in inference decoding

中文注释：低延迟 decode 路径通过 `low_latency_dispatch`/`low_latency_combine` 与 hook 机制拆分接收阶段，从而和专家计算重叠。

The low latency kernels can be used in the inference decoding phase as the below example code shows.

```python
import torch
import torch.distributed as dist
from typing import Tuple, Optional

from deep_ep import Buffer

# Communication buffer (will allocate at runtime)
# NOTES: there is no SM control API for the low-latency kernels
# 中文注释：low-latency kernel 的 SM 使用策略写在 kernel/protocol 内部，不通过 `Buffer.set_num_sms` 调。
_buffer: Optional[Buffer] = None


# You may call this function at the framework initialization
def get_buffer(group: dist.ProcessGroup, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> Buffer:
    # NOTES: the low-latency mode will consume much more space than the normal mode
    # So we recommend that `num_max_dispatch_tokens_per_rank` (the actual batch size in the decoding engine) should be less than 256
    # 中文注释：low-latency buffer 预留固定容量，容量越大 RDMA staging 和 flag 空间越大。
    global _buffer
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or _buffer.group != group or not _buffer.low_latency_mode or _buffer.num_rdma_bytes < num_rdma_bytes:
        # NOTES: for the best performance, the QP number **must** be equal to the number of the local experts
        # 中文注释：QP 数等于 local expert 数，可让每个 expert 拥有独立通信队列，降低互相阻塞。
        assert num_experts % group.size() == 0
        _buffer = Buffer(group, 0, num_rdma_bytes, low_latency_mode=True, num_qps_per_rank=num_experts // group.size())
    return _buffer


def low_latency_dispatch(hidden_states: torch.Tensor, topk_idx: torch.Tensor, num_max_dispatch_tokens_per_rank: int, num_experts: int):
    global _buffer

    # Do MoE dispatch, compatible with CUDA graph (but you may restore some buffer status once you replay)
    # 中文注释：low_latency_dispatch 返回 hook 时，真正接收 tensor 的动作可延后执行。
    recv_hidden_states, recv_expert_count, handle, event, hook = \
        _buffer.low_latency_dispatch(hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts,
                                     async_finish=False, return_recv_hook=True)

    # NOTES: the actual tensor will not be received only if you call `hook()`,
    # it is useful for double-batch overlapping, but **without any SM occupation**
    # If you don't want to overlap, please set `return_recv_hook=False`
    # Later, you can use our GEMM library to do the computation with this specific format
    # 中文注释：recv_hidden_states 的布局是低延迟专用 packed expert layout，通常直接喂给对应 GEMM。
    return recv_hidden_states, recv_expert_count, handle, event, hook


def low_latency_combine(hidden_states: torch.Tensor,
                        topk_idx: torch.Tensor, topk_weights: torch.Tensor, handle: Tuple):
    global _buffer

    # Do MoE combine, compatible with CUDA graph (but you may restore some buffer status once you replay)
    # 中文注释：low_latency_combine 同样可以返回 hook，用于把 RDMA 接收和计算阶段错开。
    combined_hidden_states, event_overlap, hook = \
        _buffer.low_latency_combine(hidden_states, topk_idx, topk_weights, handle,
                                    async_finish=False, return_recv_hook=True)

    # NOTES: the same behavior as described in the dispatch kernel
    return combined_hidden_states, event_overlap, hook
```

For two-micro-batch overlapping, you can refer to the following figure. With our receiving hook interface, the RDMA network traffic is happening in the background, without costing any GPU SMs from the computation part. But notice, the overlapped parts can be adjusted, i.e., the 4 parts of attention/dispatch/MoE/combine may not have the exact same execution time. You may adjust the stage settings according to your workload.

中文注释：这段说明 hook 的调度价值：RDMA 网络传输可以在后台进行，SM 留给 attention/MoE 计算；实际流水线切分要按模型和网络耗时调。

![low-latency](../figures/low-latency.png)

## Roadmap (V1)

中文注释：Roadmap 记录 V1 后续可能优化方向。当前主线已转向 V2，但这些条目能解释 legacy 代码里为什么同时存在旧 LD/ST、TMA、SM-free 等不同实验痕迹。

- [x] AR support
- [x] Refactor low-latency mode AR code
- [x] A100 support (intranode only)
- [x] Support BF16 for the low-latency dispatch kernel
- [x] Support NVLink protocol for intranode low-latency kernels
- [ ] TMA copy instead of LD/ST
    - [x] Intranode kernels
    - [ ] Internode kernels
    - [ ] Low-latency kernels
- [ ] SM-free kernels and refactors
- [ ] Fully remove undefined-behavior PTX instructions

## Notices

#### Easier potential overall design

The V1 implementation uses queues for communication buffers which save memory but introduce complexity and potential deadlocks. If you're implementing your own version based on DeepEP V1, consider using fixed-size buffers allocated to maximum capacity for simplicity and better performance. For a detailed discussion of this alternative approach, see https://github.com/deepseek-ai/DeepEP/issues/39.

#### Undefined-behavior PTX usage

- For extreme performance, we discover and use an undefined-behavior PTX usage: using read-only PTX `ld.global.nc.L1::no_allocate.L2::256B` to **read volatile data**. The PTX modifier `.nc` indicates that a non-coherent cache is used. But the correctness is tested to be guaranteed with `.L1::no_allocate` on Hopper architectures, and performance will be much better. The reason we guess may be: the non-coherent cache is unified with L1, and the L1 modifier is not just a hint but a strong option, so that the correctness can be guaranteed by no dirty data in L1.
- Initially, because NVCC could not automatically unroll volatile read PTX, we tried using `__ldg` (i.e., `ld.nc`). Even compared to manually unrolled volatile reads, it was significantly faster (likely due to additional compiler optimizations). However, the results could be incorrect or dirty. After consulting the PTX documentation, we discovered that L1 and non-coherent cache are unified on Hopper architectures. We speculated that `.L1::no_allocate` might resolve the issue, leading to this discovery.
- If you find kernels not working on some other platforms, you may add `DISABLE_AGGRESSIVE_PTX_INSTRS=1` to `setup.py` and disable this, or file an issue.

#### Auto-tuning on your cluster

For better performance on your cluster, we recommend to run all the tests and use the best auto-tuned configuration. The default configurations are optimized on the DeepSeek's internal cluster.
