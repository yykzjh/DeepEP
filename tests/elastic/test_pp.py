import argparse  # 解析 PP 测试参数。
import math  # 用 math.prod 计算 tensor 元素数。
import os  # 创建 profiler trace 输出目录。
import random  # 生成可复现的随机 send/recv 操作时间线。
import torch  # 构造 CUDA Tensor、sleep、spawn 多进程。
import torch.distributed as dist  # 销毁分布式进程组。

import deep_ep  # 使用 ElasticBuffer 的 PP send/recv API。
from deep_ep.utils.envs import init_dist, dist_print, get_rdma_gbs  # 初始化分布式、同步打印、估算 RDMA 带宽。
from deep_ep.utils.testing import bench_kineto  # profiler benchmark helper。


def generate_stress_ops(rank_idx: int, num_ranks: int, num_sends: int, shape: tuple):  # 为当前 rank 生成乱序 PP 操作列表。
    # 构造一组乱序但可重放的 PP send/recv 操作。send_times/recv_times 用逻辑时间
    # 模拟不同 stage 的异步调用顺序，最终 ops 按时间排序，覆盖 inflight 环形槽复用。
    send_times = {(s, d): [] for s in range(num_ranks) for d in range(num_ranks) if s != d}  # 每个 (src,dst) pair 的 send 逻辑时间列表。
    recv_times = {(s, d): [] for s in range(num_ranks) for d in range(num_ranks) if s != d}  # 每个 (src,dst) pair 的 recv 逻辑时间列表。

    for _ in range(num_sends):  # 随机生成 num_sends 个点对点消息。
        src_rank_idx = random.randint(0, num_ranks - 1)  # 随机源 rank。
        dst_rank_idx = (src_rank_idx + (1 if random.randint(0, 1) else -1)) % num_ranks  # 目标 rank 取左右邻居，形成环形通信。
        st = random.randint(0, 10 ** 8)  # send 的逻辑时间戳。
        rt = st + random.randint(1, 3 * 10 ** 6)  # recv 逻辑时间晚于 send，模拟网络/调度延迟。
        send_times[(src_rank_idx, dst_rank_idx)].append(st)  # 记录该 pair 的 send 时间。
        recv_times[(src_rank_idx, dst_rank_idx)].append(rt)  # 记录该 pair 的 recv 时间。

    ops = []  # 当前 rank 需要执行的操作列表，每项为 (op,timestamp,peer,seq,tensor)。
    for (src_rank_idx, dst_rank_idx) in send_times:  # 遍历所有有向 rank pair。
        n = len(send_times[(src_rank_idx, dst_rank_idx)])  # 该 pair 的消息数。
        sorted_send = sorted(send_times[(src_rank_idx, dst_rank_idx)])  # send 时间排序，保证同一 pair 序号一致。
        sorted_recv = sorted(recv_times[(src_rank_idx, dst_rank_idx)])  # recv 时间排序，与 send 序号一一对应。
        for i in range(n):  # 为该 pair 的每条消息构造数据。
            tensor = torch.randn(shape, dtype=torch.bfloat16, device='cuda')  # 消息内容，形状为 args 指定的 [tokens,hidden]。
            if src_rank_idx == rank_idx:  # 当前 rank 是源端时加入 send 操作。
                ops.append(('send', sorted_send[i], dst_rank_idx, i, tensor))  # peer 是目标 rank。
            if dst_rank_idx == rank_idx:  # 当前 rank 是目标端时加入 recv 操作。
                ops.append(('recv', sorted_recv[i], src_rank_idx, i, tensor))  # peer 是源 rank，tensor 是期望收到的内容。
    ops.sort(key=lambda x: (x[1], x[3]))  # 按逻辑时间和 pair 内序号排序，制造交错调用。
    return ops  # 返回当前 rank 的 send/recv 执行序列。


# noinspection PyShadowingNames
@torch.inference_mode()  # PP 测试不需要梯度。
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):  # 每个 spawn 进程执行的测试入口。
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)  # 初始化 NCCL 并得到当前 rank 信息。
    shape = (args.num_tokens, args.hidden)  # 单个 PP tensor 的形状 [num_tokens, hidden]。
    num_max_tensor_bytes = math.prod(shape) * 2  # BF16 每元素 2 字节，因此最大 tensor 字节数为 prod(shape)*2。
    num_max_inflight_tensors = args.num_max_inflight_tensors  # PP ring buffer 中允许同时在途的 tensor 数。
    buffer = deep_ep.ElasticBuffer(  # 创建 ElasticBuffer，只启用 PP 所需缓冲。
        group, explicitly_destroy=True, allow_hybrid_mode=False,  # PP 测试关闭 hybrid，使用直接 rank 间 send/recv。
        num_bytes=deep_ep.ElasticBuffer.get_pp_buffer_size_hint(  # 根据 tensor 大小和 inflight 数估算 buffer 大小。
            num_max_tensor_bytes, num_max_inflight_tensors))  # size hint 确保内部槽位足够容纳并发消息。
    buffer.pp_set_config(num_max_tensor_bytes, num_max_inflight_tensors)  # 配置 PP 最大 tensor 字节数和 inflight 槽数。

    # Print configs
    assert num_ranks > 1  # PP send/recv 至少需要两个 rank。
    dist_print(f'Config:\n'  # 打印配置块。
               f' > Ranks: {num_ranks}\n'  # 当前 world size。
               f' > Shape: {shape}\n'  # 单条消息的 tensor 形状。
               f' > Max inflight tensors: {num_max_inflight_tensors}\n',  # 最大在途消息数。
               once_in_node=True)  # 每节点只打印一次。

    # Run stress tests
    dist_print('Running stress tests:', once_in_node=True)  # 打印压力测试开始。
    for seed in range(args.num_stress_iterations):  # 多个 seed 覆盖不同交错时间线。
        dist_print(f' > Testing with {seed=} ...', once_in_node=True)  # 打印当前 seed。
        torch.manual_seed(42 + seed)  # 确保所有 rank 生成同一组 tensor 内容。
        random.seed(42 + seed)  # 确保所有 rank 生成同一组 send/recv 时间线。
        ops = generate_stress_ops(rank_idx, num_ranks, args.num_sends, shape)  # 当前 rank 的操作序列。

        prev = 0  # 上一个逻辑时间戳，用于计算 sleep 间隔。
        for j, (op, timestamp, peer, _, tensor) in enumerate(ops):  # 按时间顺序执行 send/recv。
            # send 端传入原 tensor，recv 端用同 shape 空 tensor 接收；因为 generate_stress_ops
            # 在双方 rank 上使用相同随机种子和逻辑时间，所以同一 (src,dst,i) 的 tensor 内容一致。
            if op == 'send':  # 当前操作是发送。
                buffer.pp_send(tensor, peer)  # 向 peer rank 发送 tensor。
            else:  # 当前操作是接收。
                result = torch.empty_like(tensor)  # 分配接收缓冲，形状/dtype 与期望 tensor 相同。
                buffer.pp_recv(result, peer)  # 从 peer rank 接收数据到 result。
                assert torch.equal(result, tensor), (  # 校验收到的数据与双方同 seed 生成的期望 tensor 完全一致。
                    f'Rank {rank_idx}: mismatch at op {j}'  # 报错时指出 rank 和操作序号。
                )  # 括号续行便于注释。
            if timestamp > prev:  # 根据逻辑时间差插入 GPU sleep，模拟异步调用间隔。
                torch.cuda._sleep(int((timestamp - prev) / 10 ** 8 * args.num_sleep_cycles))  # 时间戳比例映射到 CUDA sleep cycles。
            prev = timestamp  # 更新上一操作时间戳。
    dist_print(' > All stress tests passed', once_in_node=True)  # 所有 seed 正确性通过。
    dist_print(once_in_node=True)  # 空行分隔 stress 与 profiling 日志。

    # Profiling
    dist_print('Profiling PP send and recv:', once_in_node=True)  # 打印性能测试标题。
    num_approx_rdma_cycles = int(num_max_tensor_bytes * 2 / get_rdma_gbs() * 1.5)  # 估算覆盖 RDMA 延迟所需 GPU sleep cycles。

    def get_trace_path(prefix: str):  # 根据参数生成当前 rank 的 profiler trace 路径。
        return (None if not args.dump_profile_traces  # 未指定 trace 目录时不导出。
                else f'{args.dump_profile_traces}/{prefix}_rank{rank_idx}.json')  # 每个 rank 独立 trace 文件。

    for hide_rdma_latency in (True, False):  # 分别测“有计算覆盖通信”和“端到端可见延迟”。
        for num_concurrent in (1, 2, 3):  # 覆盖 1/2/3 条并发 PP 消息。
            # hide_rdma_latency=True 时在 send 和 recv 中间插入 sleep，模拟计算 kernel
            # 覆盖 RDMA 延迟；False 则测端到端可见延迟。
            send_tensors = [torch.randn(shape, dtype=torch.bfloat16, device='cuda') for _ in range(num_concurrent)]  # 待发送 tensor 列表。
            recv_tensors = [torch.empty(shape, dtype=torch.bfloat16, device='cuda') for _ in range(num_concurrent)]  # 对应接收缓冲列表。

            def loop(_hide_rdma_latency=hide_rdma_latency):  # profiler 调用的 PP benchmark 循环。
                torch.zeros((131072, 32768), dtype=torch.int, device='cuda')  # 大张量写入辅助扰动缓存/allocator 状态。
                for t in send_tensors:  # 先向下一个 rank 发出所有并发消息。
                    buffer.pp_send(t, (rank_idx + 1) % num_ranks)  # 环形拓扑：发送到右邻居。
                if _hide_rdma_latency:  # 模拟计算覆盖 RDMA 传输。
                    torch.cuda._sleep(num_approx_rdma_cycles * num_concurrent)  # 并发越多，覆盖等待越长。
                for t in recv_tensors:  # 再从上一个 rank 接收所有消息。
                    buffer.pp_recv(t, (rank_idx - 1) % num_ranks)  # 环形拓扑：从左邻居接收。

            send_t, recv_t = bench_kineto(  # 分别提取 send_impl 和 recv_impl kernel 平均耗时。
                loop, kernel_names=('send_impl', 'recv_impl'),  # profiler 中匹配的 kernel 名称。
                barrier_comm_profiling=True, barrier=buffer.barrier,  # 用 buffer barrier 对齐各 rank launch。
                trace_path=get_trace_path(f'pp_{num_concurrent}_{hide_rdma_latency}'))  # 可选导出 trace。
            dist_print(  # 打印本 rank 的 PP 性能。
                f' > EP: {rank_idx:3}/{num_ranks:3} | '  # 当前 rank/总 rank。
                f'hide={int(hide_rdma_latency)}, concurrent={num_concurrent} | '  # 延迟隐藏模式和并发数。
                f'send: {send_t * 1e6:.3f} us, '  # send kernel 微秒耗时。
                f'{2 * num_max_tensor_bytes / send_t / 1e9:.3f} GB/s | '  # send 估算双向字节带宽。
                f'recv: {recv_t * 1e6:.3f} us, '  # recv kernel 微秒耗时。
                f'{(2 if hide_rdma_latency else 1) * num_max_tensor_bytes / recv_t / 1e9:.3f} GB/s')  # recv 带宽按是否隐藏延迟调整系数。
    dist_print(once_in_node=True)  # profiling 结束空行。

    # Destroy the runtime and communication group
    buffer.destroy()  # 显式释放 ElasticBuffer 的通信和内存资源。
    dist.destroy_process_group()  # 销毁 PyTorch 分布式进程组。


if __name__ == '__main__':  # 允许脚本直接执行。
    parser = argparse.ArgumentParser(description='Test PP send/recv kernels')  # 创建参数解析器。
    parser.add_argument('--num-processes', type=int, default=4)  # 本地启动进程数。
    parser.add_argument('--num-tokens', type=int, default=4096)  # 单条 PP 消息 token 数。
    parser.add_argument('--hidden', type=int, default=7168)  # 单条 PP 消息 hidden 维度。
    parser.add_argument('--num-max-inflight-tensors', type=int, default=4)  # 最大在途 tensor 数。
    parser.add_argument('--num-stress-iterations', type=int, default=4)  # stress 测试 seed 数。
    parser.add_argument('--num-sends', type=int, default=128)  # 每个 seed 随机生成的消息数。
    parser.add_argument('--num-sleep-cycles', type=int, default=10 ** 7)  # 逻辑时间映射到 CUDA sleep 的比例。
    parser.add_argument('--dump-profile-traces', type=str, default='')  # trace 输出目录，空字符串表示不导出。
    args = parser.parse_args()  # 解析命令行参数。

    if args.dump_profile_traces:  # 用户要求导出 trace 时确保目录存在。
        os.makedirs(args.dump_profile_traces, exist_ok=True)  # 递归创建 trace 目录。

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)  # 启动多进程 PP 测试。
