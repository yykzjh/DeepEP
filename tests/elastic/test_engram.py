import argparse  # 解析 engram fetch 测试的进程数、QP 数、表大小等参数。
import os  # 创建可选的 profiler trace 输出目录。
import torch  # 构造 CUDA Tensor、启动多进程、关闭 autograd。
import torch.distributed as dist  # all_gather 参考数据和销毁分布式进程组。

import deep_ep  # 使用 ElasticBuffer 的 engram_write/engram_fetch API。
from deep_ep.utils.envs import init_dist, dist_print  # 初始化 NCCL 环境并做同步打印。
from deep_ep.utils.testing import bench_kineto  # 使用 kineto 提取 issue/wait kernel 耗时。


# noinspection PyUnboundLocalVariable,PyShadowingNames
@torch.inference_mode()  # 该测试只做通信和拷贝，不需要 autograd。
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):  # 每个 spawn 进程运行的 engram 测试入口。
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)  # 初始化分布式环境，返回全局 rank/world/group。
    num_bytes = deep_ep.ElasticBuffer.get_engram_storage_size_hint(  # 计算 engram storage + fetch workspace 所需 buffer 字节数。
        args.num_entries, args.hidden, args.num_tokens, torch.bfloat16)  # entries/hidden/tokens/dtype 共同决定 symmetric window 大小。

    # 1 QP uses 1 SM
    num_qps = args.num_qps  # 用户指定的 QP 数；0 表示自动使用最大并发。
    if num_qps == 0:  # 自动模式。
        # engram fetch 的实现是一个 QP 对应一个 block/SM；默认使用 GPU SM 数作为
        # QP 数，尽量把随机远端读取并发打满。
        num_qps = torch.cuda.get_device_properties('cuda').multi_processor_count  # 用 SM 数近似最大并发 QP 数。

    # Allocate buffer
    dist_print(f'Config:\n'  # 打印配置块。
               f' > Ranks: {num_ranks}\n'  # world size。
               f' > QPs: {num_qps}\n'  # RDMA/NVSHMEM queue pairs 数。
               f' > Entries per rank: {args.num_entries}, hidden: {args.hidden}\n'  # 每 rank KV/embedding 表形状。
               f' > Tokens to fetch: {args.num_tokens}\n'  # 本轮随机读取请求数。
               f' > Storage per rank: {args.num_entries * args.hidden * 2 / 1024 / 1024:.1f} MB\n',  # BF16 存储体积。
               once_in_node=True)  # 每节点只打印一次。
    buffer = deep_ep.ElasticBuffer(  # 创建 ElasticBuffer，并分配 engram 所需 symmetric memory。
        group,  # 覆盖所有测试 rank 的通信组。
        num_bytes=num_bytes, explicitly_destroy=True, num_allocated_qps=num_qps,  # buffer 大小、显式销毁、QP 数。
        allow_hybrid_mode=False, allow_multiple_reduction=False)  # engram fetch 使用直接读取路径，不需要 hybrid/reduction。

    # local_storage 形状 [num_entries, hidden]。all_gather 得到 global_storage
    # [num_ranks*num_entries, hidden] 作为参考；engram_write 把 local_storage 写入
    # NCCL symmetric window，供其他 rank 按 global entry id 远端读取。
    local_storage = torch.randn((args.num_entries, args.hidden), dtype=torch.bfloat16, device='cuda')  # 本 rank 表，形状 [entries,H]。
    global_storage = torch.empty((num_ranks * args.num_entries, args.hidden), dtype=torch.bfloat16, device='cuda')  # 参考全局表 [R*entries,H]。
    dist.all_gather_into_tensor(global_storage, local_storage, group)  # 用 PyTorch all_gather 构造参考全局表。
    buffer.engram_write(local_storage)  # 将本 rank 表写入 ElasticBuffer symmetric window，供远端 fetch。

    # indices 长度为 num_tokens，取值范围 [0, num_ranks*num_entries)。kernel 内会拆成
    # src_rank_idx = index // num_entries 和 src_entry_idx = index % num_entries。
    indices = torch.randint(0, num_ranks * args.num_entries, (args.num_tokens, ), device='cuda', dtype=torch.int)  # 随机全局 entry id，形状 [num_tokens]。

    # Correctness check
    ref_fetched = global_storage[indices]  # PyTorch 高级索引得到参考结果，形状 [num_tokens,H]。
    hook = buffer.engram_fetch(indices)  # issue 远端读取请求，返回等待 hook。
    fetched = hook()  # 等待读取完成并取回结果。
    if not args.skip_check:  # 默认执行正确性检查。
        assert torch.equal(ref_fetched, fetched), f'{(ref_fetched - fetched).abs().max().item()}'  # 要求 BF16 bitwise 相等。

    # Performance test
    dist_print('Running performance test ...', once_in_node=True)  # 打印性能测试开始。
    msg_bytes = args.hidden * 2  # bfloat16；单条 fetch 消息读取 hidden 个 BF16。
    num_fetched_bytes = args.num_tokens * msg_bytes  # 本轮总读取字节数。

    # bench_kineto 同时匹配 issue kernel 和 wait kernel，分别观察 GIN get 提交成本
    # 与等待远端数据完成的成本。
    def fetch_and_wait():  # profiler 里的被测函数：issue fetch 后立即 wait。
        # noinspection PyShadowingNames
        hook = buffer.engram_fetch(indices)  # 提交 engram_fetch_impl kernel。
        hook()  # 执行/等待 engram_fetch_wait_impl kernel。

    issue_t, wait_t = bench_kineto(  # 分别获取 issue 和 wait kernel 的平均耗时。
        fetch_and_wait,  # 被测函数。
        kernel_names=('engram_fetch_impl', 'engram_fetch_wait_impl'),  # profiler 中匹配的两个 kernel 名称。
        barrier_comm_profiling=True,  # 插入通信 barrier 降低 rank 间 launch 抖动。
        barrier=buffer.barrier,  # 使用 ElasticBuffer barrier 做同步。
        trace_path=f'{args.dump_profile_traces}/engram_fetch_rank{buffer.rank_idx}.json' if args.dump_profile_traces else None)  # 可选导出 trace。
    mpps = args.num_tokens / (issue_t + wait_t) / 1e6  # 每秒百万条 fetch 请求数。
    
    dist_print(f' > Rank {rank:3}/{num_ranks} | '  # 打印当前 rank。
               f'issue: {issue_t * 1e6:.1f} us, '  # issue kernel 微秒耗时。
               f'wait: {wait_t * 1e6:.1f} us, '  # wait kernel 微秒耗时。
               f'{num_fetched_bytes / (issue_t + wait_t) / 1e9:.1f} GB/s, '  # 总读取带宽。
               f'bytes: {num_fetched_bytes / 1024 / 1024:.1f} MB, '  # 本轮读取总量。
               f'{mpps:.2f} MPPS ({msg_bytes} B/msg)')  # 请求吞吐和单请求字节数。
    
    dist_print('', once_in_node=True)  # 空行分隔日志。

    # Destroy the runtime and communication group
    buffer.destroy()  # 显式释放 ElasticBuffer 资源。
    dist.destroy_process_group()  # 销毁 NCCL process group。


if __name__ == '__main__':  # 允许直接运行脚本。
    parser = argparse.ArgumentParser(description='Test engram fetch kernels')  # 创建参数解析器。
    parser.add_argument('--num-processes', type=int, default=4, help='Number of processes to spawn')  # 本地进程/GPU 数。
    parser.add_argument('--num-qps', type=int, default=0, help='Number of QPs used (0 for maximum)')  # QP 数，0 表示按 SM 数自动设置。
    parser.add_argument('--num-entries', type=int, default=524288, help='Number of entries per rank')  # 每 rank 表项数。
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension size')  # 每个 entry 的 hidden 维度。
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens to fetch')  # 随机 fetch 请求数。
    parser.add_argument('--skip-check', action='store_true', help='Skip correctness check')  # 跳过正确性验证，仅测性能。
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')  # trace 输出目录。
    args = parser.parse_args()  # 解析命令行参数。

    # Create dump trace directories
    if args.dump_profile_traces:  # 指定 trace 目录时创建目录。
        os.makedirs(args.dump_profile_traces, exist_ok=True)  # 目录已存在时不报错。

    # Launch
    num_processes = args.num_processes  # spawn 进程数。
    torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)  # 启动多进程 engram 测试。
