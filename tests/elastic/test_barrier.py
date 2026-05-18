import argparse  # 解析命令行参数，如进程数、QPs 数和压力测试开关。
import torch  # 使用 inference_mode 和 multiprocessing.spawn 启动多进程 CUDA 测试。
import torch.distributed as dist  # 测试结束时销毁 NCCL process group。

import deep_ep  # 导入 ElasticBuffer 和底层 C++/CUDA 扩展。
from deep_ep.utils.envs import init_dist, dist_print  # 初始化分布式环境并做同步打印。
from deep_ep.utils.testing import bench_kineto  # 使用 PyTorch profiler 测量 barrier kernel 时间。


def test_barrier(buffer: deep_ep.ElasticBuffer, args: argparse.Namespace):  # 对单个 ElasticBuffer 实例测 barrier 延迟。
    # barrier 使用 ElasticBuffer 的 NCCL device communicator 和 symmetric window。
    # logical domain 输出为 (scaleout_ranks, scaleup_ranks)，用于确认当前测试覆盖
    # hybrid 还是纯 scale-up 路径。
    dist_print('Profiling barrier:', once_in_node=True)  # 每个节点只打印一次测试标题。
    num_scaleout_ranks, num_scaleup_ranks = buffer.get_logical_domain_size()  # 返回逻辑域尺寸 (scaleout, scaleup)。
    dist_print(f'Config:\n'  # 打印配置块起始。
               f' > Ranks: {num_scaleout_ranks} x {num_scaleup_ranks}\n'  # 展示二维逻辑 rank 拓扑。
               f' > #QPs: {buffer.num_allocated_qps}\n',  # 展示已分配 queue pairs 数量。
               once_in_node=True)  # 每个节点只打印一次，避免多 rank 重复刷屏。

    # Test barrier time
    def loop_barrier(num_tests=1000):  # 被 profiler 调用的循环体。
        # 连续调用 1000 次放大 kernel 时间，bench_kineto 再除以 profiler 统计次数。
        for i in range(num_tests):  # 连续执行多次 barrier，提高 profiler 统计稳定性。
            buffer.barrier()  # 调用 ElasticBuffer 的设备侧/通信侧 barrier。

    t = bench_kineto(lambda: loop_barrier(), 'barrier', barrier_comm_profiling=True, barrier=buffer.barrier)  # 提取 barrier kernel 平均耗时。
    dist_print(f' > EP: {buffer.rank_idx:3}/{buffer.num_ranks:3}, '  # 打印当前 rank 与总 rank。
               f'barrier time: {t * 1e6:.3f} us')  # 秒转微秒展示。
    dist_print(once_in_node=True)  # 输出空行并同步所有 rank。


# noinspection PyShadowingNames
@torch.inference_mode()  # 测试不需要 autograd，关闭梯度可减少开销和内存占用。
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):  # spawn 后每个进程运行的入口。
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)  # 初始化 NCCL，返回全局 rank/world/group。

    do_pressure_test = args.do_pressure_test  # 是否无限循环创建/销毁 buffer 做生命周期压力测试。
    # pressure test 会反复创建/销毁 ElasticBuffer，用来暴露 communicator/window 生命周期问题。
    for i in range(int(1e9) if do_pressure_test else 1):  # 压力测试模式下近似无限循环。
        buffer = deep_ep.ElasticBuffer(  # 创建 ElasticBuffer，内部初始化 communicator/window/QP 等资源。
            group, num_bytes=2 ** 30,  # 预留 1GB buffer 空间，barrier 测试本身主要依赖控制结构。
            allow_hybrid_mode=args.allow_hybrid_mode,  # 是否允许 DeepEP 使用 hybrid logical domain。
            num_allocated_qps=args.num_allocated_qps,  # 指定或自动选择 QP 数。
            explicitly_destroy=True  # 要求用户显式 destroy，便于测试析构路径。
        )  # buffer 生命周期覆盖一次 barrier profiling。

        # Test barrier
        test_barrier(buffer, args)  # 测量该 buffer 上的 barrier 性能。

        # Destroy the runtime and communication group
        buffer.destroy()  # 显式释放 ElasticBuffer 资源。
    dist.destroy_process_group()  # 销毁 PyTorch NCCL process group。


if __name__ == '__main__':  # 允许脚本直接运行。
    parser = argparse.ArgumentParser(description='Test elastic EP barrier performance')  # 创建参数解析器。

    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')  # 本机启动的进程/GPU 数。
    parser.add_argument('--allow-hybrid-mode', type=int, default=1, help='Whether to allow hybrid mode')  # 是否允许 hybrid domain。
    parser.add_argument('--num-allocated-qps', type=int, default=8, help='Number of QPs to use (0 means auto)')  # QP 数，0 表示自动。
    parser.add_argument('--do-pressure-test', action='store_true', help='Whether to do pressure test')  # 开启长时间生命周期压力测试。
    args = parser.parse_args()  # 解析命令行参数。

    # Launch test processes
    num_processes = args.num_processes  # spawn 进程数，通常等于本机 GPU 数。
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)  # 启动多进程测试。
