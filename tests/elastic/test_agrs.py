import argparse  # 解析 all_gather 测试参数。
import math  # 用 math.prod 计算 tensor 字节数。
import random  # 随机生成 session/ag/fetch 操作序列。
import torch  # 构造 CUDA Tensor、event、spawn 多进程。
import torch.distributed as dist  # 销毁分布式进程组。
import numpy as np  # 统计多轮 CUDA event 耗时。

import deep_ep  # 使用 ElasticBuffer 的 AGRS/all_gather API。
from deep_ep.utils.envs import init_dist, dist_print  # 初始化 NCCL 环境并同步打印。


def all_gather_ref(shape: tuple, rank_idx: int, num_ranks: int, round_idx: int = 0):  # 为一次 all_gather 构造本 rank 输入和全局参考输出。
    # 每个 rank 用相同种子顺序生成所有 rank 的输入，从而不需要额外通信即可得到
    # all_gather 的参考输出 refs，refs 形状为 [num_ranks, *shape]。
    ref_list = []  # 保存每个 rank 的参考输入 tensor。
    for i in range(num_ranks):  # 按 rank 顺序生成所有 rank 的输入。
        torch.manual_seed(42 + round_idx * 43 + i)  # round_idx 和 rank id 决定唯一 seed。
        ref_list.append(torch.randn(shape, dtype=torch.bfloat16, device='cuda'))  # 单 rank 输入形状为 shape。
    return ref_list[rank_idx], torch.stack(ref_list, dim=0)  # 返回本 rank 输入和参考 all_gather 输出 [R,*shape]。


def generate_stress_ops(
    num_ops: int,
    num_max_inflight_agrs: int,
    # 中文：shape/rank 信息用于生成每个 rank 的输入和参考 all_gather 输出。
    shape: tuple,
    rank_idx: int,
    num_ranks: int,
) -> tuple[list[tuple], tuple[torch.Tensor], tuple[torch.Tensor]]:  # 随机生成 AGRS session 内的 all_gather/fetch 操作。
    # AGRS 支持一个 session 内多个 all_gather inflight。这里随机生成 ag/fetch/session
    # 切换操作，确保最多 num_max_inflight_agrs 个未 fetch 任务，覆盖 batched 和单个 handle。
    tensors, refs = zip(*(all_gather_ref(shape, rank_idx, num_ranks, round_idx=i) for i in range(num_ops)), strict=True)  # tensors/refs 长度均为 num_ops。
    unprocessed = random.sample(range(num_ops), num_ops)  # 随机排列尚未提交的 all_gather 序号。
    inflight, ops = [], [('create_session', (-1,))]  # inflight 保存尚未 fetch 的批次；ops 以创建 session 开始。
    limit = num_max_inflight_agrs  # 当前 session 剩余可提交的 inflight AGRS 数。
    while unprocessed or inflight:  # 直到所有任务都提交并 fetch 完。
        max_g = min(len(unprocessed), limit)  # 本次最多可新提交的 all_gather 数。
        choices = []  # 当前状态下允许的操作类型。
        if max_g > 0:  # session 还有容量且有未处理任务。
            choices.append('ag')  # 可以提交新的 all_gather。
        if inflight:  # 存在未 fetch 的 handle。
            choices.append('fetch')  # 可以等待并取回某个 inflight 结果。
        else:  # 没有 inflight 时允许销毁/重建 session。
            choices.append('destroy')  # 用于覆盖 session 生命周期边界。
        op = random.choice(choices)  # 随机选择下一步操作，制造不同交错。
        if op == 'ag':  # 提交一个或多个 all_gather。
            b = tuple(unprocessed[-random.randint(1, max_g):])  # 从未处理尾部取随机批量，形成一个 batch。
            limit -= len(b)  # 消耗 session inflight 容量。
            del unprocessed[-len(b):]  # 从未处理列表移除这些任务。
            inflight.append(b)  # 记录这个 batch 等待后续 fetch。
            ops.append(('ag', b))  # 写入操作序列。
        elif op == 'fetch':  # 等待某个 inflight batch。
            ops.append(('fetch', inflight.pop(random.randrange(len(inflight)))))  # 随机选择一个 batch fetch。
        else:  # 销毁并重新创建 session。
            ops.extend([('destroy_session', (-1,)), ('create_session', (-1,))])  # 插入 session 生命周期切换操作。
            limit = num_max_inflight_agrs  # 新 session 容量恢复。

    ops.append(('destroy_session', (-1,)))  # 最后销毁 session，测试资源释放路径。
    return ops, tensors, refs  # 返回操作序列、本 rank 输入 tuple、参考输出 tuple。


def do_all_gather(buffer: deep_ep.ElasticBuffer,
                  is_inplace: bool, is_batched: bool,
                  tensors: tuple[torch.Tensor, ...],
                  start_event: torch.cuda.Event | None = None):  # 执行一次单个或批量 all_gather，并返回输出和等待 handle。
    # inplace 模式先从 buffer 申请内部 tensor，再把输入拷进去；非 inplace 模式直接
    # 把用户 tensor 交给 all_gather。两条路径对应不同的内存所有权。
    if is_inplace:  # inplace 模式使用 buffer 内部分配的输入槽。
        ag_tensors = buffer.agrs_get_inplace_tensor(tuple(t.shape for t in tensors), torch.bfloat16)  # 获取一组内部 tensor，形状逐个匹配输入。
        for x, y in zip(ag_tensors, tensors, strict=True):  # 把用户输入复制到内部槽。
            x.copy_(y)  # x/y 形状相同，dtype 为 BF16。
    else:  # 非 inplace 模式直接使用调用方传入的 tensor。
        ag_tensors = tensors  # 保持 tuple[Tensor] 结构。

    # Record event
    if start_event is not None:  # profiling 模式下记录起点 event。
        torch.zeros(int(256e6 // 4), dtype=torch.int, device='cuda')  # flush L2 cache；写 256MB int32 刷新 L2。
        start_event.record()  # 在当前 stream 记录 all_gather 前的开始时间。

    # batched=True 时一次提交 tuple[tensor]，返回一个批量 handle；否则逐个 tensor
    # 提交，返回多个 handle。测试会逐个等待并比较参考输出。
    if is_batched:  # 批量模式：一次 API 调用提交多个 tensor 的 all_gather。
        *out_tensors, handle = buffer.all_gather(ag_tensors)  # 返回多个输出 tensor 和一个批量等待 handle。
        return out_tensors, [handle]  # 为统一 fetch 逻辑，把单个 handle 包成列表。
    else:  # 非批量模式：逐个 tensor 调用 all_gather。
        out_tensors, handles = [], []  # 分别保存输出 tensor 和等待 handle。
        for t in ag_tensors:  # 遍历每个输入 tensor。
            out_tensor, handle = buffer.all_gather(t)  # 返回输出 [R,*shape] 和等待 handle。
            out_tensors.append(out_tensor)  # 记录输出。
            handles.append(handle)  # 记录对应等待 handle。
        return out_tensors, handles  # 返回长度等于输入 tensor 数的输出和 handle。


# noinspection PyTypeChecker,PyCallingNonCallable,PyShadowingNames
@torch.inference_mode()  # 测试不需要梯度。
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):  # 每个 spawn 进程执行的 AGRS 测试入口。
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)  # 初始化分布式环境。

    # Print configs
    shape = (32, 64, 2048)  # stress 阶段单 rank 输入 tensor 形状，元素数约 4M。
    num_bytes_per_tensor = math.prod(shape) * 2  # BF16 每元素 2 字节。
    num_max_inflight_agrs = args.num_max_inflight_agrs  # 单个 session 内最大未完成 all_gather 数。
    num_max_session_bytes = deep_ep.ElasticBuffer.get_agrs_buffer_size_hint(  # 根据最大 session 工作集估算 buffer 大小。
        group, num_bytes_per_tensor * group.size() * num_max_inflight_agrs)  # all_gather 输出体积约为输入 * world_size * inflight。
    dist_print(f'Config:\n'  # 打印配置。
               f' > Ranks: {num_ranks}\n'  # world size。
               f' > Shape: {shape}\n'  # 单个输入 tensor 形状。
               f' > Max inflight AGRS: {num_max_inflight_agrs}\n',  # 最大在途 AGRS 数。
               once_in_node=True)  # 每节点只打印一次。

    # Create buffer
    buffer = deep_ep.ElasticBuffer(group, explicitly_destroy=True, num_bytes=num_max_session_bytes)  # 创建 AGRS 测试 buffer。
    buffer.agrs_set_config(num_max_session_bytes, num_max_inflight_agrs)  # 配置 session 最大字节数和 inflight 数。

    # Run stress tests
    dist_print('Running stress tests:', once_in_node=True)  # 打印 stress 开始。
    for seed in range(args.num_stress_iterations):  # 多个随机 seed 覆盖不同 session/handle 交错。
        random.seed(42 + seed)  # 固定 Python 随机源，使所有 rank 操作序列一致。
        num_ops = 128  # 每个 seed 的 all_gather 任务数。
        ops, tensors, refs = generate_stress_ops(num_ops, num_max_inflight_agrs, shape, rank_idx, num_ranks)  # 生成操作和参考数据。
        results = [None] * num_ops  # 保存每个 all_gather 的实际输出。
        handles = dict()  # key 为 indices batch，value 为 (out_tensors, wait_handles)。
        torch.cuda.synchronize()  # 确保上一轮 CUDA 工作完成。
        for op, indices in ops:  # 顺序执行随机生成的 session/ag/fetch 操作。
            # indices 是本次操作覆盖的 all_gather 序号。fetch 时等待对应 handle 后 clone，
            # 避免后续 session 复用 buffer 覆盖结果。
            if op == 'create_session':  # 创建一个新的 AGRS session。
                buffer.create_agrs_session()  # 后续 all_gather 都在该 session 内提交。
            elif op == 'destroy_session':  # 销毁当前 AGRS session。
                buffer.destroy_agrs_session()  # 释放 session 内部资源/状态。
            elif op == 'ag':  # 提交 all_gather。
                is_inplace, is_batched = random.random() < 0.5, random.random() < 0.8  # 随机覆盖 inplace 和 batched 两条路径。
                handles[indices] = do_all_gather(buffer, is_inplace, is_batched, tuple(tensors[i] for i in indices))  # 保存输出与等待句柄。
            elif op == 'fetch':  # 等待一批已提交 all_gather 完成。
                out_tensors, wait_handles = handles[indices]  # 取出该 batch 的输出 tensor 和 handle。
                for h in wait_handles:  # 一个 batch 可能对应一个或多个 handle。
                    h()  # 等待对应 all_gather 完成。
                for out, idx in zip(out_tensors, indices, strict=True):  # 将输出按原任务序号写回 results。
                    results[idx] = out.clone()  # clone 防止 session/buffer 复用覆盖原输出。

        for i in range(num_ops):  # 所有任务 fetch 完后统一校验。
            assert results[i] is not None and torch.equal(results[i], refs[i]), (  # 输出必须与 PyTorch 参考 all_gather 完全一致。
                f'Rank {rank_idx}: stress mismatch at seed={seed}, op={i}'  # 报错定位 rank/seed/op。
            )  # 括号续行方便注释。
        dist_print(f' > Seed {seed} passed ({num_ops} ops)', once_in_node=True)  # 打印该 seed 通过。
    dist_print(once_in_node=True)  # 空行分隔 stress 和 profiling。

    # Destroy the buffer
    dist_print(f'Profiling all-gather:', once_in_node=True)  # 打印 profiling 标题。
    buffer.destroy()  # 销毁 stress 阶段 buffer，避免旧 session 状态影响 profiling。

    # Profiling
    num_max_session_bytes = deep_ep.ElasticBuffer.get_agrs_buffer_size_hint(  # profiling 阶段按最大 64MB 输入估算 buffer。
        group, (2 ** 26) * group.size() * num_max_inflight_agrs)  # 2**26 bytes 是单输入最大字节数。
    buffer = deep_ep.ElasticBuffer(group, explicitly_destroy=True, num_bytes=num_max_session_bytes)  # 创建 profiling 专用 buffer。
    buffer.agrs_set_config(num_max_session_bytes, num_max_inflight_agrs)  # 配置 profiling session 容量。
    for num_bytes in (2 ** p for p in range(20, 27)):  # 测试 1MB 到 64MB 的输入大小。
        # Create tensors
        shape = (num_bytes // 2, )  # BF16 每元素 2 字节，因此一维元素数为 bytes/2。
        tensors = tuple(torch.randn(shape, dtype=torch.bfloat16, device='cuda') for _ in range(num_max_inflight_agrs))  # 准备 inflight 个输入 tensor。

        # Tests
        for is_inplace in (False, True):  # 覆盖用户 tensor 和 inplace buffer tensor 两种输入模式。
            for is_batched in (False, True):  # 覆盖单独提交和批量提交两种 API。
                num_tests = 50  # 每个配置测 50 轮。
                start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]  # 起点 event 列表。
                end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]  # 终点 event 列表。
                torch.cuda.synchronize()  # 清空之前 CUDA 工作。

                for i in range(num_tests):  # 正式计时循环。
                    # 每一轮都创建新 session，测试 session 生命周期与 event/handle 等待逻辑；
                    # start_event 记录在 all_gather 前，end_event 记录在所有 handle 完成后。
                    with buffer.agrs_new_session():  # 每轮创建临时 session，退出时自动销毁。
                        _, wait_handles = do_all_gather(buffer, is_inplace, is_batched, tensors, start_event=start_events[i])  # 提交 all_gather 并记录起点。
                        for h in wait_handles:  # 等所有返回 handle 完成。
                            h()  # handle 调用会等待对应通信完成。
                    end_events[i].record()  # session 退出后记录终点 event。
                torch.cuda.synchronize()  # 等所有 event 完成。

                times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events, strict=True)])[1:]  # 毫秒转秒并丢弃第一轮。
                avg_t = np.average(times)  # 当前配置平均耗时，单位秒。
                unit = ('MB', 1e6) if num_bytes >= 1e6 else ('KB', 1e3)  # 日志中选择 MB 或 KB。
                bandwidth_info = f', {num_bytes * num_ranks * num_max_inflight_agrs / avg_t / 1e9:.3f} GB/s' if num_ranks > 1 else ''  # 估算 all_gather 总带宽。

                dist_print(  # 打印当前配置性能。
                    f' > Rank: {rank_idx:3}/{num_ranks:3} | '  # 当前 rank/总 rank。
                    f'{num_ranks} x {(num_bytes / unit[1]):.0f} {unit[0]} | '  # 每 rank 输入大小和 world size。
                    f'avg: {avg_t / num_max_inflight_agrs * 1e6:.3f} us'  # 平均每个 inflight all_gather 的耗时。
                    f'{bandwidth_info}'  # 多 rank 时打印带宽。
                    f' (inplace={int(is_inplace)}, batched={int(is_batched)})')  # 打印 API 模式。
    dist_print(once_in_node=True)  # profiling 结束空行。

    # Destroy the runtime and communication group
    buffer.destroy()  # 显式释放 ElasticBuffer 资源。
    dist.destroy_process_group()  # 销毁 NCCL process group。


if __name__ == '__main__':  # 支持直接运行该测试脚本。
    parser = argparse.ArgumentParser(description='Test all_gather kernels')  # 创建命令行解析器。
    parser.add_argument('--num-processes', type=int, default=8)  # 本地进程/GPU 数。
    parser.add_argument('--num-max-inflight-agrs', type=int, default=4)  # 最大在途 AGRS 数。
    parser.add_argument('--num-stress-iterations', type=int, default=4)  # stress 随机 seed 数。
    args = parser.parse_args()  # 解析参数。

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)  # 启动多进程 all_gather 测试。
