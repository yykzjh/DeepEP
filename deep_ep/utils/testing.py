import json  # 读取 kineto 导出的 Chrome trace JSON，用于按 kernel 周期拆分耗时。
import os  # 读写环境变量，并操作底层文件描述符实现 stdout/stderr 静默。
import sys  # 暂存并替换 Python 层 stdout/stderr 对象。
import torch  # CUDA event、profiler、Tensor 分配和通信 benchmark 都依赖 PyTorch。
import numpy as np  # 对多轮 event 时间做平均、最小、最大统计。
import tempfile  # 临时保存 profiler trace，解析后自动删除。
import torch.distributed as dist  # benchmark 通信 kernel 时可插入跨 rank all_reduce barrier。
from pathlib import Path  # 读取临时 Chrome trace JSON 文件。
from typing import Callable, Optional, Union  # 标注可选回调、kernel 名称字符串或元组。


def flush_l2_cache(enabled: bool = True):  # 通过写大张量刷新 GPU L2，降低 benchmark 轮间缓存影响。
    """
    Flush the GPU L2 cache by writing a large zero-initialized tensor.

    Arguments:
        enabled: if `False`, does nothing.
    """
    l2_flush_cache_size = 256e6  # 256MB 大于多数 GPU L2，可有效驱逐旧数据。
    if enabled:  # 调用方可以关闭刷新，以观察缓存命中情况下的性能。
        # 写 256MB int tensor 迫使 L2 中大部分旧数据被替换，减少前一轮 benchmark
        # 对下一轮通信 kernel 的缓存影响。
        torch.empty(int(l2_flush_cache_size // 4), dtype=torch.int, device='cuda').zero_()  # int32 每元素 4 字节，因此元素数为 256MB/4。


def bench(fn, num_warmups: int = 50, num_tests: int = 50,  # 使用 CUDA event 测量一个 callable 的端到端 GPU 时间。
          post_fn: Optional[Callable] = None, flush_l2: bool = True):  # post_fn 可在每轮后做校验/清理。
    """
    Benchmark a function using CUDA events.

    Arguments:
        中文：bench 用 CUDA event 计时，适合测量单个 Python callable 触发的 GPU 工作。
        fn: the function to benchmark.
        num_warmups: the number of warmup iterations.
        num_tests: the number of measurement iterations.
        中文：warmup 不计入结果，num_tests 控制最终平均/最小值的样本数。
        post_fn: an optional function to call after each test iteration.
        flush_l2: whether to flush the L2 cache before each iteration.
        中文：post_fn 常用于校验或清理；flush_l2 用来降低 cache 命中造成的波动。

    Returns:
        中文：返回平均、最小和最大耗时，单位都是秒。
        avg: the average execution time in seconds.
        min: the minimum execution time in seconds.
        max: the maximum execution time in seconds.
        中文：调用方通常用 avg 算带宽，用 min/max 观察抖动范围。
    """
    torch.cuda.synchronize()  # 先清空当前 device 上已有工作，避免历史 kernel 影响 warmup。

    # warmup 不计时，用来触发 CUDA lazy init、JIT 编译、NCCL communicator 首次路径等。
    for _ in range(num_warmups):  # warmup 轮数默认 50，覆盖 CUDA lazy init 和 JIT 编译。
        fn()  # 不记录时间，只让 kernel/通信路径进入稳定状态。

    # CUDA event 只测 GPU stream 上的时间，不包含 Python 调用开销。times[1:] 丢弃
    # 第一轮，进一步避开残余 warmup 波动。
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]  # 每轮一个起点 event。
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]  # 每轮一个终点 event。
    for i in range(num_tests):  # 正式计时循环。
        flush_l2_cache(flush_l2)  # 每轮开始前可刷新 L2，提高测试可比性。
        start_events[i].record()  # 在当前 stream 记录开始事件。
        fn()  # 执行被测函数，通常会 launch 一个或多个 CUDA kernel。
        end_events[i].record()  # 在当前 stream 记录结束事件。
        if post_fn is not None:  # 如果需要在每轮后检查输出或恢复状态，则调用回调。
            post_fn()  # post_fn 不包含在 start/end event 时间内。
    torch.cuda.synchronize()  # 等所有计时 event 完成后才能读取 elapsed_time。

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]  # 毫秒转秒，并丢掉第一轮正式测试。
    return np.average(times), np.min(times), np.max(times)  # 返回平均/最小/最大耗时，单位秒。


class empty_suppress:  # 空上下文管理器，在不需要静默时保持 with 结构一致。
    """空 context manager，用来和 suppress_stdout_stderr 共享调用结构。"""

    def __enter__(self):  # 进入上下文时不做任何资源替换。
        return self  # 返回自身以满足 context manager 协议。

    def __exit__(self, *_):  # 退出上下文也不处理异常或资源。
        pass  # 返回 None，异常会按 Python 默认规则继续传播。


class suppress_stdout_stderr:  # 同时屏蔽 Python 和 C/CUDA 层 stdout/stderr 的上下文管理器。
    """
    Context manager to suppress stdout and stderr output.
    """

    def __enter__(self):  # 进入时把 stdout/stderr 重定向到 /dev/null。
        # 同时替换 Python 对象和底层 fd；否则 C++/CUDA/NCCL printf 仍会写到原 fd。
        self.outnull_file = open(os.devnull, 'w')  # 打开 stdout 的黑洞文件。
        self.errnull_file = open(os.devnull, 'w')  # 打开 stderr 的黑洞文件。

        self.old_stdout_fileno_undup = sys.stdout.fileno()  # 当前 stdout fd 编号，通常是 1。
        self.old_stderr_fileno_undup = sys.stderr.fileno()  # 当前 stderr fd 编号，通常是 2。

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())  # dup 一份 stdout 原 fd，退出时恢复用。
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())  # dup 一份 stderr 原 fd，退出时恢复用。

        self.old_stdout = sys.stdout  # 暂存 Python stdout 对象。
        self.old_stderr = sys.stderr  # 暂存 Python stderr 对象。

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)  # 底层 fd 级重定向 stdout，覆盖 C++ printf。
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)  # 底层 fd 级重定向 stderr。

        sys.stdout = self.outnull_file  # Python print 也写入 /dev/null。
        sys.stderr = self.errnull_file  # Python 异常/警告输出也写入 /dev/null。
        return self  # 返回上下文对象。

    def __exit__(self, *_):  # 退出时恢复所有 stdout/stderr 句柄。
        # 恢复顺序先恢复 Python 对象，再 dup2 回原 fd，最后关闭临时文件句柄。
        sys.stdout = self.old_stdout  # 先恢复 Python stdout 对象。
        sys.stderr = self.old_stderr  # 先恢复 Python stderr 对象。

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)  # 再恢复底层 stdout fd。
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)  # 再恢复底层 stderr fd。

        os.close(self.old_stdout_fileno)  # 关闭 dup 出来的 stdout 备份 fd。
        os.close(self.old_stderr_fileno)  # 关闭 dup 出来的 stderr 备份 fd。

        self.outnull_file.close()  # 关闭 /dev/null stdout 文件对象。
        self.errnull_file.close()  # 关闭 /dev/null stderr 文件对象。


def bench_kineto(fn,
                 kernel_names: Union[str, tuple],
                 num_tests: int = 30,
                 # 中文：下面参数控制 profiler 输出、trace 保存、L2 flush 和可选通信 barrier。
                 suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None,
                 flush_l2: bool = True,
                 # 中文：flush_l2=True 时每轮测试前冲刷 L2，减少 cache 热启动对通信带宽估计的影响。
                 barrier_comm_profiling: bool = False,
                 num_kernels_per_period: int = 1,
                 # 中文：barrier 可传通信同步函数，用于把 profiler 周期与跨 rank kernel 对齐。
                 barrier: Optional[Callable] = None):  # 使用 PyTorch profiler 提取指定 CUDA kernel 的平均耗时。
    """
    Benchmark a function using the PyTorch profiler (kineto) to get per-kernel timing.
    中文：bench_kineto 不是 wall-clock 计时，而是从 profiler event 中筛选指定 kernel 的 CUDA 时间。

    Arguments:
        中文：该工具用 kineto/profiler 从多次运行中提取目标 CUDA kernel 的平均耗时。
        fn: the function to benchmark.
        kernel_names: the CUDA kernel name(s) to profile.
        num_tests: the number of test iterations.
        中文：num_kernels_per_period 用于一次 fn 产生多个同名 kernel 时做周期平均。
        suppress_kineto_output: whether to suppress profiler output.
        trace_path: the path to save the Chrome trace (`None` to skip).
        中文：kernel_names 可为单个名称或 tuple；trace_path 非空时额外保存 Chrome trace。
        flush_l2: whether to flush the L2 cache before each iteration.
        barrier_comm_profiling: whether to insert a barrier before each iteration to reduce
            unbalanced CPU launch overhead.
        中文：barrier_comm_profiling 对多 rank 通信 benchmark 很重要，可减少 rank 间启动偏移。
        num_kernels_per_period: the number of kernels launched per test period.
        barrier: a custom barrier function to use instead of `dist.all_reduce`.
        中文：flush_l2/barrier 影响 benchmark 噪声，num_kernels_per_period 用于一个周期内有多个同名 kernel 的情况。

    Returns:
        durations: the average kernel duration(s) in seconds.
        中文：返回值单位是秒；输入多个 kernel 名称时会返回对应多个平均耗时。
    """
    assert isinstance(kernel_names, (str, tuple))  # kernel_names 只能是单个名称或名称元组。
    is_tuple = isinstance(kernel_names, tuple)  # 记录输入形态，用于保持返回值是标量还是列表。

    # NVIDIA 外部 profiler 与 kineto 同时启用会互相干扰；此时返回哨兵时间，
    # 让测试脚本继续跑正确性或外部工具采样。
    if int(os.environ.get('EP_USE_NVIDIA_TOOLS', 0)):  # 外部 nsys/ncu 采样时跳过 kineto，避免 profiler 冲突。
        return (1, ) * len(kernel_names) if is_tuple else 1  # 返回哨兵耗时，让调用方流程不中断。

    # 先跑一次 fn，触发 auto-tuning/JIT，并把可能的打印输出放在 profiler 外。
    fn()  # 预执行一次，触发 JIT/autotune/allocator 初始状态。
    torch.cuda.synchronize()  # 等预执行完成，再开启 profiler。

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress  # 按参数选择是否屏蔽 kineto 噪声输出。
    barrier_comm_profiling &= int(os.environ.get('EP_DISABLE_BARRIER_PROFILING', 0)) == 0  # 环境变量可强制禁用 barrier profiling。
    with suppress():  # 进入可选静默上下文。
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)  # 第一轮 warmup，第二轮 active 采样。
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule, acc_events=True)  # 仅采 CUDA 活动并累计事件。
        dummy = torch.ones(1, dtype=torch.float, device='cuda')  # all_reduce barrier 使用的 1 元素占位张量。
        with profiler:  # 开启 profiler 上下文。
            for i in range(2):  # 两个 profiler step：warmup step 和 active step。
                for _ in range(num_tests):  # 每个 step 内重复执行 num_tests 次被测函数。
                    # 每个 period 前刷新 L2，并可插入跨 rank barrier，让不同 rank 的
                    # kernel launch 时间尽量对齐，降低通信 benchmark 抖动。
                    flush_l2_cache(flush_l2)  # 每轮可刷新 L2 cache。

                    # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                    if barrier_comm_profiling:  # 通信 profiling 时可通过 GPU sleep + barrier 对齐各 rank。
                        torch.cuda._sleep(int(2e7)) # ~10ms  # 在 GPU 上等待约 10ms，减少 CPU launch 时间不均衡。

                        # Some network may have ring-based implement, so be careful to use `all_reduce`
                        if barrier is None:  # 未提供自定义 barrier 时使用 NCCL all_reduce。
                            dist.all_reduce(dummy)  # 这个 all_reduce 只用于同步，不关心 dummy 数值。
                        else:  # 某些网络/测试可提供更合适的 barrier 函数。
                            barrier()  # 调用用户传入的同步函数。
                    fn()  # 执行被测 kernel/通信路径。
                torch.cuda.synchronize()  # 每个 profiler step 结束前确保 kernel 全部完成。
                profiler.step()  # 推进 kineto schedule；第二个 step 会记录 active 数据。

    # 从 profiler table 中按 kernel name 找 CUDA 总耗时。要求每个 name 至多匹配
    # 一行，避免类似 dispatch 和 hybrid_dispatch 同时命中造成歧义。
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')  # 将 profiler 汇总表拆成行。
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names  # 统一转成 tuple 便于循环。
    assert all([isinstance(name, str) for name in kernel_names])  # 防止传入非字符串导致模糊匹配异常。
    for name in kernel_names:  # 检查每个目标 kernel 名称在表中最多匹配一行。
        assert sum([name in line for line in prof_lines]) <= 1, f'Errors of the kernel {name} in the profiling table: {prof_lines}'  # 多行匹配会让耗时解析歧义。

    # Save chrome traces
    if trace_path is not None:  # 调用方希望保存 trace 时导出 Chrome trace JSON。
        profiler.export_chrome_trace(trace_path)  # 文件可用 chrome://tracing 或 Perfetto 打开。

    # Return average kernel durations
    units = {'ms': 1e3, 'us': 1e6}  # profiler table 可能以 ms 或 us 展示，统一换算到秒。
    kernel_durations = []  # 存储每个 kernel name 的平均单次耗时，单位秒。
    for name in kernel_names:  # 逐个目标 kernel 解析耗时。
        total_time = 0  # 累计耗时，单位秒 * 次数。
        total_num = 0  # 累计调用次数。
        for line in prof_lines:  # 遍历 profiler 汇总表每一行。
            if name in line:  # 找到目标 kernel 对应行。
                time_str = line.split()[-2]  # 倒数第二列通常是 CUDA total time，例如 12.34us。
                num_str = line.split()[-1]  # 最后一列是调用次数。
                for unit, scale in units.items():  # 尝试识别时间单位并换算。
                    if unit in time_str:  # 当前行时间字符串包含该单位。
                        total_time += float(time_str.replace(unit, '')) / scale * int(num_str)  # total_time 累加成秒*次数。
                        total_num += int(num_str)  # 累计调用次数。
                        break  # 匹配到单位后无需继续尝试其他单位。
        kernel_durations.append(total_time / total_num if total_num > 0 else 0)  # 未找到目标 kernel 时返回 0。

    # 某些 benchmark 一次 loop 内按固定周期发多个 kernel；Chrome trace 保留了每次
    # kernel 的时间戳，可按 num_kernels_per_period 拆出各周期平均耗时。
    if num_kernels_per_period > 1:  # 一个 benchmark period 内有多个同名/相关 kernel 时，需要按时间序列拆分。
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:  # 创建临时 trace 文件。
            profiler.export_chrome_trace(tmp.name)  # 导出完整事件时间线。
            profile_data = json.loads(Path(tmp.name).read_text())  # 读回 JSON 结构用于逐事件解析。

        for i, kernel_name in enumerate(kernel_names):  # 对每个目标 kernel 单独拆分周期。
            events = [event for event in profile_data['traceEvents'] if f'::{kernel_name}' in event['name']]  # 过滤 CUDA kernel 事件。
            events = sorted(events, key=lambda event: event['ts'])  # 按 timestamp 排序，恢复实际 launch 顺序。
            durations = [event['dur'] / 1e6 for event in events]  # trace dur 单位为微秒，除以 1e6 转秒。
            assert len(durations) % num_kernels_per_period == 0  # 事件数必须能整除周期内 kernel 数。
            num_kernel_patterns = len(durations) // num_kernels_per_period  # 周期重复次数。
            kernel_durations[i] = [sum(durations[j::num_kernels_per_period]) / num_kernel_patterns for j in range(num_kernels_per_period)]  # 对每个周期位置分别求平均。

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]  # 输入是 tuple 时返回列表，否则返回单个 kernel 的秒级耗时。
