import argparse  # 解析测试脚本命令行参数，例如 token 数、hidden、rank 拓扑和性能开关。
import os  # 创建 profiler trace 输出目录时使用。
import torch  # 构造 CUDA tensor、执行 topk、同步 CUDA、启动多进程测试。
import torch.distributed as dist  # 初始化/销毁分布式进程组，并获取 rank/world_size。
from typing import Union, Tuple, Optional  # 给 helper 标注 tuple/tensor/可选类型。

import deep_ep  # 导入 ElasticBuffer、topk_idx_t 和底层 C++/CUDA 扩展。
from deep_ep.utils.math import (
    align, count_bytes, calc_diff,  # align 用于 expert padding；count_bytes/calc_diff 用于性能与误差报告。
    per_token_cast_back, per_token_cast_to_fp8,  # FP8 dispatch 测试需要 BF16<->FP8+scale 转换。
    safe_div  # 性能打印时避免除以 0。
)
from deep_ep.utils.gate import get_unbalanced_scores  # 构造可控不均衡 MoE gating 分数。
from deep_ep.utils.envs import init_dist, init_seed, dist_print  # 初始化分布式环境、固定随机种子和同步打印。
from deep_ep.utils.refs import dispatch as ref_dispatch  # PyTorch/NCCL 参考 dispatch，用于正确性对比。
from deep_ep.utils.refs import combine as ref_combine  # PyTorch/NCCL 参考 combine，用于正确性对比。
from deep_ep.utils.refs import generate_pre_combine_data, ordered_accumulate  # 构造 deterministic expert 输出和有序累加参考。
from deep_ep.utils.testing import bench_kineto  # 使用 PyTorch profiler 提取 kernel 时间。


# noinspection PyUnusedLocal,PyShadowingNames
def enumerate_ep_modes():
    # 覆盖 ElasticBuffer.dispatch/combine 的主要语义开关：
    # - do_handle_copy：handle 是否克隆 topk_idx，测试用户修改输入时的安全策略；
    # - expert_alignment：测试 expert token padding 对 GEMM 布局的影响；
    # - use_fp8_dispatch：测试 (x_fp8, scale) tuple 路径；
    # - num_bias：测试 combine epilogue 中 0/1/2 个 bias 的处理；
    # - previous_event/async/allocate_on_comm_stream：测试通信流与计算流 overlap 的事件语义。
    for do_handle_copy in (1, 0):  # 覆盖 handle 克隆和不克隆 topk_idx 两种生命周期策略。
        for expert_alignment in (128, 1):  # 覆盖 expert token 数按 128 对齐和不 padding 两种布局。
            for use_fp8_dispatch in (1, 0):  # 覆盖 FP8 dispatch 输入和 BF16 dispatch 输入。
                for num_bias in (0, 1, 2):  # 覆盖 combine 无 bias、单 bias、双 bias 三种 epilogue。
                    for with_previous_event in (0, 1):  # 覆盖是否把前序事件传给通信 API。
                        for async_with_compute_stream in (0, 1):  # 覆盖通信流是否异步等待计算流。
                            for allocate_on_comm_stream in ((1, ) if with_previous_event else (0, 1)):  # previous_event 下强制通信流分配，避免语义冲突。
                                # 返回一个完整测试模式 tuple，外层主循环逐个执行。
                                yield (do_handle_copy, expert_alignment, use_fp8_dispatch, num_bias,
                                       with_previous_event, async_with_compute_stream, allocate_on_comm_stream)


def launch(buffer: deep_ep.ElasticBuffer, name: str,
           with_previous_event: int, async_with_compute_stream: int,
           params: dict):
    if with_previous_event:  # 需要测试事件依赖时，先 capture 当前通信状态作为 previous_event。
        params.update(previous_event=buffer.capture())  # 把 previous_event 注入 dispatch/combine 参数。
    values = getattr(buffer, name)(**params)  # 动态调用 buffer.dispatch 或 buffer.combine，返回值最后一项是 event/handle。
    values[-1].current_stream_wait() if async_with_compute_stream else ()  # 异步模式下让当前流等待返回事件，模拟用户侧消费输出。
    return values  # 返回原 API tuple，调用方按具体 name 解包。


def fold_expanded(expanded: Union[Tuple[torch.Tensor], torch.Tensor],
                  indices: torch.Tensor, valid_mask: torch.Tensor):
    # expanded dispatch 会把一个 token 的不同 top-k 槽展开到不同位置。
    # fold_expanded 用 metadata 中的 indices 把展开布局折回普通布局，并检查同一 token
    # 的所有有效槽内容一致；这用于验证 do_expand=True 不改变实际 dispatch payload。
    if not isinstance(expanded, torch.Tensor):  # FP8 dispatch 返回 (x_fp8, sf) tuple，需要递归折叠每个 tensor。
        return tuple(fold_expanded(t, indices, valid_mask) for t in expanded)  # 保持 tuple 结构不变。

    gathered = expanded[indices]  # 根据 expanded metadata 把每个原 token 的 top-k 展开行收集回来，形状 [T,K,...]。
    first_valid_idx = valid_mask.to(torch.int).argmax(dim=1)  # 每个 token 找到第一个有效 top-k lane，用作代表值。
    folded = gathered[torch.arange(gathered.shape[0], device='cuda'), first_valid_idx]  # 取代表 lane 得到普通布局 [T,...]。
    result = (gathered == folded.unsqueeze(1)).all(dim=-1)  # 检查同一 token 的所有展开副本是否与代表值一致。
    result = result | (~valid_mask)  # 无效 lane 不参与一致性检查。
    assert result.all()  # 若 expanded 副本不一致，说明 dispatch expand/reduce metadata 有问题。
    return folded  # 返回折回普通 token 布局的 tensor。


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_dispatch_combine(buffer: deep_ep.ElasticBuffer, args: argparse.Namespace):
    # Settings
    num_scaleout_ranks, num_scaleup_ranks = buffer.get_logical_domain_size()  # 获取逻辑拓扑，hybrid 模式会拆成 scaleout x scaleup。
    num_max_tokens_per_rank, num_tokens, hidden = args.num_tokens, max(1, args.num_tokens - dist.get_rank()), args.hidden  # 每 rank token 数故意略不同，覆盖不均匀输入。
    num_topk, num_experts = args.num_topk, args.num_experts  # MoE 路由参数：每 token top-k 和全局 expert 数。
    num_local_experts = num_experts // buffer.num_ranks  # 每个最终 rank 管辖的 expert 数。
    num_sms = buffer.get_theoretical_num_sms(num_experts, num_topk) if args.num_sms == 0 else args.num_sms  # num_sms=0 时使用库内启发式。
    num_qps = buffer.get_theoretical_num_qps(num_sms) if args.num_qps == 0 else args.num_qps  # num_qps=0 时根据 SM 数推导 QP。
    dist_print(f'Config:\n'
               f' > Ranks: {num_scaleout_ranks} x {num_scaleup_ranks}\n'
               f' > Experts: {num_topk}/{num_experts}\n'
               # 中文：这里打印的是逻辑拓扑、MoE 路由规模和当前 rank 的有效 token 数。
               f' > Tokens: {num_tokens} (max: {num_max_tokens_per_rank}), hidden: {hidden}\n'
               f' > #SM: {num_sms}, #QPs: {num_qps}/{buffer.num_allocated_qps}\n',
               once_in_node=True)  # 每个节点只打印一次配置，避免多进程日志刷屏。

    # Construct expert selections first (may have an unbalanced ratio here)
    scores = get_unbalanced_scores(num_tokens, num_experts, buffer.num_ranks, num_topk, args.unbalanced_ratio, args.precise_unbalanced_ratio)  # 生成 [T,E] gating 分数。
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)  # 得到 [T,K] top-k 权重和 expert id。
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)  # 转成 DeepEP 编译时选择的 topk_idx_t，覆盖 32/64-bit 配置。
    if args.masked_ratio > 0:  # masked_ratio>0 时随机把部分 top-k lane 置为无效。
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)  # 与 topk_idx 同形状 [T,K] 的随机 mask。
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)  # expert id=-1 表示该 top-k lane 无效。
        topk_weights.masked_fill_(topk_idx < 0, 0)  # 无效 lane 的权重置 0，参考 combine 与 kernel 语义一致。

    # Run all tests
    dist_print('Running all test cases:', once_in_node=True)  # 开始遍历所有模式组合。
    for (do_handle_copy, expert_alignment, use_fp8_dispatch, num_bias,
         with_previous_event, async_with_compute_stream, allocate_on_comm_stream) in enumerate_ep_modes():  # 逐个测试模式 tuple。
        dist_print(f' > Testing with '
                   f'{do_handle_copy=}, {expert_alignment=}, {use_fp8_dispatch=}, {num_bias=}, '
                   f'{with_previous_event=}, {async_with_compute_stream=}, {allocate_on_comm_stream=} ...',
                   once_in_node=True)  # 打印当前模式，失败时能快速定位是哪组开关。

        # Random data
        # TODO: support top-k groups
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')  # 构造 BF16 输入 token，形状 [T,H]。
        x = per_token_cast_to_fp8(x) if use_fp8_dispatch else x  # FP8 模式把 x 转成 (x_fp8, scale) tuple。
        bias = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') if num_bias == 1 else None  # 单 bias 场景。
        if num_bias == 2:  # 双 bias 场景测试 combine epilogue 返回 tuple/bias 处理。
            bias = tuple(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') for _ in range(num_bias))  # 构造两个 [T,H] bias。
            assert len(bias) == 2   # To prevent linter warning；也确认 tuple 长度符合 num_bias。

        # Test correctness with NCCL reference
        # reference dispatch/combine 使用 torch.distributed all_to_all_single 和纯 PyTorch reduce。
        # DeepEP kernel 的 token 顺序可以不同，所以测试先比较 src_token_global_idx 的排序，
        # 再用 deterministic generate_pre_combine_data 构造 expert 输出，最终比较 combined_x。
        if not args.skip_check:  # 默认执行参考实现正确性检查；skip_check 仅用于纯性能压测。
            ref_recv_x, ref_recv_topk_idx, ref_recv_topk_weights, \
                ref_recv_src_token_idx, ref_num_recv_tokens_per_rank = \
                ref_dispatch(x, topk_idx, topk_weights, num_max_tokens_per_rank, num_experts)  # 参考 dispatch 返回 recv_x/topk/weight/src metadata。
            ref_recv_x_bf16 = per_token_cast_back(ref_recv_x[0], ref_recv_x[1]) if use_fp8_dispatch else ref_recv_x  # FP8 参考输出转回 BF16 便于比较。

            if args.allow_multiple_reduction:  # 允许多个 top-k 槽先在局部进行 reduction。
                # Should be the same as the trigger condition of DeepEP's hybrid combine, which performs intra-scaleup reduction first
                if args.allow_hybrid_mode and num_scaleout_ranks > 1:  # hybrid 多 scaleout 时先 scaleup 内 reduce，再 scaleout reduce。
                    reduced_combine_recipe = (True, True)  # expanded/reduced combine 参考路径使用 hybrid reduce 规则。
                    combine_recipe = (True, True)  # 普通 combine 也匹配 hybrid 触发条件。
                else:
                    reduced_combine_recipe = (True, False)  # 单 scaleout 或禁用 hybrid 时只做普通多重 reduction。
                    combine_recipe = (True, False)  # 普通 combine 与 reduced combine 使用同一参考规则。
            else:
                reduced_combine_recipe = (False, False)  # 不允许多重 reduction 时 expanded combine 不做提前合并。
                combine_recipe = (True, False)  # 普通 combine 参考仍按 rank 内归并后全局归并。
            ref_y = generate_pre_combine_data(
                dist.get_rank() * num_max_tokens_per_rank + torch.arange(num_tokens, device='cuda'),
                num_max_tokens_per_rank, num_topk, hidden)  # 构造 deterministic expert 输出，形状 [T,K,H]。
            ref_y[topk_idx == -1] = 0  # 无效 top-k lane 输出置零，避免参与 combine。
            ref_reduced_combined_y = ref_combine(
                ref_y, topk_idx,
                num_scaleout_ranks, num_scaleup_ranks, num_experts,
                # 中文：reduced_combine_recipe 对应 expanded/reduced combine 的规约层级。
                bias,
                *reduced_combine_recipe
            )  # reduced/expanded combine 的参考输出。
            ref_combined_y = ref_combine(
                ref_y, topk_idx,
                num_scaleout_ranks, num_scaleup_ranks,
                # 中文：普通 combine 参考路径先 rank 内规约，再做跨 rank 汇总。
                num_experts, bias,
                *combine_recipe
            )  # Reduce within rank, then globally, for non-expand combine mode
            torch.cuda.synchronize()  # 等参考实现 CUDA 工作完成，避免后续计时/比较混入异步执行。

        # Do dispatch
        dispatch_args = dict(  # DeepEP dispatch 的公共参数，后续普通/expanded/cached 模式复用。
            x=x, topk_idx=topk_idx, topk_weights=topk_weights,  # 输入 token、路由和权重。
            num_sms=num_sms, num_qps=num_qps,  # kernel SM 数和 RDMA/Gin QP 数。
            num_max_tokens_per_rank=num_max_tokens_per_rank, num_experts=num_experts,  # 容量和 expert 总数。
            expert_alignment=expert_alignment,  # expert token 对齐粒度。
            async_with_compute_stream=async_with_compute_stream,  # 是否异步 overlap。
            allocate_on_comm_stream=allocate_on_comm_stream,  # 输出 tensor 是否归属通信流。
            do_handle_copy=do_handle_copy, do_cpu_sync=args.do_cpu_sync)  # handle 克隆和 CPU sync 开关。
        recv_x, recv_topk_idx, recv_topk_weights, handle, dispatch_event = \
            launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, dispatch_args)  # 普通 dispatch，返回 recv tensor 和 handle。
        recv_x_bf16 = per_token_cast_back(recv_x[0], recv_x[1]) if use_fp8_dispatch else recv_x  # FP8 recv_x 转 BF16 以便构造 combine 输入。

        # Expanding mode
        expanded_dispatch_args = dispatch_args | dict(do_expand=True, use_tma_aligned_col_major_sf=True)  # expanded 模式按 expert/top-k 展开输出。
        expanded_recv_x, expanded_recv_topk_idx, expanded_recv_topk_weights, expanded_handle, expanded_dispatch_event = \
            launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, expanded_dispatch_args)  # 执行 expanded dispatch。
        expanded_recv_x_bf16 = per_token_cast_back(expanded_recv_x[0], expanded_recv_x[1]) if use_fp8_dispatch else expanded_recv_x  # 转回 BF16。

        # Cached mode
        cached_dispatch_args = dict(  # cached dispatch 复用第一次 dispatch 的 handle/slot/topk metadata。
            x=x,  # 输入 payload 可以变，但路由信息来自 handle。
            num_sms=num_sms, num_qps=num_qps,  # 保持与普通 dispatch 相同的 kernel 配置。
            async_with_compute_stream=async_with_compute_stream,  # 保持相同异步语义。
            allocate_on_comm_stream=allocate_on_comm_stream,  # 保持相同分配流策略。
            handle=handle)  # 复用普通 dispatch 返回的 EPHandle。
        cached_recv_x, cached_recv_topk_idx, cached_recv_topk_weights, cached_handle, cached_dispatch_event = \
            launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, cached_dispatch_args)  # 执行 cached dispatch。

        # Count the number of received tokens
        # psum_num_recv_tokens_per_scaleup_rank[-1] 是当前 rank dispatch 后真实收到的 token 数。
        # expanded layout 只改变 expert/top-k 槽位展开方式，不应改变 rank 级接收 token 总数。
        num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()  # 普通 dispatch 收到的 token 总数。
        assert num_recv_tokens == expanded_handle.psum_num_recv_tokens_per_scaleup_rank[-1].item(), \
               'Expand should not affect the number of received tokens.'  # expanded 不能改变 rank 级接收数。
        num_expanded_tokens = expanded_handle.psum_num_recv_tokens_per_expert[-1].item()  # expanded 后按 expert 展开的 token 行数。

        # Construction the input data for DeepEP combine
        src_token_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]  # 普通 dispatch metadata 第 0 列是源 token 全局 id。
        if not args.skip_check:  # 与参考 dispatch 比较源 token 集合。
            sorted_src_token_global_idx = torch.sort(src_token_global_idx).values  # DeepEP token 顺序可不同，排序后比较集合。
            assert torch.equal(ref_recv_src_token_idx, sorted_src_token_global_idx), \
                f'{ref_recv_src_token_idx=}, {sorted_src_token_global_idx=}'  # 源 token 不一致说明 dispatch 路由错误。
        local_y = generate_pre_combine_data(src_token_global_idx, num_max_tokens_per_rank, num_topk, hidden)  # [num_recv_tokens, topk, hidden] 的 deterministic expert 输出。
        local_y[recv_topk_idx[:num_recv_tokens] == -1] = 0  # 无效 top-k lane 输出置零。
        local_reduced_y = ordered_accumulate(local_y)  # 对一个 token 的 top-k 输出按确定顺序累加，构造普通 combine 输入。
        input_for_combine = torch.empty_like(recv_x_bf16, dtype=torch.bfloat16, device='cuda')  # 分配 combine 输入，形状与 recv_x 一致。
        input_for_combine[:num_recv_tokens] = local_reduced_y  # 只填真实 token 区间，padding 区不参与 combine。

        expanded_src_token_global_idx = expanded_handle.recv_src_metadata[:num_recv_tokens, 0]  # expanded metadata 仍按原 recv token 保存源 id。
        if not args.skip_check:  # expanded dispatch 也必须收到同一批源 token。
            sorted_expanded_src_token_global_idx = torch.sort(expanded_src_token_global_idx).values  # 排序后比较集合。
            assert torch.equal(ref_recv_src_token_idx, sorted_expanded_src_token_global_idx), \
                f'{ref_recv_src_token_idx=}, {sorted_expanded_src_token_global_idx=}'  # expanded 源 token 集合不一致说明 expand epilogue 错误。
        local_y_expand = generate_pre_combine_data(expanded_src_token_global_idx, num_max_tokens_per_rank, num_topk, hidden)  # [num_recv_tokens, topk, hidden]。
        # We put an extra row to conveniently handle the -1 index
        # 中文：expanded metadata 中无效 expert 槽会编码成 -1；多出的哨兵行让高级索引不越界。
        input_for_expand_combine = torch.empty((expanded_recv_x_bf16.shape[0] + 1, hidden), dtype=torch.bfloat16, device='cuda')  # 多一行让 -1 索引落到尾部占位。
        input_for_expand_combine[expanded_handle.recv_src_metadata[:num_recv_tokens, 2:].flatten()] = local_y_expand.view(-1, hidden)  # 按 expanded metadata 把每个 top-k 输出写到展开行。
        input_for_expand_combine = input_for_expand_combine[:-1, ...]  # 去掉处理 -1 的哨兵行，得到真实 expanded combine 输入。

        # Do combine
        combine_args = dict(  # 普通 combine 参数。
            x=input_for_combine, topk_weights=recv_topk_weights, bias=bias,
            handle=handle,
            # 中文：num_sms/num_qps 用同一套配置验证 dispatch 与 combine 的互逆关系。
            num_sms=num_sms, num_qps=num_qps,
            # 中文：下面两个 stream 选项覆盖同步/异步和 tensor stream 归属组合。
            async_with_compute_stream=async_with_compute_stream,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )  # 中文：普通 combine 参数组结束。
        combined_x, combined_topk_weights, combine_event = \
            launch(buffer, 'combine', with_previous_event, async_with_compute_stream, combine_args)  # 执行普通 combine。

        # Reduced combine
        reduced_combine_args = dict(  # expanded/reduced combine 参数。
            x=input_for_expand_combine, bias=bias,
            handle=expanded_handle,
            # 中文：expanded handle 不需要 topk_weights，因为输入已经按 top-k 槽展开。
            num_sms=num_sms, num_qps=num_qps,
            # 中文：reduced combine 也复用相同异步选项，确保两条路径行为一致。
            async_with_compute_stream=async_with_compute_stream,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )  # 中文：expanded/reduced combine 参数组结束。
        reduced_combined_x, reduced_combined_topk_weights, reduced_combine_event = \
            launch(buffer, 'combine', with_previous_event, async_with_compute_stream, reduced_combine_args)  # 执行 expanded handle 对应的 combine。

        assert not (args.dump_profile_traces and args.skip_perf_test), '`--skip-perf-test` should not be specified when `--dump-profile-traces` is provided'  # dump trace 需要实际跑 profiler。
        if not args.skip_perf_test:  # 性能测试开关，关闭时只跑正确性。
            # Profiling
            def get_trace_path(prefix: str):  # 根据 prefix 生成当前 rank 的 profiler trace 路径。
                return None if not args.dump_profile_traces else f'{args.dump_profile_traces}/{prefix}_rank{buffer.rank_idx}.json'  # 未指定目录时不导出 trace。

            # Calculate the number of tokens that are sent to the other scaleout peers
            dst_scaleout_rank_idx = topk_idx // (num_experts // num_scaleout_ranks)  # 将 expert id 映射到目标 scaleout rank，形状 [T,K]。
            num_scaleout_send_tokens = 0  # 统计跨 scaleout 发送 token 数，用于估算 SO 带宽。
            for i in range(num_scaleout_ranks if num_scaleout_ranks > 1 else 0):  # 单 scaleout 时没有 SO 流量，循环次数为 0。
                if args.ignore_local_traffic and i == dist.get_rank() // num_scaleup_ranks:  # 可选忽略本 scaleout 内流量。
                    continue  # 跳过本地 scaleout rank。
                num_scaleout_send_tokens += (dst_scaleout_rank_idx == i).any(dim=1).sum().item()  # 一个 token 任一 top-k 到 rank i 就计一次发送。

            # Calculate the number of tokens that are received via the other scaleup peers
            num_scaleup_recv_tokens = num_recv_tokens  # 默认所有收到 token 都计入 scale-up 接收流量。
            if args.ignore_local_traffic:  # 忽略同 scaleup rank 本地流量时，减去来源 scaleup rank 等于当前 rank 的 token。
                num_scaleup_recv_tokens -= (src_token_global_idx // num_max_tokens_per_rank % num_scaleup_ranks == dist.get_rank() % num_scaleup_ranks).sum().item()  # 根据源全局 token id 还原源 scaleup rank。

            # Test dispatch performance
            num_bytes_per_dispatch_token = safe_div(count_bytes(recv_x, recv_topk_idx, recv_topk_weights), recv_topk_idx.size(0))  # 估算单个 dispatch token payload 字节数。
            num_scaleup_bytes = num_bytes_per_dispatch_token * num_scaleup_recv_tokens  # Received via scaleup；SU 接收字节。
            num_scaleout_bytes = num_bytes_per_dispatch_token * num_scaleout_send_tokens    # Send via scaleout；SO 发送字节。
            t, copy_t = bench_kineto(lambda: buffer.dispatch(**dispatch_args),  # profile 普通 dispatch。
                                    kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),  # 分别提取主通信和 copy epilogue 时间。
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('dispatch'))  # barrier profiling 让多 rank 计时对齐。
            dist_print(f'   * EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'dispatch: '
                    # 中文：SO/SU 分别代表 scaleout 和 scaleup 通信带宽。
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'copy: {2 * num_recv_tokens * num_bytes_per_dispatch_token / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')  # 打印 SO/SU 带宽和 copy epilogue 带宽。

            # Test expanded dispatch performance
            num_bytes_per_dispatch_token_meta = safe_div(count_bytes(expanded_handle.recv_src_metadata), expanded_handle.recv_src_metadata.size(0))  # expanded copy 额外写 metadata。
            t, copy_t = bench_kineto(lambda: buffer.dispatch(**expanded_dispatch_args),  # profile expanded dispatch。
                                    kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),  # 主 kernel 和 epilogue 名称。
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('expanded_dispatch'))  # 可选导出 trace。
            dist_print(f'   - EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'expanded dispatch: '
                    # 中文：expanded dispatch 的 copy 带宽包含 metadata 写入和 expanded row 写入。
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'copy: {(num_recv_tokens * (num_bytes_per_dispatch_token_meta + num_bytes_per_dispatch_token) + num_expanded_tokens * num_bytes_per_dispatch_token) / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')  # expanded copy 包含 recv metadata 和 expanded rows。

            # Test cached dispatch performance
            t, copy_t = bench_kineto(lambda: buffer.dispatch(**cached_dispatch_args),  # profile cached dispatch。
                                    kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),  # 同样提取主通信和 epilogue。
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('cached_dispatch'))  # cached 模式复用 handle。
            dist_print(f'   # EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'cached dispatch: '
                    # 中文：cached dispatch 跳过 layout 重算，主要测 payload 搬运和 copy epilogue。
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'copy: {2 * num_scaleup_bytes / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')  # cached copy 主要衡量复用路由后的 payload 搬运。

            # Test combine performance
            num_bytes_per_combine_token = safe_div(count_bytes(recv_x_bf16, recv_topk_weights), recv_x_bf16.size(0))  # 估算 combine 单 token payload 字节。
            num_bias_bytes = count_bytes(bias)  # bias 读字节，可能为 0/单 tensor/tuple tensor。
            num_reduction_write_bytes = count_bytes(combined_x, combined_topk_weights)  # reduce epilogue 写回 combined 输出和 topk weights 的字节。

            def get_combine_bytes(is_expand_mode: bool) -> Tuple[float, float, float]:  # 估算 combine 的 SO/SU/reduce-read 三类字节数。
                num_experts_per_rank = num_experts // (num_scaleup_ranks * num_scaleout_ranks)  # 每个最终 rank 的 expert 数。
                num_experts_per_scaleout_rank = num_experts_per_rank * num_scaleup_ranks  # 每个 scaleout rank 覆盖的 expert 数。

                def get_unique_and_valid_dst_count(dst_idx: torch.Tensor,
                                                   ignored_nums_l: Optional[int] = None, ignored_nums_r: Optional[int] = None,
                                                   # 中文：max_num_in_dst_idx 用于过滤超过 expert/rank 范围的无效目标编号。
                                                   max_num_in_dst_idx: int = num_experts - 1) -> int:
                    """
                    Get the number of valid destinations, with deduplication within each token and numbers within `[ignored_nums_l, ignored_nums_r)` being ignored
                    中文：该 helper 用于估算 combine 路径里实际需要跨 scaleout/scaleup 传输的唯一目标数。

                    中文说明：
                        dst_idx 形状通常为 [num_tokens, num_topk]；函数先过滤 -1 和可选忽略区间，
                        再在单 token 内去重，用于估算 combine 通信字节数而不是校验数学结果。
                    """
                    dst_idx = dst_idx.clone()  # 复制一份，避免修改调用方的 topk_idx 派生 tensor。
                    ignore_mask = dst_idx == -1  # -1 是无效目标，不计入唯一目标数。
                    if args.ignore_local_traffic and ignored_nums_l is not None:  # 可选忽略某个本地目标区间。
                        assert ignored_nums_r is not None  # 有左边界时必须有右边界。
                        ignore_mask |= ((dst_idx >= ignored_nums_l) & (dst_idx < ignored_nums_r))  # 本地目标区间也置为忽略。
                    dst_idx = dst_idx + torch.arange(0, dst_idx.shape[0], dtype=dst_idx.dtype, device=dst_idx.device).unsqueeze(-1) * (max_num_in_dst_idx + 1)  # So that different rows will have different values；逐行加偏移实现“每 token 内去重”。
                    dst_idx[ignore_mask] = dst_idx[0][0].item()  # So that these `-1`s won't affect the count of unique numbers；忽略项都映射到同一个值。
                    return torch.unique(dst_idx, sorted=False).numel()  # 返回有效且按 token 去重后的目标数量。

                if not args.allow_multiple_reduction:  # 禁用多重 reduction 时，combine 可能需要转发所有 top-k 槽。
                    # No multiple reduction
                    if not is_expand_mode:  # 普通 combine，scale-up token 数与 dispatch recv token 数一致。
                        num_scaleup_tokens = num_scaleup_recv_tokens  # SU combine token 数。
                        num_scaleout_tokens = get_unique_and_valid_dst_count(
                            topk_idx // num_experts_per_rank, buffer.scaleout_rank_idx * num_scaleup_ranks, (buffer.scaleout_rank_idx + 1) * num_scaleup_ranks)  # 估算跨 scaleout 目标 rank 数。
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx // num_experts_per_rank)  # reduce epilogue 读取的 rank 目标数。
                    else:  # expanded combine，token 已按 top-k/expert 展开。
                        tokens_src_rank_idx = src_token_global_idx//num_max_tokens_per_rank  # 从源 token id 还原源 rank。
                        if args.ignore_local_traffic:  # 忽略本 scaleup rank 产生的本地 traffic。
                            num_scaleup_tokens = (recv_topk_idx[:num_recv_tokens] != -1)[tokens_src_rank_idx % num_scaleup_ranks != buffer.scaleup_rank_idx].sum().item()  # 只统计跨 scaleup 的有效 top-k 槽。
                        else:
                            num_scaleup_tokens = (recv_topk_idx[:num_recv_tokens] != -1).sum().item()  # 统计所有有效 top-k 槽。
                        num_scaleout_tokens = get_unique_and_valid_dst_count(
                            topk_idx, buffer.scaleout_rank_idx * num_experts_per_scaleout_rank, (buffer.scaleout_rank_idx + 1) * num_experts_per_scaleout_rank)  # SO token 按 expert 目标估算。
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx)  # expanded reduce 直接按 expert 去重。
                else:
                    # With `allow_multiple_reduction`, "combine" has exactly the same number of tokens as "dispatch"
                    num_scaleup_tokens = num_scaleup_recv_tokens  # 允许多重 reduction 时 SU token 数与 dispatch recv 数一致。
                    num_scaleout_tokens = num_scaleout_send_tokens  # SO token 数与 dispatch send 数一致。
                    if args.allow_hybrid_mode:  # hybrid 先按 scaleout 维度 reduce。
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx // num_experts_per_scaleout_rank)  # reduce 读取按 scaleout 去重。
                    else:
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx // num_experts_per_rank)  # 非 hybrid 按最终 rank 去重。
                if not args.ignore_local_traffic and num_scaleout_ranks == 1:  # 单 scaleout 没有跨 scaleout 流量。
                    num_scaleout_tokens = 0  # SO 字节置零。
                return num_scaleout_tokens * num_bytes_per_combine_token, num_scaleup_tokens * num_bytes_per_combine_token, num_reduction_read_tokens * num_bytes_per_combine_token  # 返回三类估算字节。

            num_scaleout_bytes, num_scaleup_bytes, num_reduction_read_bytes = get_combine_bytes(False)  # 普通 combine 字节估算。
            t, copy_t = bench_kineto(lambda: buffer.combine(**combine_args),  # profile 普通 combine。
                                    kernel_names=('combine_impl', 'combine_reduce_epilogue_impl'),  # 主 combine 与 reduce epilogue kernel。
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('combine'))  # 可选 trace。
            dist_print(f'   @ EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'combine: '
                    # 中文：combine 带宽分主通信和 reduce epilogue 两部分打印。
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'reduce: {(num_bias_bytes + num_reduction_read_bytes + num_reduction_write_bytes) / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')  # 打印 combine 与 reduce epilogue 带宽。

            # Test reduced combine performance
            num_scaleout_bytes, num_scaleup_bytes, num_reduction_read_bytes = get_combine_bytes(True)  # expanded/reduced combine 字节估算。
            t, copy_t = bench_kineto(lambda: buffer.combine(**reduced_combine_args),  # profile reduced combine。
                                    kernel_names=('combine_impl', 'combine_reduce_epilogue_impl'),  # 同样拆主 kernel 和 reduce epilogue。
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('reduced_combine'))  # 可选 trace。
            dist_print(f'   + EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'reduced combine: '
                    # 中文：reduced combine 的主通信字节来自 expanded/reduced layout 估算。
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'reduce: {(num_bias_bytes + num_reduction_read_bytes + num_reduction_write_bytes) / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')  # 打印 reduced combine 性能。
            dist_print(once_in_node=True)  # 打印空行分隔不同模式。

        # Checks
        # NOTES: we do checks after the performance tests, as we may modify some tensors
        if not args.skip_check:  # 只有未跳过正确性检查时执行下面所有断言。
            # Handle copy checks
            assert (topk_idx.data_ptr() != handle.topk_idx.data_ptr()) == do_handle_copy  # do_handle_copy=True 时 handle 必须拥有 topk_idx 副本。
            assert (topk_idx.data_ptr() != cached_handle.topk_idx.data_ptr()) == do_handle_copy  # cached handle 也应遵守相同复制策略。
            assert handle.topk_idx.data_ptr() == cached_handle.topk_idx.data_ptr()  # cached dispatch 应复用原 handle 中的 topk_idx。

            # Make the valid part of the whole tensor for no CPU sync mode
            if not args.do_cpu_sync:  # 无 CPU sync 时输出 tensor 可能按最大容量分配，需要裁剪真实 token 区间。
                if use_fp8_dispatch:  # FP8 recv_x 是 (data, scale) tuple。
                    recv_x = (recv_x[0][:num_recv_tokens], recv_x[1][:num_recv_tokens])  # 裁剪普通 dispatch 的 FP8 data/scale。
                    cached_recv_x = (cached_recv_x[0][:num_recv_tokens], cached_recv_x[1][:num_recv_tokens])  # 裁剪 cached dispatch 的 FP8 data/scale。
                else:
                    recv_x = recv_x[:num_recv_tokens]  # 裁剪 BF16 recv_x。
                    cached_recv_x = cached_recv_x[:num_recv_tokens]  # 裁剪 cached BF16 recv_x。
                recv_x_bf16 = recv_x_bf16[:num_recv_tokens]  # 裁剪用于比较的 BF16 视图。
                recv_topk_idx = recv_topk_idx[:num_recv_tokens]  # 裁剪 topk idx。
                recv_topk_weights = recv_topk_weights[:num_recv_tokens]  # 裁剪 topk weights。
                cached_recv_topk_idx = cached_recv_topk_idx[:num_recv_tokens]  # 裁剪 cached topk idx。
                handle.recv_src_metadata = handle.recv_src_metadata[:num_recv_tokens]  # 裁剪普通 metadata。
                expanded_handle.recv_src_metadata = expanded_handle.recv_src_metadata[:num_recv_tokens]  # 裁剪 expanded metadata。

            # Make sure deterministic mode works by doing the dispatch twice
            if args.deterministic:  # deterministic 模式要求同输入多次 dispatch 的 slot/metadata 一致。
                recv_x_twice, recv_topk_idx_twice, recv_topk_weights_twice, handle_twice, dispatch_event_twice = \
                    launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, dispatch_args)  # 第二次 dispatch 同样参数。
                if not args.do_cpu_sync:  # 无 CPU sync 时仍需裁剪第二次 metadata。
                    assert num_recv_tokens == handle_twice.psum_num_recv_tokens_per_scaleup_rank[-1].item()  # 两次真实接收数必须相同。
                    handle_twice.recv_src_metadata = handle_twice.recv_src_metadata[:num_recv_tokens]  # 裁剪第二次 metadata。
                assert torch.equal(handle.recv_src_metadata[:, :2], handle_twice.recv_src_metadata[:, :2])  # 前两列源 token/slot metadata 必须完全一致。

            # Test cumulative stats counter
            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')  # 本地 expert 真实接收计数。
            dispatch_args['cumulative_local_expert_recv_stats'] = cumulative_local_expert_recv_stats  # 把统计 tensor 传给 dispatch。
            launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, dispatch_args)  # 再跑一次 dispatch 以写统计计数。

            # Expanded checks
            assert expanded_recv_topk_idx is None  # expanded 模式不返回普通 [T,K] topk_idx，而是通过 metadata 表达展开关系。
            assert expanded_handle.recv_src_metadata.size(0) == num_recv_tokens  # expanded metadata 行数仍等于接收 token 数。
            expanded_indices = expanded_handle.recv_src_metadata[:, 2:]  # 第 2.. 列是每个 top-k lane 的 expanded 行号。
            expanded_mask = expanded_indices >= 0  # 有效 expanded 行号 mask。
            expanded_safe_indices = expanded_indices.clone()  # 拷贝索引以便替换无效 -1。
            expanded_safe_indices[~expanded_mask] = 0  # 无效 lane 暂时指向 0，后续用 mask 忽略。
            expanded_recv_x = fold_expanded(expanded_recv_x, expanded_safe_indices, expanded_mask)  # 把 expanded recv_x 折回普通布局。
            expanded_recv_topk_weights = expanded_recv_topk_weights[expanded_safe_indices]  # 按 expanded 行号取回对应权重。

            # Cached checks
            if use_fp8_dispatch:  # FP8 cached dispatch 需要 data 和 scale 都相同。
                assert torch.equal(recv_x[0], cached_recv_x[0])  # FP8 数据相同。
                assert torch.equal(recv_x[1], cached_recv_x[1])  # FP8 scale 相同。
            else:
                assert torch.equal(recv_x, cached_recv_x)  # BF16 cached dispatch payload 相同。
            assert torch.equal(recv_topk_idx, cached_recv_topk_idx)  # cached topk idx 必须相同。
            assert torch.equal(handle.dst_buffer_slot_idx, cached_handle.dst_buffer_slot_idx)  # cached slot 索引必须复用。
            assert torch.equal(handle.psum_num_recv_tokens_per_scaleup_rank, cached_handle.psum_num_recv_tokens_per_scaleup_rank)  # rank 前缀和必须一致。
            assert handle.num_recv_tokens_per_expert_list == cached_handle.num_recv_tokens_per_expert_list  # CPU sync expert count 列表必须一致。

            # Check dispatch expert count
            assert recv_x_bf16.size() == ref_recv_x_bf16.size(), f'{recv_x_bf16.size()=}, {ref_recv_x_bf16.size()=}'  # recv_x 形状与参考一致。
            assert recv_x_bf16.size(0) == num_recv_tokens  # 第一维等于真实接收 token 数。
            for i in range(num_local_experts if args.do_cpu_sync else 0):  # CPU sync 模式下才有 host expert count 列表。
                ref_count = (ref_recv_topk_idx == i).sum().item()  # 参考中本地 expert i 的真实 token 数。
                aligned_ref_count = align(ref_count, expert_alignment)  # 按 expert_alignment 对齐后的容量。
                assert ref_count == cumulative_local_expert_recv_stats[i].item(),\
                    f'{i}, {ref_count}, {cumulative_local_expert_recv_stats[i].item()}'  # stats counter 必须记录真实未对齐计数。
                assert aligned_ref_count == handle.num_recv_tokens_per_expert_list[i]  # handle 列表保存对齐后计数。
            psum_num_recv_tokens_per_expert_list = [0] + handle.psum_num_recv_tokens_per_expert.tolist()  # 普通 expert prefix 补 0 起点。
            expanded_psum_num_recv_tokens_per_expert_list = [0] + expanded_handle.psum_num_recv_tokens_per_expert.tolist()  # expanded expert prefix 补 0 起点。
            for i in range(num_local_experts):  # 遍历本地 expert。
                ref_count = (ref_recv_topk_idx == i).sum().item()  # 参考真实 token 数。
                count = psum_num_recv_tokens_per_expert_list[i + 1] - psum_num_recv_tokens_per_expert_list[i]  # 普通 prefix 差值得到对齐计数。
                expanded_count = (expanded_psum_num_recv_tokens_per_expert_list[i + 1] -
                                  align(expanded_psum_num_recv_tokens_per_expert_list[i], expert_alignment))  # expanded prefix 需要扣掉前一个 expert 的 padding。
                assert align(ref_count, expert_alignment) == count, f'{buffer.rank_idx=}, {i=}, {ref_count=}, {count=}'  # 普通计数应等于对齐后的参考。
                assert ref_count == expanded_count, f'{ref_count=}, {expanded_count=}'  # expanded 计数应等于未对齐真实参考。

            # Check dispatch scale-up received token psum
            psum_num_recv_tokens_per_scaleup_rank_list = [0] + handle.psum_num_recv_tokens_per_scaleup_rank.tolist()  # rank prefix 补 0 起点。
            for i in range(num_scaleup_ranks):  # 遍历节点内 scale-up rank。
                count = psum_num_recv_tokens_per_scaleup_rank_list[i + 1] - psum_num_recv_tokens_per_scaleup_rank_list[i]  # prefix 差值得到该 scaleup rank 接收数。
                ref_count = sum(ref_num_recv_tokens_per_rank[i::num_scaleup_ranks])  # 参考按全局 rank 间隔聚合到同一 scaleup rank。
                assert count == ref_count, f'{ref_count=}, {count=}'  # DeepEP rank prefix 必须与参考一致。

            # Check dispatch data
            for check_recv_x, check_recv_topk_idx, check_recv_topk_weights, check_handle in (  # 分别校验 expanded 折叠后和普通 dispatch 输出。
                (expanded_recv_x, None, expanded_recv_topk_weights, expanded_handle),  # Expanded
                (recv_x, recv_topk_idx, recv_topk_weights, handle),  # Unexpanded
            ):
                for i in range(buffer.num_ranks):  # 按全局源 rank 分段比较数据。
                    rank_start_idx = sum(ref_num_recv_tokens_per_rank[:i])  # 参考输出中 rank i 的起始位置。
                    rank_end_idx = rank_start_idx + ref_num_recv_tokens_per_rank[i]  # 参考输出中 rank i 的结束位置。
                    sorted_metadata = torch.sort(check_handle.recv_src_metadata[:, 0])  # 按源 token id 排序 DeepEP 输出。
                    sorted_indices = sorted_metadata.indices[rank_start_idx:rank_end_idx]  # DeepEP 中对应 rank i 的排序后索引。
                    sorted_values = sorted_metadata.values[rank_start_idx:rank_end_idx]  # DeepEP 中对应 rank i 的源 token id。
                    assert torch.equal(ref_recv_src_token_idx[rank_start_idx:rank_end_idx], sorted_values)  # 源 token 顺序集合必须匹配参考。

                    # Data should be bitwise identical
                    check_list = [(ref_recv_topk_weights, check_recv_topk_weights, True)]  # topk_weights 对无效 lane 需要 mask 后比较。
                    if check_recv_topk_idx is not None:  # 普通 dispatch 有 recv_topk_idx，expanded 折叠路径没有。
                        check_list.append((ref_recv_topk_idx, check_recv_topk_idx, False))  # topk_idx bitwise 比较。
                    if use_fp8_dispatch:  # FP8 输出需要分别比较 data 和 scale。
                        check_list.append((ref_recv_x[0], check_recv_x[0], False))  # FP8 data 比较。
                        check_list.append((ref_recv_x[1], check_recv_x[1], False))  # FP8 scale 比较。
                    else:
                        check_list.append((ref_recv_x, check_recv_x, False))  # BF16 recv_x 比较。
                    ref_mask = ref_recv_topk_idx[rank_start_idx:rank_end_idx] < 0  # 无效 top-k lane mask。
                    for ref_t, t, do_mask in check_list:  # 逐个 tensor 比较参考和 DeepEP 输出。
                        ref_t = ref_t[rank_start_idx:rank_end_idx]  # 取参考 rank i 分段。
                        t = t[sorted_indices]  # 按源 token 排序后的 DeepEP 分段。
                        if do_mask:  # weight tensor 对无效 lane 置零后比较。
                            ref_t = ref_t.masked_fill(ref_mask, 0)  # 参考无效 weight 置零。
                            t = t.masked_fill(ref_mask, 0)  # DeepEP 无效 weight 置零。
                        assert torch.equal(ref_t, t), f'{ref_t=}, {t=}'  # 要求 bitwise 一致。

            # Combined data should also be bitwise-identical
            assert torch.equal(combined_x, ref_combined_y), \
                f'Diff: {calc_diff(combined_x, ref_combined_y)}'  # 普通 combine 输出必须与参考 bitwise 一致。
            assert torch.equal(reduced_combined_x, ref_reduced_combined_y), \
                f'Diff: {calc_diff(reduced_combined_x, ref_reduced_combined_y)}'  # reduced/expanded combine 输出必须与参考一致。
            assert torch.equal(combined_topk_weights, topk_weights), \
                f'{calc_diff(combined_topk_weights, topk_weights)}'  # combine 返回的 topk_weights 应保持原始权重。

        # Break on the first test case
        if args.test_first_only:  # 调试模式只跑第一个组合，缩短迭代时间。
            break  # 跳出模式循环。
    dist_print('', once_in_node=True)  # 所有模式结束后打印空行分隔日志。


# noinspection PyUnboundLocalVariable,PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):  # torch.multiprocessing.spawn 的每进程入口。
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)  # 初始化分布式环境并返回 rank/world/group。
    def construct_elastic_buffer():  # 构造 ElasticBuffer，压力测试中可能反复销毁重建。
        return deep_ep.ElasticBuffer(group,  # 使用当前进程所属 NCCL ProcessGroup。
                                     num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,  # buffer 容量按最大 token 和 hidden 维度分配。
                                     deterministic=args.deterministic,  # 是否启用 deterministic slot 分配。
                                     allow_hybrid_mode=args.allow_hybrid_mode,  # 是否允许 scaleout+scaleup hybrid kernel。
                                     allow_multiple_reduction=args.allow_multiple_reduction,  # 是否允许 combine 本地多重 reduction。
                                     prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),  # 是否偏向通信/计算 overlap。
                                     sl_idx=args.sl_idx,  # NCCL/Gin service level index。
                                     num_allocated_qps=max(args.num_allocated_qps, args.num_qps),  # 实际分配 QP 数至少覆盖请求 QP 数。
                                     explicitly_destroy=True,  # 测试中手动 destroy，避免析构顺序影响进程退出。
                                     num_gpu_timeout_secs=args.num_gpu_timeout_secs,  # GPU 侧通信等待超时。
                                     num_cpu_timeout_secs=args.num_cpu_timeout_secs)  # CPU 侧同步等待超时。

    buffer = construct_elastic_buffer()  # 创建本进程 ElasticBuffer。

    # Warning in case of precise unbalanced ratio
    if args.precise_unbalanced_ratio:  # precise 模式人工构造 top-k 分布，提醒它与真实 gating 随机性不同。
        dist_print('\033[33mWarning: Using precise unbalanced ratio mode. '
                   'Test data is manually constructed and may differ from real world distribution.\033[0m',
                   once_in_node=True)  # 每节点只打印一次 warning。

    # Test MoE kernels
    test_dispatch_combine(buffer, args)  # 执行一次完整 dispatch/combine 正确性与性能测试。

    # Pressure tests
    for seed in range(int(1e9) if args.do_pressure_test else 0):  # 压力测试模式下用大量 seed 持续跑。
        if not args.reuse_elastic_buffer:  # 如果不复用 buffer，则每轮重建以覆盖初始化/销毁路径。
            # Recreate elastic buffer
            buffer.destroy()  # 显式释放旧通信资源。
            buffer = construct_elastic_buffer()  # 重新创建 ElasticBuffer。

        assert not args.skip_check  # 压力测试必须保留正确性检查，否则无法发现随机种子触发的问题。
        dist_print(f'Testing with {seed=} ...', once_in_node=True)  # 打印当前压力测试 seed。
        init_seed(seed)  # 使用当前 seed 重置随机数。
        test_dispatch_combine(buffer, args)  # 用该 seed 再跑完整测试。

    # Destroy the runtime and communication group
    buffer.destroy()  # 显式销毁 ElasticBuffer 内部 NCCL/Gin/JIT 资源。
    dist.destroy_process_group()  # 销毁 PyTorch 分布式进程组。


if __name__ == '__main__':  # 支持直接以 python tests/elastic/test_ep.py 运行。
    parser = argparse.ArgumentParser(description='Test elastic EP kernels')  # 创建命令行参数解析器。

    # Resource settings
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')  # 本机启动的 rank 数；通常等于本机 GPU 数，world_size 由 torchrun/spawn 组合决定。
    parser.add_argument('--num-sms', type=int, default=0, help='Number of SMs to use (0 means auto)')  # 通信 kernel 占用的 SM 数；0 表示 Buffer 内部按 GPU/模式自动选择。
    parser.add_argument('--num-qps', type=int, default=0, help='Number of QPs to use (0 means auto)')  # 实际参与传输的 RDMA/NVSHMEM QP 数；影响并行网络通道数。
    parser.add_argument('--num-allocated-qps', type=int, default=0, help='Number of QPs to allocate (0 means auto)')  # 预分配 QP 容量；可大于 num_qps，便于测试运行期选择较少 QP。
    parser.add_argument('--num-gpu-timeout-secs', type=int, default=100, help='Timeout in seconds (GPU side)')  # GPU kernel 内部等待信号/计数器的超时时间，避免通信异常时无限自旋。
    parser.add_argument('--num-cpu-timeout-secs', type=int, default=100, help='Timeout in seconds (CPU side)')  # Python/CPU 同步等待超时，覆盖 init、sync、destroy 等 host 侧流程。
    parser.add_argument('--sl-idx', type=int, default=0, help='SL index')  # NCCL/Gin service level index，用于选择底层网络服务等级。

    # Model settings
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens')  # 每 rank 输入 token 数，x 的第 0 维为 [num_tokens, hidden]。
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size')  # token hidden 维度，决定每个 token 需要搬运的字节数。
    parser.add_argument('--num-topk', type=int, default=6, help='Number of top-k experts')  # 每个 token 路由到的专家数，topk_idx/topk_weights 形状为 [num_tokens, num_topk]。
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')  # 全局专家总数，dispatch 会按 expert_id 映射到目标 rank 和本地 expert。

    # Scenario settings
    parser.add_argument('--do-cpu-sync', type=int, default=1, help='Whether to do CPU sync')  # 是否在 dispatch/combine 间插入 CPU 同步；关闭后更贴近异步流水。
    parser.add_argument('--allow-hybrid-mode', type=int, default=1, help='Whether to allow hybrid mode')  # 允许跨节点 scaleout 与节点内 scaleup 混合 kernel。
    parser.add_argument('--allow-multiple-reduction', type=int, default=1, help='Whether to allow multiple reductions')  # combine 是否允许同一 token 多路贡献在本地做 reduction。
    parser.add_argument('--prefer-overlap-with-compute', type=int, default=0, help='Whether to prefer overlap with compute')  # 选择更偏向通信/计算 overlap 的参数组合，例如更多异步等待。
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithm')  # 使用确定性分配策略，便于复现 slot 顺序和比对问题。

    # Test settings
    parser.add_argument('--seed', type=int, default=0, help='Default seed for pressure tests')  # 初始随机种子；压力测试会不断覆盖为循环 seed。
    parser.add_argument('--skip-check', action='store_true', help='Whether to skip correctness checks')  # 跳过 correctness check，仅保留性能/冒烟路径。
    parser.add_argument('--skip-perf-test', action='store_true', help='Whether to skip performance tests')  # 跳过 benchmark/profiler，适合只验证功能正确性。
    parser.add_argument('--do-pressure-test', action='store_true', help='Whether to do pressure test')  # 连续运行大量随机 seed，覆盖稀有路由分布和资源重建路径。
    parser.add_argument('--reuse-elastic-buffer', action='store_true', help='Whether to reuse elastic buffer for each test')  # 压力测试中复用 ElasticBuffer；关闭则每轮测试 destroy/recreate。
    parser.add_argument('--test-first-only', action='store_true', help='Only test the first case')  # 只跑第一个模式组合，便于缩小 debug 面。
    parser.add_argument('--unbalanced-ratio', type=float, default=1.0, help='The MoE unbalanced ratio')  # 控制 expert 负载倾斜程度；越大越容易压测热点 expert。
    parser.add_argument('--precise-unbalanced-ratio', action='store_true', help='Generate topk index with precise unbalanced ratio')  # 人工构造精确倾斜 top-k 分布，用来复现特定负载比例。
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')  # 将部分 top-k entry 标为无效，测试 mask 与 padded slot 的处理。
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')  # profiler trace 输出目录，非空时会保存 JSON 供 Chrome trace 打开。
    parser.add_argument('--ignore-local-traffic', action='store_true', help='Whether to ignore local traffic during bandwidth calculation')  # 统计带宽时忽略本地 rank 间流量，更关注跨节点网络。
    args = parser.parse_args()  # 解析命令行参数。

    # Create dump trace directories
    if args.dump_profile_traces:  # 用户指定 profiler trace 输出目录时创建它。
        os.makedirs(args.dump_profile_traces, exist_ok=True)  # exist_ok=True 允许目录已存在。

    # Launch test processes
    num_processes = args.num_processes  # 要启动的本机进程数。
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)  # 每个进程绑定一个 local_rank 执行 test_loop。
