import argparse  # 解析 legacy intranode 测试参数。
import time  # layout benchmark 后短暂 sleep，避免多进程日志挤在一起。
import torch  # 构造 CUDA Tensor、spawn 多进程。
import torch.distributed as dist  # all_reduce/barrier/destroy process group。

# noinspection PyUnresolvedReferences
import deep_ep  # legacy Buffer、Config、topk_idx_t 顶层 API。
from deep_ep.utils.envs import init_dist  # 初始化 NCCL 分布式环境。
from deep_ep.utils.math import calc_diff, inplace_unique, per_token_cast_to_fp8, per_token_cast_back  # 数值比较、rank 去重、FP8 转换。
from deep_ep.utils.testing import bench  # CUDA event benchmark helper。

# Test compatibility with low latency functions
import test_low_latency  # 可选复用低延迟测试，验证同一个 Buffer 与 LL API 兼容。


# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, num_sms: int, local_rank: int, num_ranks: int, rank: int, buffer: deep_ep.Buffer,
              group: dist.ProcessGroup):  # legacy 单节点 dispatch/combine 正确性与调优测试。
    # Settings：legacy intranode 只覆盖单节点 num_ranks。x 用 rank 常数填充，
    # 后续 check_data 可通过每行 min/max 判断 token 是否来自预期 rank。
    num_tokens, hidden = args.num_tokens, args.hidden  # token 数 T 与 hidden 维度 H。
    num_topk, num_experts = args.num_topk, args.num_experts  # top-k 宽度 K 与 expert 总数 E。

    assert num_experts % num_ranks == 0  # legacy intranode 假设 expert 均匀分布到 rank。
    if local_rank == 0:  # 只让本地 rank0 打印配置。
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}', flush=True)  # 输出 T/H/K。

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank  # 可校验输入 [T,H]，每行值都等于源 rank。
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')  # 随机输入 [T,H]，用于数值 diff 测试。
    x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None  # SM90 编译可用时测试 FP8 dispatch。
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None  # 保证 scale-factor 的 stride/layout 符合 kernel 需求。
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1  # gate 分数 [T,E]。
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]  # 每 token 选出的 expert id [T,K]。
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)  # 转成 C++ kernel 期望 dtype。
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank  # 可校验权重 [T,K]，值编码源 rank。
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')  # 随机权重用于数值 diff。
    rank_idx = topk_idx // (num_experts // num_ranks)  # 将 expert id 映射成目标 rank id，形状 [T,K]。
    rank_idx = rank_idx.to(torch.int64)  # inplace_unique 期望可索引整数类型。
    rank_idx.masked_fill_(topk_idx == -1, -1)  # 无效 expert lane 对应 rank 也设为 -1。
    inplace_unique(rank_idx, num_ranks)  # 每个 token 对目标 rank 去重，得到“是否发送到 rank”的紧凑表示。

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')  # 当前 rank 发送给每个 expert 的 token 计数 [E]。
    for i in range(num_experts):  # 逐 expert 统计命中次数。
        num_tokens_per_expert[i] = (topk_idx == i).sum()  # 当前 rank 中 expert i 被选中的次数。
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()  # 全局计数副本。
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)  # 聚合所有 rank 的 per-expert token 计数。

    # Rank layout meta：token_idx_in_rank 最终转成 [num_tokens, num_ranks]，
    # 表示每个 token 发往某 rank 时在该 rank 接收序列中的位置，-1 表示不发送。
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda')  # 当前 rank 需要发送到各目标 rank 的 token 数 [R]。
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')  # 临时矩阵 [R,T]，记录 token 在目标 rank 的局部序号。
    for i in range(num_ranks):  # 逐目标 rank 构造布局。
        num_tokens_per_rank[i] = (rank_idx == i).sum()  # 统计 top-k lane 命中 rank i 的次数。
        token_sel = (rank_idx == i).max(dim=-1)[0]  # 每个 token 是否需要发往 rank i，形状 [T]。
        count = token_sel.sum().item()  # 发往 rank i 的 token 数。
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]  # True token 排到前面，得到 token id 排列。
        tokens[:count] = torch.sort(tokens[:count])[0]  # 有效 token id 再按升序排列，匹配 kernel 的稳定顺序。
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')  # 写入 token 在目标 rank 的局部位置。
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)  # 转成 [T,R]，与 kernel 输出 is_token_in_rank 对齐。
    is_token_in_rank = token_idx_in_rank >= 0  # 布尔矩阵 [T,R]，表示 token 是否发往目标 rank。
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()  # 全局接收计数副本。
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)  # 聚合所有源 rank 后，gbl_num_tokens_per_rank[rank] 是当前 rank 接收总数。

    ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)  # 调用 C++/CUDA layout kernel，返回 rank/expert/token-in-rank 元数据。
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)  # kernel rank 计数必须匹配手工参考。
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)  # kernel expert 计数必须匹配手工参考。
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)  # kernel token->rank 布尔矩阵必须匹配。
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]  # 单独 benchmark layout kernel。
    if local_rank == 0:  # 只让本地 rank0 打印性能。
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)  # layout 耗时转成 ms。
        print('', flush=True)  # 空行分隔 layout 与 dispatch 日志。
    group.barrier()  # 所有 rank 等待日志输出和 layout 测试完成。
    time.sleep(1)  # 给多进程 stdout 一点缓冲时间，减少日志交错。

    # Config
    nvl_buffer_size = 256  # legacy intranode NVLink buffer token 容量 hint。
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)  # 默认测试配置：num_sms、NVL chunk、NVL buffer size。

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, rank_prefix_matrix):
        # rank_prefix_matrix [src_rank, dst_rank] 是前缀和；对当前 dst=rank，
        # check_x 中来自 src i 的行区间是 [prev_prefix, prefix_i)。
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))  # 每行所有 hidden 都应等于同一个源 rank 编码值。
        check_start = 0  # 当前源 rank 区间起点。
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()  # 源 rank i 在当前目标 rank 的区间终点。
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0  # 该区间必须全部来自 rank i。
            check_start = check_end  # 下一源 rank 区间从这里开始。

    for previous_mode in (False, True):
        for async_mode in (False, True):  # 覆盖同步返回和异步 EventOverlap 返回。
            for current_x in filter(lambda elem: elem is not None, (x_pure_rand, x, x_e4m3)):  # 依次测试随机 BF16、rank 编码 BF16、可选 FP8。
                for with_topk in (False, True):  # 分别测试只搬 payload 与同时搬 top-k metadata。
                    if local_rank == 0:  # rank0 打印当前 case。
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            # 中文：intranode 测试同样覆盖 dtype、top-k metadata、异步和 previous_event 组合。
                            flush=True,
                            end='')
                    dispatch_args = {  # 首次 dispatch 使用显式 layout 元数据。
                        'x': current_x,  # 输入 payload，可能是 BF16 tensor 或 FP8(data,scale) tuple。
                        'num_tokens_per_rank': num_tokens_per_rank,  # 每目标 rank token 数 [R]。
                        'is_token_in_rank': is_token_in_rank,  # token->rank 布尔矩阵 [T,R]。
                        'num_tokens_per_expert': num_tokens_per_expert,  # 每 expert token 数 [E]。
                        'config': config,  # legacy kernel 调优配置。
                        'async_finish': async_mode  # 是否异步返回 event。
                    }
                    if with_topk:  # 带 metadata 时 dispatch 会返回 recv_topk_idx/recv_topk_weights。
                        dispatch_args.update({
                            'topk_idx': topk_idx,  # 原始 top-k expert id [T,K]。
                            'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights  # 随机 payload 搭配随机权重，否则用 rank 编码权重。
                        })
                    if previous_mode:  # previous_event 覆盖“通信等待前置事件”的路径。
                        dispatch_args.update({'previous_event': buffer.capture()})  # 捕获当前 stream event 作为依赖。
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(
                        **dispatch_args)  # 执行 legacy intranode dispatch。
                    event.current_stream_wait() if async_mode else ()  # async 模式下让当前 stream 等待通信 event。
                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x  # FP8 接收结果反量化回 BF16 便于检查。

                    # Checks
                    rank_prefix_matrix = handle[0]  # handle[0] 是 [src_rank,dst_rank] 前缀矩阵。
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                        0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list  # 当前 rank 本地 experts 接收计数必须匹配全局参考。
                    if current_x is not x_pure_rand:  # rank 编码输入可直接检查来源 rank。
                        check_data(recv_x, rank_prefix_matrix)  # 验证每个 prefix 区间的 payload 都等于源 rank。
                    recv_topk_weights_clone = None  # num_worst_tokens 分支需要保留未修改的权重副本。
                    if with_topk:
                        # recv_topk_idx 是本 rank 内局部 expert id，范围
                        # [0, num_experts/num_ranks)，无效 lane 为 -1；每个 expert 的计数
                        # 应等于 recv_num_tokens_per_expert_list。
                        assert (recv_topk_idx.eq(-1) |
                                ((recv_topk_idx >= 0) &
                                 (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):  # 遍历当前 rank 本地 expert。
                            assert recv_topk_idx.eq(i).sum().item() == count  # topk_idx 中 expert i 的出现次数必须等于计数。

                        # Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()  # 后续会原地填充无效 lane，因此先 clone。
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(
                                dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]  # 无效 lane 填成本行最大值，避免 -1 lane 干扰 rank 编码检查。
                            check_data(recv_topk_weights, rank_prefix_matrix)  # 权重也应按源 rank 分段。

                    # Test `num_worst_tokens != 0`
                    if with_topk:
                        num_worst_tokens = num_tokens * num_ranks  # 固定容量 dispatch 的最坏接收 token 数。
                        dispatch_args.update({'num_worst_tokens': num_worst_tokens})  # 启用无 CPU sync 的固定容量布局。
                        recv_worst_x, recv_worst_topk_idx, recv_worst_topk_weights, empty_list, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()  # 异步模式等待固定容量 dispatch 完成。
                        recv_worst_x = per_token_cast_back(*recv_worst_x) if isinstance(recv_worst_x, tuple) else recv_worst_x  # FP8 输出反量化。
                        assert len(empty_list) == 0  # 固定容量模式不返回 per-expert Python list。
                        assert num_worst_tokens == recv_worst_x.size(0)  # 接收 x 第一维等于最坏容量。
                        assert num_worst_tokens == recv_worst_topk_idx.size(0)  # topk_idx 同样扩展到最坏容量。
                        assert num_worst_tokens == recv_worst_topk_weights.size(0)  # topk_weights 同样扩展。
                        assert torch.equal(recv_x, recv_worst_x[:recv_x.size(0)])  # 有效前缀必须与普通 dispatch 一致。
                        assert torch.equal(recv_topk_idx, recv_worst_topk_idx[:recv_x.size(0)])  # 有效 topk_idx 前缀一致。
                        assert torch.equal(recv_topk_weights_clone, recv_worst_topk_weights[:recv_x.size(0)])  # 有效权重前缀一致。
                        assert torch.all(recv_worst_topk_idx[recv_x.size(0):] == -1).item()  # padding 区域 expert id 必须为 -1。

                    # cached dispatch 复用 handle 中的 rank_prefix_matrix/channel layout，
                    # 因此不能重新传 topk_idx/topk_weights。
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}  # cached dispatch 只传 payload 和 handle。
                        if previous_mode:  # 覆盖 previous_event 等待路径。
                            dispatch_args.update({'previous_event': buffer.capture()})  # 当前 stream 捕获事件作为通信前置依赖。
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()  # async cached dispatch 等待 event。
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x  # FP8 cached 输出反量化。
                        if current_x is not x_pure_rand:  # rank 编码输入可校验来源。
                            check_data(recv_x, rank_prefix_matrix)  # cached dispatch 必须与首次 dispatch 路由一致。

                    # Test combine
                    combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode}  # combine 使用 dispatch handle 反向路由。
                    if with_topk:  # 带 top-k 时 combine 也回传权重。
                        combine_args.update({'topk_weights': recv_topk_weights})  # 使用 dispatch 接收端权重作为专家侧输入权重。
                    if previous_mode:  # combine 也测试 previous_event 依赖。
                        combine_args.update({'previous_event': buffer.capture()})  # 捕获当前 stream 事件。
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)  # 执行反向 combine。
                    event.current_stream_wait() if async_mode else ()  # async combine 等待事件。
                    check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)  # combine 是无权重求和，除以贡献 rank 数得到原输入。
                    ref_x = x_pure_rand if current_x is x_pure_rand else x  # 随机输入用随机参考，否则用 rank 编码 x。
                    assert calc_diff(check_x, ref_x) < 5e-6  # combine 结果应接近参考输入。
                    if with_topk:
                        check_topk_weights = combined_topk_weights if (current_x
                                                                       is x_pure_rand) else (combined_topk_weights /
                                                                                             # 中文：rank 编码输入需除以贡献 rank 数还原原始权重。
                                                                                             is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights  # 选择对应参考权重。
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9  # 权重 combine 要求几乎精确一致。

                    # For later tuning
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2  # BF16 dispatch 接收字节数，用于 NVLink 带宽估算。
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes  # combine 发送字节数与 dispatch 接收近似相同。

                    if local_rank == 0:  # 当前 case 全部断言通过。
                        print(' passed', flush=True)  # 打印 passed。
    if local_rank == 0:  # 所有 correctness case 结束。
        print('', flush=True)  # 空行分隔调参日志。

    # Tune dispatch performance
    best_dispatch_results = None  # 保存 rank0 选出的最佳 dispatch config。
    fp8_factor = (1 + 4 / 128) / 2  # FP8 数据约为 BF16 一半，但每 128 元素有 4 字节 scale。
    for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):  # 分别调优 FP8/BF16 dispatch。
        best_time, best_results = 1e10, None  # 当前 dtype 下最佳耗时和配置。
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes  # 当前 dtype 的估算字节数。
        for nvl_chunk_size in tuple(range(4, 33, 2)) + (0, ):  # 遍历 NVLink chunk 大小，0 表示默认配置。
            if nvl_chunk_size > 0:
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)  # 构造手动调优 config。
            else:
                # Test default config as well
                deep_ep.Buffer.set_num_sms(num_sms)  # 默认配置表读取类变量 num_sms。
                config = deep_ep.Buffer.get_dispatch_config(num_ranks)  # 获取默认 dispatch config。
            tune_args = {'x': current_x, 'handle': handle, 'config': config}  # cached dispatch benchmark 参数。
            t = bench(lambda: buffer.dispatch(**tune_args))[0]  # noqa: B023  # 只测平均耗时。
            if t < best_time and nvl_chunk_size > 0:
                # 中文：nvl_chunk_size=0 表示默认配置，只打印不覆盖手动搜索最佳值。
                best_time, best_results = t, (num_sms, nvl_chunk_size)
            if local_rank == 0:
                print(
                    f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '  # 当前调参点。
                    f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us',  # 估算 NVL 带宽和耗时。
                    flush=True)  # 立即刷新日志。
        if local_rank == 0:  # 当前 dtype 的全部 chunk 配置测完后打印最佳结果。
            print(
                f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us',
                flush=True)
            print('', flush=True)  # 空行分隔 FP8/BF16 调参结果。

        # Gather the best config from rank 0 and the first test setting
        if best_dispatch_results is None:  # 只保存第一轮 dtype 的最佳 dispatch 配置，通常是 FP8；无 FP8 时为 BF16。
            best_dispatch_results = torch.tensor([best_results[0], best_results[1]], dtype=torch.int32, device='cuda')  # 配置向量 [num_sms,nvl_chunk]。
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]  # all_gather 输出列表，每 rank 一个 [2]。
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)  # 收集所有 rank 的最佳配置。
            best_dispatch_results = all_best_fp8_results_list[0].tolist()  # 统一采用 rank0 配置，保证后续所有 rank 一致。
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size)  # 用最佳 dispatch chunk 构造配置。

    dispatch_args = {  # 为 combine 调参准备一次正式 BF16 dispatch。
        'x': x,  # 使用 rank 编码 BF16 输入，避免 FP8 误差影响 combine benchmark。
        'num_tokens_per_rank': num_tokens_per_rank,  # 每 rank token 计数 [R]。
        'is_token_in_rank': is_token_in_rank,  # token->rank 布尔矩阵 [T,R]。
        'num_tokens_per_expert': num_tokens_per_expert,  # 每 expert token 计数 [E]。
        'config': dispatch_config if dispatch_config is not None else config  # 正常使用最佳 dispatch config；兜底保留上一轮 config。
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)  # 得到 combine benchmark 所需的专家侧 recv_x 和 handle。

    # Tune combine performance
    best_time, best_results = 1e10, None  # combine 最佳耗时和配置。
    for nvl_chunk_size in tuple(range(1, 17, 1)) + (0, ):  # combine 搜索 1..16 的 NVL chunk，并额外测试默认配置。
        if nvl_chunk_size > 0:  # 手动 chunk 配置。
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)  # 构造 combine config。
        else:
            # Test default config as well
            deep_ep.Buffer.set_num_sms(num_sms)  # 默认配置表读取全局 num_sms。
            config = deep_ep.Buffer.get_combine_config(num_ranks)  # 获取默认 combine config。
        tune_args = {'x': recv_x, 'handle': handle, 'config': config}  # cached combine benchmark 参数。
        t = bench(lambda: buffer.combine(**tune_args))[0]  # noqa: B023  # 测量 combine 平均耗时。
        if local_rank == 0:  # 只让 rank0 打印调参日志。
            print(
                f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                # 中文：intranode combine 只统计 NVL send 带宽，不涉及 RDMA。
                f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us',
                flush=True)
            if t < best_time and nvl_chunk_size > 0:  # 默认配置只对比打印，不作为手动最佳配置。
                best_time, best_results = t, (num_sms, nvl_chunk_size)  # 记录最佳手动配置。

    if local_rank == 0:  # combine 搜索结束后打印最佳结果。
        print(
            f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}: {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us',
            flush=True)
        print('', flush=True)  # 空行结束 test_main。


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)  # 初始化单节点分布式环境。
    test_ll_compatibility, num_rdma_bytes = False, 0  # intranode 默认不测 low-latency，因此 RDMA bytes 为 0。
    if test_ll_compatibility:  # 该分支当前默认关闭，保留给手动兼容性验证。
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9  # low-latency 小规模测试 shape。
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(ll_num_tokens, ll_hidden, num_ranks, ll_num_experts)  # 估算 LL RDMA buffer。

    buffer = deep_ep.Buffer(group,  # legacy Buffer 封装 NVLink/RDMA workspace 与 C++ runtime。
                            int(2e9),  # NVLink buffer 字节数，单节点主路径使用。
                            num_rdma_bytes,  # RDMA buffer 字节数，默认 0。
                            low_latency_mode=test_ll_compatibility,  # 是否初始化 LL 额外资源。
                            num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1),  # LL 模式按 expert/rank 分配 QP，否则占位 1。
                            explicitly_destroy=True,  # 测试末尾显式释放资源。
                            allow_mnnvl=args.allow_mnnvl)  # 是否允许 MNNVL 支持。
    torch.manual_seed(rank)  # 每个 rank 使用不同 seed，但固定可复现。

    for i in (24, ):  # 当前 legacy benchmark 固定测试 24 个 SM。
        test_main(args, i, local_rank, num_ranks, rank, buffer, group)  # 运行完整 correctness + tuning。
        if local_rank == 0:  # 分隔不同 num_sms 的日志。
            print('', flush=True)  # 空行。

    # Test compatibility with low latency functions
    if test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)  # 清理 LL 信号区。
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)  # 复用同一个 Buffer 跑 LL 测试。

    # Destroy the buffer runtime and communication group
    buffer.destroy()  # 先释放 DeepEP Buffer，避免 process group 销毁后仍需通信。
    dist.barrier()  # 等待所有 rank 完成释放。
    dist.destroy_process_group()  # 销毁 PyTorch 分布式进程组。


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')  # 命令行入口：单节点 legacy EP 测试。
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')  # 本节点进程/GPU 数。
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens (default: 4096)')  # 每 rank token 数 T。
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')  # hidden 维度 H。
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')  # 每 token top-k 数 K。
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts (default: 256)')  # expert 总数 E。
    parser.add_argument('--allow-mnnvl', action="store_true", help='Enable MNNVL support')  # 是否启用 MNNVL 支持。
    args = parser.parse_args()  # 解析命令行参数。

    num_processes = args.num_processes  # spawn 进程数。
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)  # 拉起 local_rank=0..num_processes-1 的子进程。
