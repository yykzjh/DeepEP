import argparse  # 解析 legacy internode 测试参数。
import os  # 创建 profiler trace 输出目录。
import time  # 在布局 benchmark 后 sleep，避免多进程日志重叠。
import torch  # 构造 CUDA Tensor、topk、spawn 多进程。
import torch.distributed as dist  # all_reduce/barrier/destroy process group。

# noinspection PyUnresolvedReferences
import deep_ep  # legacy Buffer、Config、topk_idx_t 以及底层 C++/CUDA 扩展。
from deep_ep.utils.envs import init_dist  # 初始化 NCCL 分布式环境。
from deep_ep.utils.math import calc_diff, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back, hash_tensor  # 数值比较、分组 gating、去重、FP8 转换和 hash。
from deep_ep.utils.testing import bench, bench_kineto  # CUDA event benchmark 和 profiler benchmark。

# Test compatibility with low latency functions
import test_low_latency  # 复用 low-latency 测试，确保 internode buffer 与 LL API 兼容。


# noinspection PyShadowingNames
def test_main(args: argparse.Namespace,
              num_sms: int,  # legacy high-throughput kernel 使用的 SM 数。
              local_rank: int,  # 节点内 rank。
              num_local_ranks: int,  # 每节点 GPU/rank 数。
              num_ranks: int,  # 全局 rank 数。
              num_nodes: int,  # 节点数。
              rank: int,  # 当前全局 rank。
              buffer: deep_ep.Buffer,  # legacy Buffer runtime。
              group: dist.ProcessGroup,  # torch distributed 组。
              skip_benchmark: bool = False):  # 是否只跑正确性不跑 benchmark。
    # Settings：legacy internode 假设每节点 8 张卡，所以 num_local_ranks 必须为 8。
    # num_nodes = num_ranks / 8，对应 RDMA rank 数。
    num_tokens, hidden = args.num_tokens, args.hidden  # token 数 T 与 hidden 维度 H。
    num_topk_groups, num_topk, num_experts = args.num_topk_groups, args.num_topk, args.num_experts  # 分组 top-k、实际 top-k 和 expert 总数。

    assert num_experts % num_ranks == 0 and num_local_ranks == 8  # legacy internode 固定每节点 8 卡并要求 expert 均分。
    if local_rank == 0:  # 只让每节点 local rank0 打印配置。
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)  # 输出核心测试参数。

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank  # 可校验输入 [T,H]，每行值都编码源 rank。
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')  # 随机 BF16 输入，用于 hash/数值路径。
    x_e4m3 = per_token_cast_to_fp8(x)  # 将可校验输入转成 FP8 data+scale。
    x_pure_rand_e4m3 = per_token_cast_to_fp8(x_pure_rand)  # 将随机输入转成 FP8 data+scale。
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)  # 调整 scale stride，覆盖非默认布局。
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1  # gate 分数 [T,E]，取正值避免特殊负数。
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)  # 每个节点一组，取组内最大分数得到 [T,num_nodes]。
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices  # 每 token 选择若干目标节点组。
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)  # 只保留被选中节点组内的 expert 分数。
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]  # 实际 top-k expert id，形状 [T,K]。
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)  # 转成 C++ kernel 编译期选择的 topk_idx_t。
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank  # 可校验权重 [T,K]，值编码源 rank。
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')  # 随机权重用于数值 diff。
    rank_idx = topk_idx // (num_experts // num_ranks)  # expert id 映射到全局 rank id，形状 [T,K]。
    rank_idx = rank_idx.to(torch.int64)  # inplace_unique 需要可索引整型。
    rank_idx.masked_fill_(topk_idx == -1, -1)  # 无效 expert lane 的 rank 也置为 -1。
    inplace_unique(rank_idx, num_ranks)  # 对同一 token 内重复 rank 去重，模拟 dispatch 只向同 rank 发送一次 token。
    rdma_rank_idx = rank_idx // num_local_ranks  # 全局 rank 映射到 RDMA/node rank。
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)  # 无效 rank lane 对应 RDMA rank 置 -1。
    inplace_unique(rdma_rank_idx, num_nodes)  # 对同一 token 内重复 RDMA rank 去重。
    hash_value = 0  # 多轮 dispatch/combine 输出 hash 累加，防止编译器/运行时偷懒路径未执行。

    # RDMA dispatch counts：rdma_idx 是 token 命中的节点 id，先对同 token 去重，
    # num_rdma_token_sent 估算跨节点 BF16/FP8 通信字节数。
    rdma_idx = topk_idx // (num_experts // num_nodes)  # expert id 映射到节点/RDMA rank。
    rdma_idx.masked_fill_(topk_idx == -1, -1)  # 无效 expert lane 不计入 RDMA 发送。
    inplace_unique(rdma_idx, num_nodes)  # 同一 token 对同一 RDMA rank 只计一次。
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()  # 跨节点发送 token 槽数量，用于性能带宽估算。

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')  # 当前 rank 发送给每个 expert 的计数 [E]。
    for i in range(num_experts):  # 逐 expert 统计 top-k 命中次数。
        num_tokens_per_expert[i] = (topk_idx == i).sum()  # expert i 在当前 rank 输入中出现的次数。
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()  # 全局 expert 计数副本。
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)  # 聚合所有 rank，得到全局每 expert token 数。

    # Rank layout meta：同时构造 rank 级和 RDMA rank 级计数，分别校验
    # buffer.get_dispatch_layout 返回的 num_tokens_per_rank/num_tokens_per_rdma_rank。
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda')  # 当前 rank 发往每个全局 rank 的 token 数 [R]。
    num_tokens_per_rdma_rank = torch.empty((num_nodes, ), dtype=torch.int, device='cuda')  # 当前 rank 发往每个 RDMA/node rank 的 token 数 [N]。
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')  # [R,T]，记录 token 在目标 rank 内的局部序号。
    for i in range(num_ranks):  # 遍历每个目标全局 rank。
        num_tokens_per_rank[i] = (rank_idx == i).sum()  # 当前 rank top-k lane 去重后发往 rank i 的 token 数。
        token_sel = (rank_idx == i).max(dim=-1)[0]  # 每个 token 是否需要发往 rank i，形状 [T]。
        count = token_sel.sum().item()  # 需要发往 rank i 的 token 个数。
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]  # True token 排在前面，得到候选 token id。
        tokens[:count] = torch.sort(tokens[:count])[0]  # 有效 token id 再升序排列，匹配 layout kernel 的稳定排序。
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')  # 写 token 在目标 rank 内的 slot。
    for i in range(num_nodes):  # 遍历每个 RDMA/node rank。
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()  # 当前 rank 发往节点 i 的去重 token 数。
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)  # 转成 [T,R]，与 buffer.get_dispatch_layout 输出对齐。
    is_token_in_rank = token_idx_in_rank >= 0  # 布尔矩阵 [T,R]，表示 token 是否发往 rank。
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()  # 全局 rank 接收计数副本。
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)  # 聚合所有源 rank，得到每个目标 rank 的接收总数。

    ref_num_tokens_per_rank, ref_num_tokens_per_rdma_rank, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)  # 调用 legacy layout kernel，生成 dispatch 所需元数据。
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)  # rank 计数必须匹配手工参考。
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)  # RDMA rank 计数必须匹配手工参考。
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)  # expert 计数必须匹配手工参考。
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)  # token->rank 布尔布局必须匹配。
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]  # benchmark layout kernel。
    if local_rank == 0:  # 只让 local rank0 打印 layout 性能。
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)  # 输出毫秒耗时。
        print('', flush=True)  # 空行分隔日志。
    group.barrier()  # 等所有 rank 完成 layout benchmark。
    time.sleep(1)  # 短暂停顿，减少后续日志交错。

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (24, 48, 96, 144, 160) else 512)  # 特定 rank 数需要更大的 NVL buffer 以容纳 chunk。
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)  # legacy dispatch/combine 配置：SM、NVL peers、buffer chunk 等。

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        # recv_gbl_rank_prefix_sum [num_ranks] 给出当前 rank 接收序列中每个源 rank 的
        # 结束位置；x 的每行全等于源 rank，因此可直接校验数据路由。
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0  # 当前源 rank 区间的起点。
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()  # 源 rank i 的区间终点。
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0  # 区间内所有行都应编码为源 rank i。
            check_start = check_end  # 下一段从当前终点开始。

    for previous_mode in (False, True):
        for async_mode in (False, True):  # 覆盖同步和异步 dispatch/combine。
            for current_x in (x_pure_rand, x, x_pure_rand_e4m3, x_e4m3):  # 覆盖随机/可校验 BF16 与 FP8 输入。
                for with_topk in (False, True):  # 覆盖是否返回/携带 top-k metadata。
                    # is_rand=True 表示输入没有 rank 编码，后续比较走纯数值容差路径。
                    is_rand = current_x is x_pure_rand or current_x is x_pure_rand_e4m3
                    # 中文：rank0 负责打印 case 维度，避免所有 rank 重复刷屏。
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            # 中文：每个 case 会覆盖 dtype、top-k metadata、异步和 previous_event 四个维度。
                            flush=True,  # 立即刷新，避免多 rank benchmark 日志堆积。
                            end='')  # 不换行，后续耗时/结果接在同一行。
                    dispatch_args = {
                        'x': current_x,  # 当前测试输入，可能是 BF16 tensor 或 FP8 tuple。
                        'num_tokens_per_rank': num_tokens_per_rank,  # rank 级发送计数。
                        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,  # RDMA rank 级发送计数。
                        'is_token_in_rank': is_token_in_rank,  # token 是否发送到某 rank 的 bool 矩阵。
                        'num_tokens_per_expert': num_tokens_per_expert,  # expert 级发送计数。
                        'config': config,  # legacy kernel 配置。
                        'async_finish': async_mode  # 是否异步结束。
                    }
                    if with_topk:  # with_topk=True 时测试 metadata/topk_weights 随 dispatch 一起返回。
                        dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if is_rand else topk_weights})  # 随机输入配随机权重，可校验输入无 rank 编码时的数值路径。
                    if previous_mode:  # previous_mode 测试 legacy event 依赖。
                        dispatch_args.update({'previous_event': buffer.capture()})  # 捕获当前 buffer event 作为前序事件。
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(
                        **dispatch_args)  # 执行 legacy internode dispatch。
                    event.current_stream_wait() if async_mode else ()  # async 模式下当前流等待 dispatch 完成。

                    if current_x is x_pure_rand or current_x is x:  # BF16 输入直接 hash recv_x。
                        hash_value += hash_tensor(recv_x)  # 累加 hash，压力测试会比较多轮一致性。
                    else:
                        hash_value += hash_tensor(recv_x[0])  # FP8 data hash。
                        hash_value += hash_tensor(recv_x[1])  # FP8 scale hash。

                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x  # FP8 recv_x 转 BF16 后校验。

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-4]  # legacy handle 中倒数第 4 项是源 rank prefix。
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), \
                        f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'  # 当前 rank 接收总数必须等于全局参考。
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list  # 本 rank local experts 计数必须匹配。
                    if not is_rand:  # rank 编码输入可直接检查每行值是否等于源 rank。
                        check_data(recv_x, recv_gbl_rank_prefix_sum)  # 校验 dispatch 数据路由。
                    recv_topk_weights_clone = None  # 保存 dispatch 返回权重副本，num_worst_tokens 路径会比较。
                    if with_topk:  # 只有带 top-k metadata 的 dispatch 才检查 topk_idx/weights。
                        # internode dispatch 后 topk_idx 也会被转换成本 rank 内局部 expert id。
                        # 计数校验覆盖 RDMA+NVLink 两级路径是否把所有 expert lane 送到位。
                        assert (recv_topk_idx.eq(-1) |
                                ((recv_topk_idx >= 0) &
                                 (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):  # 遍历本地 expert。
                            assert recv_topk_idx.eq(i).sum().item() == count  # recv_topk_idx 中 expert i 出现次数必须等于计数。

                        # Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()  # 保存原始返回权重，后续会临时修改无效 lane。
                        if not is_rand:  # rank 编码权重可用 check_data 验证来源。
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(
                                dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]  # 无效 lane 填成该行最大值，避免 -1 lane 影响 rank 编码检查。
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)  # 校验权重也按源 rank 路由。

                    # Test `num_worst_tokens != 0`
                    if with_topk:  # num_worst_tokens 只在带 top-k metadata 的 dispatch 输出中有意义。
                        num_worst_tokens = num_tokens * num_ranks  # 最坏容量设置成每 rank 都收到所有 token。
                        dispatch_args.update({'num_worst_tokens': num_worst_tokens})  # 强制 dispatch 输出按 worst capacity 分配。
                        recv_worst_x, recv_worst_topk_idx, recv_worst_topk_weights, empty_list, _, event = buffer.dispatch(**dispatch_args)  # 运行 worst capacity dispatch。
                        event.current_stream_wait() if async_mode else ()  # async 模式等待完成。
                        recv_worst_x = per_token_cast_back(*recv_worst_x) if isinstance(recv_worst_x, tuple) else recv_worst_x  # FP8 转 BF16。
                        assert len(empty_list) == 0  # worst capacity 模式不返回 per-expert list。
                        assert num_worst_tokens == recv_worst_x.size(0)  # recv_x 第一维应等于 worst capacity。
                        assert num_worst_tokens == recv_worst_topk_idx.size(0)  # topk_idx 第一维也等于 worst capacity。
                        assert num_worst_tokens == recv_worst_topk_weights.size(0)  # topk_weights 第一维也等于 worst capacity。
                        assert torch.equal(recv_x, recv_worst_x[:recv_x.size(0)])  # 有效前缀数据与普通 dispatch 一致。
                        assert torch.equal(recv_topk_idx, recv_worst_topk_idx[:recv_x.size(0)])  # 有效前缀 topk_idx 一致。
                        assert torch.equal(recv_topk_weights_clone, recv_worst_topk_weights[:recv_x.size(0)])  # 有效前缀权重一致。
                        assert torch.all(recv_worst_topk_idx[recv_x.size(0):] == -1).item()  # padding 区 topk_idx 必须填 -1。

                    # cached dispatch 复用 handle 中 RDMA/NVL prefix 与 SourceMeta，
                    # 因此只用于不重新生成 top-k 元数据的路径。
                    if not with_topk:  # 无 top-k 的 dispatch 可进入 cached payload-only 路径。
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}  # 复用 handle 中的布局。
                        if previous_mode:  # cached dispatch 同样覆盖 previous_event。
                            dispatch_args.update({'previous_event': buffer.capture()})  # 注入前序事件。
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)  # 执行 cached dispatch。
                        event.current_stream_wait() if async_mode else ()  # async 模式等待。
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x  # FP8 转 BF16。
                        if not is_rand:  # rank 编码输入可检查路由。
                            check_data(recv_x, recv_gbl_rank_prefix_sum)  # cached 数据路由也必须正确。

                    # Test combine
                    bias_0 = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')  # combine bias0 [T,H]。
                    bias_1 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')  # combine bias1 [T,H]。
                    combine_args = {'x': recv_x, 'bias': (bias_0, bias_1), 'handle': handle, 'config': config, 'async_finish': async_mode}  # legacy combine 参数。
                    if with_topk:  # 带 top-k 权重时测试权重回传。
                        combine_args.update({'topk_weights': recv_topk_weights})  # 将 dispatch 返回权重传给 combine。
                    if previous_mode:  # combine 也覆盖 previous_event。
                        combine_args.update({'previous_event': buffer.capture()})  # 注入前序事件。
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)  # 执行 legacy combine。
                    event.current_stream_wait() if async_mode else ()  # async 模式等待 combine 完成。
                    check_x = (combined_x.float() - bias_0.float() - bias_1.float()) / is_token_in_rank.sum(dim=1).unsqueeze(1)  # 去掉 bias 并按重复 rank 数求平均。
                    ref_x = x_pure_rand if is_rand else x  # 随机输入用 x_pure_rand，rank 编码输入用 x。
                    assert calc_diff(check_x, ref_x) < 5e-4 if current_x is x_pure_rand_e4m3 else 5e-6  # FP8 随机路径误差阈值更宽，其余要求更严。
                    if with_topk:  # 只有 dispatch 带 top-k 时 combine 才返回 combined_topk_weights。
                        check_topk_weights = combined_topk_weights if is_rand else (combined_topk_weights /
                                                                                    is_token_in_rank.sum(dim=1).unsqueeze(1))  # rank 编码权重需要除以重复累加次数。
                        ref_topk_weights = topk_weights_pure_rand if is_rand else topk_weights  # 选择参考权重。
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9  # 权重要求几乎 bitwise 一致。

                    hash_value += hash_tensor(recv_x)  # 累加 combine 前输入 hash，压力测试用于一致性比较。

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2  # BF16 dispatch RDMA 发送字节估算。
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2  # BF16 dispatch NVL 接收字节估算。
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes  # combine NVL 发送量与 dispatch NVL 接收量对称。
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes  # combine RDMA 接收量与 dispatch RDMA 发送量对称。

                    if local_rank == 0:  # 当前模式通过时只让 local rank0 打印。
                        print(' passed', flush=True)  # 补齐前面 testing 行的结尾。
    if local_rank == 0:  # 所有模式结束后打印空行。
        print('', flush=True)  # 分隔 dispatch/combine correctness 与 tuning 日志。

    if skip_benchmark:  # 压力测试模式可跳过调参 benchmark。
        return hash_value  # 返回 hash 供多轮一致性比较。

    # Tune dispatch performance
    # `best_dispatch_results` 保存 FP8 dispatch 的最优配置，后面会复用它来生成 BF16 combine 的输入。
    best_dispatch_results = None
    # FP8 输入由 E4M3 数据和每 128 个元素一个 scale 组成，因此近似字节比例为 (1 byte 数据 + 4/128 byte scale) / 2 byte BF16。
    fp8_factor = (1 + 4 / 128) / 2
    # 先调 FP8，再调 BF16：`x_e4m3` 是 tuple，`x` 是普通 BF16 tensor。
    for current_x in (x_e4m3, x):
        # 每种精度单独维护最佳时间和最佳参数，避免 FP8/BF16 的吞吐差异互相污染。
        best_time, best_results = 1e10, None
        # 如果当前输入是 FP8 tuple，RDMA 发送字节按 FP8 压缩比例折算；BF16 则直接使用前面估算值。
        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        # NVLink 接收的 payload 同样随精度变化；这里仅用于打印有效带宽，不参与正确性判断。
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        # `nvl_chunk_size` 控制节点内 NVLink 分片粒度，步长 4 是 legacy kernel 的经验调参范围。
        for nvl_chunk_size in range(4, 45, 4):
            # `rdma_chunk_size` 控制跨节点 RDMA 分片粒度，和 NVL chunk 共同决定 SM 内流水线节奏。
            for rdma_chunk_size in range(4, 33, 4):
                # Config 五元组含义：SM 数、NVL chunk、NVL 缓冲、RDMA chunk、RDMA 缓冲。
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                # 调参阶段复用前面 correctness dispatch 返回的 handle，只替换输入精度和 kernel 配置。
                # 中文：复用 handle 可以隔离 payload 搬运性能，避免 layout 重新计算影响调参。
                tune_args = {'x': current_x, 'handle': handle, 'config': config}
                # `bench_kineto` 同时测 dispatch 主体和 notify 阶段；notify 是跨 rank 元信息同步的成本。
                t, notify_t = bench_kineto(
                    lambda: buffer.dispatch(**tune_args),  # noqa: B023
                    ('dispatch', 'notify'),
                    # 中文：只保留 profiler 结果，不打印 kineto 的详细表格。
                    suppress_kineto_output=True)
                # 以 dispatch 主体时间作为排序指标；notify 只随结果一起记录，便于打印分解耗时。
                if t < best_time:
                    # best_results 的维度是 4 个标量：[num_sms, nvl_chunk_size, rdma_chunk_size, notify_t]。
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size, notify_t)
                # 只让每个节点的 local rank 0 打印，避免 8 张卡重复刷屏。
                if local_rank == 0:
                    print(
                        f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: '  # 当前调参点。
                        f'{notify_t * 1e6:.0f} + {t * 1e6:.0f} us, '  # notify 与 dispatch 主体耗时。
                        f'{rdma_send_bytes / 1e9 / t:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL) ',  # 估算带宽。
                        flush=True)  # 立即刷新日志。
        # 当前精度的全部 chunk 组合测完后，打印该精度下最优配置。
        if local_rank == 0:
            print(
                f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: '  # 最优配置。
                f'{best_results[3] * 1e6:.0f} + {best_time * 1e6:.0f} us, '  # 最优 notify+dispatch 耗时。
                f'{rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)',  # 最优带宽。
                flush=True)  # 立即刷新日志。
            # 空行让多组 seed 或精度的日志更容易肉眼分隔。
            print('', flush=True)

        # FP8 的最佳 dispatch 配置会作为后续一次正式 BF16 dispatch 的配置来源。
        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            # 每个 rank 都构造一个 3 元 tensor，内容是 [num_sms, nvl_chunk_size, rdma_chunk_size]。
            best_dispatch_results = torch.tensor([best_results[0], best_results[1], best_results[2]], dtype=torch.int32, device='cuda')
            # all_gather 的输出列表长度等于全局 rank 数，每个元素形状都是 `[3]`。
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]
            # 跨所有进程收集配置；下面统一取 rank0 的配置，保证所有 rank 后续 dispatch 使用同一组参数。
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            # `tolist()` 把 CUDA tensor 转成 Python list，便于创建 deep_ep.Config。
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    # 使用 FP8 调出来的最佳 chunk 配置做一次 BF16 dispatch，获得 combine benchmark 需要的 `recv_x` 和 `handle`。
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size, best_dispatch_results[2],
                                     rdma_buffer_size)

    # 这里重新走非 cached dispatch，因为 combine 调参需要一个和当前 best dispatch 配置匹配的 handle。
    dispatch_args = {
        # combine benchmark 使用 BF16 payload，避免 FP8 误差影响 combine 正确性与带宽估算。
        'x': x,
        # 每个目标 rank 的 token 数，shape `[num_ranks]`。
        'num_tokens_per_rank': num_tokens_per_rank,
        # 每个目标 RDMA rank 的 token 数，shape `[num_rdma_ranks]`。
        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
        # token 是否发往某个 rank 的布尔矩阵，shape `[num_tokens, num_ranks]`。
        'is_token_in_rank': is_token_in_rank,
        # 每个 expert 的 token 数，shape `[num_experts]`。
        'num_tokens_per_expert': num_tokens_per_expert,
        # 正常情况下使用 FP8 调出的 dispatch_config；兜底分支保留旧变量 config。
        'config': dispatch_config if dispatch_config is not None else config
    }
    # `recv_x` 是 dispatch 后每个 rank 收到的 token buffer，`handle` 记录 combine 还原路由所需的元信息。
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    # combine 的最佳配置单独搜索，因为反向聚合的 NVL/RDMA 方向和 dispatch 相反，瓶颈也可能不同。
    best_time, best_results = 1e10, None
    # combine 的 NVL chunk 搜索范围更小，通常聚合写回对节点内分片不需要太大粒度。
    for nvl_chunk_size in range(1, 8, 1):
        # 两节点场景从 12 开始是 legacy benchmark 经验值，多节点则允许从 8 开始探索。
        for rdma_chunk_size in range(12 if num_nodes == 2 else 8, 33, 4):
            # 为当前 chunk 组合创建 combine 配置；buffer 大小沿用 dispatch 的估算结果。
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
            # combine 只需要收到的 token 和 handle，handle 内含原 token 的 rank/expert/slot 映射。
            tune_args = {'x': recv_x, 'handle': handle, 'config': config}
            # Kineto 计时标签里同时抓 combine 和 notify，便于把同步开销与数据搬运开销拆开看。
            t, notify_t = bench_kineto(
                lambda: buffer.combine(**tune_args),  # noqa: B023
                ('combine', 'notify'),
                # 中文：notify_t 单独显示 prefix/head 同步成本，t 表示 combine 主体搬运成本。
                suppress_kineto_output=True)
            # 只在 local rank 0 打印调参日志，并在同一个分支内更新最佳值以减少重复输出。
            if local_rank == 0:
                print(
                    f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: '  # 当前 combine 调参点。
                    f'{notify_t * 1e6:.0f} + {t * 1e6:.0f} us, '  # notify 与 combine 主体耗时。
                    f'{combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (RDMA), '  # RDMA 接收带宽估算。
                    f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL) ',  # NVL 发送带宽估算。
                    flush=True)  # 立即刷新日志。
                # combine 以主 kernel 时间作为最优指标，notify_t 只是记录下来用于日志解释。
                if t < best_time:
                    # best_results 维度同 dispatch：[num_sms, nvl_chunk_size, rdma_chunk_size, notify_t]。
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size, notify_t)

    # 调参结束后打印 combine 的全局最佳结果；这个 benchmark 只用于性能观察，不改变测试 hash。
    if local_rank == 0:
        print(
            # 中文：combine 调参日志把 notify 和主体时间拆开显示，便于观察控制面开销。
            f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, '
            f'{best_results[3] * 1e6:.2f} + {best_time * 1e6:.2f} us, '
            # 中文：这里同时打印 RDMA 接收带宽和 NVL 发送带宽，便于判断瓶颈在哪一层。
            # 中文：第一个时间是 notify，第二个时间是 combine 主体。
            # 中文：best_results[3] 是 notify 时间，best_time 是 combine 主 kernel 时间。
            f'{combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)',
            flush=True)
        # 空行分隔后续 seed 或后续测试项。
        print('', flush=True)
    # 返回 correctness 阶段累计出的 hash；调参循环不参与 hash，避免性能测量改变正确性基准。
    return hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    # `WORLD_SIZE` 在跨节点 torchrun/mpi 启动时通常表示节点数，这里用于推导 top-k group 默认值和 RDMA 拓扑。
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    # 初始化分布式进程组；返回的 rank 是全局 rank，num_ranks 是全局 GPU/进程数，group 是通信组句柄。
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    # 低延迟兼容测试需要额外构造一组较小的 token/expert 参数，用于复用同一个 Buffer 测 LL kernel。
    if args.test_ll_compatibility:
        # 维度含义：16 个 token、hidden=5120、256 个 expert、每个 token 选 9 个 expert。
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9

    # legacy internode 测试固定使用 24 个 SM，和历史调参结果保持一致。
    num_sms = 24
    # num_qps_per_rank 要覆盖普通 EP 的 SM 并发，也要覆盖低延迟模式下每 rank 负责的 expert 数。
    num_qps_per_rank = max(num_sms, ll_num_experts // num_ranks if args.test_ll_compatibility else 0)

    # Buffer 是 Python API 对 C++/CUDA 通信运行时的封装，内部会分配 NVLink/RDMA 缓冲并注册 IPC/NVSHMEM 资源。
    buffer = deep_ep.Buffer(group,
                            # 第一个容量是节点内 NVL buffer，上限 2GB，足够容纳本测试的 dispatch/combine 中间数据。
                            int(2e9),
                            # 第二个容量是跨节点 RDMA buffer，上限 1GB，和 test_main 中的容量估算互相约束。
                            int(1e9),
                            # 开启后 Buffer 会额外初始化 low-latency kernel 需要的队列/信号槽。
                            low_latency_mode=args.test_ll_compatibility,
                            # 每 rank 的队列数影响 low-latency 与 normal kernel 的并发资源划分。
                            num_qps_per_rank=num_qps_per_rank,
                            # 测试脚本显式 destroy，方便发现资源释放顺序问题。
                            explicitly_destroy=True)
    # internode legacy 测试假设每节点 8 卡，且至少两个节点；不满足时测试语义会偏离 RDMA 路径。
    assert num_local_ranks == 8 and num_ranks > 8

    # pressure test 模式下 seed 会一直递增；非压力模式只跑第 0 个 seed 后 break。
    for seed in range(int(1e9)):
        # 每轮 seed 开头打印一次，方便在长时间压力测试日志里定位失败轮次。
        if local_rank == 0:
            print(f'Testing with seed {seed} ...', flush=True)
        # 每个 rank 使用 rank+seed，保证不同 rank 数据不同，同时同一 seed 重跑可复现。
        torch.manual_seed(rank + seed)
        # ref_hash 是本 seed 的正确性基准，后续 20 次重放必须完全一致。
        ref_hash = 0
        # 这里保留 tuple 写法是为了兼容过去一次测试多个 num_sms 的结构；当前只测试固定 24 SM。
        for i in (num_sms, ):
            # `pressure_test_mode == 1` 表示只做 correctness，不跑耗时调参，从而提高压力测试频率。
            ref_hash += test_main(args, i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group,
                                  args.pressure_test_mode == 1)
            # 每个 num_sms 测完留空行，让 rank0 日志更易读。
            if local_rank == 0:
                print('', flush=True)
        # 普通模式跑完首轮 correctness/benchmark 即退出。
        if args.pressure_test_mode == 0:
            break

        # 压力模式打印本轮参考 hash，失败时可用它和后续 current_hash 对照。
        if local_rank == 0:
            print(f'{ref_hash=}')
            print('', flush=True)

        # 每个 seed 再重复 20 次相同随机输入，主要检测异步通信、缓冲复用和元信息复用的稳定性。
        for _ in range(20):
            # 重置同一个随机种子，确保 current_hash 应与 ref_hash 完全一致。
            torch.manual_seed(rank + seed)
            # 当前重放轮的累计 hash，维度上是若干 num_sms 测试结果的标量和。
            current_hash = 0
            # 和参考轮使用同样的 num_sms 列表，避免配置差异导致 hash 不可比。
            for i in (num_sms, ):
                current_hash += test_main(args, i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group,
                                          args.pressure_test_mode == 1)
                # 空行分隔每次 test_main 的日志。
                if local_rank == 0:
                    print('', flush=True)
            # 这是压力测试的核心断言：只要通信顺序或缓冲复用有非确定性 bug，hash 通常会在这里暴露。
            assert current_hash == ref_hash

    # Test compatibility with low latency functions
    # 兼容性测试会在 normal EP 测完后复用 Buffer，但先清理 LL buffer，避免旧信号槽影响低延迟路径。
    if args.test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        # 这里调用 legacy low-latency 测试入口，验证同一个 Buffer 同时支持 normal 和 low-latency 两类 API。
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the buffer runtime and communication group
    # 先销毁 Buffer，让 CUDA/NVSHMEM/IPC 资源在 process group 还有效时完成释放。
    buffer.destroy()
    # barrier 保证所有 rank 都完成释放，防止某个 rank 提前 destroy process group 导致其他 rank 通信失败。
    dist.barrier()
    # 最后销毁 PyTorch 分布式进程组，进程退出前不再进行 collective。
    dist.destroy_process_group()


if __name__ == '__main__':
    # 命令行入口只负责解析参数和拉起本节点多进程，真正的 EP 测试逻辑在 `test_loop`。
    parser = argparse.ArgumentParser(description='Test internode EP kernels')
    # 每节点启动的进程数，默认 8 对应 DGX/HGX 单节点 8 GPU，也是 test_loop 断言的前提。
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    # 每个 rank 初始 token 数，输入张量 x 的形状是 `[num_tokens, hidden]`。
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens (default: 4096)')
    # hidden 是 token 向量宽度，DeepSeek 系列 MoE 常见 hidden=7168，本测试沿用该维度压测带宽。
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    # top-k group 控制 group-limited routing 的组数；None 时按节点数自动选择。
    parser.add_argument('--num-topk-groups', type=int, default=None, help='Number of top-k groups (default: `min(num_nodes, 4)`)')
    # 每个 token 选择的 expert 数，topk_idx 的形状是 `[num_tokens, num_topk]`。
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    # pressure-test-mode=0 跑一次完整 benchmark；1 跳过 benchmark 高频重放；2 连 benchmark 也一起压力测试。
    parser.add_argument(
        # 中文：压力测试参数单独换行，便于后续扩展 choices 或说明。
        '--pressure-test-mode',
        type=int,
        default=0,
        # 中文：压力测试模式控制是否重复执行 correctness 和 benchmark，用于暴露偶发竞态。
        # 中文：模式 1 跳过 benchmark 加速压力测试，模式 2 连 benchmark 一起重复执行。
        # 中文：默认 0 保持单次 correctness/benchmark，适合常规开发验证。
        help='Pressure test mode. 0: don\'t do pressure test, 1: do pressure test without benchmarks, 2: do pressure test with benchmarks')
    # expert 总数决定路由分布和每 rank/local expert 的切分，默认 256 与 internode 大规模 MoE 配置一致。
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts (default: 256')
    # 打开后在 normal EP 后追加 low-latency API 兼容性测试，确认同一 Buffer 的资源复用安全。
    parser.add_argument('--test-ll-compatibility', action='store_true', help='whether to test compatibility with low-latency kernels')
    # argparse.Namespace 会被传给每个 spawn 子进程；子进程只读参数，不修改共享状态。
    args = parser.parse_args()

    # Set default `num_topk_groups` if not provided
    # 默认 top-k group 不超过 4，节点数不足时跟随节点数，保证 group-limited routing 不产生空 group。
    if args.num_topk_groups is None:
        # WORLD_SIZE 在本脚本语义下表示节点数；和 test_loop 中的 num_nodes 保持同一来源。
        num_nodes = int(os.getenv('WORLD_SIZE', 1))
        # 例如 2 节点默认 2 个 top-k group，8 节点默认 4 个 top-k group。
        args.num_topk_groups = min(num_nodes, 4)

    # 本节点进程数与 GPU 数一一对应，local_rank 会由 multiprocessing.spawn 自动传给 test_loop。
    num_processes = args.num_processes
    # spawn 会创建 `num_processes` 个 Python 子进程，每个子进程调用 test_loop(local_rank, num_processes, args)。
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
