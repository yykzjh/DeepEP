import argparse  # 解析低延迟 legacy 测试参数。
import random  # 生成随机 mask 位置和 LogFMT 测试输入。
import torch  # 构造 CUDA Tensor、spawn 多进程、执行 GEMM/sleep 等。
import torch.distributed as dist  # 初始化后的 group 用于 all_gather/barrier/destroy。
from functools import partial  # benchmark 时固定 test_func 的 return_recv_hook 参数。
from typing import Literal, Set  # 标注 API 名称只能是 dispatch/combine/clean，masked rank 集合类型。

import deep_ep  # legacy Buffer 和 topk_idx_t/Config 等顶层 API。
from deep_ep.utils.envs import init_dist  # 初始化 NCCL 分布式环境。
from deep_ep.utils.math import calc_diff, per_token_cast_back, hash_tensor  # 数值比较、FP8 反量化、结果 hash。
from deep_ep.utils.testing import bench, bench_kineto  # CUDA event benchmark 和 profiler benchmark。


def simulate_failure_and_skip(rank: int, api: Literal["dispatch", "combine", "clean"], expected_masked_ranks: Set[int]):  # shrink 测试中模拟某 rank 在指定 API 首次失败。
    # 在 shrink_test 中模拟 rank 第一次进入某个通信 API 时失败。失败 rank 之后会被
    # 加入 expected_masked_ranks，其他 rank 通过 mask buffer 跳过它。
    failed_api_ranks = {  # 每类 API 预设一个会失败的 rank。
        # API -> rank to fail (rank fails when it first calls the corresponding communication API)
        'dispatch': 1,  # 第一次 dispatch 时模拟 rank1 失败。
        'combine': 3,  # 第一次 combine 时模拟 rank3 失败。
        'clean': 5  # 第一次 clean 时模拟 rank5 失败。
    }  # 这些 rank 会逐步加入 mask buffer。
    if rank in expected_masked_ranks:  # 当前 rank 已经被标记失效。
        # Rank already failed
        return True  # 已失效 rank 跳过后续通信 API。
    if api in failed_api_ranks.keys():  # 当前 API 有预设失败 rank。
        expected_masked_ranks.add(failed_api_ranks[api])  # 所有 rank 的预期集合都加入该失败 rank。
        if failed_api_ranks[api] == rank:  # 如果当前 rank 就是本轮失败者。
            print(f"Rank {rank} failed when first calling {api} communication API, exit...", flush=True)  # 打印模拟失败信息。
            return True  # 当前 rank 跳过本次 API，模拟退出通信。
    return False  # 当前 rank 正常执行 API。


def query_mask_buffer_and_check(api: Literal["dispatch", "combine", "clean"], buffer: deep_ep.Buffer, mask_status: torch.Tensor,
                                expected_masked_ranks: Set[int]):  # 查询低延迟 mask buffer 并与预期失败集合比较。
    # mask_status 形状 [num_ranks]，1 表示该 rank 已被 low-latency barrier 标记为失效。
    buffer.low_latency_query_mask_buffer(mask_status)  # C++ 将 mask 状态写入 [num_ranks] int tensor。
    assert set(mask_status.nonzero().squeeze(-1).tolist()) == expected_masked_ranks  # 非零 rank 集合必须等于预期失败集合。


def test_main(num_tokens: int,  # 当前 rank 生成并 dispatch 的 token 数。
              hidden: int,  # token hidden 维度。
              num_experts: int,  # 全局 expert 数。
              num_topk: int,  # 每 token top-k 数。
              rank: int,  # 当前全局 rank。
              num_ranks: int,  # 全局 rank 数。
              group: dist.ProcessGroup,  # torch distributed 通信组。
              buffer: deep_ep.Buffer,  # 已初始化的 legacy low-latency Buffer。
              use_logfmt: bool = False,  # combine 是否测试 LogFMT 压缩路径。
              shrink_test: bool = False,  # 是否测试 shrink/mask 容错路径。
              seed: int = 0):  # 低延迟 dispatch/combine 正确性与性能主测试。
    torch.manual_seed(seed + rank)  # 每个 rank 用不同但可复现的 PyTorch seed。
    random.seed(seed + rank)  # Python random 同步 seed，用于 mask/LogFMT 随机输入。

    # low-latency dispatch 的 packed_recv_x 以 expert 为第一维：
    # [num_local_experts, num_ranks, num_max_dispatch_tokens_per_rank, hidden]。
    # 最后 128 维写 token id，方便 FP8 round_scale 与 BF16 路径做路由校验。
    assert num_experts % num_ranks == 0  # 低延迟测试假设 expert 均匀分配给 rank。
    num_local_experts = num_experts // num_ranks  # 当前 rank 拥有的 expert 数。

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128  # 用 rank-rank_offset 写入 BF16，避免正负值范围太大。
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'  # BF16 对大整数精度有限。

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)  # 基础输入 [T,H]，主体值编码源 rank。
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)  # 尾部 128 列编码 token id。
    x_list = [x]  # 测试输入列表，第一项用于严格路由检查。
    for _ in range(4 if use_logfmt else 0):  # LogFMT 模式额外构造多组随机输入，覆盖动态 cast。
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.5 * random.random())  # 随机 BF16 输入 [T,H]。
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.1)  # 性能测试输入，值较小以触发更多 LogFMT cast。

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1  # gate 分数 [T,E]，全为正。
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]  # 路由 expert id [T,K]。
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)  # 转成 C++ kernel 期望的 topk_idx_t。
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()  # combine 权重 [T,K]。

    # Randomly mask some positions
    for _ in range(10):  # 随机制造无效 expert lane，测试 -1 路由处理。
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1  # 某个 token 的某个 top-k 置为无效。

    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='cuda')  # 收集所有 rank 路由 [R,T,K]。
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)  # 用 PyTorch all_gather 构造参考路由全集。

    # For failure simulation and shrink testing
    mask_status = torch.zeros((num_ranks, ), dtype=torch.int, device='cuda')  # 查询 mask buffer 的输出 [R]。
    expected_masked_ranks = set()  # Python 侧预期失效 rank 集合。

    # Check dispatch correctness：遍历 BF16/FP8/UE8M0、是否返回 hook、是否模拟失败。
    # handle[0] 是 src_info，handle[1] 是 layout_range，combine 复用它们反向发送。
    do_check = True  # 控制是否做逐 expert 正确性检查。
    hash_value, num_times = 0, 0  # hash_value 用于压力测试跨重复运行校验确定性；num_times 用于交替重复次数。
    for current_x in x_list:  # 遍历严格检查输入、LogFMT 输入和性能输入。
        for return_recv_hook in (False, True):  # 覆盖立即等待和返回 recv hook 两种模式。
            for dispatch_use_fp8 in (False, True):  # 覆盖 BF16 dispatch 与 FP8 dispatch。
                for round_scale in (False, True) if dispatch_use_fp8 else (False, ):  # FP8 模式下覆盖 scale 是否取 2 的幂。
                    for use_ue8m0 in (False, True) if round_scale else (False, ):  # round_scale 时覆盖 UE8M0 packed scale。
                        if shrink_test and simulate_failure_and_skip(rank, "dispatch", expected_masked_ranks):  # shrink 模式下模拟 dispatch 失败。
                            break  # 当前 rank 失效后跳出本层配置。
                        num_times += 1  # 记录执行配置次数。
                        for _ in range((num_times % 2) + 1):  # 交替执行 1/2 次，覆盖双缓冲复用。
                            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')  # per-expert 累计接收统计。
                            # dispatch 返回固定容量接收 buffer、计数、handle、事件和可选 hook。
                            packed_recv_x, packed_recv_count, handle, event, hook = buffer.low_latency_dispatch(
                                current_x, topk_idx, num_tokens, num_experts,  # 输入 [T,H] 与路由 [T,K]。
                                use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,  # 覆盖 FP8/scale 格式组合。
                                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,  # 统计每个本地 expert 接收数。
                                async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)  # hook 模式下拆分 issue/recv。
                            hook() if return_recv_hook else event.current_stream_wait()  # hook 模式显式等数据，否则等待事件。
                        if shrink_test:  # shrink 测试需要检查 dispatch 后 mask buffer。
                            query_mask_buffer_and_check("dispatch", buffer, mask_status, expected_masked_ranks)  # 验证失效 rank 集合。
                        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x  # FP8 scale 转 contiguous 便于 view。
                        simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape) \
                            if dispatch_use_fp8 else packed_recv_x.clone()  # 模拟 expert GEMM 输出；FP8 先反量化回 BF16 形状。
                        for i in range(num_local_experts if do_check else 0):  # 逐本地 expert 校验接收结果。
                            expert_id = rank * num_local_experts + i  # 当前本地 expert 的全局 expert id。
                            recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]  # 当前 expert 的接收 token。
                            recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]  # count、src_info、layout_range 均按 expert 切片。

                            # layout_range[j] 打包 (offset,count)：低 32 位 count，高 32 位 offset。
                            # 它描述源 rank j 在当前 local expert 接收区中的连续片段。
                            int_mask = (2**32) - 1  # 低 32 位掩码，用于从 layout_range 中取 count。
                            num_valid_tokens = recv_count.item()  # 当前 expert 实际收到的 token 数。
                            assert cumulative_local_expert_recv_stats[i].item(  # 累计统计必须等于当前 expert 有效 token 数。
                            ) == num_valid_tokens, f'{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}'  # 失败时打印统计值与期望。
                            assert num_valid_tokens == (
                                recv_layout_range  # layout_range 的低 32 位是每个源 rank count。
                                & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'  # 所有源 rank count 之和应等于 recv_count。
                            assert num_valid_tokens == (all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status == 0].sum().item(
                            ), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status==0].sum().item()}'  # 参考 all_topk 统计也必须一致。

                            if num_valid_tokens == 0:  # 没有 token 发给该 expert 时跳过数据内容检查。
                                continue  # count/layout 上面的断言已经覆盖空 expert。
                            # Check received data
                            if current_x is x:
                                recv_x = recv_x[:num_valid_tokens]  # 只检查有效 token 区间，固定容量尾部可能是旧数据。
                                recv_x_amin = recv_x[:, :-128].amin(dim=-1)  # 主体列最小值应等于源 rank 编码。
                                recv_src_info = recv_src_info[:num_valid_tokens]  # 源信息也只截取有效 token。
                                assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))  # 主体列 min/max 相等说明 rank 编码一致。
                                if round_scale:  # round_scale 会带来可接受量化误差。
                                    assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007  # token id 近似匹配源信息。
                                else:  # 非 round_scale 路径应 bitwise 保持 token id。
                                    assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0  # 尾部 token id 校验。
                                for j in range(num_ranks):  # 逐源 rank 检查 layout_range 切片。
                                    if shrink_test and mask_status[j]:  # shrink 模式下失效 rank 不再提供 token。
                                        continue  # 跳过已 mask 的源 rank。
                                    begin_idx, count = (recv_layout_range[j] >> 32).item(), (recv_layout_range[j] & int_mask).item()  # 解包 offset/count。
                                    if not round_scale:  # 非 round_scale 下可做严格源 rank 编码检查。
                                        assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()  # 该源 rank token 数匹配路由。
                                        assert (recv_x[begin_idx:begin_idx + count, :-128] - j + rank_offset).sum().item() == 0  # 对应连续片段编码正确。
                            if dispatch_use_fp8:  # FP8 路径 hash 数据和 scale 两部分。
                                hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])  # 数据 tensor 指纹。
                                hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])  # scale tensor 指纹。
                            else:  # BF16 路径只 hash 数据。
                                hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])  # 有效 token 指纹。

                        # combine correctness：把专家输出按 topk_weights 加权归并回原 token。
                        # shrink_test 下先把失效 rank 拥有的 expert lane 置 -1，参考值同步降级。
                        if shrink_test and simulate_failure_and_skip(rank, "combine", expected_masked_ranks):
                            break
                        for zero_copy in (False, ) if use_logfmt else (False, True):  # LogFMT 不覆盖 zero-copy；普通模式覆盖两条路径。
                            if zero_copy:  # zero-copy 模式要求调用方预先写入下一块 combine RDMA buffer。
                                buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x  # 将模拟 GEMM 输出写入 registered buffer。
                            out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')  # 可选 in-place 输出 [T,H]。
                            combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,  # 模拟 expert GEMM 输出。
                                                                                 topk_idx,  # 原始路由矩阵。
                                                                                 topk_weights,  # combine 加权权重。
                                                                                 handle,  # dispatch 返回的低延迟 handle。
                                                                                 use_logfmt=use_logfmt,  # 是否测试 LogFMT。
                                                                                 async_finish=not return_recv_hook,  # hook 模式下不直接等待 event。
                                                                                 zero_copy=zero_copy,  # 是否使用预写 RDMA buffer。
                                                                                 return_recv_hook=return_recv_hook,  # 是否返回接收 hook。
                                                                                 out=out)  # 原地输出 tensor。
                            hook() if return_recv_hook else event.current_stream_wait()  # 等待 combine 接收完成。
                            if shrink_test:  # shrink 模式检查 combine 后 mask 状态。
                                query_mask_buffer_and_check("combine", buffer, mask_status, expected_masked_ranks)  # 验证 mask buffer。
                            if do_check:  # 执行数值正确性检查。
                                if shrink_test:  # shrink 模式下失效 rank 的 expert lane 应从参考中移除。
                                    owner_by_expert = (torch.arange(num_experts, device='cuda') // num_local_experts)  # 每个 expert 所属 rank。
                                    fail_owner_mask = (mask_status == 1).index_select(0, owner_by_expert)  # 每个 expert 是否属于失败 rank。
                                    valid_topk_idx = topk_idx >= 0  # 有效 top-k lane。
                                    failed_topk_idx = torch.zeros_like(topk_idx, device='cuda', dtype=torch.bool)  # 标记需要失效的 top-k lane。
                                    failed_topk_idx[valid_topk_idx] = fail_owner_mask.index_select(0, topk_idx[valid_topk_idx])  # 将失败 owner 映射到 lane。
                                    topk_idx[failed_topk_idx] = -1  # 参考路由中移除失败 rank 的 expert。
                                diff = calc_diff(current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)  # 参考值是输入乘有效权重和。
                                assert torch.isnan(combined_x).sum().item() == 0  # combine 输出不能包含 NaN。
                                if not round_scale:  # round_scale 有额外量化误差，不走严格阈值。
                                    assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}'  # FP8 阈值更宽。
                                hash_value ^= hash_tensor(combined_x)  # 将 combine 输出纳入确定性 hash。

                        # Clean buffer API
                        # 中文：clean_low_latency_buffer 覆盖低延迟控制面清零与 shrink mask 交互。
                        if shrink_test:
                            if simulate_failure_and_skip(rank, "clean", expected_masked_ranks):
                                break

                            # 中文：shrink 测试中清理 buffer 也要覆盖失败 rank 跳过逻辑。
                            buffer.clean_low_latency_buffer(num_tokens, hidden, num_experts)  # 清理低延迟 buffer 的零初始化区域。
                            query_mask_buffer_and_check("clean", buffer, mask_status, expected_masked_ranks)  # clean 后也验证 mask 状态。

    if shrink_test:  # shrink 测试只验证故障和 mask 流程，不跑性能 benchmark。
        return  # 提前结束并返回 None。

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):  # 用大 GEMM 模拟 expert 计算，并在计算后等待通信 hook。
        # 这个函数只用于 benchmark hook 模式：真实 MoE 中这里会是 expert MLP。
        # 先启动一个足够大的 GEMM，让通信 send/recv 有机会与计算重叠；随后调用 hook，
        # 等价于“计算完成前不阻塞，真正需要结果时再等待通信到达”。
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)  # 构造 GEMM 左矩阵。
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)  # 构造 GEMM 右矩阵。
        mat_0 @ mat_1  # 触发一次大矩阵乘，模拟通信/计算 overlap。
        hook()  # 等待低延迟通信的 recv 阶段完成。

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):  # benchmark 用的 dispatch+combine 闭包。
        # return_recv_hook=False 时，dispatch/combine kernel 自己等待接收完成，计时更接近端到端延迟。
        # return_recv_hook=True 时，kernel 只发起请求并返回 hook，后面的 large_gemm_with_hook
        # 模拟上层把接收等待推迟到 expert 计算之后，从而观察 overlap 后的拆分时间。
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,  # 发起 FP8 低延迟 dispatch。
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,  # 复用最近一次统计 tensor。
                                        use_fp8=True, async_finish=False, return_recv_hook=return_recv_hook)  # benchmark 分 hook/non-hook 两类。
        large_gemm_with_hook(hook) if return_recv_hook else None  # hook 模式用 GEMM 覆盖通信接收。
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,  # 模拟 expert 输出。
                                                             topk_idx,  # 原始 top-k 路由。
                                                             topk_weights,  # top-k 权重。
                                                             handle,  # dispatch handle。
                                                             use_logfmt=use_logfmt,  # 是否启用 LogFMT。
                                                             return_recv_hook=return_recv_hook)  # 是否拆分接收等待。
        large_gemm_with_hook(hook) if return_recv_hook else None  # combine hook 模式同样模拟 overlap。

    # Calculate bandwidth：dispatch 按 FP8 message 字节估算，combine 按 BF16 或 LogFMT10
    # message 字节估算；只统计 topk_idx 有效的 expert selection。
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2  # 单个 selection 的 FP8/BF16 估算字节数。
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4  # LogFMT10 单 selection 字节数，含 scale。
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0  # 累计 dispatch/combine 通信字节。
    for i in range(num_tokens):  # 遍历每个 token。
        # 每个有效 top-k selection 都对应一次“发给 expert 所在 rank”的逻辑通信。
        # 如果某个 top-k 为 -1，该槽在 kernel 中会被跳过，带宽估算也不应把它算进去。
        num_selections = (topk_idx[i] != -1).sum().item()  # 当前 token 的有效 expert selection 数。
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections  # dispatch 按 FP8 消息估算。
        num_combine_comm_bytes += (num_logfmt10_bytes if use_logfmt else num_bf16_bytes) * num_selections  # combine 按格式估算。

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))  # 用 CUDA event 测 dispatch+combine 总时间。
    print(
        f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
        # 中文：这里是端到端 dispatch+combine 的总带宽，包含两段通信时间。
        f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us',
        flush=True)

    # Separate profiling
    for return_recv_hook in (False, True):  # 分别 profile 整体 kernel 与拆分 send/recv hook。
        group.barrier()  # profile 前所有 rank 同步。
        dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=return_recv_hook),
                                             kernel_names=('dispatch', 'combine'),
                                             # 中文：barrier_comm_profiling=True 会在 profiler 周期之间插入跨 rank 同步。
                                             barrier_comm_profiling=True,
                                             suppress_kineto_output=True,
                                             # 中文：hook 模式下 dispatch/combine 各拆成 issue 和 recv 两个 kernel period。
                                             num_kernels_per_period=2 if return_recv_hook else 1)
        if not return_recv_hook:  # 非 hook 模式返回单个 dispatch/combine kernel 时间。
            print(
                f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                # 中文：combine 带宽用估算通信字节除以 profiler 提取的 combine kernel 时间。
                f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us',
                flush=True)
        else:  # hook 模式按 issue/recv 两段返回。
            print(
                f'[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | '
                # 中文：hook 模式输出 issue 时间 + recv 等待时间，便于观察 overlap 空间。
                f'Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us',
                flush=True)
    return hash_value  # 压力测试用该 hash 验证相同 seed 重复运行结果一致。


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):  # spawn 后每个进程运行的入口。
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)  # 初始化分布式环境。
    num_tokens, hidden = args.num_tokens, args.hidden  # 读取 token 数和 hidden 维度。
    num_topk, num_experts = args.num_topk, args.num_experts  # 读取 top-k 和 expert 总数。

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)  # 估算低延迟 RDMA buffer 大小。
    if local_rank == 0:  # 每节点 local rank0 打印一次。
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)  # 输出 buffer MB 数。
    buffer = deep_ep.Buffer(group,  # 创建 legacy low-latency Buffer。
                            num_rdma_bytes=num_rdma_bytes,  # 只需要 RDMA buffer。
                            low_latency_mode=True,  # 启用 IBGDA 低延迟路径。
                            num_qps_per_rank=num_experts // num_ranks,  # QP 数等于本地 expert 数。
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink,  # 是否允许 NVLink P2P。
                            explicitly_destroy=True,  # 测试结束显式 destroy。
                            allow_mnnvl=args.allow_mnnvl,  # 是否允许 MNNVL。
                            enable_shrink=args.shrink_test)  # shrink 测试需要 mask buffer。
    test_main(num_tokens,  # 当前 rank token 数。
              hidden,  # hidden 维度。
              num_experts,  # 全局 expert 数。
              num_topk,  # top-k 数。
              rank,  # 当前 rank。
              num_ranks,  # 总 rank 数。
              group,  # torch distributed 组。
              buffer,  # low-latency Buffer。
              use_logfmt=args.use_logfmt,  # 是否测试 LogFMT。
              shrink_test=args.shrink_test,  # 是否测试 shrink/mask。
              seed=1)  # 初始正确性 seed。

    do_pressure_test = args.pressure_test  # 是否进入长时间压力测试。
    for seed in range(int(1e9) if do_pressure_test else 0):  # 压力测试近似无限循环 seed。
        # 压力测试每个 seed 先跑一次作为参考 hash，再重复 20 次比较。
        # 这样能同时覆盖随机路由、双缓冲轮转、hook/非 hook 组合下的确定性。
        if local_rank == 0:  # 只打印一次 seed。
            print(f'Testing with seed {seed} ...', flush=True)  # 输出当前 seed。
        ref_hash = test_main(num_tokens,  # 当前压力测试 token 数。
                             hidden,  # hidden 维度。
                             num_experts,  # expert 数。
                             num_topk,  # top-k 数。
                             rank,  # 当前 rank。
                             num_ranks,  # 总 rank 数。
                             group,  # 分布式通信组。
                             buffer,  # low-latency Buffer。
                             use_logfmt=args.use_logfmt,  # 是否启用 LogFMT。
                             seed=seed)  # 当前随机 seed。
        for _ in range(20):  # 同一 seed 重复运行，验证 hash 稳定性。
            assert test_main(num_tokens,  # 与参考运行相同 token 数。
                             hidden,  # 与参考运行相同 hidden。
                             num_experts,  # 与参考运行相同 expert 数。
                             num_topk,  # 与参考运行相同 top-k。
                             rank,  # 当前 rank。
                             num_ranks,  # 总 rank 数。
                             group,  # 通信组。
                             buffer,  # 复用同一 low-latency Buffer。
                             use_logfmt=args.use_logfmt,  # 与参考运行相同 LogFMT 开关。
                             seed=seed) == ref_hash, f'Error: seed={seed}'  # 同 seed 多次运行 hash 必须一致。

    # Destroy the buffer runtime and communication group
    buffer.destroy()  # 显式释放 legacy Buffer runtime。
    dist.barrier()  # 销毁 process group 前做一次同步。
    dist.destroy_process_group()  # 释放 NCCL process group。


if __name__ == '__main__':  # 允许脚本直接执行。
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    # 这个脚本通常用 torch.multiprocessing 在单机多卡上启动；跨节点时需要外层启动器
    # 负责设置 MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE 等环境变量，init_dist 会读取这些信息。
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')  # 创建参数解析器。
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')  # 本地进程/GPU 数。
    parser.add_argument('--num-tokens', type=int, default=128, help='Number of tokens (default: 128)')  # 每 rank token 数。
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')  # hidden 维度。
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')  # top-k 宽度。
    parser.add_argument('--num-experts', type=int, default=288, help='Number of experts (default: 288)')  # expert 总数。
    parser.add_argument('--allow-mnnvl', action="store_true", help='Allow MNNVL for communication')  # 允许 MNNVL。
    parser.add_argument('--disable-nvlink', action='store_true', help='Whether to disable NVLink for testing')  # 禁用低延迟中的 NVLink。
    parser.add_argument('--use-logfmt', action='store_true', help='Whether to test LogFMT combine')  # 测试 LogFMT combine。
    parser.add_argument("--pressure-test", action='store_true', help='Whether to do pressure test')  # 开启压力测试。
    parser.add_argument("--shrink-test", action='store_true', help='Whether to simulate failure and test shrink mode')  # 开启 shrink/failure 测试。
    args = parser.parse_args()  # 解析命令行。

    num_processes = args.num_processes  # spawn 进程数。
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)  # 启动多进程测试。
