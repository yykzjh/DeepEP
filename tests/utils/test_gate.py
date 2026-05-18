import torch  # 测试 gate score 生成逻辑时使用 CUDA tensor/topk/scatter 等操作。

from deep_ep.utils.math import ceil_div  # 计算 top-k 至少会覆盖多少个 rank。
from deep_ep.utils.gate import get_unbalanced_scores  # 被测函数：生成不均衡 MoE gate 分数。


def test_unbalanced_scores():  # 验证不均衡 gate score 生成函数在多组配置下接近期望 ratio。
    print('Testing gate score generation (Output with num_tokens = 4096, num_experts = 512):')  # 打印测试标题。
    # get_unbalanced_scores 生成一个偏向 rank0 的 gate 分数矩阵，测试覆盖不同
    # num_topk/num_ranks/experts_per_rank 组合，验证实际 rank0 流量比例接近期望 ratio。
    for num_tokens in [1, 4096]:  # 覆盖极小 batch 和常规 batch。
        for num_experts_per_rank in [1, 4, 8, 16]:  # 覆盖每 rank expert 数的不同粒度。
            for num_ranks in [2, 4, 8, 16, 64, 72]:  # 覆盖小规模、本地 8 卡、大规模和非 2 次幂 rank 数。
                num_experts = num_experts_per_rank * num_ranks  # 总 expert 数 E=experts_per_rank*R。
                for num_topk in [1, 2, 4, 6, 8, 9]:  # 覆盖常见 MoE top-k 宽度。
                    if num_topk > num_experts:  # top-k 不能超过总 expert 数。
                        continue  # 不合法组合直接跳过。
                    for ratio in [1.0, 2.0, 4.0]:  # rank0 相对其他 rank 的目标流量倍数。
                        for precise in [1, 0]:  # precise=1 为精确构造，0 为随机近似构造。
                            total_rank_count = torch.zeros(num_ranks, device='cuda')  # 累计各 rank 被 token 触达次数，形状 [R]。

                            # precise 算法要求每个 token 至少覆盖 lower_bound_per_token 个 rank，
                            # 且最多覆盖 upper_bound_per_token 个 rank；不可满足时跳过该组合。
                            lower_bound_per_token = max(1, ceil_div(num_topk, num_experts_per_rank))  # top-k 塞入 rank 时至少需要的 rank 数。
                            upper_bound_per_token = min(min(num_topk, num_ranks), int((num_ranks - 1) / ratio) + 1)  # precise 算法允许的每 token 最大 rank 数。
                            if lower_bound_per_token > upper_bound_per_token:  # 目标 ratio 与 top-k/expert 布局不可同时满足。
                                continue  # 跳过不可构造的 precise 配置。

                            # 重复 num_ranks 次是为了平滑随机性；rank_count 形状
                            # [num_tokens, num_ranks]，每行表示该 token 是否发往某 rank。
                            for rank_idx in range(num_ranks):  # 多次采样平滑随机波动；rank_idx 仅作为重复计数变量。
                                scores = get_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio, precise)  # scores 形状 [T,E]。
                                _topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)  # topk_idx 形状 [T,K]。
                                topk_idx = topk_idx // num_experts_per_rank  # expert id 转 rank id，形状仍为 [T,K]。
                                row_indices = torch.arange(num_tokens).unsqueeze(1).expand(num_tokens, num_topk).flatten()  # 展平后的 token 行索引 [T*K]。
                                topk_idx = topk_idx.flatten()  # 展平 rank id 为 [T*K]，与 row_indices 对齐。
                                rank_count = torch.zeros((num_tokens, num_ranks), device='cuda')  # 每个 token 是否触达各 rank 的矩阵 [T,R]。
                                rank_count[row_indices, topk_idx] = 1  # 同一 token 多次命中同 rank 只记一次触达。
                                rank_count = rank_count.sum(dim=0)  # 汇总到 rank 维，得到 [R]。
                                total_rank_count += rank_count  # 累计多轮采样结果。

                            # practical_ratio 衡量 rank0 与其他 rank 最小流量的比值，
                            # inequality 衡量非 rank0 之间是否均衡。
                            practical_ratio = total_rank_count[0].item() / max(total_rank_count[1:].min().item(), 1)  # rank0 / 最冷 normal rank。
                            inequality = total_rank_count[1:].max().item() / max(total_rank_count[1:].min().item(), 1)  # normal ranks 之间的最热/最冷比。
                            total_sent_tokens = int(total_rank_count.sum().item())  # 所有 rank 触达次数总和。
                            if num_tokens > 1000:  # 大 batch 统计更稳定，才打印/断言比例。
                                if num_ranks in [8, 64] and num_experts_per_rank == 8:  # 只打印代表性配置，避免日志过长。
                                    print(f' > {precise=}, {num_ranks=:2d}, {num_topk=}, expected_ratio={ratio} | '  # 打印配置前半段。
                                        f'ratio={practical_ratio:6.3f}, {inequality=:6.3f}, {total_sent_tokens=:7d}')  # 打印统计结果。

                                # Only check the ratio and inequality in precise mode
                                if precise:  # 精确构造模式应满足严格比例要求；随机模式只观察不强断言。
                                    assert abs(practical_ratio - ratio) / ratio < 0.1 and inequality < 1.02, (  # ratio 误差 <10%，normal rank 不均衡 <2%。
                                            f'Failed to generate unbalanced scores with following config: \n'  # 失败信息第一行。
                                            f'{precise=}, {num_tokens=}, {num_experts=:3d}, {num_ranks=:2d}, {num_topk=}, expected_ratio={ratio} | '  # 配置详情。
                                            f'ratio={practical_ratio:6.3f}, {inequality=:6.3f}, {total_sent_tokens=:7d}'  # 统计详情。
                                    )  # 用括号续行，便于行尾中文注释。
    print()  # 测试结束后输出空行，分隔后续日志。


if __name__ == '__main__':  # 允许直接 python tests/utils/test_gate.py 运行。
    test_unbalanced_scores()  # 执行 gate 分数生成测试。
