import torch  # 构造 MoE gating 测试数据时依赖 CUDA Tensor、topk、scatter 等操作。


def generate_topk_idx(rank_count: torch.Tensor, num_tokens: int, num_experts: int, num_ranks: int, num_topk: int) -> torch.Tensor:  # 根据每个 rank 的命中次数生成具体 expert id。
    """
    Map rank count to expert indices
    """
    assert torch.equal(torch.sum(rank_count, dim=1), torch.ones(num_tokens, dtype=torch.int, device='cuda') * num_topk)  # 每个 token 的 rank_count 总和必须等于 top-k 数。
    assert (num_tokens, num_ranks) == rank_count.shape  # rank_count 形状为 [num_tokens, num_ranks]。
    num_experts_per_rank = num_experts // num_ranks  # 假设 expert 均匀分布到各 rank，每个 rank 管 num_experts/num_ranks 个 expert。

    # Generate base value
    # rank_count: [num_tokens, num_ranks]，表示每个 token 在每个 rank 上要选择几个 expert。
    # base_vals 把 expert id 组织成 [num_tokens, num_ranks, num_experts_per_rank]，
    # 后续会在每个 rank 内随机打乱，再按 rank_count 截取前若干个 expert。
    base_vals = torch.arange(num_experts, device='cuda').view(1, num_ranks, num_experts_per_rank).expand(num_tokens, num_ranks, num_experts_per_rank)  # 形状 [T,R,E/R]。

    # Randomize the ordering within each row
    rand_vals = torch.rand(num_tokens, num_ranks, num_experts_per_rank, device='cuda')  # 为每个 token-rank 的内部 expert 生成随机排序分数。
    perm_indices = torch.argsort(rand_vals, dim=-1)  # 沿 expert_per_rank 维排序，得到随机排列索引。
    permuted = torch.gather(base_vals, 2, perm_indices)  # 按随机索引重排 expert id，形状仍为 [T,R,E/R]。

    # Create the mask
    k_idx = torch.arange(num_experts_per_rank, device='cuda').view(1, 1, num_experts_per_rank).expand(num_tokens, num_ranks, num_experts_per_rank)  # 每个 rank 内的位置编号。
    rank_count_expanded = rank_count.unsqueeze(2).expand(num_tokens, num_ranks, num_experts_per_rank)  # 把 [T,R] 扩展到 [T,R,E/R] 以便逐位置比较。
    mask = k_idx < rank_count_expanded  # 对每个 token-rank 只保留前 rank_count 个随机 expert。

    # Get the final indices by masking and reshaping
    selected = permuted[mask]  # (num_tokens * num_topk,)；布尔索引会把每行选中的 expert 展平成一维。
    topk_idx = selected.view(num_tokens, num_topk)  # 恢复为 [num_tokens, num_topk]，每行都是该 token 的 expert id。

    return topk_idx  # 返回值用于把这些 expert 的 score 抬高，从而强制 topk 选择符合 rank_count。


def generate_rank_count(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float) -> torch.Tensor:  # 构造每个 token 触达各 rank 的次数。
    """
    Generate rank count tensor for a given number of tokens, experts, ranks, and top-k.

    中文：返回矩阵按 token 行、rank 列记录每个 token 在各 rank 上命中的 expert 槽数。
    This function generates a tensor of shape `(num_tokens, num_ranks)` where each element `[i, j]` represents
    中文：矩阵每行求和等于 num_topk，是后续把 rank 命中次数展开成 expert id 的依据。
    the number of topk experts that token `i` have on rank `j`. The distribution is such that
    one special rank gets `ratio` times more traffic than the others.
    """
    num_experts_per_rank = num_experts // num_ranks  # 每个 rank 拥有的 expert 数量。
    num_normal_ranks = num_ranks - 1  # rank0 被当成特殊高流量 rank，其余 rank 作为 normal ranks。

    assert ratio >= 1.0, 'ratio must be no less than 1.0'  # ratio 小于 1 与“特殊 rank 更热”的建模目标矛盾。

    # Generate rank count of each token from random distribution
    # 先用完全随机 score 估计每个 token 通常会触达多少个不同 rank，得到 a: [num_tokens]。
    # a[i] 表示 token i 的 top-k experts 分布在多少个 rank 上。后续再人为让 rank0 获得
    # ratio 倍流量，同时尽量保持每个 token 触达 rank 的数量分布接近随机 gating。
    random_scores = torch.rand(num_tokens, num_experts, device='cuda')  # 完全随机 gating 分数，形状 [T,E]。
    topk_weights_, topk_indices = torch.topk(random_scores, num_topk, dim=1, largest=True, sorted=False)  # topk_indices 形状 [T,K]。
    topk_indices //= num_experts_per_rank  # 将 expert id 映射成 rank id，仍是 [T,K]。
    sorted_topk_indices = torch.sort(topk_indices, dim=1)[0]  # 对每个 token 的 rank id 排序，便于统计唯一 rank 数。
    topk_indices_diff_mask = sorted_topk_indices[:, 1:] != sorted_topk_indices[:, :-1]  # 相邻 rank id 变化处表示出现一个新 rank。
    a = topk_indices_diff_mask.sum(dim=1) + 1  # a 形状 [T]，表示每个 token 的 top-k 横跨多少个 rank。

    # Upper bound for this generating algorithm
    upper_bound_per_token = int(num_normal_ranks / ratio) + 1  # ratio 越大，单 token 分配给 normal rank 的上限越小。

    # Clamp the value in range [1, upper_bound_per_token] for each token
    a = torch.clamp(a, None, upper_bound_per_token)  # 限制每个 token 涉及的 rank 数，给特殊 rank 留出流量空间。

    # Consider the special rank
    sum_a = torch.sum(a).item()  # 所有 token 的 rank 槽总数，是后续分配 normal/special 流量的总预算。
    normal_token_count = int(sum_a / (num_normal_ranks + ratio))  # 估算每个 normal rank 的 token 槽数。
    special_token_count = sum_a - normal_token_count * num_normal_ranks  # 剩余槽数给特殊 rank0。
    special_token_count = min(special_token_count, int(normal_token_count * ratio) + 1)  # 控制 rank0 不超过目标 ratio 太多。

    # Tokens that the special rank must be in topk
    must_mask = (a == num_ranks)  # 若某 token 需要覆盖所有 rank，则特殊 rank0 必须包含在内。
    must_count = int(must_mask.sum().item())  # 必选 rank0 的 token 数。
    special_token_count = max(must_count, special_token_count)  # 至少保证所有 must token 都能放入 rank0。
    assert must_count <= special_token_count, 'Too many tokens with full rank assignment'  # 防御性断言，理论上上一行已保证成立。

    # Tokens that the special rank can optionally be in topk
    optional_token_indices = torch.where(must_mask == 0)[0]  # 可以选择是否包含 rank0 的 token 行号。
    optional_token_indices = optional_token_indices[torch.randperm(num_tokens - must_count, device='cuda')][:special_token_count - must_count]  # 随机抽取一部分可选 token 给 rank0。
    must_token_indices = torch.where(must_mask != 0)[0]  # 必须包含 rank0 的 token 行号。
    special_token_row_index = torch.cat(([must_token_indices, optional_token_indices]))  # 合并得到最终包含特殊 rank0 的 token 行。

    # Generate permutations for normal ranks
    rank_perm = (torch.randperm(num_normal_ranks, device='cuda') + 1).repeat(num_tokens * num_topk // num_normal_ranks + 1)  # 生成 normal rank 的循环随机序列，值域为 [1,R)。

    # Compute cumulative sum of a to get starting indices in b for each row
    a_cumsum = torch.cumsum(torch.cat((torch.tensor([0], device='cuda'), a)), dim=0)  # 前缀和长度 T+1，用来把每行 rank 槽映射到扁平数组。
    row_starts = a_cumsum[:-1]  # Starting indices for each row in b, shape (n,)

    # Insert special rank index into the permutation for special tokens
    # special rank 固定为 rank0。special_token_mask 在扁平化的 [num_tokens * num_topk]
    # 空间标记哪些 token 的第一个 rank 槽必须放 rank0，从而精确控制 rank0 的流量占比。
    rank_perm_with_special_rank = torch.zeros(num_tokens * num_topk, dtype=torch.long, device='cuda')  # 扁平 rank 槽数组，长度 T*K。
    special_token_mask = torch.zeros(num_tokens * num_topk, dtype=torch.bool, device='cuda')  # 标记哪些扁平槽固定为 rank0。
    special_token_flattened_row_index = row_starts[special_token_row_index]  # 每个 special token 的第一个 rank 槽位置。
    special_token_mask[special_token_flattened_row_index] = 1  # 把这些槽留给特殊 rank0；数组默认值 0 即 rank0。
    all_indices = torch.arange(num_tokens * num_topk, device='cuda')  # 所有扁平槽位置 [0, T*K)。
    non_special_indices = all_indices[special_token_mask != True]  # 其他槽由 normal rank 随机序列填充。
    rank_perm_with_special_rank[non_special_indices] = rank_perm[:len(non_special_indices)]  # 写入 normal rank id，特殊槽保持 0。

    # Create column index grids
    col_idx = torch.arange(num_topk, device='cuda').view(1, num_topk)  # (1, num_topk)，广播到每个 token 的 top-k 列。

    # Compute modulo indices: col_idx % a[i] for each row
    # torch.max is used to avoid zeros in case a[i] = 0 (which happens when the only topk rank is the special rank)
    mod_idx = col_idx % a.view(num_tokens, 1)  # (T,K)，让每行只在自己的 a[i] 个 rank 槽中循环取值。

    # Compute indices in b: row_start + (col % a[i])
    b_indices = row_starts.view(num_tokens, 1) + mod_idx  # (T,K)，每个 token top-k 对应扁平 rank 槽索引。

    # Gather values from b using computed indices
    result = rank_perm_with_special_rank[b_indices]  # result 形状 [T,K]，每个元素是该 top-k expert 应落在哪个 rank。

    # Shuffle rows randomly to avoid any pattern
    shuffle_indices = torch.randperm(num_tokens, device='cuda')  # 随机打乱 token 行，避免构造数据呈现明显块状模式。
    result = result[shuffle_indices]  # Shuffle rows；形状仍为 [T,K]。

    # Create rank count tensor
    rank_count = torch.zeros((num_tokens, num_ranks), dtype=torch.int32, device='cuda')  # 输出矩阵 [T,R]，每行统计各 rank 出现次数。
    rank_count.scatter_add_(dim=1, index=result, src=torch.ones_like(result, dtype=torch.int32))  # 把 [T,K] rank id 计数到 [T,R]。
    return rank_count  # 每行和为 num_topk，后续 generate_topk_idx 会据此采样 expert id。


def get_precise_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float):  # 生成严格控制 rank0 热点比例的分数矩阵。
    """
    Generate precise unbalanced scores for testing.

    中文：该函数构造可控热点路由分布，用于压测某个 rank 流量倾斜时的 dispatch 行为。
    Note that this function generates a distribution with precise unbalanced distribution,
    中文：它服务测试/benchmark，不模拟真实 gate 分布的随机性。
    which **differs from real distribution**.
    """
    # Generate num topk experts for each rank
    # 中文：先生成每个 token 在各 rank 的 top-k 命中次数，再把次数展开成具体 expert 分数。
    rank_count = generate_rank_count(num_tokens, num_experts, num_ranks, num_topk, ratio)  # rank_count 形状 [T,R]，指定每个 token top-k 横跨哪些 rank。

    # Generate scores in a low distribution
    threshold = 0.9  # 低分/高分的分界线；低分在 [0,0.9)，高分在 (0.9,1]。
    scores = torch.empty((num_tokens, num_experts), dtype=torch.float32, device='cuda')  # 分数矩阵形状 [T,E]。
    scores.uniform_(to=threshold)  # 先把所有 expert 初始化为低分，确保默认不会进入 top-k。

    # Generate topk indices and change their scores to a high distribution
    topk_idx = generate_topk_idx(rank_count, num_tokens, num_experts, num_ranks, num_topk)  # 目标 expert id 形状 [T,K]。
    topk_scores = torch.empty((num_tokens, num_topk), dtype=torch.float32, device='cuda')  # 为目标 expert 生成高分，形状 [T,K]。
    topk_scores.uniform_(threshold + 1e-6, 1.0)  # 高分严格大于 threshold，保证这些位置被 topk 选中。
    row_idx = torch.arange(num_tokens).unsqueeze(1).expand(num_tokens, num_topk)  # 行索引形状 [T,K]，与 topk_idx 配合高级索引。
    scores[row_idx, topk_idx] = topk_scores  # 将指定 expert 的分数抬高，使 top-k 分布严格匹配 rank_count。
    return scores  # 返回 [T,E] 分数矩阵；torch.topk(scores,K) 会得到预设的不均衡路由。


def get_scores_by_factor(num_tokens: int, num_experts: int, num_ranks: int, factor: float) -> torch.Tensor:  # 用分数放大因子制造 rank0 热点。
    num_experts_per_rank = num_experts // num_ranks  # rank0 的 expert 范围是 [0, num_experts_per_rank)。
    scores = torch.empty((num_tokens, num_experts), dtype=torch.float32, device='cuda')  # 分数矩阵形状 [T,E]。
    scores[:, :num_experts_per_rank].uniform_(to=factor)  # rank0 expert 分数范围 [0,factor)，factor 越大越容易进 top-k。
    scores[:, num_experts_per_rank:].uniform_(to=1)  # 其他 rank expert 分数范围 [0,1)，作为普通流量基线。
    return scores  # 返回随机不均衡分数；不保证精确 ratio，只保证统计倾向。


def map_unbalanced_ratio_to_factor(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float) -> float:  # 用二分搜索把目标 ratio 映射为分数放大因子。
    num_iterations = 20  # 固定迭代次数，足够把 factor 收敛到测试可用精度。
    factor_l, factor_r = 1.0, 100.0  # factor 搜索区间；1 表示均衡，100 表示 rank0 极热。

    num_experts_per_rank = num_experts // num_ranks  # 把 expert id 映射到 rank id 时需要这个分段宽度。
    for _i in range(num_iterations):  # 每轮生成一批随机 score 并估算当前 factor 的 rank0 热度。
        factor_mid = (factor_l + factor_r) / 2  # 二分候选 factor。
        scores = get_scores_by_factor(num_tokens, num_experts, num_ranks, factor_mid)  # 构造 [T,E] 随机分数。
        _, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)  # topk_idx 形状 [T,K]。
        rank_idx = topk_idx // num_experts_per_rank  # 将 expert id 转成 rank id，形状 [T,K]。
        one_hot = torch.nn.functional.one_hot(rank_idx, num_ranks)  # 形状 [T,K,R]，每个 top-k 槽一个 rank one-hot。
        counts = one_hot.any(dim=1).to(torch.float).sum(dim=0)  # counts 形状 [R]，统计有多少 token 触达各 rank。
        if counts[0].item() > counts[1:].mean().item() * ratio:  # rank0 过热时降低 factor 上界。
            factor_r = factor_mid  # 收缩右边界，让下一轮 rank0 分数稍弱。
        else:  # rank0 未达到目标 ratio 时需要增强 factor。
            factor_l = factor_mid  # 抬高左边界，让下一轮 rank0 更容易进入 top-k。
    return factor_l  # 返回保守下界，避免随机波动让 rank0 超过目标过多。


def get_random_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float):  # 生成统计意义上的不均衡 gating 分数。
    """Generate unbalanced scores with a given ratio.
    """
    factor = 1.0  # ratio=1 时不放大 rank0，得到接近均衡的随机分数。
    if ratio != 1.0:  # 只有目标不均衡时才需要二分搜索 factor。
        factor = map_unbalanced_ratio_to_factor(num_tokens, num_experts, num_ranks, num_topk, ratio)  # 将目标流量比转成 score 放大系数。
    return get_scores_by_factor(num_tokens, num_experts, num_ranks, factor)  # 返回 [T,E] 分数矩阵。


def get_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float, precise: bool):  # 统一入口：按 precise 选择精确或随机不均衡数据。
    if precise:  # precise=True 用构造式方法严格控制 rank_count。
        return get_precise_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio)  # 返回严格 top-k 分布的 [T,E] scores。
    else:  # precise=False 更接近真实随机 gating，但 ratio 只是统计近似。
        return get_random_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio)  # 返回随机放大 rank0 的 [T,E] scores。
