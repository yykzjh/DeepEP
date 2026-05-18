import math  # generate_pre_combine_data 使用 sin 构造确定性的参考输出。
import torch  # 参考实现使用 PyTorch tensor/all_to_all/topk 等算子模拟 DeepEP kernel 行为。
import torch.distributed as dist  # dispatch 参考实现通过 all_to_all_single 在 rank 间交换 token。
from typing import Optional, Tuple, Union  # 标注 FP8 tuple 输入、可选 bias/topk_weights 等类型。

from .envs import get_global_seed  # 参考输出数据使用全局 seed，确保所有 rank 生成一致函数。
from .math import ceil_div  # 计算 FP8 scale-factor 列数 hidden_sf=ceil(hidden/128)。


def dispatch(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
             topk_idx: torch.Tensor, topk_weights: Optional[torch.Tensor],
             num_max_tokens_per_rank: int, num_experts: int):  # PyTorch 版 dispatch 参考实现，用于验证 CUDA kernel。
    """
    The reference implementation of dispatching tokens to experts across multiple ranks.

    Not expanded. Sorted by rank and then by token within each rank (i.e. sorted by `src_token_global_idx`).

    Arguments:
    中文：该参考实现按 rank 排序 token，用于测试 DeepEP dispatch 的路由和 metadata。
    - `x`: Input tokens, `[num_tokens, hidden]` or (`[num_tokens, hidden], [num_tokens, hidden_sf]`)
      中文：FP8 输入时 x 是 data/scale 二元组，参考实现保持同构返回。
    - `topk_idx`: Top-k expert indices for each token, `[num_tokens, num_topk]`
      中文：topk_idx 的 -1 表示空路由槽，参考实现会跳过它。
    - `topk_weights`: Top-k weights for each token, can be None, `[num_tokens, num_topk]`
    - `num_max_tokens_per_rank`: Maximum number of tokens per rank, must >= actual number of tokens per rank and aligned with DeepEP's `num_max_tokens_per_rank` since we're going to calculate `src_token_global_idx = src_rank_idx*num_max_tokens_per_rank + src_token_local_idx`
    - `num_experts`: Total number of experts across all ranks
    中文：num_max_tokens_per_rank 必须和 kernel 容量一致，否则 src_token_global_idx 编码无法对齐。

    Returns:
    中文：返回 dispatch 后的 token、路由 metadata、源 token 编码和每 rank 接收计数。
    - `recv_x`, `recv_topk_idx`, and `recv_topk_weights`: Received tokens, top-k indices, and top-k weights after dispatching. Out of range `recv_topk_idx` (i.e. that expert is not on the current rank) are set to -1.
    - `recv_src_token_idx`: Received `src_token_global_idx` for each received token
    - `num_recv_tokens_per_rank`: Number of received tokens from each rank, `[num_ranks]`

    中文说明：
    - 这是 correctness reference，不追求性能；它用 torch.distributed.all_to_all_single
      模拟 DeepEP dispatch 后的可观察结果。
    - 输入 x 的形状是 [T,H]；FP8 时 x=(data, sf)，其中 sf 形状为 [T, ceil(H/128)]。
    - topk_idx 形状 [T,K]，任一 top-k expert 属于目标 rank 时，该 token 会被发送到该 rank。
    - 发送给目标 rank 后，recv_topk_idx 会从全局 expert id 转成目标 rank 内的 local expert id；
      不属于当前 rank 的 top-k 槽被置为 -1，这与 CUDA dispatch epilogue 输出语义一致。
    - recv_src_token_idx 采用 src_rank*num_max_tokens_per_rank+src_local_token 的编码，
      它是 combine 参考实现和 CUDA metadata 对齐的关键字段。
    """
    # TODO: support forwarding
    # TODO: make top-k weight fully optional
    rank_idx = dist.get_rank()  # 当前全局 rank，决定本 rank 拥有哪些 expert。
    num_ranks = dist.get_world_size()  # 总 rank 数，expert 会均匀切分到这些 rank 上。

    assert num_experts % num_ranks == 0  # 参考实现假设 expert 能被 rank 均分。
    num_experts_per_rank = num_experts // num_ranks  # 每个 rank 的 expert 数量。

    # Unpack SF
    use_fp8 = isinstance(x, tuple)  # FP8 输入用 (x_fp8, scale_factor) 表示；普通 BF16 输入是单 Tensor。
    x, sf = x if use_fp8 else (x, None)  # 解包后 x 形状 [num_tokens, hidden]，sf 可能为 [num_tokens, hidden_sf]。

    # TODO: use SF bytes instead of hardcoded recipe
    num_tokens, hidden = x.size()  # token 激活矩阵形状 [T,H]。
    num_tokens_, num_topk = topk_idx.size()  # 路由矩阵形状 [T,K]。
    assert num_tokens == num_tokens_  # x 和 topk_idx 必须描述同一批 token。
    if sf is not None:  # FP8 路径需要额外检查 scale-factor 形状。
        num_tokens__, hidden_sf = sf.size()  # sf 形状 [T,ceil(H/128)]。
        assert num_tokens == num_tokens__  # scale-factor token 数必须与 x 一致。
        assert hidden_sf == ceil_div(hidden, 128)  # 每 128 hidden 元素对应一个 scale。
    if topk_weights is not None:  # topk_weights 可选，但当前 TODO 中尚未完全支持 None。
        num_tokens__, num_topk_ = topk_weights.size()  # 权重形状 [T,K]。
        assert num_tokens == num_tokens__  # 权重 token 数必须与 x 一致。
        assert num_topk == num_topk_  # 权重 top-k 宽度必须与 topk_idx 一致。

    # Prepare per-peer send buffers
    # 参考实现故意使用 PyTorch all_to_all_single，而不是 DeepEP kernel。
    # 它按目标 rank 构造 send_x/send_topk_idx/send_src_token_idx：
    # - mask_to_send: [num_tokens]，只要 token 的任一 top-k expert 属于 dst_rank 就发送；
    # - send_src_token_idx: 本地 token index 加上 rank_idx * num_max_tokens_per_rank，得到全局 token id。
    # 这个全局 id 与 C++/CUDA metadata 中的 src token 编码一致，测试可直接比较。
    send_x_list = []  # 分 rank 收集待发送 token 激活，每个元素形状 [tokens_to_dst,H]。
    send_sf_list = []  # FP8 路径收集 scale-factor，每个元素形状 [tokens_to_dst,hidden_sf]。
    send_topk_idx_list = []  # 分 rank 收集 mask 后的 topk_idx，每个元素形状 [tokens_to_dst,K]。
    send_topk_weights_list = []  # 分 rank 收集 topk 权重，每个元素形状 [tokens_to_dst,K]。
    send_src_token_idx_list = []  # 分 rank 收集源 token 本地 index，后面会编码成全局 id。
    num_send_tokens_per_rank = torch.zeros((num_ranks, ), dtype=torch.int, device=x.device)  # 发送计数向量 [R]。
    for dst_rank_idx in range(num_ranks):  # 逐个目标 rank 构造待发送数据。
        expert_start_idx = dst_rank_idx * num_experts_per_rank  # 目标 rank 拥有的 expert 起始全局 id。
        expert_end_idx = expert_start_idx + num_experts_per_rank  # 目标 rank 拥有的 expert 结束全局 id，左闭右开。

        # Get the indices of tokens
        mask_to_send = ((expert_start_idx <= topk_idx) & (topk_idx < expert_end_idx)).any(dim=1)  # [T]，只要一个 top-k expert 属于 dst rank 就发送。
        indices_to_send = mask_to_send.nonzero(as_tuple=True)[0]  # 待发送 token 的本地行号，形状 [tokens_to_dst]。
        num_send_tokens_per_rank[dst_rank_idx] = indices_to_send.numel()  # 记录发给该 rank 的 token 数。

        # Select the data for tokens
        x_to_send = x[indices_to_send]  # 选择激活，形状 [tokens_to_dst,H]。
        sf_to_send = sf[indices_to_send] if use_fp8 else None  # FP8 scale-factor 同步选择，形状 [tokens_to_dst,hidden_sf]。
        topk_idx_to_send = topk_idx[indices_to_send]  # 对应 token 的全部 top-k expert id，形状 [tokens_to_dst,K]。
        topk_weights_to_send = topk_weights[indices_to_send]  # 对应 token 的 top-k 权重，形状 [tokens_to_dst,K]。
        masked_topk_idx = torch.where((expert_start_idx <= topk_idx_to_send) & (topk_idx_to_send < expert_end_idx),  # 本目标 rank 内的 expert 保留。
                                      topk_idx_to_send, torch.full_like(topk_idx_to_send, -1))  # 不属于目标 rank 的 expert 置 -1。

        send_x_list.append(x_to_send)  # 追加到按目标 rank 排列的发送列表。
        send_sf_list.append(sf_to_send)  # 普通 BF16 路径这里追加 None，后续不会 cat。
        send_topk_idx_list.append(masked_topk_idx)  # 追加 mask 后路由信息。
        send_topk_weights_list.append(topk_weights_to_send)  # 追加权重信息。
        send_src_token_idx_list.append(indices_to_send)  # 追加源 token 本地 index。

    send_x = torch.cat(send_x_list, dim=0)  # 按 dst_rank 顺序拼接，形状 [sum_send,H]。
    send_sf = torch.cat(send_sf_list, dim=0) if use_fp8 else None  # FP8 scale-factor 拼接形状 [sum_send,hidden_sf]。
    send_topk_idx = torch.cat(send_topk_idx_list, dim=0)  # 拼接后的路由矩阵 [sum_send,K]。
    send_topk_weights = torch.cat(send_topk_weights_list, dim=0)  # 拼接后的权重矩阵 [sum_send,K]。
    send_src_token_idx = torch.cat(send_src_token_idx_list, dim=0).to(torch.int)  # 拼接源本地 token id，形状 [sum_send]。
    send_src_token_idx += rank_idx * num_max_tokens_per_rank  # 编码成全局 token id：src_rank * max_tokens + local_idx。

    # Exchange size
    num_recv_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device=x.device)  # 接收计数向量 [R]。
    dist.all_to_all_single(num_recv_tokens_per_rank, num_send_tokens_per_rank)  # 先交换每个 peer 的发送/接收数量。
    num_recv_tokens = int(num_recv_tokens_per_rank.sum().item())  # 当前 rank 总接收 token 数，Python int 用于分配输出。

    # Exchange main data
    num_send_tokens_per_rank = num_send_tokens_per_rank.tolist()  # PyTorch all_to_all_single split_sizes 需要 Python list。
    num_recv_tokens_per_rank = num_recv_tokens_per_rank.tolist()  # 接收 split sizes 也转成 list。
    recv_x = torch.empty((num_recv_tokens, hidden), dtype=x.dtype, device=x.device)  # 接收激活缓冲 [recv_T,H]。
    recv_sf = torch.empty((num_recv_tokens, hidden_sf), dtype=sf.dtype, device=x.device) if use_fp8 else None  # FP8 接收 scale [recv_T,hidden_sf]。
    recv_topk_idx = torch.empty((num_recv_tokens, num_topk), dtype=topk_idx.dtype, device=x.device)  # 接收路由 [recv_T,K]。
    recv_topk_weights = torch.empty((num_recv_tokens, num_topk), dtype=topk_weights.dtype, device=x.device)  # 接收权重 [recv_T,K]。
    recv_src_token_idx = torch.empty((num_recv_tokens, ), dtype=torch.int, device=x.device)  # 接收源全局 token id [recv_T]。
    dist.all_to_all_single(recv_x, send_x, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # 交换主激活。
    if use_fp8:  # FP8 路径额外交换 scale-factor。
        dist.all_to_all_single(recv_sf, send_sf, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # 交换 scale-factor。
    dist.all_to_all_single(recv_topk_idx, send_topk_idx, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # 交换路由 metadata。
    dist.all_to_all_single(recv_topk_weights, send_topk_weights, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # 交换 top-k 权重。
    dist.all_to_all_single(recv_src_token_idx, send_src_token_idx, num_recv_tokens_per_rank, num_send_tokens_per_rank)  # 交换源 token id。

    # Mask top-k indices
    expert_start_idx = rank_idx * num_experts_per_rank  # 当前 rank 的 expert 起始全局 id。
    expert_end_idx = expert_start_idx + num_experts_per_rank  # 当前 rank 的 expert 结束全局 id。
    mask = (expert_start_idx <= recv_topk_idx) & (recv_topk_idx < expert_end_idx)  # 标记哪些 top-k 槽属于当前 rank。
    recv_topk_idx = recv_topk_idx - expert_start_idx  # 把全局 expert id 转成本 rank 内局部 expert id。
    recv_topk_idx.masked_fill_(~mask, -1)  # 不属于本 rank 的 top-k 槽置 -1，和 CUDA kernel 输出语义一致。

    # Pack SF
    recv_x = (recv_x, recv_sf) if use_fp8 else recv_x  # FP8 路径把激活和 scale-factor 重新打包成 tuple。

    return (recv_x, recv_topk_idx, recv_topk_weights,  # 返回接收激活、局部 expert id、权重。
            recv_src_token_idx, torch.tensor(num_recv_tokens_per_rank, dtype=torch.int))  # 同时返回源 token id 和每 peer 接收计数。


def generate_pre_combine_data(src_token_global_idx: torch.Tensor,
                              num_max_tokens_per_rank: int, num_topk: int, hidden: int) -> torch.Tensor:  # 根据源 token id 生成 deterministic expert 输出。
    """
    Generate data needed for combine from `src_token_global_idx`.
    Recall that `src_token_global_idx = src_rank_idx * num_max_tokens_per_rank + src_token_local_idx`.
    中文：这个编码和 C++ SourceMeta 的源 token 还原逻辑保持一致。
    The generated data (denoted as `y`) of the i-th token has a shape of [num_topk, hidden], with
    中文：该函数故意用确定性三角函数生成 expert 输出，避免随机误差影响 combine 校验。

    `y[j, k] = sin((token_seeds * P % max_seed + 1) / max_seed * (k + 1) + sin(seed))`

    where `P=131071` is a large prime, `token_seeds` is calculated via `token_seeds = src_token_global_idx[i] * num_topk + j`,
    and `max_seed = num_ranks * num_max_tokens_per_rank * num_topk`.

    Arguments:
    - `src_token_global_idx`: Source token global indices, `[num_tokens]`

    Returns:
    - Generated data, `[num_tokens, num_topk, hidden]`

    中文说明：
    - 该函数根据 src_token_global_idx 构造确定性 expert 输出，而不是使用随机数。
    - 输出形状为 [T,K,H]，其中 T=len(src_token_global_idx)，K=num_topk。
    - 同一个 src_token_global_idx/top-k lane 在所有 rank 上都会生成相同数据，
      因此可以在 dispatch 改变 token 顺序后仍稳定验证 combine 正确性。
    """
    # 这里不用随机数直接生成 y，而是用 src_token_global_idx 构造确定性函数。
    # 这样 dispatch 后 token 顺序即使被 DeepEP kernel 重排，只要 src_token_global_idx 相同，
    # reference 和 kernel 输入就能得到完全一致的 expert 输出，便于验证 combine。
    num_ranks = dist.get_world_size()  # 总 rank 数，用于计算 token seed 的最大范围。
    token_seeds = (src_token_global_idx.unsqueeze(1) * num_topk +  # [num_tokens,1] 扩展出每个 token 的 top-k seed 基础值。
                   torch.arange(num_topk, device=src_token_global_idx.device).unsqueeze(0))  # 加上 top-k 槽 id，得到 [num_tokens,num_topk]。
    max_seed = num_ranks * num_max_tokens_per_rank * num_topk  # seed 空间上界，覆盖所有 rank/token/top-k 槽。
    result = torch.sin(  # 构造 [num_tokens,num_topk,hidden] 的确定性浮点模式。
        (((token_seeds * 131071 % max_seed).float() + 1) / max_seed).unsqueeze(-1) *  # 质数扰动后归一化到 (0,1]，形状 [T,K,1]。
        torch.arange(1, hidden + 1, device=src_token_global_idx.device, dtype=torch.float32).broadcast_to(1, 1, hidden) +  # hidden 维频率项 [1,1,H]。
        math.sin(float(get_global_seed()))  # 全局 seed 相位偏移，使不同测试 seed 得到不同参考数据。
    )  # result 初始 dtype 为 FP32，便于 sin 计算稳定。
    return result.to(torch.bfloat16)  # 返回 BF16 [T,K,H]，与 DeepEP 常用激活 dtype 对齐。


def ordered_accumulate(data: torch.Tensor, initial_value: Optional[torch.Tensor] = None) -> torch.Tensor:  # 按 top-k 顺序累加专家输出。
    """
    Accumulate `data` in order along the num_topk dimension.

    Arguments:
    - `data`: Data to be accumulated, `[num_tokens, num_topk, hidden]`
    - `initial_value`: Initial value for accumulation, `[num_tokens, hidden]`

    Returns:
    - Result, `[num_tokens, hidden]`

    中文说明：
    - data 形状为 [T,K,H]，函数严格按 K 维从左到右累加。
    - initial_value 可作为 bias 或已有 partial sum，形状 [T,H]。
    - 使用 FP32 累加再转回原 dtype，尽量贴近 CUDA combine 中的数值处理。
    """
    num_topk = data.shape[1]  # data 形状 [T,K,H]，第二维是需要按序累加的 top-k。
    if initial_value is None:  # 没有 bias/初值时从 0 开始累加。
        result = torch.zeros((data.shape[0], data.shape[2]), dtype=torch.float32, device=data.device)  # 累加缓冲 [T,H]，使用 FP32 降低误差。
    else:  # combine 支持传入 bias 或已有部分和。
        result = initial_value.clone()  # clone 避免原始 bias 被原地修改。
    for i in range(num_topk):  # 严格按 top-k 维度从左到右累加，匹配 CUDA kernel 的顺序语义。
        result += data[:, i, :].float()  # 当前 top-k 槽 [T,H] 转 FP32 后加入结果。
    return result.to(data.dtype)  # 转回输入 dtype，通常为 BF16。


def combine(y: torch.Tensor, topk_idx: torch.Tensor,
            num_scaleout_ranks: int, num_scaleup_ranks: int, num_experts: int,
            bias: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
            reduce_in_local: bool, reduce_in_scaleup: bool) -> torch.Tensor:  # PyTorch 版 combine 参考实现。
    """
    The reference implementation of (possibly multi-level reduction) combining tokens.

    Arguments:
    中文：该函数模拟 Elastic combine 的多层规约顺序，用于测试 hybrid/non-hybrid 行为。
    - `y`: Input tokens to be combined, `[num_tokens, num_topk, hidden]`
    - `topk_idx`: `[num_tokens, num_topk]`
    - `reduce_in_local` and `reduce_in_scaleup`: Whether to do reduction within rank and within scale-up group.
       中文：下面三种组合对应 Elastic combine 中不同的规约层级与 bias 注入时机。
       - `(True, True)` -> Hybrid combine
       - `(True, False)` -> Non-hybrid combine
       - `(False, False)` -> Equivalent to `allow_multiple_reduction` is False
       中文：第三种组合保留更多独立贡献，通常通信量更大但规约顺序更接近单次累加。
       Pay attention that `reduce_in_scaleup` = `True` or `False` is NOT equivalent even if `num_scaleout_ranks == 1` due to `bias` handling.
       中文：即使只有一个 scaleout rank，bias 注入位置不同也会让两种规约顺序不等价。

    Returns:
    - Combined result, `[num_tokens, hidden]`
    中文：输出按原 token 顺序排列。

    中文说明：
    - y 是每个原 token 的 top-k expert 输出，形状 [T,K,H]。
    - reduce_in_local=True 时先把同一 rank 内多个 expert 贡献合并；
      reduce_in_scaleup=True 时再把同一 scaleout 域内的贡献合并。
    - 最后调用 ordered_accumulate 沿 top-k 维规约成 [T,H]，并可叠加 bias。
    - 这个参考函数对应 elastic/hybrid combine 的不同 reduction 策略，用于测试中比较 kernel 输出。
    """
    num_ranks = num_scaleout_ranks * num_scaleup_ranks  # 逻辑 rank 总数 = scaleout 维度 * scaleup 维度。
    num_tokens, hidden = y.shape[0], y.shape[2]  # y 形状 [T,K,H]，这里取 T 和 H。
    num_topk = y.shape[1]  # top-k 宽度 K。
    assert not (not reduce_in_local and reduce_in_scaleup), 'Invalid reduction configuration'  # 不能跳过 local reduce 却做 scaleup reduce。

    def grouped_reduce(data_to_reduce: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:  # 按 group_id 对同一 token 的 top-k 槽做分段归约。
        """
        Perform in-place grouped reduction on `data_to_reduce` according to `group_id`.
        The summation within each group are performed in strict order along the `num_topk` dimension.
        中文：严格 top-k 顺序用于模拟 kernel 中固定顺序累加，避免浮点加法交换律差异。
        The result for each group is stored at the rightmost token of that group, and other tokens are set to zero.

        Arguments:
        - `data_to_reduce`: Data to be reduced, `[num_tokens, num_topk, hidden]`
        - `group_id`: group IDs for each token, `[num_tokens, num_topk]`

        中文说明：
        - group_id 与 data_to_reduce 的前两维相同，表示每个 top-k 槽属于哪个规约组。
        - stable sort 后，同组槽位在同一 token 行内相邻；函数把每组的和写到该组最右侧槽位。
        - 组内中间槽会被置零，避免后续 ordered_accumulate 再次重复累加。
        """
        # Shuffle to make tokens with the same group_id contiguous
        # group_id 形状为 [num_tokens, num_topk]。例如 hybrid combine 中，第一次 reduce
        # 可按 rank 分组，第二次可按 scaleout group 分组。stable sort 保证同组内 top-k 顺序稳定，
        # 对齐 device kernel 中“按 top-k 顺序累加”的数值语义。
        group_id, src_indices = torch.sort(group_id, dim=-1, stable=True)  # 每行按 group id 稳定排序，返回排序后的 id 和源列索引。
        # transformed_src_indices[i, j] = i * num_topk + src_indices[i, j]
        transformed_src_indices = (  # 把二维 [token,topk] 源列索引转换为扁平索引。
            (src_indices + torch.arange(0, num_tokens, device=y.device).unsqueeze(-1) * num_topk).flatten())  # 形状 [T*K]。
        data_to_reduce = data_to_reduce.view(-1, hidden)[transformed_src_indices].view(num_tokens, num_topk, hidden)  # 按排序索引重排 y，仍为 [T,K,H]。
        # Perform segmented reduce within each group
        # data_to_reduce: [num_tokens, num_topk, hidden]。遍历 top-k 维度时，cur_accum_buf
        # 保存当前 segment 的 FP32 累加值；遇到 segment break 才写回右侧槽位，其余槽位置零。
        # 这模拟了 CUDA combine 在 allow_multiple_reduction 下的局部归约行为。
        cur_accum_buf = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=y.device)  # 当前 segment 的累加缓冲 [T,H]。
        for i in range(num_topk):  # 沿排序后的 top-k 槽逐列扫描。
            is_segment_break = (  # 标记每个 token 在第 i 列是否到达当前 group segment 的末尾。
                torch.full((num_tokens, ), True, dtype=torch.bool, device=y.device)  # 最后一列一定是 segment 结束。
                if i == num_topk - 1 else group_id[:, i] != group_id[:, i + 1]  # 否则与下一列 group_id 不同即为结束。
            )  # 形状 [T]。
            cur_accum_buf += data_to_reduce[:, i, :].float()  # 当前槽贡献加入 segment 累加缓冲。
            # For one token, if `is_segment_break` is True,
            # save the accumulated value and clear the buffer, otherwise, clear `data_to_reduce[:, i, :]`
            segment_break_token_indices = torch.where(is_segment_break)[0]  # 到达 segment 末尾的 token 行号。
            data_to_reduce[segment_break_token_indices, i] = cur_accum_buf[segment_break_token_indices].to(data_to_reduce.dtype)  # 把 segment 和写回右端槽。
            cur_accum_buf[segment_break_token_indices] = 0.0  # 完成写回的行清空累加缓冲，准备下一个 segment。
            non_segment_break_token_indices = torch.where(~is_segment_break)[0]  # 还没到 segment 末尾的 token 行号。
            data_to_reduce[non_segment_break_token_indices, i] = 0.0  # segment 中间槽置零，避免后续 ordered_accumulate 重复计入。
        # Unshuffle
        # noinspection PyShadowingNames
        result = torch.empty_like(data_to_reduce)  # 分配反排序后的结果缓冲 [T,K,H]。
        result.view(-1, hidden)[transformed_src_indices] = data_to_reduce.view(-1, hidden)  # 用同一扁平索引把数据放回原 top-k 槽位。
        return result.view(num_tokens, num_topk, hidden)  # 返回形状 [T,K,H]，但同组贡献只保留在 segment 右端槽。
    
    num_experts_per_rank = num_experts // num_ranks  # 每个逻辑 rank 拥有的 expert 数。
    if reduce_in_local:  # local reduce 把同一 rank 内多个 expert 的贡献先合并。
        y = grouped_reduce(y, topk_idx // num_experts_per_rank)  # expert id 除以每 rank expert 数，得到 rank group id [T,K]。
    if reduce_in_scaleup:  # scaleup reduce 再把同一 scaleout 域内的贡献合并。
        y = grouped_reduce(y, topk_idx // (num_experts_per_rank * num_scaleup_ranks))  # 得到 scaleout group id [T,K]。
    bias_sum = bias[0].float() + bias[1].float() if isinstance(bias, tuple) else bias.float() if bias is not None else None  # bias tuple 表示两段 bias 先相加。
    result = ordered_accumulate(y, bias_sum)  # 最后沿 top-k 维度按顺序累加成 [T,H]。
    return result  # 返回 combine 后 token 激活，形状 [num_tokens, hidden]。
