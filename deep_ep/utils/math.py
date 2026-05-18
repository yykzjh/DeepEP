import torch  # DeepEP 的 Python 工具函数全部围绕 CUDA Tensor 和 torch.compile 展开。
from typing import Tuple  # 标注 FP8 转换函数同时返回量化张量和 scale 张量。


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:  # 计算两个张量的归一化差异，测试中用于判断数值是否接近。
    x, y = x.double() + 1, y.double() + 1  # 转 double 降低误差，并整体加 1 避免零附近相对误差过度放大。
    denominator = (x * x + y * y).sum()  # 分母是两个张量 L2 能量之和；x/y 形状相同，结果是标量 Tensor。
    sim = 2 * (x * y).sum() / denominator  # Dice/余弦风格相似度：完全相等时 sim 接近 1。
    return (1 - sim).item()  # 返回 Python float，越接近 0 表示差异越小。


def safe_div(a, b) -> float:  # 安全除法，主要用于测试统计中处理 0/0 的吞吐或误差比例。
    try:  # 先走普通除法，保持 Python 对 int/float 的原生行为。
        return a / b  # b 非零时直接返回商。
    except ZeroDivisionError as e:  # 只有分母为 0 时进入这里；保留 e 便于调试器查看原始异常。
        if a == 0:  # 0/0 在一些“没有 token/没有字节”的测试统计里代表空结果。
            return 0  # 约定空结果的比例为 0，避免测试日志中断。
        else:  # 非 0 除以 0 是真实错误，应交给调用方看到异常。
            raise  # 保留原始 ZeroDivisionError 栈信息。


def ceil_div(x: int, y: int) -> int:  # 整数上取整除法，用于 block 数、对齐段数等 CUDA 参数计算。
    return (x + y - 1) // y  # 例如 ceil_div(129, 128)=2；要求 y 为正数。


def align(x: int, y: int) -> int:  # 把 x 向上对齐到 y 的倍数。
    return ceil_div(x, y) * y  # 例如 hidden=7168 已经是 128 倍数，aligned_n 仍为 7168。


@torch.compile(dynamic=True)  # 让 PyTorch 为动态 token 数生成图，减少 Python 循环/调度开销。
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # 将 BF16/FP32 token 矩阵按 128 列分组量化成 FP8。
    # 输入 x 形状为 [num_tokens, hidden]。FP8 E4M3 的最大有限值约为 448，
    # 因此这里按“每个 token、每 128 个 hidden 元素”为一组求 amax，并存一个 scale。
    # 例：hidden=7168 时，x_scales 形状为 [num_tokens, 56]；dispatch FP8 路径会把
    # x_fp8 与 x_scales 一起传给 CUDA kernel，kernel 再按 scale 还原或转发。
    assert x.dim() == 2  # 只接受二维矩阵：[num_tokens, hidden]，否则 scale 维度定义不成立。
    m, n = x.shape  # m=num_tokens，n=hidden；后续所有 view 都围绕这两个维度构造。
    aligned_n = align(n, 128)  # hidden 向上补齐到 128 的倍数，使每组恰好 128 个元素。
    x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode='constant', value=0)  # 在最后一维右侧补零，形状 [m, aligned_n]。
    x_padded_view = x_padded.view(m, -1, 128)  # 分组视图形状 [m, aligned_n/128, 128]。
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)  # 每组一个 amax，形状 [m, aligned_n/128]。
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, aligned_n)[:, :n].contiguous(), (x_amax / 448.0).view(m, -1)  # 返回 FP8 矩阵 [m,n] 与 scale [m,ceil(n/128)]。


@torch.compile(dynamic=True)  # 反量化路径也编译，以便测试/参考实现与 kernel 对齐时减少开销。
def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor) -> torch.Tensor:  # 用 per-token-per-128 scale 把 FP8 恢复为 BF16。
    if x_fp8.numel() == 0:  # 空 token 批次在 MoE 路由中可能出现，需要避免 view/reshape 产生异常。
        return x_fp8.to(torch.bfloat16)  # 空张量只转换 dtype，形状保持不变。

    assert x_fp8.dim() == 2  # 输入 FP8 张量形状必须为 [num_tokens, hidden]。
    m, n = x_fp8.shape  # m=num_tokens，n=hidden，用来恢复 padding 前的列数。
    aligned_n = align(n, 128)  # 与量化时相同，把 hidden 补齐到 128 倍数。
    x_fp8_padded = torch.nn.functional.pad(x_fp8, (0, aligned_n - n), mode='constant', value=0)  # 形状 [m, aligned_n]，便于按 128 分组。
    if x_scales.dtype == torch.int:  # int scale 表示 UE8M0 压缩格式，需要先还原为浮点 scale。
        # UE8M0 packed scale 的 trick：scale 本质上只保存指数位。把 uint8 扩展到 int 后左移 23 位，
        # 再 view 成 float，就能构造出 2^e 形式的 FP32 scale，避免逐元素调用 pow/exp。
        # 这个格式与 low_latency_dispatch(use_ue8m0=True) 的 scale 输出维度联动。
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23  # 把 8 位指数放到 FP32 指数字段位置。
        x_scales = x_scales.view(dtype=torch.float)  # 通过 dtype 视图把位模式解释为 FP32 数值。
    x_fp32_padded = x_fp8_padded.to(torch.float32).view(x_fp8.shape[0], -1, 128)  # FP8 值扩展成 [m, groups, 128] 的 FP32。
    x_scales = x_scales.view(x_fp8.shape[0], -1, 1)  # scale 形状 [m, groups, 1]，通过广播乘到每组 128 个元素。
    return (x_fp32_padded * x_scales).view(x_fp8_padded.shape).to(torch.bfloat16)[:, :n].contiguous()  # 去掉 padding 后返回 BF16 [m,n]。


def inplace_unique(x: torch.Tensor, num_slots: int) -> None:  # 对二维 id 表按行去重，并把结果写回原张量。
    # 对每一行去重并按出现频次排序，常用于构造“每个 token 触达哪些 rank/group”的紧凑表示。
    # 输入 x: [num_tokens, k]，负数表示无效槽；num_slots 是合法 id 范围。
    # 先把无效槽临时映射到 num_slots，再用 scatter_add 得到直方图，最后把出现过的 id 写回前面。
    assert x.dim() == 2  # x 必须是 [rows, width]，每行独立去重。
    mask = x < 0  # 负数槽表示无效 id，后续不会作为合法 slot 保留。
    x_padded = x.masked_fill(mask, num_slots)  # 把无效 id 暂时改成额外 bin=num_slots，便于 scatter_add 统计。
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)  # 直方图形状 [rows, num_slots+1]，最后一列承接无效槽。
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))  # 对每行的 slot id 计数，重复 id 会累加出现次数。
    bin_count = bin_count[:, :num_slots]  # 丢掉无效槽对应的额外 bin，剩余形状 [rows, num_slots]。
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)  # 按出现次数降序排列 slot id。
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)  # 未出现过的 slot 标成 -1，避免被当作有效 id。
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values  # 再按 id 降序整理，有效 id 在前、-1 在后。
    x[:, :].fill_(-1)  # 先清空原张量，保证尾部旧值不会残留。
    valid_len = min(num_slots, x.size(1))  # 输出宽度不能超过原 x 的列数，也不能超过合法 slot 数。
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]  # 将去重结果写回 x 的前 valid_len 列。


def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:  # 根据 group_idx 过滤 expert 分数。
    # scores: [num_tokens, num_experts]，group_idx: [num_tokens, num_selected_groups]。
    # 这里把 expert 维度 reshape 为 [num_tokens, num_groups, experts_per_group]，
    # 只保留 group_idx 指定的 group，其他 group 的 expert score 置零，用于 group-limited gating 测试。
    num_tokens, num_experts = scores.shape  # scores 原始形状 [num_tokens, num_experts]。
    scores = scores.view(num_tokens, num_groups, -1)  # 变为 [num_tokens, num_groups, experts_per_group]。
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)  # 每个 token 一个 group 级布尔掩码。
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)  # 选中的 group 扩展到其内部所有 expert。
    return (scores * mask).view(num_tokens, num_experts)  # 未选 group 的 expert 分数清零后恢复二维形状。


def hash_tensor(t: torch.Tensor) -> int:  # 给张量构造轻量 hash，测试中用于快速比较结果是否变化。
    return t.view(torch.int).sum().item()  # 按 int 视图求和；这不是密码学 hash，只是低成本一致性指纹。


def hash_tensors(*tensors) -> int:  # 对多个张量或嵌套列表/元组递归计算组合 hash。
    value = 0  # 使用异或聚合，顺序变化可能抵消，因此只用于测试日志/快速校验。
    for t in tensors:  # 遍历所有输入对象。
        if isinstance(t, (tuple, list)):  # 支持 kernel 返回的 tuple/list 结构。
            value ^= hash_tensors(*t)  # 对嵌套结构递归展开并异或到总值。
        elif t is not None and isinstance(t, torch.Tensor):  # 忽略 None 和非 Tensor 参数。
            value ^= hash_tensor(t)  # 单个 Tensor 的指纹参与聚合。
    return value  # 返回 Python int，方便打印或参与断言。


def count_bytes(*tensors) -> int:  # 统计一组张量占用的理论字节数，用于带宽/吞吐计算。
    total = 0  # 累计字节数，单位 byte。
    for t in tensors:  # 支持传入多个 Tensor 或嵌套结构。
        if isinstance(t, (tuple, list)):  # kernel 返回多个张量时通常是 tuple/list。
            total += count_bytes(*t)  # 递归统计嵌套元素。
        elif t is not None:  # None 表示可选输出缺失，不参与统计。
            total += t.numel() * t.element_size()  # numel 是元素数，element_size 是单元素字节数。
    return total  # 返回总字节数，调用方可除以耗时得到 GB/s。
