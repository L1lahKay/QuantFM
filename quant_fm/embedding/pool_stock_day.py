"""将逐事件隐状态序列池化为单向量股日嵌入。"""

from __future__ import annotations

import torch


def pool_hidden(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    *,
    method: str = "mean",
    last_k: int = 256,
) -> torch.Tensor:
    """
    将 ``[L, d]`` 隐状态约简为单个 ``[d]`` 嵌入。

    参数
    ----------
    hidden
        逐事件隐状态，形状 ``[L, d]``。
    mask
        布尔有效性掩码，形状 ``[L]``（True = 真实事件）。
    method
        ``"mean"``（掩码平均）、``"last"``（最后有效事件）或
        ``"lastk_mean"``（最后 ``last_k`` 个有效事件的均值）。
    last_k
        ``"lastk_mean"`` 的窗口大小。

    返回
    -------
    torch.Tensor
        形状 ``[d]`` 的池化嵌入。
    """
    valid = mask.bool()
    if valid.sum() == 0:
        return torch.zeros(hidden.size(-1), device=hidden.device)

    if method == "mean":
        h = hidden[valid]
        return h.mean(dim=0)
    if method == "last":
        idx = torch.nonzero(valid, as_tuple=False).max()
        return hidden[idx]
    if method == "lastk_mean":
        h = hidden[valid]
        return h[-last_k:].mean(dim=0)
    msg = f"unknown pooling method {method!r}"
    raise ValueError(msg)
