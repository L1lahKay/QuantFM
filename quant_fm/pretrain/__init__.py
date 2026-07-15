"""解码器多任务下一事件预训练（torch）。"""

from __future__ import annotations

__all__ = [
    "EventWindowDataset",
    "OrderFlowFM",
    "OrderFlowFMConfig",
    "collate_windows",
]


def __getattr__(name: str):
    """惰性暴露 torch 相关符号，保持导入轻量。"""
    if name in {"EventWindowDataset", "collate_windows"}:
        from quant_fm.pretrain import dataset as _d

        return getattr(_d, name)
    if name in {"OrderFlowFM", "OrderFlowFMConfig"}:
        from quant_fm.pretrain import model as _m

        return getattr(_m, name)
    msg = f"module 'quant_fm.pretrain' has no attribute {name!r}"
    raise AttributeError(msg)
