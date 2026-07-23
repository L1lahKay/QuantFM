"""生产日频 score 信号的稳定接口。"""

from __future__ import annotations

from typing import Any

__all__ = ["generate_scores", "validate_scores"]


def __getattr__(name: str) -> Any:
    """延迟导入 torch 相关实现，保持包导入轻量。"""
    if name == "generate_scores":
        from quant_fm.signal.generate import generate_scores

        return generate_scores
    if name == "validate_scores":
        from quant_fm.signal.schema import validate_scores

        return validate_scores
    msg = f"module 'quant_fm.signal' has no attribute {name!r}"
    raise AttributeError(msg)
