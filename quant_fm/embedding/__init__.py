"""从预训练 OrderFlow FM 提取冻结的股日嵌入。"""

from __future__ import annotations

__all__ = ["extract_stock_day_embeddings", "pool_hidden"]


def __getattr__(name: str):
    """惰性暴露 torch 相关符号。"""
    if name == "extract_stock_day_embeddings":
        from quant_fm.embedding.extract_hidden import extract_stock_day_embeddings

        return extract_stock_day_embeddings
    if name == "pool_hidden":
        from quant_fm.embedding.pool_stock_day import pool_hidden

        return pool_hidden
    msg = f"module 'quant_fm.embedding' has no attribute {name!r}"
    raise AttributeError(msg)
