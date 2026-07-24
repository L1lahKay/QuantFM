"""同步后的跨股票上下文组件。"""

from __future__ import annotations

__all__ = [
    "CrossAssetContext",
    "add_clock_interval",
    "build_synchronous_context",
    "clock_interval_id",
]


def __getattr__(name: str):
    """延迟导入 numpy/polars/torch 相关实现。"""
    if name in {"add_clock_interval", "clock_interval_id"}:
        from quant_fm.cross_asset.clock_grid import (
            add_clock_interval,
            clock_interval_id,
        )

        return {
            "add_clock_interval": add_clock_interval,
            "clock_interval_id": clock_interval_id,
        }[name]
    if name in {"CrossAssetContext", "build_synchronous_context"}:
        from quant_fm.cross_asset.context_pool import (
            CrossAssetContext,
            build_synchronous_context,
        )

        return {
            "CrossAssetContext": CrossAssetContext,
            "build_synchronous_context": build_synchronous_context,
        }[name]
    msg = f"module 'quant_fm.cross_asset' has no attribute {name!r}"
    raise AttributeError(msg)
