"""
门控 1：将重建订单簿与参考 L2 快照对账。

PyLOB 引擎可通过
:meth:`pylob.result_mixin.ResultMixin.process_workflow_with_book` 输出逐事件订单簿，
行含 ``ap_array / av_array / bp_array / bv_array``（卖价升序、卖量、
买价降序、买量）及 ``int_time``。交易所亦约每 3 秒发布 10 档快照
（MinIO ``default/1``）。
本模块按时点对齐（重建簿在时刻 t 的最后一次更新，满足时间 <= 快照时间），
并报告各档位一致率，作为数据质量门控。

参考快照列名因供应商而异，加载器先规范为 :class:`BookSnapshot` 列表；
自动检测不足时，调用方可显式传入 ``column_map``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BookSnapshot:
    """单一时点的 Top-N 订单簿快照。"""

    int_time: int
    bid_px: list[float]
    bid_qty: list[float]
    ask_px: list[float]
    ask_qty: list[float]


@dataclass(slots=True)
class ReconcileReport:
    """门控 1 的汇总一致统计。"""

    n_compared: int = 0
    n_price_match: int = 0
    n_qty_match: int = 0
    per_level_match: list[float] = field(default_factory=list)

    @property
    def price_consistency(self) -> float:
        """在容差内价格一致的被比较档位占比。"""
        return self.n_price_match / self.n_compared if self.n_compared else 0.0

    @property
    def qty_consistency(self) -> float:
        """在容差内成交量一致的被比较档位占比。"""
        return self.n_qty_match / self.n_compared if self.n_compared else 0.0

    def passed(self, threshold: float) -> bool:
        """价格一致率是否达到 ``threshold``（如 0.99）。"""
        return self.price_consistency >= threshold

    def as_dict(self) -> dict[str, float]:
        """返回可 JSON 序列化的摘要。"""
        return {
            "n_compared": float(self.n_compared),
            "price_consistency": self.price_consistency,
            "qty_consistency": self.qty_consistency,
        }


def rebuilt_book_to_snapshots(all_orderbook: pl.DataFrame) -> list[BookSnapshot]:
    """
    将 ``process_workflow_with_book`` 输出转为 :class:`BookSnapshot`。

    接受含列表列 ``ap_array, av_array, bp_array, bv_array`` 及 ``int_time`` 的
    polars 或 pandas 数据帧。
    """
    if not isinstance(all_orderbook, pl.DataFrame):
        all_orderbook = pl.from_pandas(all_orderbook)
    snaps: list[BookSnapshot] = []
    for row in all_orderbook.iter_rows(named=True):
        snaps.append(
            BookSnapshot(
                int_time=int(row["int_time"]),
                bid_px=list(row.get("bp_array") or []),
                bid_qty=list(row.get("bv_array") or []),
                ask_px=list(row.get("ap_array") or []),
                ask_qty=list(row.get("av_array") or []),
            )
        )
    snaps.sort(key=lambda s: s.int_time)
    return snaps


def _detect_level_columns(cols: list[str], side: str, kind: str) -> list[str]:
    """尽力查找有序的 ``{side}_{kind}_{i}`` 风格列。"""
    keys = {
        ("bid", "px"): ["bid_price", "bp", "buy_price", "bidpx"],
        ("bid", "qty"): ["bid_volume", "bid_vol", "bv", "buy_volume", "bidsz"],
        ("ask", "px"): ["ask_price", "ap", "sell_price", "askpx", "offer_price"],
        ("ask", "qty"): ["ask_volume", "ask_vol", "av", "sell_volume", "asksz"],
    }[(side, kind)]
    matched = [c for c in cols if any(c.lower().startswith(k) for k in keys)]
    return sorted(matched, key=lambda c: (len(c), c))


def reference_to_snapshots(
    reference: pl.DataFrame,
    *,
    n_levels: int = 10,
    time_col: str = "int_time",
    column_map: dict[str, list[str]] | None = None,
) -> list[BookSnapshot]:
    """将供应商快照数据帧规范为 :class:`BookSnapshot` 列表。"""
    cols = reference.columns
    if column_map is None:
        column_map = {
            "bid_px": _detect_level_columns(cols, "bid", "px")[:n_levels],
            "bid_qty": _detect_level_columns(cols, "bid", "qty")[:n_levels],
            "ask_px": _detect_level_columns(cols, "ask", "px")[:n_levels],
            "ask_qty": _detect_level_columns(cols, "ask", "qty")[:n_levels],
        }
    if time_col not in cols:
        msg = f"reference snapshot missing time column {time_col!r}"
        raise KeyError(msg)

    snaps: list[BookSnapshot] = []
    for row in reference.iter_rows(named=True):
        snaps.append(
            BookSnapshot(
                int_time=int(row[time_col]),
                bid_px=[float(row[c]) for c in column_map["bid_px"]],
                bid_qty=[float(row[c]) for c in column_map["bid_qty"]],
                ask_px=[float(row[c]) for c in column_map["ask_px"]],
                ask_qty=[float(row[c]) for c in column_map["ask_qty"]],
            )
        )
    snaps.sort(key=lambda s: s.int_time)
    return snaps


def _latest_before(rebuilt: list[BookSnapshot], t: int) -> BookSnapshot | None:
    """时点查询：满足 ``int_time <= t`` 的最后一次重建订单簿。"""
    lo, hi, found = 0, len(rebuilt) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if rebuilt[mid].int_time <= t:
            found = rebuilt[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return found


def reconcile_snapshots(
    rebuilt: list[BookSnapshot],
    reference: list[BookSnapshot],
    *,
    n_levels: int = 10,
    price_tol: float = 1e-4,
    qty_rtol: float = 1e-6,
) -> ReconcileReport:
    """按档位、按时点比较重建簿与参考簿。"""
    report = ReconcileReport(per_level_match=[0.0] * n_levels)
    level_seen = [0] * n_levels
    level_hit = [0] * n_levels

    for ref in reference:
        book = _latest_before(rebuilt, ref.int_time)
        if book is None:
            continue
        for side_px, side_qty, ref_px, ref_qty in (
            (book.bid_px, book.bid_qty, ref.bid_px, ref.bid_qty),
            (book.ask_px, book.ask_qty, ref.ask_px, ref.ask_qty),
        ):
            for lvl in range(n_levels):
                have = lvl < len(side_px) and lvl < len(ref_px)
                if not have:
                    continue
                report.n_compared += 1
                level_seen[lvl] += 1
                price_ok = abs(side_px[lvl] - ref_px[lvl]) <= price_tol
                qty_ok = np.isclose(
                    side_qty[lvl] if lvl < len(side_qty) else 0.0,
                    ref_qty[lvl] if lvl < len(ref_qty) else 0.0,
                    rtol=qty_rtol,
                    atol=1.0,
                )
                report.n_price_match += int(price_ok)
                report.n_qty_match += int(qty_ok)
                level_hit[lvl] += int(price_ok)

    report.per_level_match = [
        (level_hit[i] / level_seen[i]) if level_seen[i] else 0.0
        for i in range(n_levels)
    ]
    logger.info(
        "snapshot reconcile: compared=%d price=%.4f qty=%.4f",
        report.n_compared,
        report.price_consistency,
        report.qty_consistency,
    )
    return report
