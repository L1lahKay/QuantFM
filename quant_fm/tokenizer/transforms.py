"""
cn_l2_v1 事件的因果、尺度不变特征变换。

此处每个变换均为**因果**：仅使用截至当前事件（含）的信息，
避免未来数据泄漏到 token 流。

产生的连续字段：

* ``price_rel``  -- 相对因果 EW-VWAP 中间价的对数价格
* ``log_volume`` -- ``log1p(qty / 100)``
* ``log_delta_t``-- ``log1p(elapsed_ms)``，elapsed 为解码后的真实时间
"""

from __future__ import annotations

import numpy as np
import polars as pl

DERIVED_CONTINUOUS = ("price_rel", "log_volume", "log_delta_t")
FEATURE_TRANSFORM_LEGACY_V1 = "ew_vwap_future_backfill_v1"
FEATURE_TRANSFORM_CAUSAL_V2 = "ew_vwap_causal_nan_v2"
DEFAULT_FEATURE_TRANSFORM_VERSION = FEATURE_TRANSFORM_CAUSAL_V2
SUPPORTED_FEATURE_TRANSFORM_VERSIONS = frozenset(
    {FEATURE_TRANSFORM_LEGACY_V1, FEATURE_TRANSFORM_CAUSAL_V2}
)


def validate_feature_transform_version(version: str) -> str:
    """Return a supported derived-feature contract or raise explicitly."""
    normalized = str(version)
    if normalized not in SUPPORTED_FEATURE_TRANSFORM_VERSIONS:
        msg = (
            f"unsupported feature_transform_version={normalized!r}; expected one of "
            f"{sorted(SUPPORTED_FEATURE_TRANSFORM_VERSIONS)}"
        )
        raise ValueError(msg)
    return normalized


def reference_price_initialization(version: str) -> str:
    """Describe the frozen EW-VWAP initialization policy for audit output."""
    normalized = validate_feature_transform_version(version)
    if normalized == FEATURE_TRANSFORM_LEGACY_V1:
        return "future_first_valid_backfill"
    return "causal_nan_until_current_or_past_price"


def int_time_to_ms(int_time: np.ndarray) -> np.ndarray:
    """将打包的 ``HHMMSSmmm`` 整数解码为自午夜起的毫秒数。"""
    t = int_time.astype(np.int64)
    ms = t % 1000
    ss = (t // 1000) % 100
    mm = (t // 100_000) % 100
    hh = (t // 10_000_000) % 100
    return ((hh * 3600 + mm * 60 + ss) * 1000 + ms).astype(np.int64)


def ew_vwap_mid(
    price: np.ndarray,
    qty: np.ndarray,
    is_trade: np.ndarray,
    elapsed_ms: np.ndarray,
    *,
    half_life_ms: float = 5_000.0,
    transform_version: str = DEFAULT_FEATURE_TRANSFORM_VERSION,
) -> np.ndarray:
    """
    时间感知的指数加权 VWAP 中间价估计（因果）。

    遵循 TradeFM 思路：仅成交事件携带价格信息；中间价为 ``price * qty`` 对 ``qty``
    的 EMA，并带时间衰减，使更大、更近的成交权重更高。非成交事件沿用上一中间价。

    参数
    ----------
    price, qty
        逐事件价格（元）与数量。
    is_trade
        布尔掩码，成交处为 ``True``。
    elapsed_ms
        逐事件自午夜起的毫秒数（来自 :func:`int_time_to_ms`）。
    half_life_ms
        EMA 半衰期（毫秒）。
    transform_version
        起始缺失策略。新默认值保持 NaN 直到当前或历史事件出现有效价格；
        legacy 值仅用于复现旧 token，会用未来首个有效值回填。

    返回
    -------
    numpy.ndarray
        与输入事件对齐的因果中间价估计。
    """
    transform_version = validate_feature_transform_version(transform_version)
    if half_life_ms <= 0 or not np.isfinite(half_life_ms):
        msg = "half_life_ms must be finite and positive"
        raise ValueError(msg)
    n = len(price)
    mid = np.empty(n, dtype=np.float64)
    num = 0.0  # price*qty 的 EMA
    den = 0.0  # qty 的 EMA
    last_mid = np.nan
    last_t = elapsed_ms[0] if n else 0
    decay_base = 0.5

    for i in range(n):
        if is_trade[i] and qty[i] > 0 and np.isfinite(price[i]) and price[i] > 0:
            dt = max(elapsed_ms[i] - last_t, 0)
            w = decay_base ** (dt / half_life_ms)
            num = num * w + price[i] * qty[i]
            den = den * w + qty[i]
            last_t = elapsed_ms[i]
            if den > 0:
                last_mid = num / den
        if np.isnan(last_mid) and np.isfinite(price[i]) and price[i] > 0:
            last_mid = price[i]
        mid[i] = last_mid

    # 旧 artifact 的显式复现路径。严格因果版本保留开头 NaN，绝不从未来读值。
    if transform_version == FEATURE_TRANSFORM_LEGACY_V1 and np.isnan(mid).any():
        valid = np.where(np.isfinite(mid))[0]
        if valid.size:
            mid[: valid[0]] = mid[valid[0]]
        else:
            mid[:] = 1.0
    return mid


def add_derived_fields(
    events: pl.DataFrame,
    *,
    transform_version: str = DEFAULT_FEATURE_TRANSFORM_VERSION,
) -> pl.DataFrame:
    """
    为规范事件数据帧追加因果连续字段。

    期望单标的单日数据帧，含列 ``price, qty, evt_type, int_time``，按事件顺序排序。
    """
    price = events["price"].to_numpy().astype(np.float64)
    qty = events["qty"].to_numpy().astype(np.float64)
    is_trade = (events["evt_type"] == "EXEC").to_numpy()
    int_time = events["int_time"].to_numpy()

    elapsed = int_time_to_ms(int_time)
    transform_version = validate_feature_transform_version(transform_version)
    mid = ew_vwap_mid(
        price,
        qty,
        is_trade,
        elapsed,
        transform_version=transform_version,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        valid_reference = (
            np.isfinite(price) & (price > 0) & np.isfinite(mid) & (mid > 0)
        )
        ratio = np.full(price.shape, np.nan, dtype=np.float64)
        ratio[valid_reference] = price[valid_reference] / mid[valid_reference]
        price_rel = np.log(np.clip(ratio, 1e-6, 1e6))
    if transform_version == FEATURE_TRANSFORM_LEGACY_V1:
        # v1 encoded missing/non-price rows as the neutral relative-price value.
        price_rel = np.nan_to_num(price_rel, nan=0.0)
    log_volume = np.log1p(np.clip(qty, 0, None) / 100.0)

    delta_ms = np.diff(elapsed, prepend=elapsed[0] if len(elapsed) else 0)
    delta_ms = np.clip(delta_ms, 0, None)
    log_delta_t = np.log1p(delta_ms.astype(np.float64))

    return events.with_columns(
        pl.Series("mid", mid),
        pl.Series("price_rel", price_rel),
        pl.Series("log_volume", log_volume),
        pl.Series("log_delta_t", log_delta_t),
    )
