"""
Create reproducible, stratified validation-window sample plans.

The token manifest is ordered by file-system traversal.  Evaluating only the first
few batches therefore overweights the first dates and symbols.  This module builds
an explicit list of dataset window indices, balanced as far as possible across
``date x exchange x board x liquidity x activity`` strata.  Persist the plan once
and reuse it for every architecture in an experiment.

Liquidity is not stored in the v1 token manifest.  Callers may provide a point-in-
time liquidity value keyed by ``(date, market, symbol)``.  Missing values form an
explicit ``unknown`` bucket instead of being silently imputed from future data.
Activity uses the manifest row count and is therefore always available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch.utils.data import Sampler

from quant_fm.manifest.build_manifest import Manifest
from quant_fm.schema.cn_l2_v1 import board_of, exchange_of

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from quant_fm.manifest.build_manifest import ShardEntry

logger = logging.getLogger(__name__)

PLAN_VERSION = 1
ShardKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class ValidationWindow:
    """One selected window and its audit metadata."""

    dataset_index: int
    shard_index: int
    start: int
    length: int
    date: str
    exchange: str
    market: str
    board: str
    symbol: str
    liquidity_bucket: str
    activity_bucket: str
    shard_sha256: str

    @property
    def stratum(self) -> tuple[str, str, str, str, str]:
        """Return the full balancing stratum."""
        return (
            self.date,
            self.exchange,
            self.board,
            self.liquidity_bucket,
            self.activity_bucket,
        )


@dataclass(frozen=True, slots=True)
class ValidationSamplePlan:
    """Serializable fixed list of validation windows."""

    version: int
    seed: int
    context: int
    stride: int
    min_len: int
    source_fingerprint: str
    total_candidate_windows: int
    windows: tuple[ValidationWindow, ...]

    @property
    def indices(self) -> tuple[int, ...]:
        """Return dataset indices in deterministic evaluation order."""
        return tuple(window.dataset_index for window in self.windows)

    @property
    def stratum_counts(self) -> dict[str, int]:
        """Return selected counts by human-readable stratum key."""
        counts = Counter("|".join(window.stratum) for window in self.windows)
        return dict(sorted(counts.items()))

    def save(self, path: Path) -> None:
        """Write the plan as stable JSON."""
        payload = {
            "version": self.version,
            "seed": self.seed,
            "context": self.context,
            "stride": self.stride,
            "min_len": self.min_len,
            "source_fingerprint": self.source_fingerprint,
            "total_candidate_windows": self.total_candidate_windows,
            "selected_windows": len(self.windows),
            "stratum_counts": self.stratum_counts,
            "windows": [asdict(window) for window in self.windows],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ValidationSamplePlan:
        """Load a plan and reject unsupported versions."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = int(payload.get("version", 0))
        if version != PLAN_VERSION:
            msg = f"unsupported validation sample plan version: {version}"
            raise ValueError(msg)
        return cls(
            version=version,
            seed=int(payload["seed"]),
            context=int(payload["context"]),
            stride=int(payload["stride"]),
            min_len=int(payload["min_len"]),
            source_fingerprint=str(payload["source_fingerprint"]),
            total_candidate_windows=int(payload["total_candidate_windows"]),
            windows=tuple(ValidationWindow(**row) for row in payload["windows"]),
        )

    def validate(
        self,
        shards: Sequence[ShardEntry],
        *,
        context: int,
        stride: int | None,
        min_len: int,
    ) -> None:
        """Fail fast if a plan is reused with another dataset or window layout."""
        effective_stride = stride or context
        expected = manifest_fingerprint(shards)
        if self.source_fingerprint != expected:
            msg = "validation plan does not match the current manifest split"
            raise ValueError(msg)
        actual_config = (self.context, self.stride, self.min_len)
        expected_config = (context, effective_stride, min_len)
        if actual_config != expected_config:
            msg = (
                "validation plan window config mismatch: "
                f"plan={actual_config}, requested={expected_config}"
            )
            raise ValueError(msg)
        if (
            self.windows
            and self.windows[-1].dataset_index >= self.total_candidate_windows
        ):
            msg = "validation plan contains an out-of-range dataset index"
            raise ValueError(msg)


class FixedValidationSampler(Sampler[int]):
    """PyTorch sampler that emits exactly the indices stored in a plan."""

    def __init__(self, plan: ValidationSamplePlan) -> None:
        self.plan = plan

    def __iter__(self) -> Iterator[int]:
        return iter(self.plan.indices)

    def __len__(self) -> int:
        return len(self.plan.windows)


def manifest_fingerprint(shards: Sequence[ShardEntry]) -> str:
    """Hash split identity and ordering without depending on machine-local paths."""
    digest = hashlib.sha256()
    for index, shard in enumerate(shards):
        row = (
            index,
            shard.market,
            shard.symbol,
            shard.date,
            int(shard.rows),
            shard.sha256,
            shard.split,
        )
        digest.update(json.dumps(row, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def shard_key(shard: ShardEntry) -> ShardKey:
    """Return the canonical key used by optional point-in-time metadata."""
    return shard.date, shard.market.upper(), str(shard.symbol).zfill(6)


def _bucket_labels(n_buckets: int) -> tuple[str, ...]:
    if n_buckets == 3:
        return "low", "mid", "high"
    return tuple(f"q{i + 1}_of_{n_buckets}" for i in range(n_buckets))


def _quantile_buckets(values: Mapping[int, float], *, n_buckets: int) -> dict[int, str]:
    """Bucket finite values by empirical quantiles; preserve unknown separately."""
    if n_buckets < 1:
        msg = "n_buckets must be >= 1"
        raise ValueError(msg)
    labels = _bucket_labels(n_buckets)
    finite = sorted(value for value in values.values() if math.isfinite(value))
    if not finite:
        return dict.fromkeys(values, "unknown")
    boundaries = [
        finite[min(math.ceil(len(finite) * q / n_buckets) - 1, len(finite) - 1)]
        for q in range(1, n_buckets)
    ]
    out: dict[int, str] = {}
    for index, value in values.items():
        if not math.isfinite(value):
            out[index] = "unknown"
            continue
        bucket = sum(value > boundary for boundary in boundaries)
        out[index] = labels[bucket]
    return out


def _enumerate_windows(
    shards: Sequence[ShardEntry],
    *,
    context: int,
    stride: int,
    min_len: int,
    liquidity_values: Mapping[ShardKey, float] | None,
    n_liquidity_buckets: int,
    n_activity_buckets: int,
) -> list[ValidationWindow]:
    activity = _quantile_buckets(
        {index: float(shard.rows) for index, shard in enumerate(shards)},
        n_buckets=n_activity_buckets,
    )
    liquidity_raw = {
        index: float(liquidity_values.get(shard_key(shard), math.nan))
        if liquidity_values is not None
        else math.nan
        for index, shard in enumerate(shards)
    }
    liquidity = _quantile_buckets(liquidity_raw, n_buckets=n_liquidity_buckets)

    windows: list[ValidationWindow] = []
    dataset_index = 0
    for shard_index, shard in enumerate(shards):
        if shard.rows < min_len:
            continue
        start = 0
        while start < shard.rows:
            length = min(context, shard.rows - start)
            if length >= min_len:
                windows.append(
                    ValidationWindow(
                        dataset_index=dataset_index,
                        shard_index=shard_index,
                        start=start,
                        length=length,
                        date=shard.date,
                        exchange=exchange_of(shard.market),
                        market=shard.market.upper(),
                        board=board_of(shard.symbol, shard.market),
                        symbol=str(shard.symbol).zfill(6),
                        liquidity_bucket=liquidity[shard_index],
                        activity_bucket=activity[shard_index],
                        shard_sha256=shard.sha256,
                    )
                )
                dataset_index += 1
            start += stride
    return windows


def _balanced_sample(
    candidates: Sequence[ValidationWindow],
    *,
    seed: int,
    max_windows: int | None,
    windows_per_stratum: int | None,
) -> tuple[ValidationWindow, ...]:
    groups: dict[tuple[str, str, str, str, str], list[ValidationWindow]] = defaultdict(
        list
    )
    for window in candidates:
        groups[window.stratum].append(window)

    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
        if windows_per_stratum is not None:
            groups[key] = groups[key][:windows_per_stratum]

    limit = (
        len(candidates) if max_windows is None else min(max_windows, len(candidates))
    )
    selected: list[ValidationWindow] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            group = groups[key]
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1

    # Sequential order avoids needless shard-cache churn during evaluation.
    return tuple(sorted(selected, key=lambda window: window.dataset_index))


def build_validation_sample_plan(
    shards: Sequence[ShardEntry],
    *,
    context: int,
    stride: int | None = None,
    min_len: int = 16,
    seed: int = 42,
    max_windows: int | None = None,
    windows_per_stratum: int | None = None,
    liquidity_values: Mapping[ShardKey, float] | None = None,
    n_liquidity_buckets: int = 3,
    n_activity_buckets: int = 3,
) -> ValidationSamplePlan:
    """
    Build a deterministic, approximately balanced validation sample plan.

    Parameters
    ----------
    shards
        Manifest entries in the exact order used to construct the dataset.
    context, stride, min_len
        Window layout; these must match :class:`EventWindowDataset`.
    seed
        Seed controlling within-stratum selection.
    max_windows
        Optional global cap.  Round-robin sampling gives each non-empty stratum at
        most one additional window per pass.
    windows_per_stratum
        Optional hard cap before applying ``max_windows``.
    liquidity_values
        Optional point-in-time values keyed by ``(date, market, six_digit_symbol)``.
    n_liquidity_buckets, n_activity_buckets
        Number of empirical quantile buckets for the two continuous strata.
    """
    effective_stride = stride or context
    if context < 1 or effective_stride < 1 or min_len < 1:
        msg = "context, stride and min_len must be positive"
        raise ValueError(msg)
    if max_windows is not None and max_windows < 1:
        msg = "max_windows must be >= 1 when provided"
        raise ValueError(msg)
    if windows_per_stratum is not None and windows_per_stratum < 1:
        msg = "windows_per_stratum must be >= 1 when provided"
        raise ValueError(msg)

    candidates = _enumerate_windows(
        shards,
        context=context,
        stride=effective_stride,
        min_len=min_len,
        liquidity_values=liquidity_values,
        n_liquidity_buckets=n_liquidity_buckets,
        n_activity_buckets=n_activity_buckets,
    )
    selected = _balanced_sample(
        candidates,
        seed=seed,
        max_windows=max_windows,
        windows_per_stratum=windows_per_stratum,
    )
    return ValidationSamplePlan(
        version=PLAN_VERSION,
        seed=seed,
        context=context,
        stride=effective_stride,
        min_len=min_len,
        source_fingerprint=manifest_fingerprint(shards),
        total_candidate_windows=len(candidates),
        windows=selected,
    )


def _load_liquidity_values(path: Path | None) -> dict[ShardKey, float] | None:
    if path is None:
        return None
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        msg = "liquidity JSON must be a list of date/market/symbol/liquidity rows"
        raise TypeError(msg)
    out: dict[ShardKey, float] = {}
    for row in payload:
        key = (
            str(row["date"]),
            str(row["market"]).upper(),
            str(row["symbol"]).zfill(6),
        )
        out[key] = float(row["liquidity"])
    return out


def main() -> None:
    """Build a reusable validation-window JSON plan from a manifest split."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, required=True)
    parser.add_argument("--windows-per-stratum", type=int)
    parser.add_argument("--liquidity-json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    shards = Manifest.load(args.manifest).split(args.split)
    if not shards:
        msg = f"no shards for split={args.split}"
        raise SystemExit(msg)
    plan = build_validation_sample_plan(
        shards,
        context=args.context,
        stride=args.stride,
        min_len=args.min_len,
        seed=args.seed,
        max_windows=args.max_windows,
        windows_per_stratum=args.windows_per_stratum,
        liquidity_values=_load_liquidity_values(args.liquidity_json),
    )
    plan.save(args.out)
    logger.info(
        "wrote %s with %d/%d windows across %d strata",
        args.out,
        len(plan.windows),
        plan.total_candidate_windows,
        len(plan.stratum_counts),
    )


if __name__ == "__main__":
    main()
