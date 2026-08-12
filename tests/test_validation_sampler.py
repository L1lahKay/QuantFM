import json
from pathlib import Path

import pytest

from quant_fm.manifest.build_manifest import ShardEntry
from quant_fm.pretrain.dataset import EventWindowDataset
from quant_fm.pretrain.validation_sampler import (
    FixedValidationSampler,
    ValidationSamplePlan,
    build_validation_sample_plan,
    shard_key,
)


def _shard(
    market: str,
    symbol: str,
    date: str,
    *,
    rows: int = 32,
) -> ShardEntry:
    return ShardEntry(
        market=market,
        symbol=symbol,
        date=date,
        path=f"/{market}/{symbol}/{date}.parquet",
        rows=rows,
        sha256=f"sha-{market}-{symbol}-{date}-{rows}",
        split="val",
    )


def test_stratified_plan_is_fixed_balanced_and_matches_dataset_indices() -> None:
    shards = [
        _shard("SH", "600000", "2026-01-05"),
        _shard("SH", "688001", "2026-01-05"),
        _shard("SZ", "000001", "2026-01-06"),
        _shard("SZ", "300001", "2026-01-06"),
    ]
    plan_a = build_validation_sample_plan(
        shards, context=8, stride=8, min_len=4, seed=7, max_windows=8
    )
    plan_b = build_validation_sample_plan(
        shards, context=8, stride=8, min_len=4, seed=7, max_windows=8
    )
    dataset = EventWindowDataset(shards, context=8, stride=8, min_len=4)

    assert plan_a == plan_b
    assert len(plan_a.windows) == 8
    assert list(plan_a.indices) == sorted(plan_a.indices)
    assert len(set(plan_a.indices)) == 8
    assert max(plan_a.indices) < len(dataset)
    # Four date/exchange/board strata receive two windows each.
    assert set(plan_a.stratum_counts.values()) == {2}
    assert list(FixedValidationSampler(plan_a)) == list(plan_a.indices)


def test_liquidity_unknown_is_explicit_and_plan_round_trips(tmp_path: Path) -> None:
    shards = [
        _shard("SH", "600000", "2026-01-05", rows=10),
        _shard("SH", "600001", "2026-01-05", rows=30),
        _shard("SH", "600002", "2026-01-05", rows=50),
    ]
    liquidity = {
        shard_key(shards[0]): 1.0,
        shard_key(shards[1]): 2.0,
        # The third shard intentionally has no point-in-time liquidity observation.
    }
    plan = build_validation_sample_plan(
        shards,
        context=10,
        min_len=4,
        seed=11,
        max_windows=6,
        liquidity_values=liquidity,
    )
    buckets = {window.symbol: window.liquidity_bucket for window in plan.windows}
    assert buckets["600002"] == "unknown"
    assert {window.activity_bucket for window in plan.windows} == {
        "low",
        "mid",
        "high",
    }

    path = tmp_path / "validation-plan.json"
    plan.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["plan_sha256"] == plan.sha256
    loaded = ValidationSamplePlan.load(path)
    assert loaded == plan
    loaded.validate(
        shards,
        context=10,
        stride=None,
        min_len=4,
        liquidity_values=liquidity,
    )

    changed_liquidity = {**liquidity, shard_key(shards[1]): 9.0}
    with pytest.raises(ValueError, match="stratification inputs"):
        loaded.validate(
            shards,
            context=10,
            stride=None,
            min_len=4,
            liquidity_values=changed_liquidity,
        )


def test_plan_sha_binds_seed_exact_windows_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    shards = [
        _shard("SH", "600000", "2026-01-05"),
        _shard("SZ", "000001", "2026-01-06"),
    ]
    first = build_validation_sample_plan(
        shards, context=8, stride=8, min_len=4, seed=7, max_windows=3
    )
    second = build_validation_sample_plan(
        shards, context=8, stride=8, min_len=4, seed=8, max_windows=3
    )
    assert first.sha256 != second.sha256

    path = tmp_path / "validation-plan.json"
    first.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["windows"][0]["dataset_index"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical SHA-256"):
        ValidationSamplePlan.load(path)


def test_plan_validation_rejects_manifest_or_window_changes() -> None:
    shards = [_shard("SH", "600000", "2026-01-05")]
    plan = build_validation_sample_plan(
        shards, context=8, stride=8, min_len=4, max_windows=1
    )
    with pytest.raises(ValueError, match="window config mismatch"):
        plan.validate(shards, context=16, stride=8, min_len=4)

    changed = [_shard("SH", "600000", "2026-01-05", rows=33)]
    with pytest.raises(ValueError, match="current manifest"):
        plan.validate(changed, context=8, stride=8, min_len=4)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"context": 0}, "context must be a positive integer"),
        ({"stride": 0}, "stride must be a positive integer"),
        ({"min_len": 0}, "min_len must be a positive integer"),
        ({"cache_size": 0}, "cache_size must be a positive integer"),
        ({"context": 4, "min_len": 5}, "cannot exceed context"),
    ],
)
def test_event_window_dataset_rejects_invalid_geometry(options, message) -> None:
    kwargs = {"context": 8, "stride": 8, "min_len": 4, "cache_size": 1}
    kwargs.update(options)

    with pytest.raises(ValueError, match=message):
        EventWindowDataset([_shard("SH", "600000", "2026-01-05")], **kwargs)
