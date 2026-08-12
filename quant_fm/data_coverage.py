"""Exact symbol-day coverage receipts for formal V2 datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from quant_fm.manifest.build_manifest import Manifest

COVERAGE_VERSION = "1.0"
VALID_MARKETS = frozenset({"SH", "SZ"})


def symbol_key(market: str, symbol: str) -> str:
    """Return the stable serialized identity of one requested instrument."""
    normalized_market = str(market).upper()
    normalized_symbol = str(symbol).strip()
    if normalized_market not in VALID_MARKETS or not normalized_symbol:
        msg = f"invalid market/symbol pair: {market!r}/{symbol!r}"
        raise ValueError(msg)
    return f"{normalized_market}:{normalized_symbol}"


def expected_symbol_keys(
    symbols_sz: Sequence[str],
    symbols_sh: Sequence[str],
) -> tuple[str, ...]:
    """Build a unique, stable universe identity from per-market symbol lists."""
    cross_market_duplicates = sorted(set(symbols_sz) & set(symbols_sh))
    if cross_market_duplicates:
        msg = (
            "formal V2 universe repeats bare symbols across markets, which the "
            f"cleaner cannot disambiguate: {cross_market_duplicates[:8]}"
        )
        raise ValueError(msg)
    values = [
        *(symbol_key("SZ", symbol) for symbol in symbols_sz),
        *(symbol_key("SH", symbol) for symbol in symbols_sh),
    ]
    if len(values) != len(set(values)):
        msg = "formal V2 universe contains duplicate market/symbol entries"
        raise ValueError(msg)
    return tuple(sorted(values))


def keys_sha256(values: Iterable[str]) -> str:
    """Hash a canonical ordered collection of symbol keys."""
    payload = "\n".join(sorted(set(values))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coverage_receipt_path(workdir: Path, date: str) -> Path:
    """Return the conventional per-date coverage receipt path."""
    return Path(workdir) / "data" / "coverage" / f"{date}.json"


def coverage_set_sha256(workdir: Path) -> str:
    """Hash every coverage receipt name and byte identity as one generation."""
    receipt_dir = Path(workdir) / "data" / "coverage"
    digest = hashlib.sha256()
    for path in sorted(receipt_dir.glob("*.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\n")
    return digest.hexdigest()


def _materialized_clean_keys(clean_dir: Path) -> set[str]:
    materialized: set[str] = set()
    for market in sorted(VALID_MARKETS):
        market_dir = Path(clean_dir) / market
        if not market_dir.is_dir():
            continue
        for events_path in market_dir.glob("*/events.parquet"):
            materialized.add(symbol_key(market, events_path.parent.name))
    return materialized


def write_coverage_receipt(
    *,
    workdir: Path,
    clean_dir: Path,
    date: str,
    symbols_sz: Sequence[str],
    symbols_sh: Sequence[str],
    failed_symbols: Sequence[str] = (),
) -> Path:
    """Classify every requested symbol as materialized, empty, or failed."""
    expected = set(expected_symbol_keys(symbols_sz, symbols_sh))
    materialized = _materialized_clean_keys(clean_dir)
    unexpected = sorted(materialized - expected)
    if unexpected:
        msg = f"clean output contains symbols outside requested universe: {unexpected[:8]}"
        raise RuntimeError(msg)

    failed_names = {str(symbol).strip() for symbol in failed_symbols}
    failed = {key for key in expected if key.split(":", 1)[1] in failed_names}
    if failed_names - {key.split(":", 1)[1] for key in failed}:
        unknown = sorted(failed_names - {key.split(":", 1)[1] for key in failed})
        msg = f"cleaner reported failures outside requested universe: {unknown[:8]}"
        raise RuntimeError(msg)
    if materialized & failed:
        msg = (
            "coverage receipt cannot classify a symbol as both materialized and failed"
        )
        raise RuntimeError(msg)
    empty = expected - materialized - failed
    payload: dict[str, Any] = {
        "coverage_version": COVERAGE_VERSION,
        "date": date,
        "expected_sha256": keys_sha256(expected),
        "expected": sorted(expected),
        "materialized": sorted(materialized),
        "empty": sorted(empty),
        "failed": sorted(failed),
    }
    destination = coverage_receipt_path(workdir, date)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_coverage_receipt(path: Path) -> dict[str, Any]:
    """Load and structurally validate one exact-coverage receipt."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("coverage_version") != COVERAGE_VERSION:
        msg = f"unsupported coverage receipt version in {source}"
        raise ValueError(msg)
    expected = set(payload.get("expected", []))
    materialized = set(payload.get("materialized", []))
    empty = set(payload.get("empty", []))
    failed = set(payload.get("failed", []))
    if any(
        not isinstance(value, str) or ":" not in value
        for value in expected | materialized | empty | failed
    ):
        msg = f"invalid symbol identity in coverage receipt {source}"
        raise ValueError(msg)
    if materialized & empty or materialized & failed or empty & failed:
        msg = f"coverage classifications overlap in {source}"
        raise ValueError(msg)
    if materialized | empty | failed != expected:
        msg = f"coverage classifications do not exactly partition expected in {source}"
        raise ValueError(msg)
    if payload.get("expected_sha256") != keys_sha256(expected):
        msg = f"coverage expected-universe hash mismatch in {source}"
        raise ValueError(msg)
    return payload


def verify_dataset_coverage(
    workdir: Path,
    *,
    expected_dates: Sequence[str],
    expected_keys: Sequence[str] | None = None,
    manifest: Manifest | None = None,
) -> dict[str, Any]:
    """Verify every requested date and optionally bind receipts to a manifest."""
    expected_date_set = set(expected_dates)
    receipt_dir = Path(workdir) / "data" / "coverage"
    receipt_paths = sorted(receipt_dir.glob("*.json"))
    actual_dates = {path.stem for path in receipt_paths}
    missing = sorted(expected_date_set - actual_dates)
    extra = sorted(actual_dates - expected_date_set)
    if missing or extra:
        msg = (
            f"coverage receipt dates mismatch: missing={missing[:8]} extra={extra[:8]}"
        )
        raise ValueError(msg)

    requested = set(expected_keys) if expected_keys is not None else None
    manifest_by_date: dict[str, set[str]] = {}
    if manifest is not None:
        for shard in manifest.shards:
            manifest_by_date.setdefault(shard.date, set()).add(
                symbol_key(shard.market, shard.symbol)
            )

    universe_hashes: set[str] = set()
    empty_count = materialized_count = 0
    for path in receipt_paths:
        receipt = load_coverage_receipt(path)
        if receipt.get("date") != path.stem:
            msg = f"coverage receipt date disagrees with filename: {path}"
            raise ValueError(msg)
        expected = set(receipt["expected"])
        materialized = set(receipt["materialized"])
        failed = set(receipt["failed"])
        if failed:
            msg = f"coverage receipt records failed symbols for {path.stem}: {sorted(failed)[:8]}"
            raise ValueError(msg)
        if requested is not None and expected != requested:
            msg = f"coverage universe mismatch for {path.stem}"
            raise ValueError(msg)
        if (
            manifest is not None
            and manifest_by_date.get(path.stem, set()) != materialized
        ):
            msg = f"manifest symbol coverage disagrees with receipt for {path.stem}"
            raise ValueError(msg)
        universe_hashes.add(str(receipt["expected_sha256"]))
        empty_count += len(receipt["empty"])
        materialized_count += len(materialized)
    if len(universe_hashes) != 1:
        msg = "formal V2 coverage receipts do not share one fixed universe"
        raise ValueError(msg)
    return {
        "dates": len(receipt_paths),
        "universe_sha256": next(iter(universe_hashes)),
        "materialized_symbol_days": materialized_count,
        "empty_symbol_days": empty_count,
    }


__all__ = [
    "COVERAGE_VERSION",
    "coverage_receipt_path",
    "coverage_set_sha256",
    "expected_symbol_keys",
    "keys_sha256",
    "load_coverage_receipt",
    "symbol_key",
    "verify_dataset_coverage",
    "write_coverage_receipt",
]
