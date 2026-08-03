"""
将 PyLOB 清洗产物转换为版本化的 QuantFM 规范事件 parquet。

繁重工作（沪深订单簿重建）已由 :func:`pylob.pipeline.workflow.build_clean_dataset`
完成，并写出 ``{market}/{symbol}/events.parquet``。本模块读取这些事件并
重新投影到 ``cn_l2_v1`` 或 ``cn_l2_v2`` 规范空间，*不*进行分箱
（分箱在后续仅于训练窗口进行，以防泄漏）。V2 强制读取清洗回放期间捕获的
逐事件 ``book_features.parquet``，绝不以占位值冒充真实盘口。

输出布局::

    {out_dir} / {market} / {symbol} / {date}.parquet

以便每个标的可累积多个交易日，供分片训练使用。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from quant_fm.schema.cn_l2_v1 import SCHEMA_VERSION as V1_SCHEMA_VERSION
from quant_fm.schema.cn_l2_v1 import events_to_canonical as events_to_canonical_v1
from quant_fm.schema.cn_l2_v2 import SCHEMA_VERSION as V2_SCHEMA_VERSION
from quant_fm.schema.cn_l2_v2 import events_to_canonical as events_to_canonical_v2

logger = logging.getLogger(__name__)


def default_canon_workers() -> int:
    """Default worker count for parallel canonicalization."""
    env = os.environ.get("CANON_WORKERS")
    if env:
        return max(1, int(env))
    cpu = os.cpu_count() or 8
    return max(1, min(16, cpu // 4))


def _canonicalize_one(payload: dict) -> tuple[str, str, int, str | None]:
    """Read clean events and return ``(status, path, n_events, error)``."""
    events_path = Path(payload["events_path"])
    dst = Path(payload["dst"])
    date = payload["date"]
    market = payload["market"]
    try:
        events = pl.read_parquet(events_path)
        if events.is_empty():
            return "empty", str(dst), 0, None
        book_features = _read_book_features(payload, events.height)
        canonical = events_frame_to_canonical(
            events,
            date=date,
            market=market,
            schema_version=payload.get("schema_version", V1_SCHEMA_VERSION),
            book_features=book_features,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_parquet(dst)
        return "written", str(dst), canonical.height, None
    except Exception as exc:
        return "error", str(dst), 0, str(exc)


def _canonicalize_tokenize_one(payload: dict) -> tuple[str, str, int, str | None]:
    """Fuse clean→canonical→token to avoid two intermediate parquet round trips."""
    from quant_fm.tokenizer.artifact_contract import write_token_contract

    events_path = Path(payload["events_path"])
    dst = Path(payload["dst"])
    try:
        events = pl.read_parquet(events_path)
        if events.is_empty():
            return "empty", str(dst), 0, None
        vocab = _load_vocab(Path(payload["vocab_path"]))
        book_features = _read_book_features(payload, events.height)
        canonical = events_frame_to_canonical(
            events,
            date=payload["date"],
            market=payload["market"],
            schema_version=vocab.schema_version,
            book_features=book_features,
        )
        storage_encoding = None
        if vocab.schema_version == V2_SCHEMA_VERSION:
            from quant_fm.tokenizer.storage_encoding_v2 import quantize_frame_v2
            from quant_fm.tokenizer.tokenize_events_v2 import tokenize_frame_v2

            tokens = tokenize_frame_v2(canonical, vocab)
            tokens, storage_metadata = quantize_frame_v2(tokens, vocab)
            storage_encoding = storage_metadata.to_dict()
        else:
            from quant_fm.tokenizer.tokenize_events import tokenize_frame

            tokens = tokenize_frame(canonical, vocab)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".parquet.tmp")
        tokens.write_parquet(
            tmp,
            compression="zstd",
            compression_level=3,
            statistics=True,
        )
        tmp.replace(dst)
        write_token_contract(dst, vocab, storage_encoding=storage_encoding)
        return "written", str(dst), tokens.height, None
    except Exception as exc:
        return "error", str(dst), 0, str(exc)


def events_frame_to_canonical(
    events: pl.DataFrame,
    *,
    date: str,
    market: str,
    schema_version: str = V1_SCHEMA_VERSION,
    book_features: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Convert one cleaned symbol-day to the explicitly requested schema."""
    if schema_version == V1_SCHEMA_VERSION:
        if book_features is not None:
            msg = "cn_l2_v1 canonicalization must not receive V2 book features"
            raise ValueError(msg)
        return events_to_canonical_v1(events, date=date, market=market)
    if schema_version == V2_SCHEMA_VERSION:
        if book_features is None:
            msg = "cn_l2_v2 canonicalization requires real aligned book features"
            raise ValueError(msg)
        return events_to_canonical_v2(
            events,
            date=date,
            market=market,
            book_features=book_features,
        )
    msg = f"unsupported schema_version: {schema_version!r}"
    raise ValueError(msg)


def _read_book_features(payload: dict, expected_rows: int) -> pl.DataFrame | None:
    """Read required V2 book features and fail closed on absent/misaligned data."""
    if payload.get("schema_version", V1_SCHEMA_VERSION) != V2_SCHEMA_VERSION:
        return None
    raw_path = payload.get("book_features_path")
    if not raw_path:
        msg = "cn_l2_v2 payload is missing book_features_path"
        raise ValueError(msg)
    path = Path(raw_path)
    if not path.is_file():
        msg = f"cn_l2_v2 requires captured book features: {path}"
        raise FileNotFoundError(msg)
    features = pl.read_parquet(path)
    if features.height != expected_rows:
        msg = (
            "book features/events row mismatch: "
            f"features={features.height}, events={expected_rows}, path={path}"
        )
        raise ValueError(msg)
    return features


def _load_vocab(path: Path):
    """Load a V1 or V2 vocab without guessing from the filename."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("vocab_version") == "2.0":
        from quant_fm.tokenizer.vocab_v2 import VocabV2

        return VocabV2.load(path)
    from quant_fm.tokenizer.vocab import Vocab

    return Vocab.load(path)


def _assert_existing_schema(path: Path, expected: str) -> None:
    """Reject resume across V1/V2 roots instead of silently reusing a shard."""
    try:
        values = (
            pl.read_parquet(path, columns=["schema_version"])
            .get_column("schema_version")
            .drop_nulls()
            .unique()
            .to_list()
        )
    except Exception as exc:
        msg = f"cannot validate existing canonical shard {path}: {exc}"
        raise RuntimeError(msg) from exc
    if values != [expected]:
        msg = (
            f"refusing to reuse {path}: schema={values}, expected={expected}; "
            "use a new workdir for V2"
        )
        raise RuntimeError(msg)


def canonicalize_clean_dir(
    clean_dir: Path,
    out_dir: Path,
    *,
    date: str,
    markets: tuple[str, ...] = ("SH", "SZ"),
    symbols: tuple[str, ...] | None = None,
    skip_existing: bool = False,
    n_workers: int | None = None,
    strict: bool = False,
    schema_version: str = V1_SCHEMA_VERSION,
) -> list[Path]:
    """
    规范化 ``clean_dir`` 下所有 ``events.parquet``。

    参数
    ----------
    clean_dir
        ``build_clean_dataset`` 写入的根目录（含 ``{market}/{symbol}``）。
    out_dir
        规范 ``{market}/{symbol}/{date}.parquet`` 的目标根目录。
    date
        交易日（``YYYY-MM-DD``），写入每一行。
    markets
        要扫描的市场。
    symbols
        可选的 6 位代码白名单；``None`` 表示扫描到的全部。

    返回
    -------
    list[pathlib.Path]
        已写入的规范 parquet 路径列表。
    """
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)
    symbol_set = set(symbols) if symbols is not None else None
    pending: list[dict] = []
    skipped = 0

    for market in markets:
        market_dir = clean_dir / market
        if not market_dir.is_dir():
            continue
        for symbol_dir in sorted(market_dir.iterdir()):
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name
            if symbol_set is not None and symbol not in symbol_set:
                continue
            events_path = symbol_dir / "events.parquet"
            if not events_path.exists():
                logger.warning("missing %s, skipping", events_path)
                continue

            dst = out_dir / market / symbol / f"{date}.parquet"
            if skip_existing and dst.exists():
                _assert_existing_schema(dst, schema_version)
                skipped += 1
                continue

            pending.append(
                {
                    "events_path": str(events_path),
                    "dst": str(dst),
                    "date": date,
                    "market": market,
                    "schema_version": schema_version,
                    "book_features_path": str(symbol_dir / "book_features.parquet"),
                }
            )

    if skipped:
        logger.info(
            "skip_existing: reused %d canonical symbol-days for %s", skipped, date
        )

    if not pending:
        logger.info("nothing to canonicalize for %s", date)
        return []

    workers = max(
        1, int(n_workers if n_workers is not None else default_canon_workers())
    )
    written: list[Path] = []
    errors = 0

    if workers == 1:
        for payload in pending:
            status, path, n_events, err = _canonicalize_one(payload)
            if status == "written":
                written.append(Path(path))
                logger.info("wrote %s (%d events)", path, n_events)
            elif status == "empty":
                logger.warning("empty %s, skipping", payload["events_path"])
            else:
                errors += 1
                logger.error("failed %s: %s", path, err)
        if strict and errors:
            msg = f"canonicalize failed for {errors} symbol(s) on {date}"
            raise RuntimeError(msg)
        return written

    logger.info(
        "canonicalizing %d symbols for %s with n_workers=%d",
        len(pending),
        date,
        workers,
    )
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(_canonicalize_one, p) for p in pending]
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), start=1):
            status, path, n_events, err = fut.result()
            if status == "written":
                written.append(Path(path))
            elif status == "empty":
                logger.warning("empty shard skipped -> %s", path)
            else:
                errors += 1
                logger.error("failed %s: %s", path, err)
            if i % 200 == 0 or i == total:
                logger.info(
                    "canonicalize progress %s %d/%d written=%d",
                    date,
                    i,
                    total,
                    len(written),
                )

    logger.info("canonicalized %d symbol-days for %s", len(written), date)
    if strict and errors:
        msg = f"canonicalize failed for {errors} symbol(s) on {date}"
        raise RuntimeError(msg)
    return written


def canonicalize_and_tokenize_clean_dir(
    clean_dir: Path,
    tokens_dir: Path,
    *,
    vocab_path: Path,
    date: str,
    markets: tuple[str, ...] = ("SH", "SZ"),
    symbols: tuple[str, ...] | None = None,
    skip_existing: bool = False,
    n_workers: int | None = None,
) -> list[Path]:
    """Directly convert clean shards to final tokens in one worker pass."""
    from quant_fm.tokenizer.artifact_contract import assert_token_contract_matches

    clean_dir = Path(clean_dir)
    tokens_dir = Path(tokens_dir)
    vocab = _load_vocab(vocab_path)
    symbol_set = set(symbols) if symbols is not None else None
    pending: list[dict[str, str]] = []
    written: list[Path] = []

    for market in markets:
        market_dir = clean_dir / market
        if not market_dir.is_dir():
            continue
        for symbol_dir in sorted(market_dir.iterdir()):
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name
            if symbol_set is not None and symbol not in symbol_set:
                continue
            events_path = symbol_dir / "events.parquet"
            if not events_path.exists():
                continue
            dst = tokens_dir / market / symbol / f"{date}.parquet"
            if skip_existing and dst.exists():
                try:
                    assert_token_contract_matches(dst, vocab)
                except ValueError as exc:
                    msg = (
                        f"refusing to overwrite incompatible token shard during "
                        f"resume: {dst}; use a new tokens root ({exc})"
                    )
                    raise RuntimeError(msg) from exc
                written.append(dst)
                continue
            pending.append(
                {
                    "events_path": str(events_path),
                    "dst": str(dst),
                    "date": date,
                    "market": market,
                    "vocab_path": str(vocab_path),
                    "schema_version": vocab.schema_version,
                    "book_features_path": str(symbol_dir / "book_features.parquet"),
                }
            )

    if not pending:
        return written
    workers = max(
        1, int(n_workers if n_workers is not None else default_canon_workers())
    )
    errors: list[str] = []
    if workers == 1:
        results = (_canonicalize_tokenize_one(payload) for payload in pending)
        for status, path, _rows, error in results:
            if status == "written":
                written.append(Path(path))
            elif status == "error":
                errors.append(f"{path}: {error}")
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(_canonicalize_tokenize_one, item) for item in pending
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                status, path, _rows, error = future.result()
                if status == "written":
                    written.append(Path(path))
                elif status == "error":
                    errors.append(f"{path}: {error}")
                if index % 200 == 0 or index == len(futures):
                    logger.info(
                        "fused tokenize progress %s %d/%d written=%d errors=%d",
                        date,
                        index,
                        len(futures),
                        len(written),
                        len(errors),
                    )
    if errors:
        msg = f"fused canonicalize/tokenize failed on {date}: {errors[:5]}"
        raise RuntimeError(msg)
    logger.info("fused canonicalize/tokenize done %s shards=%d", date, len(written))
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--markets", default="SH,SZ")
    parser.add_argument("--symbols", default=None)
    parser.add_argument(
        "--schema-version",
        choices=(V1_SCHEMA_VERSION, V2_SCHEMA_VERSION),
        default=V2_SCHEMA_VERSION,
    )
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    symbols = tuple(args.symbols.split(",")) if args.symbols else None
    markets = tuple(m.strip().upper() for m in args.markets.split(","))
    paths = canonicalize_clean_dir(
        args.clean_dir,
        args.out_dir,
        date=args.date,
        markets=markets,
        symbols=symbols,
        schema_version=args.schema_version,
    )
    logger.info("canonicalized %d symbol-days", len(paths))


if __name__ == "__main__":
    main()
