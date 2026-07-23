"""
从 tokens 目录**动态**生成一个临时 manifest（所有分片归入单一 split，默认 ``test``）。

用途：增量出 score 时，不等 ``run_medium`` 在全部天跑完后才写正式 manifest，而是把
「当前已 tokenize 的天」即时组装成 manifest 交给 ``extract_hidden`` 抽 embedding。

与 ``build_manifest.scan_token_dir`` 的区别：
* **先按日期过滤，再登记**——只收本轮要抽的天，避免重复；
* **跳过 sha256**（置空）——embedding 抽取只用到 path/date/symbol/market，
  逐文件哈希在这里纯属浪费（长窗口下省掉数千次全文件读）。

用法::

    uv run python -m quant_fm.scripts.make_adhoc_manifest \
        --tokens-dir quant_fm/runs/oos2026/tokens \
        --out       quant_fm/runs/oos2026/embeddings/incr/_cycle/manifest.json \
        --skip-dates-file quant_fm/runs/oos2026/embeddings/incr/embedded_dates.txt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq

from quant_fm.manifest.build_manifest import Manifest, ShardEntry

logger = logging.getLogger(__name__)


def write_day_index(tokens_dir: Path, day: str) -> Path:
    """Persist token metadata once so incremental embedding avoids tree rescans."""
    entries: list[dict[str, object]] = []
    for path in sorted(tokens_dir.rglob(f"{day}.parquet")):
        try:
            relative = path.relative_to(tokens_dir)
            market, symbol = relative.parts[:2]
            rows = pq.ParquetFile(path).metadata.num_rows
        except (OSError, ValueError):
            logger.warning("跳过不可读 token 分片: %s", path)
            continue
        entries.append(
            {
                "market": market,
                "symbol": symbol,
                "date": day,
                "path": str(path.resolve()),
                "rows": int(rows),
            }
        )
    index_dir = tokens_dir.parent / "data" / "shard_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    out = index_dir / f"{day}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, separators=(",", ":")), encoding="utf-8")
    tmp.replace(out)
    logger.info("token index day=%s shards=%d → %s", day, len(entries), out)
    return out


def _read_dates(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _entries_from_day_indexes(
    tokens_dir: Path,
    dates: set[str],
    *,
    markets: tuple[str, ...],
    split: str,
    min_rows: int,
) -> list[ShardEntry] | None:
    """Load O(number-of-new-shards) indexes; return None if any date lacks one."""
    index_dir = tokens_dir.parent / "data" / "shard_index"
    paths = {date: index_dir / f"{date}.json" for date in dates}
    if not dates or any(not path.exists() for path in paths.values()):
        return None
    entries: list[ShardEntry] = []
    market_set = set(markets)
    try:
        for date in sorted(dates):
            rows = json.loads(paths[date].read_text(encoding="utf-8"))
            for item in rows:
                if item["market"] not in market_set or int(item["rows"]) < min_rows:
                    continue
                shard = Path(item["path"])
                if not shard.exists():
                    logger.warning("索引中的 token 已不存在: %s", shard)
                    continue
                entries.append(
                    ShardEntry(
                        market=str(item["market"]),
                        symbol=str(item["symbol"]),
                        date=date,
                        path=str(shard),
                        rows=int(item["rows"]),
                        sha256="",
                        split=split,
                    )
                )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("日期索引不可用，退回目录扫描", exc_info=True)
        return None
    logger.info("命中日期 shard index：%d 天，%d 分片", len(dates), len(entries))
    return entries


def scan_dates(
    tokens_dir: Path,
    *,
    markets: tuple[str, ...] = ("SH", "SZ"),
    skip_dates: set[str] | None = None,
    include_dates: set[str] | None = None,
    split: str = "test",
    min_rows: int = 1,
) -> list[ShardEntry]:
    """
    扫描 ``{tokens_dir}/{market}/{symbol}/{date}.parquet``，按日期过滤后登记。

    ``include_dates`` 非空时为**白名单**（只收这些天）；``skip_dates`` 为黑名单。
    两者可叠加：先过白名单再过黑名单。
    """
    skip = skip_dates or set()
    allow = include_dates or None
    if allow is not None:
        indexed = _entries_from_day_indexes(
            tokens_dir,
            allow - skip,
            markets=markets,
            split=split,
            min_rows=min_rows,
        )
        if indexed is not None:
            return indexed
    entries: list[ShardEntry] = []
    for market in markets:
        mkt_dir = tokens_dir / market
        if not mkt_dir.is_dir():
            continue
        for sym_dir in sorted(mkt_dir.iterdir()):
            if not sym_dir.is_dir():
                continue
            for shard in sorted(sym_dir.glob("*.parquet")):
                date = shard.stem
                if allow is not None and date not in allow:
                    continue
                if date in skip:
                    continue
                try:
                    rows = pq.ParquetFile(shard).metadata.num_rows
                except Exception:
                    logger.warning("跳过不可读分片: %s", shard)
                    continue
                if rows < min_rows:
                    continue
                entries.append(
                    ShardEntry(
                        market=market,
                        symbol=sym_dir.name,
                        date=date,
                        path=str(shard.resolve()),
                        rows=int(rows),
                        sha256="",  # embedding 不需要，省掉全文件哈希
                        split=split,
                    )
                )
    return entries


def build_adhoc_manifest(
    *,
    tokens_dir: Path,
    out: Path,
    split: str = "test",
    skip_dates_file: Path | None = None,
    include_dates_file: Path | None = None,
    vocab_path: Path | None = None,
) -> tuple[Path, list[str]]:
    """生成临时 manifest，返回 (路径, 本次纳入的排序日期列表)。"""
    entries = scan_dates(
        tokens_dir,
        skip_dates=_read_dates(skip_dates_file),
        include_dates=_read_dates(include_dates_file) or None,
        split=split,
    )
    manifest = Manifest(
        shards=entries,
        train_end=None,
        val_end=None,
        vocab_path=str(vocab_path) if vocab_path else None,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(out)
    dates = sorted({e.date for e in entries})
    logger.info(
        "adhoc manifest → %s: %d shards, %d 新天 %s",
        out,
        len(entries),
        len(dates),
        f"{dates[0]}..{dates[-1]}" if dates else "(无)",
    )
    return out, dates


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--skip-dates-file", type=Path, default=None)
    p.add_argument(
        "--include-dates-file",
        type=Path,
        default=None,
        help="白名单：只纳入该文件中列出的日期（每行一个 YYYY-MM-DD）",
    )
    p.add_argument("--vocab", type=Path, default=None)
    p.add_argument(
        "--print-new-dates",
        action="store_true",
        help="额外把本次纳入的新日期以 JSON 打到 stdout（供 shell 读取）",
    )
    args = p.parse_args()
    _, dates = build_adhoc_manifest(
        tokens_dir=args.tokens_dir,
        out=args.out,
        split=args.split,
        skip_dates_file=args.skip_dates_file,
        include_dates_file=args.include_dates_file,
        vocab_path=args.vocab,
    )
    if args.print_new_dates:
        print(json.dumps(dates))


if __name__ == "__main__":
    main()
