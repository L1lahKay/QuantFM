"""
在**全新解释器**里并行 tokenize 某个交易日的 event 分片。

单独成一个 CLI 是为了规避一个死锁：``run_medium`` 在同一进程里先做 clean/
canonicalize（polars/rayon 累积上百个线程、占用上百 GB 常驻内存），随后再用
``ProcessPoolExecutor(spawn)`` 开池时，fork+exec 那一瞬间会被父进程的线程锁 /
巨大页表拖死（worker 全部 futex_wait、0 产出）。由 ``run_medium`` 用 subprocess
调用本模块，可保证每天的 tokenize 都在干净解释器里跑，池必然正常启动。

用法::

    uv run python -m quant_fm.scripts.tokenize_dir \
        --events-dir <events_root> --tokens-dir <tokens_root> \
        --vocab <vocab.json> --day 2026-01-05 --workers 16 --drop-events
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant_fm.scripts.make_adhoc_manifest import write_day_index
from quant_fm.scripts.run_medium import _load_vocab, _tokenize_shards_parallel
from quant_fm.tokenizer.artifact_contract import assert_token_contract_matches

logger = logging.getLogger(__name__)


def tokenize_day(
    *,
    events_dir: Path,
    tokens_dir: Path,
    vocab_path: Path,
    day: str,
    workers: int,
    drop_events: bool,
    resume: bool,
) -> int:
    """收集某日未 tokenize 的分片并并行处理，返回处理数。"""
    vocab = _load_vocab(vocab_path)
    jobs: list[tuple[Path, Path]] = []
    for p in sorted(events_dir.rglob(f"{day}.parquet")):
        dst = tokens_dir / p.relative_to(events_dir)
        if dst.exists() and resume:
            try:
                assert_token_contract_matches(dst, vocab)
            except ValueError as exc:
                msg = (
                    f"refusing to overwrite incompatible token shard during resume: "
                    f"{dst}; use a new tokens root ({exc})"
                )
                raise RuntimeError(msg) from exc
            if drop_events:
                p.unlink(missing_ok=True)
            continue
        jobs.append((p, dst))
    return _tokenize_shards_parallel(
        jobs, vocab_path=vocab_path, drop_events=drop_events, n_workers=workers
    )


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events-dir", type=Path, required=True)
    p.add_argument("--tokens-dir", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--day", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--drop-events", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    n = tokenize_day(
        events_dir=args.events_dir,
        tokens_dir=args.tokens_dir,
        vocab_path=args.vocab,
        day=args.day,
        workers=args.workers,
        drop_events=args.drop_events,
        resume=not args.no_resume,
    )
    write_day_index(args.tokens_dir, args.day)
    logger.info("tokenize_dir day=%s shards=%d", args.day, n)


if __name__ == "__main__":
    main()
