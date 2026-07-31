"""Run multi-GPU Dense230M stock-day embedding extraction inside one K8s pod."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

from quant_fm.embedding.contract import propagate_embedding_contract


def _worker_command(
    *,
    checkpoint: Path,
    manifest: Path,
    split: str,
    out: Path,
    batch_size: int,
    num_gpus: int,
    gpu: int,
    dtype: str,
    context: int | None,
    pooling: str | None,
    stride: int | None,
) -> list[str]:
    """Build one worker command without overriding frozen representation defaults."""
    command = [
        sys.executable,
        "-m",
        "quant_fm.embedding.extract_hidden",
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(manifest),
        "--split",
        split,
        "--out",
        str(out),
        "--dtype",
        dtype,
        "--batch-size",
        str(batch_size),
        "--num-parts",
        str(num_gpus),
        "--part-index",
        str(gpu),
        "--device",
        "cuda:0",
    ]
    if context is not None:
        command.extend(("--context", str(context)))
    if pooling is not None:
        command.extend(("--pooling", pooling))
    if stride is not None:
        command.extend(("--stride", str(stride)))
    return command


def _merge_parts(parts_dir: Path, split: str, num_gpus: int, out: Path) -> None:
    paths = [
        parts_dir / f"{split}.part{gpu}of{num_gpus}.parquet" for gpu in range(num_gpus)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        message = f"missing embedding parts: {missing}"
        raise RuntimeError(message)
    frames = [pl.read_parquet(path) for path in paths]
    merged = pl.concat(frames, how="vertical_relaxed").sort(["date", "symbol"])
    temporary = out.with_suffix(out.suffix + ".tmp")
    merged.write_parquet(temporary)
    temporary.replace(out)
    propagate_embedding_contract(
        paths,
        out,
        context=f"K8s embedding parts for split={split}",
    )
    print(f"merged split={split} rows={merged.height} -> {out}", flush=True)


def _extract_split(
    *,
    checkpoint: Path,
    manifest: Path,
    emb_dir: Path,
    split: str,
    num_gpus: int,
    batch_size: int,
    dtype: str,
    context: int | None,
    pooling: str | None,
    stride: int | None,
) -> None:
    parts_dir = emb_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
    for gpu in range(num_gpus):
        out = parts_dir / f"{split}.part{gpu}of{num_gpus}.parquet"
        log_path = parts_dir / f"{split}.part{gpu}of{num_gpus}.log"
        log_handle = log_path.open("wb")
        command = _worker_command(
            checkpoint=checkpoint,
            manifest=manifest,
            split=split,
            out=out,
            batch_size=batch_size,
            num_gpus=num_gpus,
            gpu=gpu,
            dtype=dtype,
            context=context,
            pooling=pooling,
            stride=stride,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(
            command, stdout=log_handle, stderr=subprocess.STDOUT, env=env
        )
        processes.append((gpu, process, log_handle))
        print(
            f"launched split={split} gpu={gpu} pid={process.pid} -> {out}", flush=True
        )

    failures: list[tuple[int, int]] = []
    for gpu, process, log_handle in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode != 0:
            failures.append((gpu, returncode))
    if failures:
        message = f"embedding workers failed for split={split}: {failures}"
        raise RuntimeError(message)
    _merge_parts(parts_dir, split, num_gpus, emb_dir / f"{split}.parquet")


def main() -> None:
    """Extract requested splits across visible GPUs and merge their parts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--emb-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="explicit override; default uses checkpoint context_horizon",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "last", "lastk_mean", "multi_scale"],
        default=None,
        help="explicit override; default uses checkpoint pooling.method",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="explicit override; default uses checkpoint pooling.stride",
    )
    args = parser.parse_args()
    if args.num_gpus < 1:
        parser.error("--num-gpus must be positive")
    if args.context is not None and args.context < 1:
        parser.error("--context must be positive")
    if args.stride is not None and args.stride < 1:
        parser.error("--stride must be positive")
    if (
        args.context is not None
        and args.stride is not None
        and args.stride > args.context
    ):
        parser.error("--stride cannot exceed --context")
    args.emb_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        _extract_split(
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            emb_dir=args.emb_dir,
            split=split,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
            dtype=args.dtype,
            context=args.context,
            pooling=args.pooling,
            stride=args.stride,
        )

    split_paths = [args.emb_dir / f"{split}.parquet" for split in args.splits]
    frames = [pl.read_parquet(path) for path in split_paths]
    merged = pl.concat(frames, how="vertical_relaxed").sort(["date", "symbol"])
    out = args.emb_dir / "all.parquet"
    temporary = out.with_suffix(out.suffix + ".tmp")
    merged.write_parquet(temporary)
    temporary.replace(out)
    propagate_embedding_contract(
        split_paths,
        out,
        context="K8s embedding splits",
    )
    print(
        f"merged all rows={merged.height} dates={merged['date'].n_unique()} -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
