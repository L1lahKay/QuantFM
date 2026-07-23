"""兼容入口：用冻结 Ranker 生成仅含 score 的生产交付。"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_fm.signal.generate import generate_scores


def main() -> None:
    """将参数转交给稳定的 signal 生成器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--ranker-metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fm-checkpoint", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--universe", type=Path)
    parser.add_argument("--allow-in-sample", action="store_true")
    args = parser.parse_args()
    generate_scores(
        embeddings_path=args.embeddings,
        ranker_path=args.ranker,
        ranker_metadata_path=args.ranker_metadata,
        out_dir=args.out_dir,
        device=args.device,
        fm_checkpoint_path=args.fm_checkpoint,
        vocab_path=args.vocab,
        universe_path=args.universe,
        allow_in_sample=args.allow_in_sample,
    )


if __name__ == "__main__":
    main()
