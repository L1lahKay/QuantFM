import json

import polars as pl

from quant_fm.downstream.run_judge import run_judge


def _embeddings(dates: list[str], symbols: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "market": "SZ",
                "emb_0": float(index),
                "emb_1": float(index % 3),
            }
            for date in dates
            for index, symbol in enumerate(symbols)
        ]
    )


def test_dev_only_judge_does_not_require_or_read_test_embeddings(tmp_path) -> None:
    symbols = [f"{index:06d}" for index in range(20)]
    train_dates = ["2025-01-01", "2025-01-02"]
    val_dates = ["2025-01-03"]
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    _embeddings(train_dates, symbols).write_parquet(emb_dir / "train.parquet")
    _embeddings(val_dates, symbols).write_parquet(emb_dir / "val.parquet")
    assert not (emb_dir / "test.parquet").exists()

    panel = pl.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "fwd_ret": float(index) / 100.0,
            }
            for date in (*train_dates, *val_dates)
            for index, symbol in enumerate(symbols)
        ]
    )
    panel_path = tmp_path / "panel.parquet"
    panel.write_parquet(panel_path)
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint identity only")

    report_path = run_judge(
        workdir=tmp_path / "quick",
        checkpoint=checkpoint,
        panel_path=panel_path,
        emb_dir=emb_dir,
        epochs=1,
        device="cpu",
        dev_only=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["evaluation_scope"] == "development_only"
    assert report["test"]["error"] == "empty"
    assert report["test_random_baseline_mean_ic"] is None
