import json

import polars as pl

from quant_fm.downstream.run_judge import run_judge
from quant_fm.moe.artifact import load_regime_moe_artifact


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


def test_dev_only_judge_trains_and_persists_temporal_regime_moe(tmp_path) -> None:
    symbols = [f"{index:06d}" for index in range(20)]
    train_dates = ["2025-01-01", "2025-01-02"]
    val_dates = ["2025-01-03"]
    all_dates = [*train_dates, *val_dates]
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    _embeddings(train_dates, symbols).write_parquet(emb_dir / "train.parquet")
    _embeddings(val_dates, symbols).write_parquet(emb_dir / "val.parquet")

    panel_path = tmp_path / "panel.parquet"
    pl.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "fwd_ret": float(index) / 100.0,
            }
            for date in all_dates
            for index, symbol in enumerate(symbols)
        ]
    ).write_parquet(panel_path)
    regime_path = tmp_path / "regime.parquet"
    pl.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "asof_date": date,
                "market_vol": float(day),
                "stock_ofi": float(index - 10),
            }
            for day, date in enumerate(all_dates)
            for index, symbol in enumerate(symbols)
        ]
    ).write_parquet(regime_path)
    config_path = tmp_path / "regime.yaml"
    config_path.write_text(
        """
moe_router_version: "1.0"
placement: temporal_aggregator
enabled: true
n_experts: 2
top_k: 1
expert_hidden: 8
router_hidden: 4
dropout: 0.0
capacity_factor: 2.0
regime_features:
  - {name: market_vol, availability_lag: 0}
  - {name: stock_ofi, availability_lag: 0}
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint identity only")

    report_path = run_judge(
        workdir=tmp_path / "quick-regime",
        checkpoint=checkpoint,
        panel_path=panel_path,
        emb_dir=emb_dir,
        epochs=1,
        device="cpu",
        dev_only=True,
        regime_config_path=config_path,
        regime_features_path=regime_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    regime = report["temporal_regime_moe"]
    assert regime["config"]["placement"] == "temporal_aggregator"
    assert regime["feature_normalizer"]["fit_end"] == "2025-01-02"
    assert report["ranker"]["training_history"][0]["train_moe_aux_loss"] > 0
    model, normalizer, _payload = load_regime_moe_artifact(regime["path"])
    assert model.hidden_dim == 2
    assert normalizer.fit_end == "2025-01-02"
