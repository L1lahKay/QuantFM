"""quant_fm 流水线单元测试（模式、分词器、清单、门控）。"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fm.schema.cn_l2_v1 import (
    CANONICAL_COLUMNS,
    board_of,
    events_to_canonical,
    exchange_of,
)
from quant_fm.tokenizer.fit_bins import fit_bins
from quant_fm.tokenizer.tokenize_events import (
    assert_no_leakage,
    tokenize_frame,
)
from quant_fm.tokenizer.vocab import PAD_ID, Vocab, default_vocab


def _fake_events(n: int = 200, symbol: str = "000001") -> pl.DataFrame:
    rng = np.random.default_rng(0)
    int_time = np.linspace(93000000, 145600000, n).astype(np.int64)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "market": ["SZ"] * n,
            "event_idx": np.arange(n, dtype=np.int64),
            "int_time": int_time,
            "local_time": int_time,
            "serial": np.arange(n, dtype=np.int64),
            "delta_t": np.zeros(n, dtype=np.int64),
            "session_phase": ["CONTINUOUS_AM"] * n,
            "event_type": rng.choice(["ADD", "CANCEL", "TRADE"], size=n),
            "side": rng.choice(["BUY", "SELL", "UNKNOWN"], size=n),
            "price": rng.integers(90000, 110000, size=n),
            "volume": rng.integers(100, 5000, size=n),
            "log_volume": np.zeros(n),
            "orderorino": np.arange(n, dtype=np.int64),
            "buy_id": np.zeros(n, dtype=np.int64),
            "sell_id": np.zeros(n, dtype=np.int64),
        }
    )


def test_exchange_and_board():
    assert exchange_of("SH") == "XSHG"
    assert exchange_of("SZ") == "XSHE"
    assert board_of("688981", "SH") == "STAR"
    assert board_of("600519", "SH") == "MAIN"
    assert board_of("300750", "SZ") == "CHINEXT"
    assert board_of("000001", "SZ") == "MAIN"


def test_events_to_canonical_columns():
    canonical = events_to_canonical(_fake_events(), date="2024-07-03", market="SZ")
    assert tuple(canonical.columns) == CANONICAL_COLUMNS
    assert canonical["symbol"][0] == "000001.SZ"
    assert set(canonical["evt_type"].unique()) <= {"ADD", "CANCEL", "EXEC"}
    # 价格由 1e4 缩放整数还原。
    assert 5.0 < canonical["price"].mean() < 15.0


def test_vocab_roundtrip(tmp_path):
    vocab = default_vocab(n_bins=8)
    vocab.edges = {k: list(np.linspace(-1, 1, 7)) for k in vocab.edges}
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocab.load(path)
    assert loaded.field_sizes() == vocab.field_sizes()
    assert loaded.size("evt_type") == vocab.size("evt_type")


def test_tokenize_determinism(tmp_path):
    canonical = events_to_canonical(_fake_events(), date="2024-07-03", market="SZ")
    canonical.write_parquet(tmp_path / "d.parquet")
    vocab = fit_bins([tmp_path / "d.parquet"], n_bins=16, fit_dates=["2024-07-03"])
    t1 = tokenize_frame(canonical, vocab)
    t2 = tokenize_frame(canonical, vocab)
    assert t1.equals(t2)
    # 真实 token 中不会出现 PAD id。
    for col in [c for c in t1.columns if c.startswith("tok_")]:
        assert (t1[col] > PAD_ID).all()


def test_fit_bins_rejects_false_date_provenance(tmp_path):
    canonical = events_to_canonical(_fake_events(), date="2024-07-03", market="SZ")
    path = tmp_path / "events.parquet"
    canonical.write_parquet(path)
    with pytest.raises(ValueError, match="fit_dates must exactly match"):
        fit_bins([path], n_bins=8, fit_dates=["2024-07-04"])


def test_no_leakage_raises():
    vocab = default_vocab()
    vocab.fit_dates = ("2024-01-01", "2024-01-02")
    with pytest.raises(AssertionError):
        assert_no_leakage(vocab, ["2024-01-02"], ["2024-01-03"])


def test_manifest_split(tmp_path):
    from quant_fm.manifest.build_manifest import build_manifest

    for date in ["2024-01-01", "2024-01-05", "2024-01-10"]:
        canonical = events_to_canonical(_fake_events(), date=date, market="SZ")
        dst = tmp_path / "tokens" / "SZ" / "000001" / f"{date}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        tokenize_frame(canonical, default_vocab(8)).write_parquet(dst)

    manifest = build_manifest(
        tmp_path / "tokens", train_end="2024-01-01", val_end="2024-01-05"
    )
    assert len(manifest.split("train")) == 1
    assert len(manifest.split("val")) == 1
    assert len(manifest.split("test")) == 1
    assert manifest.shards[0].sha256


def test_pretrain_split_contract_rejects_thin_validation_and_vocab_leakage():
    from quant_fm.manifest.build_manifest import Manifest, ShardEntry
    from quant_fm.pretrain.train import validate_pretrain_split_contract

    def shard(date: str, split: str) -> ShardEntry:
        return ShardEntry("SZ", "000001", date, "unused", 10, "hash", split)

    manifest = Manifest(
        shards=[
            shard("2025-01-02", "train"),
            shard("2025-01-03", "train"),
            shard("2025-01-06", "val"),
            shard("2025-01-07", "test"),
        ]
    )
    vocab = default_vocab()
    vocab.fit_dates = ("2025-01-02", "2025-01-07")
    with pytest.raises(ValueError, match="validation split is too short") as error:
        validate_pretrain_split_contract(
            manifest,
            vocab,
            require_validation=True,
            min_validation_dates=2,
            min_test_dates=1,
        )
    assert "vocab was fitted on validation/test dates" in str(error.value)


def test_pretrain_split_contract_accepts_chronological_date_blocks():
    from quant_fm.manifest.build_manifest import Manifest, ShardEntry
    from quant_fm.pretrain.train import validate_pretrain_split_contract

    shards = []
    for index in range(15):
        split = "train" if index < 5 else "val" if index < 10 else "test"
        shards.append(
            ShardEntry(
                "SZ",
                "000001",
                f"2025-01-{index + 1:02d}",
                "unused",
                10,
                "hash",
                split,
            )
        )
    vocab = default_vocab()
    vocab.fit_dates = tuple(item.date for item in shards if item.split == "train")
    contract = validate_pretrain_split_contract(
        Manifest(shards=shards),
        vocab,
        require_validation=True,
    )
    assert contract["train_dates"] == 5
    assert contract["validation_dates"] == 5
    assert contract["test_dates"] == 5


def test_pretrain_split_contract_rejects_vocab_dates_absent_from_manifest():
    from quant_fm.manifest.build_manifest import Manifest, ShardEntry
    from quant_fm.pretrain.train import validate_pretrain_split_contract

    shards = [
        ShardEntry("SZ", "000001", "2025-01-02", "unused", 10, "hash", "train"),
        ShardEntry("SZ", "000001", "2025-01-03", "unused", 10, "hash", "val"),
        ShardEntry("SZ", "000001", "2025-01-06", "unused", 10, "hash", "test"),
    ]
    vocab = default_vocab()
    # 该日期即使早于正式 OOS，只要不属于 manifest 的训练切分，也不能用于拟合。
    vocab.fit_dates = ("2025-01-02", "2025-01-05")

    with pytest.raises(ValueError, match="not contained in the manifest training"):
        validate_pretrain_split_contract(
            Manifest(shards=shards),
            vocab,
            require_validation=True,
            min_validation_dates=1,
            min_test_dates=1,
        )


def test_pretrain_split_contract_rejects_manifest_date_plan_mismatch():
    from quant_fm.manifest.build_manifest import Manifest, ShardEntry
    from quant_fm.pretrain.train import validate_pretrain_split_contract

    shards = [
        ShardEntry("SZ", "000001", "2025-01-02", "unused", 10, "hash", "train"),
        ShardEntry("SZ", "000001", "2025-01-03", "unused", 10, "hash", "val"),
        ShardEntry("SZ", "000001", "2025-01-06", "unused", 10, "hash", "test"),
    ]
    vocab = default_vocab()
    vocab.fit_dates = ("2025-01-02",)
    with pytest.raises(ValueError, match="dates differ from frozen plan"):
        validate_pretrain_split_contract(
            Manifest(shards=shards),
            vocab,
            require_validation=True,
            min_validation_dates=1,
            min_test_dates=1,
            expected_dates={"train": {"2025-01-02", "2025-01-03"}},
        )


def test_cpcv_and_dsr():
    from quant_fm.downstream.evaluate import cpcv_splits, deflated_sharpe_ratio

    dates = [f"2024-01-{d:02d}" for d in range(1, 25)]
    splits = cpcv_splits(dates, n_groups=6, n_test_groups=2, purge=1, embargo=1)
    assert len(splits) == 15  # C(6,2)
    for train, test in splits:
        assert not (set(train) & set(test))

    dsr_high = deflated_sharpe_ratio(0.15, n_trials=5, n_obs=250, sr_variance=0.01)
    dsr_low = deflated_sharpe_ratio(0.02, n_trials=500, n_obs=250, sr_variance=0.25)
    assert dsr_high > dsr_low


def test_model_forward_and_loss():
    torch = pytest.importorskip("torch")
    from quant_fm.pretrain.dataset import DEFAULT_TARGET_FIELDS, FIELD_ORDER
    from quant_fm.pretrain.heads import next_event_loss
    from quant_fm.pretrain.model import OrderFlowFM

    vocab = default_vocab(n_bins=8)
    vocab.edges = {k: list(np.linspace(-1, 1, 7)) for k in vocab.edges}
    model = OrderFlowFM.from_vocab(
        vocab, d_model=32, n_layers=2, n_heads=4, max_seq_len=64
    )
    b, length = 2, 20
    batch = {f: torch.randint(1, 4, (b, length)) for f in FIELD_ORDER}
    batch["attention_mask"] = torch.ones(b, length, dtype=torch.bool)
    logits = model(batch)
    assert set(logits) == set(DEFAULT_TARGET_FIELDS)
    loss = next_event_loss(logits, batch, DEFAULT_TARGET_FIELDS)
    assert torch.isfinite(loss.total)
    assert model.num_parameters() > 0
