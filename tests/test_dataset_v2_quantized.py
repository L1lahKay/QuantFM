from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from quant_fm.manifest.build_manifest import ShardEntry  # noqa: E402
from quant_fm.pretrain.dataset_v2 import EventWindowDatasetV2  # noqa: E402
from quant_fm.tokenizer.artifact_contract import write_token_contract  # noqa: E402
from quant_fm.tokenizer.field_spec import FieldSpec  # noqa: E402
from quant_fm.tokenizer.storage_encoding_v2 import (  # noqa: E402
    quantize_frame_v2,
)
from quant_fm.tokenizer.vocab_v2 import default_vocab_v2  # noqa: E402


def _vocab():
    specs = (
        FieldSpec("evt_type", "evt_type", "categorical", is_target=True),
        FieldSpec("feature", "feature", "ordinal", n_bins=4, is_target=True),
    )
    return default_vocab_v2(
        specs,
        categorical={"evt_type": ("ADD", "EXEC")},
    )


def _float_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tok_evt_type": pl.Series([6, 7, 6], dtype=pl.Int64),
            "tok_feature_bin": pl.Series([6, 6, 6], dtype=pl.Int64),
            "val_feature": pl.Series([-5.0, 0.25, 5.0], dtype=pl.Float32),
        }
    )


def _shard(path) -> ShardEntry:
    return ShardEntry(
        market="SH",
        symbol="600000",
        date="2025-01-02",
        path=str(path),
        rows=3,
        sha256="test",
    )


def _write_storage_contract(path, vocab, metadata) -> None:
    write_token_contract(path, vocab, storage_encoding=metadata.to_dict())


def test_dataset_decodes_q16_scalars_and_keeps_model_dtypes(tmp_path) -> None:
    vocab = _vocab()
    original = _float_frame()
    encoded, metadata = quantize_frame_v2(original, vocab)
    path = tmp_path / "q16.parquet"
    encoded.write_parquet(path)
    _write_storage_contract(path, vocab, metadata)

    sample = EventWindowDatasetV2(
        [_shard(path)],
        vocab=vocab,
        context=3,
        min_len=1,
    )[0]

    assert encoded["val_feature"].dtype == pl.Int16
    assert sample["tok_evt_type"].dtype == torch.int64
    assert sample["tok_feature_bin"].dtype == torch.int64
    assert sample["val_feature"].dtype == torch.float32
    np.testing.assert_allclose(
        sample["val_feature"].numpy(),
        original["val_feature"].to_numpy(),
        rtol=0.0,
        atol=5.0 / 32767.0,
    )


def test_dataset_keeps_legacy_float32_scalar_compatible(tmp_path) -> None:
    vocab = _vocab()
    original = _float_frame()
    path = tmp_path / "legacy_float.parquet"
    original.write_parquet(path)

    sample = EventWindowDatasetV2(
        [_shard(path)],
        vocab=vocab,
        context=3,
        min_len=1,
    )[0]

    assert sample["val_feature"].dtype == torch.float32
    np.testing.assert_array_equal(
        sample["val_feature"].numpy(),
        original["val_feature"].to_numpy(),
    )


def test_dataset_rejects_int16_scalar_without_storage_contract(tmp_path) -> None:
    vocab = _vocab()
    encoded, _ = quantize_frame_v2(_float_frame(), vocab)
    path = tmp_path / "uncontracted_q16.parquet"
    encoded.write_parquet(path)

    dataset = EventWindowDatasetV2(
        [_shard(path)],
        vocab=vocab,
        context=3,
        min_len=1,
    )
    with pytest.raises(ValueError, match=r"(?i)(storage|metadata|int16|quantiz)"):
        dataset[0]
