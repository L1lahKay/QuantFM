import json
from dataclasses import replace
from pathlib import Path

import polars as pl

from quant_fm.manifest.build_manifest import Manifest, ShardEntry
from quant_fm.scripts.audit_v2_artifacts import audit_v2_artifacts
from quant_fm.tokenizer.artifact_contract import (
    token_contract_path,
    token_contract_payload,
)
from quant_fm.tokenizer.field_spec import FULL_FIELD_SPECS_V2
from quant_fm.tokenizer.storage_encoding_v2 import (
    StorageEncodingMetadataV2,
    quantize_frame_v2,
)
from quant_fm.tokenizer.vocab_v2 import (
    BinnedFieldVocab,
    ContinuousNormalizer,
    VocabV2,
)


def _full_vocab(*, wide_token: bool = False) -> VocabV2:
    date = "2025-01-02"
    categorical: dict[str, tuple[str, ...]] = {}
    categorical_occupancy: dict[str, tuple[int, ...]] = {}
    binned: dict[str, BinnedFieldVocab] = {}
    for spec in FULL_FIELD_SPECS_V2:
        if spec.kind in {"categorical", "context"}:
            width = 300 if wide_token and spec.name == "evt_type" else 1
            categorical[spec.name] = tuple(
                f"category_{index}" for index in range(width)
            )
            categorical_occupancy[spec.name] = (
                2,
                *(0 for _ in range(width - 1)),
            )
            continue
        binned[spec.name] = BinnedFieldVocab(
            requested_n_bins=int(spec.n_bins or 1),
            occupancy=(2,),
            normalizer=ContinuousNormalizer(
                mean=0.0,
                std=1.0,
                clip=5.0,
                count=2,
            ),
            min_value=-0.5,
            max_value=0.5,
            n_observed=2,
        )
    return VocabV2(
        field_specs=FULL_FIELD_SPECS_V2,
        categorical=categorical,
        categorical_occupancy=categorical_occupancy,
        binned=binned,
        fit_dates=(date,),
    )


def _semantic_frame(vocab: VocabV2) -> pl.DataFrame:
    data: dict[str, pl.Series] = {
        "int_time": pl.Series("int_time", [93_000_000, 93_000_001], dtype=pl.Int64),
        "source_seqnum": pl.Series("source_seqnum", [1, 2], dtype=pl.Int64),
        "event_idx": pl.Series("event_idx", [0, 1], dtype=pl.Int64),
    }
    for column in vocab.token_field_sizes():
        data[column] = pl.Series(column, [6, 6], dtype=pl.Int64)
    for spec in vocab.field_specs:
        if spec.value_column is not None:
            column = str(spec.value_column)
            data[column] = pl.Series(column, [-0.5, 0.5], dtype=pl.Float32)
    return pl.DataFrame(data)


def _write_contract(
    path: Path,
    vocab: VocabV2,
    metadata: StorageEncodingMetadataV2 | None,
) -> None:
    payload = token_contract_payload(vocab)
    if metadata is not None:
        payload["storage_encoding"] = metadata.to_dict()
    token_contract_path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _complete_artifacts(
    tmp_path: Path,
    *,
    q16: bool,
    wide_token: bool = False,
) -> tuple[Path, VocabV2, Path, StorageEncodingMetadataV2 | None]:
    root = tmp_path / "v2_shared"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    vocab = _full_vocab(wide_token=wide_token)
    vocab_path = data_dir / "vocab_v2.json"
    vocab.save(vocab_path)
    token_path = data_dir / "tokens.parquet"
    frame = _semantic_frame(vocab)
    metadata = None
    if q16:
        frame, metadata = quantize_frame_v2(frame, vocab)
    frame.write_parquet(token_path)
    _write_contract(token_path, vocab, metadata)
    Manifest(
        shards=[
            ShardEntry(
                market="SH",
                symbol="600000",
                date="2025-01-02",
                path=str(token_path),
                rows=frame.height,
                sha256="test",
                split="train",
            )
        ],
        vocab_path=str(vocab_path),
        schema_version=vocab.schema_version,
        event_ordering_version=vocab.event_ordering_version,
        feature_transform_version=vocab.feature_transform_version,
    ).save(data_dir / "manifest.json")
    (root / "validation_windows.json").write_text("{}\n", encoding="utf-8")
    return root, vocab, token_path, metadata


def test_v2_audit_reports_missing_required_artifacts(tmp_path: Path) -> None:
    result = audit_v2_artifacts(tmp_path / "v2_shared")
    assert result["contract_ready"] is False
    assert {item["code"] for item in result["issues"]} == {
        "missing_manifest",
        "missing_vocab_v2",
    }


def test_v2_audit_rejects_invalid_vocab_without_scanning_tokens(
    tmp_path: Path,
) -> None:
    data = tmp_path / "v2_shared" / "data"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text(
        json.dumps({"schema_version": "cn_l2_v2", "shards": []}),
        encoding="utf-8",
    )
    (data / "vocab_v2.json").write_text("{}", encoding="utf-8")
    result = audit_v2_artifacts(tmp_path / "v2_shared")
    assert result["contract_ready"] is False
    assert result["issues"][0]["code"] == "invalid_vocab"


def test_v2_audit_accepts_legacy_float32_storage(tmp_path: Path) -> None:
    root, _, _, _ = _complete_artifacts(tmp_path, q16=False)

    result = audit_v2_artifacts(root)

    assert result["contract_ready"] is True
    encoding = result["sampled_shards"][0]["storage_encoding"]
    assert encoding["mode"] == "legacy_float32"
    assert encoding["validated"] is True


def test_v2_audit_validates_q16_and_narrow_token_storage(tmp_path: Path) -> None:
    root, _, _, _ = _complete_artifacts(tmp_path, q16=True, wide_token=True)

    result = audit_v2_artifacts(root)

    assert result["contract_ready"] is True
    encoding = result["sampled_shards"][0]["storage_encoding"]
    assert encoding["mode"] == "q16"
    assert encoding["validated"] is True
    assert encoding["physical_dtypes"]["tok_evt_type"] == "uint16"
    assert encoding["physical_dtypes"]["tok_side"] == "uint8"
    assert encoding["physical_dtypes"]["val_price"] == "int16"
    assert len(encoding["metadata_sha256"]) == 64


def test_v2_audit_rejects_integer_scalars_without_storage_metadata(
    tmp_path: Path,
) -> None:
    root, vocab, token_path, metadata = _complete_artifacts(tmp_path, q16=True)
    assert metadata is not None
    _write_contract(token_path, vocab, metadata=None)

    result = audit_v2_artifacts(root)

    assert result["contract_ready"] is False
    assert "token_storage_encoding_missing" in {
        issue["code"] for issue in result["issues"]
    }


def test_v2_audit_rejects_tampered_or_wrong_vocab_storage_metadata(
    tmp_path: Path,
) -> None:
    root, vocab, token_path, metadata = _complete_artifacts(tmp_path, q16=True)
    assert metadata is not None
    payload = token_contract_payload(vocab)
    tampered = metadata.to_dict()
    tampered["scalar_fields"][0]["clip"] = 4.0
    payload["storage_encoding"] = tampered
    token_contract_path(token_path).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    result = audit_v2_artifacts(root)
    assert "token_storage_metadata_invalid" in {
        issue["code"] for issue in result["issues"]
    }

    wrong_vocab = replace(metadata, vocab_sha256="0" * 64)
    _write_contract(token_path, vocab, wrong_vocab)
    result = audit_v2_artifacts(root)
    assert "token_storage_metadata_invalid" in {
        issue["code"] for issue in result["issues"]
    }


def test_v2_audit_rejects_q16_dtype_and_value_range_violations(
    tmp_path: Path,
) -> None:
    root, vocab, token_path, metadata = _complete_artifacts(tmp_path, q16=True)
    assert metadata is not None
    first_token = next(iter(vocab.token_field_sizes()))
    first_scalar = str(
        next(spec.value_column for spec in vocab.field_specs if spec.value_column)
    )
    frame = pl.read_parquet(token_path).with_columns(
        pl.col(first_token).cast(pl.UInt16),
        pl.col(first_scalar).cast(pl.Int32),
    )
    frame.write_parquet(token_path)

    result = audit_v2_artifacts(root)
    assert "token_storage_dtype_mismatch" in {
        issue["code"] for issue in result["issues"]
    }

    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(vocab.token_field_sizes()[first_token])
        .otherwise(pl.col(first_token))
        .cast(pl.UInt8)
        .alias(first_token),
        pl.when(pl.int_range(pl.len()) == 0)
        .then(-32_768)
        .otherwise(pl.col(first_scalar))
        .cast(pl.Int16)
        .alias(first_scalar),
    )
    frame.write_parquet(token_path)

    result = audit_v2_artifacts(root)
    assert "token_storage_value_out_of_range" in {
        issue["code"] for issue in result["issues"]
    }
