import json
from pathlib import Path

import pytest

from quant_fm.data_coverage import (
    coverage_set_sha256,
    expected_symbol_keys,
    write_coverage_receipt,
)
from quant_fm.manifest.build_manifest import Manifest, ShardEntry
from quant_fm.manifest.validation import sha256_file
from quant_fm.scripts.run_v2_parallel_data import (
    artifacts_ready,
    build_finalize_command,
    build_prepare_command,
    split_dates_round_robin,
    validate_dates,
    validate_worker_budget,
    verify_canonical_events,
    write_date_chunks,
)
from quant_fm.tokenizer.artifact_contract import token_contract_path


def test_round_robin_chunks_are_stable_and_non_overlapping(tmp_path: Path) -> None:
    dates = [f"2026-01-{day:02d}" for day in range(5, 11)]

    chunks = split_dates_round_robin(dates, 2)
    paths = write_date_chunks(tmp_path, chunks)

    assert chunks == [dates[::2], dates[1::2]]
    assert paths[0].read_text(encoding="utf-8").splitlines() == dates[::2]
    assert paths[1].read_text(encoding="utf-8").splitlines() == dates[1::2]


def test_prepare_command_is_fast_events_only_without_vocab() -> None:
    command = build_prepare_command(
        dates_file=Path("dates.txt"),
        workdir=Path("work"),
        symbols_sz_file=Path("sz.txt"),
        symbols_sh_file=Path("sh.txt"),
        train_end="2026-01-07",
        val_end="2026-01-08",
        build_regime_data=True,
    )

    assert "--fast-clean" in command
    assert "--events-only" in command
    assert "--resume" in command
    assert "--reuse-vocab" not in command
    assert "--drop-events" not in command
    assert "--build-regime-data" in command


def test_finalize_command_is_the_only_global_vocab_stage() -> None:
    command = build_finalize_command(
        dates_file=Path("dates.txt"),
        workdir=Path("work"),
        symbols_sz_file=Path("sz.txt"),
        symbols_sh_file=Path("sh.txt"),
        train_end="2026-01-07",
        val_end="2026-01-08",
        n_bins=32,
        max_samples_per_field=1000,
        seed=7,
        drop_events=True,
        build_regime_data=True,
    )

    assert "--events-only" not in command
    assert "--skip-clean" in command
    assert "--drop-events" in command
    assert "--v2-full-audit" in command
    assert "--build-regime-data" in command


def test_canonical_event_gate_fails_before_global_vocab(tmp_path: Path) -> None:
    event = tmp_path / "events" / "SZ" / "000001" / "2026-01-05.parquet"
    event.parent.mkdir(parents=True)
    event.touch()

    with pytest.raises(RuntimeError, match="2026-01-06"):
        verify_canonical_events(tmp_path / "events", ["2026-01-05", "2026-01-06"])


def test_canonical_event_gate_rejects_recorded_symbol_gaps(tmp_path: Path) -> None:
    for date in ("2026-01-05", "2026-01-06"):
        event = tmp_path / "events" / "SZ" / "000001" / f"{date}.parquet"
        event.parent.mkdir(parents=True, exist_ok=True)
        event.touch()
    failure = tmp_path / "data" / ".failed" / "2026-01-06.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('["000002"]\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"recorded symbol gaps.*2026-01-06"):
        verify_canonical_events(tmp_path / "events", ["2026-01-05", "2026-01-06"])


def test_canonical_event_gate_binds_exact_universe_receipt(tmp_path: Path) -> None:
    date = "2026-01-05"
    clean = tmp_path / "clean" / date / "SZ" / "000001" / "events.parquet"
    clean.parent.mkdir(parents=True)
    clean.touch()
    write_coverage_receipt(
        workdir=tmp_path,
        clean_dir=tmp_path / "clean" / date,
        date=date,
        symbols_sz=("000001", "000002"),
        symbols_sh=(),
    )
    canonical = tmp_path / "events" / "SZ" / "000001" / f"{date}.parquet"
    canonical.parent.mkdir(parents=True)
    canonical.touch()
    keys = expected_symbol_keys(("000001", "000002"), ())

    verify_canonical_events(tmp_path / "events", [date], expected_keys=keys)

    canonical.unlink()
    wrong = tmp_path / "events" / "SZ" / "000002" / f"{date}.parquet"
    wrong.parent.mkdir(parents=True)
    wrong.touch()
    with pytest.raises(RuntimeError, match="coverage disagrees"):
        verify_canonical_events(tmp_path / "events", [date], expected_keys=keys)


def test_formal_artifact_resume_requires_full_path_audit(tmp_path: Path) -> None:
    token_path = tmp_path / "tokens" / "SZ" / "000001" / "2026-01-05.parquet"
    token_path.parent.mkdir(parents=True)
    token_path.touch()
    sidecar = token_contract_path(token_path)
    sidecar.write_text("{}\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    vocab_path = tmp_path / "data" / "vocab_v2.json"
    vocab_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "data" / "manifest.json"
    Manifest(
        shards=[
            ShardEntry(
                market="SZ",
                symbol="000001",
                date="2026-01-05",
                path=str(token_path),
                rows=1,
                sha256=sha256_file(token_path),
                split="train",
                data_contract_sha256=sha256_file(sidecar),
            )
        ],
        train_end="2026-01-05",
        val_end="2026-01-05",
    ).save(manifest_path)
    clean = tmp_path / "clean" / "2026-01-05" / "SZ" / "000001" / "events.parquet"
    clean.parent.mkdir(parents=True)
    clean.touch()
    write_coverage_receipt(
        workdir=tmp_path,
        clean_dir=tmp_path / "clean" / "2026-01-05",
        date="2026-01-05",
        symbols_sz=("000001",),
        symbols_sh=(),
    )
    audit_path = tmp_path / "artifact_audit.json"
    audit_path.write_text(
        '{"contract_ready": true, "checked_all_paths": false}', encoding="utf-8"
    )

    assert artifacts_ready(tmp_path) is False

    audit_path.write_text(
        json.dumps(
            {
                "audit_version": "2.0",
                "contract_ready": True,
                "checked_all_paths": True,
                "coverage_sha256": coverage_set_sha256(tmp_path),
                "manifest_sha256": sha256_file(manifest_path),
                "vocab_file_sha256": sha256_file(vocab_path),
            }
        ),
        encoding="utf-8",
    )
    assert artifacts_ready(tmp_path) is True
    assert (
        artifacts_ready(
            tmp_path,
            expected_dates=["2026-01-06"],
            train_end="2026-01-05",
            val_end="2026-01-05",
        )
        is False
    )

    receipt = tmp_path / "data" / "coverage" / "2026-01-05.json"
    receipt.write_text(
        receipt.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    assert artifacts_ready(tmp_path) is False

    token_path.unlink()
    assert artifacts_ready(tmp_path) is False


def test_formal_events_only_fails_before_marking_clean_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fm.lob_rebuild.clean_day_fast as clean_fast_module
    import quant_fm.scripts.run_medium as run_medium_module

    monkeypatch.setattr(run_medium_module, "load_read_config", lambda: object())
    monkeypatch.setattr(run_medium_module, "read_bucket", lambda: "fixture")
    monkeypatch.setattr(
        clean_fast_module,
        "clean_day_fast",
        lambda **_kwargs: {
            "written": 0,
            "empty": 0,
            "errors": 1,
            "skipped": 0,
            "failed_symbols": ["000001"],
        },
    )

    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    with pytest.raises(RuntimeError, match="formal V2 preparation failed"):
        run_medium_module.run(
            dates=dates,
            symbols_sz=("000001",),
            symbols_sh=(),
            workdir=tmp_path,
            train_end=dates[0],
            val_end=dates[1],
            n_bins=8,
            skip_clean=False,
            drop_clean=True,
            drop_events=False,
            fit_sample_days=None,
            resume=True,
            estimate_only=False,
            fast_clean=True,
            events_only=True,
        )

    assert not (tmp_path / "data" / ".clean_done" / dates[0]).exists()
    assert not (tmp_path / "data" / ".done" / dates[0]).exists()


def test_formal_parallel_input_and_cpu_budget_fail_closed() -> None:
    with pytest.raises(ValueError, match="strictly chronological"):
        validate_dates(["2026-01-06", "2026-01-05", "2026-01-06"])
    with pytest.raises(ValueError, match="only 64 CPUs"):
        validate_worker_budget(3, 30, cpu_count=64)
    with pytest.raises(ValueError, match="repeats bare symbols across markets"):
        expected_symbol_keys(("000001",), ("000001",))


def test_parallel_regime_mode_requires_global_finalize_inputs(tmp_path: Path) -> None:
    from quant_fm.scripts.run_v2_parallel_data import run

    dates = tmp_path / "dates.txt"
    symbols_sz = tmp_path / "sz.txt"
    symbols_sh = tmp_path / "sh.txt"
    dates.write_text("2026-01-05\n2026-01-06\n2026-01-07\n", encoding="utf-8")
    symbols_sz.write_text("000001\n", encoding="utf-8")
    symbols_sh.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Regime finalize requires inputs"):
        run(
            dates_file=dates,
            workdir=tmp_path / "work",
            symbols_sz_file=symbols_sz,
            symbols_sh_file=symbols_sh,
            groups=1,
            clean_workers=1,
            canon_workers=1,
            tokenize_workers=1,
            train_end=None,
            val_end=None,
            n_bins=8,
            max_samples_per_field=100,
            seed=0,
            drop_events=True,
            build_regime_data=True,
        )
