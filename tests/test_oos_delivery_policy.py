from __future__ import annotations

import sys
from datetime import date as calendar_date
from pathlib import Path

import polars as pl
import pytest

from quant_fm.downstream.build_panel_from_minio import build_execution_panel
from quant_fm.downstream.return_spec import get_return_spec
from quant_fm.downstream.train_ranker import RankerObjectiveConfig
from quant_fm.scripts import build_oos_delivery as delivery


def _api_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "train_emb_dir": tmp_path / "train_embeddings",
        "train_panel": tmp_path / "train_panel.parquet",
        "test_emb": tmp_path / "test_embeddings.parquet",
        "out_dir": tmp_path / "delivery",
    }


def _lineage_report(
    *,
    training_end: str = "2024-12-31",
    acceptance_sha256: str = "a" * 64,
    checkpoint_sha256: str = "b" * 64,
) -> dict[str, object]:
    return {
        "format_version": "strict_pretrain_lineage_v1",
        "status": "verified",
        "effective_training_end": training_end,
        "acceptance": {
            "path": "/tmp/acceptance.json",
            "sha256": acceptance_sha256,
        },
        "fm_checkpoint": {
            "path": "/tmp/fm.pt",
            "sha256": checkpoint_sha256,
        },
    }


def _alternate_execution_panel(tmp_path: Path) -> tuple[Path, Path]:
    calendar = ["2025-01-02", "2025-01-03", "2025-01-06"]
    daily = pl.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "market": "SZ",
                "open": 10.0 + day_index,
                "close": 10.5 + day_index,
                "vwap": 10.25 + day_index,
                "is_st": False,
                "is_new": False,
                "is_halt": False,
                "limit_locked": False,
            }
            for day_index, date in enumerate(calendar)
            for symbol in ("000001", "000002", "000003")
        ]
    )
    panel = build_execution_panel(
        daily,
        signal_dates=[calendar[0]],
        trading_calendar=calendar,
        spec=get_return_spec("open_t1_close_t1"),
    )
    panel_path = tmp_path / "alternate_panel.parquet"
    calendar_path = tmp_path / "calendar.txt"
    panel.write_parquet(panel_path)
    calendar_path.write_text("\n".join(calendar) + "\n", encoding="utf-8")
    return panel_path, calendar_path


def test_direct_api_rejects_custom_objective_even_with_legacy_inputs(
    tmp_path: Path,
) -> None:
    changed = RankerObjectiveConfig(global_ic_weight=0.31)
    with pytest.raises(ValueError, match="frozen RankerObjectiveConfig"):
        delivery.build_oos_delivery(
            **_api_kwargs(tmp_path),
            objective=changed,
            allow_legacy_training_panel=True,
        )
    assert not (tmp_path / "delivery").exists()


def test_direct_api_accepts_custom_objective_only_with_named_research_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_locked(**kwargs: object) -> Path:
        captured.update(kwargs)
        return Path(kwargs["out_dir"])

    monkeypatch.setattr(delivery, "_build_oos_delivery_locked", fake_locked)
    changed = RankerObjectiveConfig(global_ic_weight=0.31)
    result = delivery.build_oos_delivery(
        **_api_kwargs(tmp_path),
        objective=changed,
        allow_legacy_training_panel=True,
        allow_research_objective_return_spec_override=True,
        fm_training_end_date="2025-12-31",
    )
    assert result == tmp_path / "delivery"
    assert captured["objective"] == changed
    assert captured["allow_research_objective_return_spec_override"] is True
    assert not (result / ".score.lock").exists()


@pytest.mark.parametrize("value", ["0", "2025-1-01", "2025-02-30", "20250101"])
def test_direct_api_rejects_noncanonical_fm_training_end_date(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=r"fm_training_end_date.*YYYY-MM-DD"):
        delivery.build_oos_delivery(
            **_api_kwargs(tmp_path),
            allow_legacy_training_panel=True,
            fm_training_end_date=value,
        )
    assert not (tmp_path / "delivery").exists()


def test_strict_api_rejects_alternate_return_spec_without_research_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel_path, calendar_path = _alternate_execution_panel(tmp_path)
    kwargs = {
        **_api_kwargs(tmp_path),
        "train_panel": panel_path,
        "train_calendar": calendar_path,
        "train_universe": tmp_path / "train_universe.parquet",
        "test_universe": tmp_path / "test_universe.parquet",
        "pretrain_acceptance_path": tmp_path / "acceptance.json",
    }
    with pytest.raises(ValueError, match=r"execution_contract\.return_spec"):
        delivery.build_oos_delivery(**kwargs)

    lineage_call: dict[str, object] = {}

    def fake_lineage(**lineage_kwargs: object) -> dict[str, object]:
        lineage_call.update(lineage_kwargs)
        return _lineage_report()

    def reached_after_policy(_path: Path) -> pl.DataFrame:
        msg = "reached data loading after research policy"
        raise RuntimeError(msg)

    monkeypatch.setattr(delivery, "validate_pretrain_lineage", fake_lineage)
    monkeypatch.setattr(delivery, "_load_emb", reached_after_policy)
    with pytest.raises(
        RuntimeError, match="reached data loading after research policy"
    ):
        delivery.build_oos_delivery(
            **kwargs,
            allow_research_objective_return_spec_override=True,
        )
    assert lineage_call == {
        "acceptance_path": tmp_path / "acceptance.json",
        "train_embeddings": tmp_path / "train_embeddings" / "all.parquet",
        "oos_embeddings": tmp_path / "test_embeddings.parquet",
        "expected_training_end": None,
    }


def test_nonlegacy_research_mode_requires_pretrain_acceptance(tmp_path: Path) -> None:
    panel_path, calendar_path = _alternate_execution_panel(tmp_path)
    with pytest.raises(ValueError, match="requires --pretrain-acceptance"):
        delivery.build_oos_delivery(
            **{
                **_api_kwargs(tmp_path),
                "train_panel": panel_path,
                "train_calendar": calendar_path,
                "train_universe": tmp_path / "train_universe.parquet",
                "test_universe": tmp_path / "test_universe.parquet",
            },
            allow_research_objective_return_spec_override=True,
        )


def test_strict_lineage_cutoff_is_derived_and_manual_date_is_only_assertion() -> None:
    report = _lineage_report(training_end="2025-12-30")
    provenance = delivery._foundation_model_provenance(
        lineage_report=report,
        parsed_training_end=None,
        parsed_oos_start=calendar_date(2026, 1, 5),
    )

    assert provenance["lineage_mode"] == "verified_pretrain_lineage"
    assert provenance["training_end_date"] == "2025-12-30"
    assert provenance["pretrain_acceptance_sha256"] == "a" * 64
    assert provenance["fm_checkpoint_sha256"] == "b" * 64
    assert provenance["lineage_report"] == report
    assert len(provenance["lineage_report_sha256"]) == 64

    with pytest.raises(ValueError, match="does not match verified lineage"):
        delivery._foundation_model_provenance(
            lineage_report=report,
            parsed_training_end=calendar_date(2025, 12, 31),
            parsed_oos_start=calendar_date(2026, 1, 5),
        )


def test_lineage_hashes_rekey_ranker_identity() -> None:
    first = delivery._foundation_model_provenance(
        lineage_report=_lineage_report(acceptance_sha256="a" * 64),
        parsed_training_end=None,
        parsed_oos_start=calendar_date(2026, 1, 5),
    )
    second = delivery._foundation_model_provenance(
        lineage_report=_lineage_report(acceptance_sha256="c" * 64),
        parsed_training_end=None,
        parsed_oos_start=calendar_date(2026, 1, 5),
    )

    assert delivery._stable_key({"foundation_model": first}) != delivery._stable_key(
        {"foundation_model": second}
    )


def test_cli_wires_explicit_research_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build(**kwargs: object) -> Path:
        captured.update(kwargs)
        return Path(kwargs["out_dir"])

    monkeypatch.setattr(delivery, "build_oos_delivery", fake_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_oos_delivery",
            "--train-emb-dir",
            str(tmp_path / "train"),
            "--train-panel",
            str(tmp_path / "panel.parquet"),
            "--test-emb",
            str(tmp_path / "test.parquet"),
            "--out-dir",
            str(tmp_path / "out"),
            "--global-ic-weight",
            "0.31",
            "--allow-legacy-training-panel",
            "--allow-research-objective-return-spec-override",
            "--pretrain-acceptance",
            str(tmp_path / "acceptance.json"),
            "--fm-training-end-date",
            "2025-12-31",
        ],
    )
    delivery.main()
    assert captured["allow_legacy_training_panel"] is True
    assert captured["allow_research_objective_return_spec_override"] is True
    assert captured["pretrain_acceptance_path"] == tmp_path / "acceptance.json"
    assert captured["objective"] == RankerObjectiveConfig(global_ic_weight=0.31)


def test_cli_custom_objective_fails_closed_without_research_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_oos_delivery",
            "--train-emb-dir",
            str(tmp_path / "train"),
            "--train-panel",
            str(tmp_path / "panel.parquet"),
            "--test-emb",
            str(tmp_path / "test.parquet"),
            "--out-dir",
            str(tmp_path / "out"),
            "--global-ic-weight",
            "0.31",
        ],
    )
    with pytest.raises(ValueError, match="frozen RankerObjectiveConfig"):
        delivery.main()


def test_cli_rejects_invalid_fm_training_end_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_oos_delivery",
            "--train-emb-dir",
            str(tmp_path / "train"),
            "--train-panel",
            str(tmp_path / "panel.parquet"),
            "--test-emb",
            str(tmp_path / "test.parquet"),
            "--out-dir",
            str(tmp_path / "out"),
            "--allow-legacy-training-panel",
            "--fm-training-end-date",
            "0",
        ],
    )
    with pytest.raises(ValueError, match=r"fm_training_end_date.*YYYY-MM-DD"):
        delivery.main()


def test_cli_help_marks_legacy_and_research_modes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["build_oos_delivery", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        delivery.main()
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--pretrain-acceptance" in help_text
    assert "--allow-research-objective-return-spec-override" in help_text
    assert "legacy_diagnostic" in help_text
    assert "不得生产交付" in help_text
