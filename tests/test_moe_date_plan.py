from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_fm.scripts.build_moe_date_plan import build_training_plan


def _weekdays(count: int, *, start: date = date(2024, 1, 2)) -> list[str]:
    values: list[str] = []
    cursor = start
    while len(values) < count:
        if cursor.weekday() < 5:
            values.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return values


def test_training_plan_keeps_benchmark_and_adaptation_disjoint_contracts() -> None:
    dates = _weekdays(613)
    cutoff = dates[561]
    plan, files = build_training_plan(dates, adaptation_cutoff=cutoff)

    benchmark = plan["strict_benchmark"]
    assert benchmark["train"]["count"] == 300
    assert benchmark["validation"]["count"] == 60
    assert benchmark["test"]["count"] == 100
    assert benchmark["locked_oos"]["count"] == 153
    assert files["benchmark_train_dates"][-1] < files["benchmark_validation_dates"][0]
    assert files["benchmark_validation_dates"][-1] < files["benchmark_test_dates"][0]
    assert files["benchmark_test_dates"][-1] < files["benchmark_locked_oos_dates"][0]

    adaptation = plan["adaptation_2026q1"]
    assert adaptation["development_train"]["count"] == 258
    assert adaptation["purge"]["count"] == 2
    assert adaptation["validation"]["count"] == 40
    assert adaptation["final_refit_train"]["count"] == 300
    assert adaptation["shadow_oos"]["count"] == 51
    assert adaptation["eligible_for_full_oos_claim"] is False
    assert files["adaptation_final_train_dates"][-1] == cutoff
    assert len(files["adaptation_development_pipeline_dates"]) == 349
    assert len(files["adaptation_refit_pipeline_dates"]) == 351
    assert not (
        set(files["adaptation_purge_dates"])
        & set(files["adaptation_development_pipeline_dates"])
    )


def test_training_plan_rejects_too_few_dates() -> None:
    with pytest.raises(ValueError, match="benchmark needs 460 dates"):
        build_training_plan(_weekdays(459))


def test_training_plan_requires_252_development_days() -> None:
    dates = _weekdays(613)
    with pytest.raises(ValueError, match="at least 252 dates"):
        build_training_plan(
            dates,
            adaptation_cutoff=dates[561],
            adaptation_validation_days=60,
            adaptation_purge_days=2,
        )
