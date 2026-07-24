import pytest

from quant_fm.experiments.registry import ExperimentRecord, ExperimentRegistry


def _record(identifier: str, base: str | None = None) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=identifier,
        base_experiment_id=base,
        changed_factor="ffn_hidden",
        config_path="config.yaml",
        git_commit="abc123",
        seed=42,
        effective_tokens=1000,
        validation_plan_sha256="deadbeef",
        metrics={"val_loss": 1.0},
    )


def test_registry_round_trip_and_lineage(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments.json")
    registry.register(_record("D0"))
    registry.register(_record("D1", "D0"))
    assert [item.experiment_id for item in registry.load()] == ["D0", "D1"]
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(_record("D1", "D0"))
    with pytest.raises(ValueError, match="base experiment"):
        registry.register(_record("D2", "missing"))
