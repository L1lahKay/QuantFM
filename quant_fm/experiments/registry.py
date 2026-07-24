"""小型、版本化的消融实验 registry。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """一个只改变单项因素、可追溯到代码与数据的实验。"""

    experiment_id: str
    base_experiment_id: str | None
    changed_factor: str
    config_path: str
    git_commit: str
    seed: int
    effective_tokens: int
    validation_plan_sha256: str
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = (
            self.experiment_id,
            self.changed_factor,
            self.config_path,
            self.git_commit,
            self.validation_plan_sha256,
        )
        if not all(required):
            msg = "experiment identity and reproducibility fields are required"
            raise ValueError(msg)
        if self.effective_tokens < 0:
            msg = "effective_tokens must be non-negative"
            raise ValueError(msg)


class ExperimentRegistry:
    """JSON artifact：拒绝重复 id，并以稳定顺序写盘。"""

    VERSION = "1.0"

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> tuple[ExperimentRecord, ...]:
        """加载所有已登记实验并验证版本。"""
        if not self.path.exists():
            return ()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("registry_version") != self.VERSION:
            msg = "unsupported experiment registry version"
            raise ValueError(msg)
        return tuple(ExperimentRecord(**item) for item in payload["experiments"])

    def register(self, record: ExperimentRecord) -> None:
        """原子追加一个实验，拒绝重复 id 和悬空 lineage。"""
        records = list(self.load())
        if any(item.experiment_id == record.experiment_id for item in records):
            msg = f"duplicate experiment id: {record.experiment_id}"
            raise ValueError(msg)
        if record.base_experiment_id is not None and not any(
            item.experiment_id == record.base_experiment_id for item in records
        ):
            msg = "base experiment must already exist in the registry"
            raise ValueError(msg)
        records.append(record)
        payload = {
            "registry_version": self.VERSION,
            "experiments": [asdict(item) for item in records],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
