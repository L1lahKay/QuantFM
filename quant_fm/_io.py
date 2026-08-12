"""Small, dependency-free file-writing helpers shared across entrypoints."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def atomic_write_text(path: Path, value: str) -> None:
    """Write text through a sibling temporary file and atomically replace ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: object) -> None:
    """Write stable, human-readable JSON with the repository's standard format."""
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
