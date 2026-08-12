from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _target_block(name: str) -> str:
    lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{name}:")
    selected: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(("\t", " ")) and line.endswith(":"):
            break
        selected.append(line)
    return "\n".join(selected)


def test_signal_make_entrypoint_supplies_strict_identity_inputs() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    block = _target_block("signal")

    assert "SIGNAL_UNIVERSE ?=" in makefile
    assert "SIGNAL_REGIME_FEATURES ?=" in makefile
    assert "SIGNAL_UNIVERSE must point to the daily PIT scoring universe" in block
    assert '--fm-checkpoint "$(SIGNAL_FM_CHECKPOINT)"' in block
    assert '--vocab "$(SIGNAL_VOCAB)"' in block
    assert '--universe "$(SIGNAL_UNIVERSE)"' in block
