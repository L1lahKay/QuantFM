from __future__ import annotations

import math

import polars as pl
import pytest

from quant_fm.signal.schema import validate_scores


def test_validate_scores_normalises_and_sorts() -> None:
    scores = pl.DataFrame(
        {
            "date": ["2026-01-06", "2026-01-05"],
            "symbol": [1, 600000],
            "score": [0, 1],
        }
    )
    result = validate_scores(scores)
    assert result.schema == {
        "date": pl.String,
        "symbol": pl.String,
        "score": pl.Float64,
    }
    assert result["symbol"].to_list() == ["600000", "000001"]


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (
            pl.DataFrame(
                {"date": ["2026-02-30"], "symbol": ["000001"], "score": [0.1]}
            ),
            "valid YYYY-MM-DD",
        ),
        (
            pl.DataFrame({"date": ["2026-01-05"], "symbol": ["1.SZ"], "score": [0.1]}),
            "six-digit",
        ),
        (
            pl.DataFrame(
                {
                    "date": ["2026-01-05", "2026-01-05"],
                    "symbol": ["000001", "000001"],
                    "score": [0.1, 0.2],
                }
            ),
            "duplicate",
        ),
        (
            pl.DataFrame(
                {
                    "date": ["2026-01-05"],
                    "symbol": ["000001"],
                    "score": [math.inf],
                }
            ),
            "finite",
        ),
    ],
)
def test_validate_scores_rejects_invalid_rows(frame: pl.DataFrame, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_scores(frame)
