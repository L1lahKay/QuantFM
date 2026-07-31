from types import SimpleNamespace

import polars as pl
import pytest

from quant_fm.lob_rebuild import clean_day_fast


class _FailingLazyFrame:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def collect_schema(self):
        raise self.error

    def collect(self):
        raise self.error


def test_read_one_projected_retries_502_then_succeeds(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_scan(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            return _FailingLazyFrame(OSError("HEAD failed: 502 Bad Gateway"))
        return pl.DataFrame({"symbol": ["000001"], "price": [10]}).lazy()

    monkeypatch.setattr(clean_day_fast.pl, "scan_parquet", fake_scan)
    monkeypatch.setattr(clean_day_fast.time, "sleep", sleeps.append)
    reader = SimpleNamespace(storage_options={})

    result = clean_day_fast._read_one_projected(
        reader,
        "bucket",
        ("one.parquet",),
        ("000001",),
        project=True,
        max_attempts=4,
        base_delay_seconds=0.5,
    )

    assert calls == 3
    assert sleeps == [0.5, 1.0]
    assert result.height == 1


@pytest.mark.parametrize("message", ["403 AccessDenied", "404 NoSuchKey"])
def test_read_one_projected_does_not_retry_permanent_errors(
    monkeypatch, message
) -> None:
    calls = 0

    def fake_scan(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _FailingLazyFrame(OSError(message))

    monkeypatch.setattr(clean_day_fast.pl, "scan_parquet", fake_scan)
    monkeypatch.setattr(
        clean_day_fast.time,
        "sleep",
        lambda _delay: pytest.fail("permanent errors must not sleep"),
    )
    reader = SimpleNamespace(storage_options={})

    with pytest.raises(OSError, match=message.split()[0]):
        clean_day_fast._read_one_projected(
            reader,
            "bucket",
            ("one.parquet",),
            (),
            project=True,
            max_attempts=4,
            base_delay_seconds=0,
        )

    assert calls == 1


def test_read_one_projected_raises_after_retry_budget(monkeypatch) -> None:
    calls = 0

    def fake_scan(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _FailingLazyFrame(OSError("504 Gateway Timeout"))

    monkeypatch.setattr(clean_day_fast.pl, "scan_parquet", fake_scan)
    monkeypatch.setattr(clean_day_fast.time, "sleep", lambda _delay: None)
    reader = SimpleNamespace(storage_options={})

    with pytest.raises(OSError, match="504"):
        clean_day_fast._read_one_projected(
            reader,
            "bucket",
            ("one.parquet",),
            (),
            project=True,
            max_attempts=3,
            base_delay_seconds=0,
        )

    assert calls == 3
