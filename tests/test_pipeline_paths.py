"""Tests for pipeline object-key helpers."""

from __future__ import annotations

import pytest
from pylob.pipeline.paths import (
    archive_object_key,
    hds_order_prefix,
    hds_trade_prefix,
    normalize_endpoint,
    symbol_object_keys,
)


def test_normalize_endpoint_strips_scheme() -> None:
    assert normalize_endpoint("http://192.168.2.11:9000") == "192.168.2.11:9000"
    assert (
        normalize_endpoint("https://minio.example.com:9000/")
        == "minio.example.com:9000"
    )


def test_hds_prefixes() -> None:
    trade = hds_trade_prefix("2026-02-02", mic="XSHE")
    order = hds_order_prefix("2026-02-02", mic="XSHE")
    assert trade.endswith("/TYPE=transaction/")
    assert order.endswith("/TYPE=order/")
    assert "DATE=2026-02-02" in trade


def test_archive_object_key() -> None:
    assert (
        archive_object_key("2026-02-02") == "2026.02/2026.02.02/default/1/all.parquet"
    )


def test_symbol_object_keys() -> None:
    keys = symbol_object_keys("prefix/", "SZ", ("000001", "000002"))
    assert keys == (
        "prefix/SZ000001.parquet",
        "prefix/SZ000002.parquet",
    )


def test_archive_layout_requires_date_or_combined_key() -> None:
    from pylob.pipeline.config import PipelineConfig

    config = PipelineConfig(
        bucket="zeus-cn-quote",
        trade_prefix="raw/trade/",
        order_prefix="raw/order/",
        output_dir="data/clean",
        symbols=("000001",),
        market="SZ",
        layout="archive",
    )
    with pytest.raises(ValueError, match="PYLOB_DATE"):
        config.resolved_trade_keys()
