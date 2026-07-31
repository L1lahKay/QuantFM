import os

from pylob.pipeline.config import MinioConfig
from pylob.pipeline.s3_io import build_storage_options

from quant_fm.scripts.minio_config import (
    configure_minio_proxy_bypass,
    load_read_config,
)


def test_configure_minio_proxy_bypass_preserves_proxy_and_existing_entries(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("NO_PROXY", "localhost,192.168.2.11")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("MINIO_BYPASS_PROXY", raising=False)

    hosts = configure_minio_proxy_bypass(
        "http://192.168.2.11:9000/", "minio.internal:9100"
    )
    configure_minio_proxy_bypass("192.168.2.11:9000")

    assert hosts == ("192.168.2.11", "minio.internal")
    assert os.environ["HTTP_PROXY"] == "http://proxy.invalid:3128"
    assert os.environ["HTTPS_PROXY"] == "http://proxy.invalid:3128"
    assert os.environ["NO_PROXY"].split(",") == [
        "localhost",
        "192.168.2.11",
        "minio.internal",
    ]
    assert os.environ["no_proxy"].split(",") == [
        "localhost",
        "192.168.2.11",
        "minio.internal",
    ]


def test_load_read_config_adds_configured_endpoint_to_proxy_bypass(monkeypatch) -> None:
    monkeypatch.setenv("MINIO_READ_ACCESS_KEY", "read-key")
    monkeypatch.setenv("MINIO_READ_SECRET_KEY", "read-secret")
    monkeypatch.setenv("MINIO_READ_ENDPOINT", "10.20.30.40:9000")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("MINIO_BYPASS_PROXY", raising=False)

    config = load_read_config()

    assert config.endpoint == "10.20.30.40:9000"
    assert os.environ["NO_PROXY"] == "10.20.30.40"
    assert os.environ["no_proxy"] == "10.20.30.40"


def test_configure_minio_proxy_bypass_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("MINIO_BYPASS_PROXY", "false")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    assert configure_minio_proxy_bypass("192.168.2.11:9000") == ()
    assert "NO_PROXY" not in os.environ
    assert "no_proxy" not in os.environ


def test_configure_minio_proxy_bypass_auto_ignores_public_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("MINIO_BYPASS_PROXY", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    assert configure_minio_proxy_bypass("s3.example.com:443") == ()
    assert "NO_PROXY" not in os.environ
    assert "no_proxy" not in os.environ


def test_storage_options_set_object_store_retry_budget() -> None:
    options = build_storage_options(
        MinioConfig(
            endpoint="192.168.2.11:9000",
            access_key="read-key",
            secret_key="read-secret",
        )
    )

    assert options["aws_endpoint_url"] == "http://192.168.2.11:9000"
    assert options["aws_allow_http"] == "true"
    assert options["max_retries"] == 2
