"""
MinIO 读写分离配置（开箱即用）。

默认策略：

- **读** 原始 L2：``192.168.2.11:9000`` → bucket ``zeus-cn-quote``
- **写** 训练产物：``192.168.2.11:9100`` → bucket ``model-cache``

读写可使用**不同密钥**（``MINIO_*`` vs ``MINIO_WRITE_*``）。
凭据优先级：环境变量 → ``~/.mc/config.json`` 别名 ``myminio``。
"""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

from pylob.pipeline.config import MinioConfig

DEFAULT_READ_ENDPOINT = "192.168.2.11:9000"
DEFAULT_WRITE_ENDPOINT = "192.168.2.11:9100"
DEFAULT_READ_BUCKET = "zeus-cn-quote"
DEFAULT_WRITE_BUCKET = "model-cache"
DEFAULT_OUTPUT_PREFIX = f"fm-pretrain/{getpass.getuser()}"
DEFAULT_AWS_REGION = "us-east-1"


def _credentials_from_env(
    *,
    access_vars: tuple[str, ...] = ("MINIO_ACCESS_KEY",),
    secret_vars: tuple[str, ...] = ("MINIO_SECRET_KEY",),
) -> tuple[str, str] | None:
    key = next((os.getenv(v) for v in access_vars if os.getenv(v)), None)
    secret = next((os.getenv(v) for v in secret_vars if os.getenv(v)), None)
    if key and secret:
        return key, secret
    return None


def _credentials_from_mc() -> tuple[str, str]:
    cfg_path = Path.home() / ".mc/config.json"
    if not cfg_path.is_file():
        msg = (
            "MinIO 凭据未配置：请设置 MINIO_ACCESS_KEY/MINIO_SECRET_KEY，"
            "或配置 mc alias 'myminio'（见 quant_fm/scripts/minio_env.example.sh）"
        )
        raise RuntimeError(msg)
    alias = json.loads(cfg_path.read_text())["aliases"]["myminio"]
    return alias["accessKey"], alias["secretKey"]


def _resolve_read_credentials() -> tuple[str, str]:
    creds = _credentials_from_env(
        access_vars=("MINIO_READ_ACCESS_KEY", "MINIO_ACCESS_KEY"),
        secret_vars=("MINIO_READ_SECRET_KEY", "MINIO_SECRET_KEY"),
    )
    if creds is not None:
        return creds
    return _credentials_from_mc()


def _resolve_write_credentials() -> tuple[str, str]:
    # 写端优先用专用密钥（9100/model-cache 与 9000 常不同）
    creds = _credentials_from_env(
        access_vars=("MINIO_WRITE_ACCESS_KEY", "MINIO_ACCESS_KEY"),
        secret_vars=("MINIO_WRITE_SECRET_KEY", "MINIO_SECRET_KEY"),
    )
    if creds is not None:
        return creds
    return _credentials_from_mc()


def _endpoint(name: str, default: str) -> str:
    """Resolve endpoint env with legacy ``MINIO_ENDPOINT`` fallback for read."""
    if val := os.getenv(name):
        return val
    if name == "MINIO_READ_ENDPOINT" and (legacy := os.getenv("MINIO_ENDPOINT")):
        return legacy
    return default


def _secure_flag() -> bool:
    return os.getenv("MINIO_SECURE", "false").lower() == "true"


def load_read_config() -> MinioConfig:
    """Config for reading raw L2 from ``zeus-cn-quote`` on port **9000**."""
    key, secret = _resolve_read_credentials()
    return MinioConfig(
        endpoint=_endpoint("MINIO_READ_ENDPOINT", DEFAULT_READ_ENDPOINT),
        access_key=key,
        secret_key=secret,
        secure=_secure_flag(),
    )


def load_write_config() -> MinioConfig:
    """Config for writing artifacts to ``model-cache`` on port **9100**."""
    key, secret = _resolve_write_credentials()
    return MinioConfig(
        endpoint=_endpoint("MINIO_WRITE_ENDPOINT", DEFAULT_WRITE_ENDPOINT),
        access_key=key,
        secret_key=secret,
        secure=_secure_flag(),
    )


def load_minio_config() -> MinioConfig:
    """Backward-compatible alias: read config."""
    return load_read_config()


def read_bucket() -> str:
    """Return the raw L2 source bucket."""
    return os.environ.get("MINIO_BUCKET", DEFAULT_READ_BUCKET)


def output_bucket() -> str:
    """Return the model artifact destination bucket."""
    return os.environ.get("MINIO_OUTPUT_BUCKET", DEFAULT_WRITE_BUCKET)


def output_prefix(*parts: str) -> str:
    """Build an artifact prefix beneath the configured output root."""
    base = os.environ.get("MINIO_OUTPUT_PREFIX", DEFAULT_OUTPUT_PREFIX).strip("/")
    extra = "/".join(p.strip("/") for p in parts if p)
    return f"{base}/{extra}" if extra else base


def storage_options_for_read() -> dict[str, str]:
    """Polars S3 options for reading raw L2."""
    from pylob.pipeline.s3_io import build_storage_options

    return build_storage_options(load_read_config())


def storage_options_for_write() -> dict[str, str]:
    """Polars S3 options for writing to model-cache."""
    from pylob.pipeline.s3_io import build_storage_options

    return build_storage_options(load_write_config())


def describe_config() -> str:
    """Human-readable summary for logging / diagnostics."""
    read_cfg = load_read_config()
    write_cfg = load_write_config()
    return (
        f"read  s3://{read_bucket()} @ {read_cfg.endpoint}\n"
        f"write s3://{output_bucket()} @ {write_cfg.endpoint}/{output_prefix()}/"
    )
