"""Print MinIO read/write config and probe connectivity."""

from __future__ import annotations

import argparse
import urllib.request

from quant_fm.scripts.minio_config import (
    DEFAULT_READ_ENDPOINT,
    DEFAULT_WRITE_ENDPOINT,
    describe_config,
    load_read_config,
    load_write_config,
    output_bucket,
    read_bucket,
)


def _health(endpoint: str) -> str:
    host = endpoint if endpoint.startswith("http") else f"http://{endpoint}"
    url = f"{host.rstrip('/')}/minio/health/live"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return f"OK ({resp.status})"
    except Exception as exc:
        return f"FAIL ({exc})"


def main() -> None:
    """Print effective MinIO endpoints and probe service health."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    read_cfg = load_read_config()
    write_cfg = load_write_config()

    print("== MinIO 配置（代码默认，读写分离）==")
    print(describe_config())
    print()
    print(f"read  health @ {read_cfg.endpoint}:  {_health(read_cfg.endpoint)}")
    print(f"write health @ {write_cfg.endpoint}: {_health(write_cfg.endpoint)}")
    print()
    print("defaults:")
    print(f"  READ  {DEFAULT_READ_ENDPOINT}  bucket={read_bucket()}")
    print(f"  WRITE {DEFAULT_WRITE_ENDPOINT}  bucket={output_bucket()}")


if __name__ == "__main__":
    main()
