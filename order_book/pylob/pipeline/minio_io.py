"""MinIO object listing and dataframe loading helpers."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pylob.pipeline.config import MinioConfig

logger = logging.getLogger(__name__)


class MinioDataSource:
    """Read parquet or CSV objects from a MinIO bucket into Polars."""

    def __init__(self, config: MinioConfig):
        try:
            from minio import Minio
        except ImportError as exc:
            msg = "MinIO support requires the optional 'minio' package."
            raise ImportError(msg) from exc

        self.client = Minio(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
            region=config.region,
        )

    def list_objects(
        self,
        bucket: str,
        prefix: str,
        suffixes: tuple[str, ...] = (".parquet", ".csv"),
    ) -> list[str]:
        """Return sorted object names under ``prefix`` matching ``suffixes``."""
        objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
        names = [
            obj.object_name
            for obj in objects
            if obj.object_name is not None and obj.object_name.endswith(suffixes)
        ]
        logger.info("Listed %d objects under %s/%s", len(names), bucket, prefix)
        return sorted(names)

    def read_objects(self, bucket: str, object_names: list[str]) -> pl.DataFrame:
        """Download objects to temp files and concatenate them diagonally."""
        frames = []
        for index, name in enumerate(object_names, start=1):
            logger.info(
                "Downloading %s/%s (%d/%d)", bucket, name, index, len(object_names)
            )
            frames.append(self._read_one(bucket, name))
        if not frames:
            msg = f"no readable objects found in bucket={bucket}"
            raise FileNotFoundError(msg)
        return pl.concat(frames, how="diagonal_relaxed")

    def _read_one(self, bucket: str, object_name: str) -> pl.DataFrame:
        suffix = Path(object_name).suffix.lower()
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / Path(object_name).name
            self.client.fget_object(bucket, object_name, str(local_path))
            if suffix == ".parquet":
                return pl.read_parquet(local_path)
            if suffix == ".csv":
                return pl.read_csv(local_path, infer_schema_length=10000)
        msg = f"unsupported object suffix for {object_name}"
        raise ValueError(msg)
