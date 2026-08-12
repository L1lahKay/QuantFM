"""
构建带内容哈希与时间切分的分片数据集清单。

【新手】manifest.json = 训练器的「图书目录」：
  - 列出每个 tokens parquet 的路径、行数、sha256
  - 标注属于 train / val / test（按日期切，防止用未来数据训练）
  - train.py 通过 Manifest.load() 读取，不直接扫文件夹
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date as calendar_date
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
from pylob.event_ordering import DEFAULT_EVENT_ORDERING_VERSION

from quant_fm.tokenizer.artifact_contract import (
    assert_token_contract_matches,
    stable_vocab_sha256,
    token_contract_path,
)
from quant_fm.tokenizer.transforms import DEFAULT_FEATURE_TRANSFORM_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_VALID_SPLITS = frozenset({"train", "val", "test"})


def _canonical_date(value: str, *, field_name: str) -> str:
    """Return a canonical ISO date or reject ambiguous lexical comparisons."""
    try:
        parsed = calendar_date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        msg = f"{field_name} must be a canonical YYYY-MM-DD date, got {value!r}"
        raise ValueError(msg) from exc
    if parsed.isoformat() != value:
        msg = f"{field_name} must be a canonical YYYY-MM-DD date, got {value!r}"
        raise ValueError(msg)
    return value


@dataclass(slots=True)
class ShardEntry:
    """数据集中的一条 token 分片文件。"""

    market: str
    symbol: str
    date: str
    path: str
    rows: int
    sha256: str
    split: str = "train"
    data_contract_sha256: str | None = None


@dataclass(slots=True)
class Manifest:
    """分片条目集合，含切分边界与 CPCV 参数。"""

    shards: list[ShardEntry] = field(default_factory=list)
    train_end: str | None = None
    val_end: str | None = None
    purge_days: int = 5
    embargo_days: int = 2
    vocab_path: str | None = None
    vocab_sha256: str | None = None
    schema_version: str = "cn_l2_v1"
    event_ordering_version: str | None = DEFAULT_EVENT_ORDERING_VERSION
    feature_transform_version: str | None = DEFAULT_FEATURE_TRANSFORM_VERSION

    def split(self, name: str) -> list[ShardEntry]:
        """返回属于 ``name``（train/val/test）的所有分片。"""
        return [s for s in self.shards if s.split == name]

    def dates(self, name: str) -> list[str]:
        """返回某切分的排序后唯一日期列表。"""
        return sorted({s.date for s in self.split(name)})

    def validate_split_contract(self, *, context: str = "manifest") -> None:
        """Validate split labels and bind them to declared date boundaries."""
        if (self.train_end is None) != (self.val_end is None):
            msg = f"{context} must declare train_end and val_end together"
            raise ValueError(msg)

        train_end = val_end = None
        if self.train_end is not None and self.val_end is not None:
            train_end = _canonical_date(self.train_end, field_name="train_end")
            val_end = _canonical_date(self.val_end, field_name="val_end")
            if train_end > val_end:
                msg = (
                    f"{context} requires train_end <= val_end, got "
                    f"{train_end} > {val_end}"
                )
                raise ValueError(msg)

        seen_paths: set[str] = set()
        for index, shard in enumerate(self.shards):
            prefix = f"{context} shard[{index}]"
            _canonical_date(shard.date, field_name=f"{prefix}.date")
            if shard.split not in _VALID_SPLITS:
                msg = (
                    f"{prefix}.split must be one of {sorted(_VALID_SPLITS)}, "
                    f"got {shard.split!r}"
                )
                raise ValueError(msg)
            if shard.path in seen_paths:
                msg = f"{context} contains duplicate shard path: {shard.path}"
                raise ValueError(msg)
            seen_paths.add(shard.path)
            if train_end is not None and val_end is not None:
                expected = _assign_split(shard.date, train_end, val_end)
                if shard.split != expected:
                    msg = (
                        f"{prefix}.split={shard.split!r} disagrees with declared "
                        f"boundaries; expected {expected!r} for {shard.date}"
                    )
                    raise ValueError(msg)

    def save(self, path: Path) -> None:
        """将清单序列化为 JSON。"""
        self.validate_split_contract(context="manifest save")
        payload = {
            "schema_version": self.schema_version,
            "train_end": self.train_end,
            "val_end": self.val_end,
            "purge_days": self.purge_days,
            "embargo_days": self.embargo_days,
            "vocab_path": self.vocab_path,
            "vocab_sha256": self.vocab_sha256,
            "event_ordering_version": self.event_ordering_version,
            "feature_transform_version": self.feature_transform_version,
            "shards": [asdict(s) for s in self.shards],
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """从 JSON 加载清单。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        shards = [ShardEntry(**s) for s in data["shards"]]
        manifest = cls(
            shards=shards,
            train_end=data.get("train_end"),
            val_end=data.get("val_end"),
            purge_days=data.get("purge_days", 5),
            embargo_days=data.get("embargo_days", 2),
            vocab_path=data.get("vocab_path"),
            vocab_sha256=data.get("vocab_sha256"),
            schema_version=data.get("schema_version", "cn_l2_v1"),
            event_ordering_version=data.get("event_ordering_version"),
            feature_transform_version=data.get("feature_transform_version"),
        )
        manifest.validate_split_contract(context=f"manifest {path}")
        return manifest


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    """流式计算文件 SHA-256，无需全量载入内存。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def scan_token_dir(
    tokens_dir: Path,
    *,
    markets: Sequence[str] = ("SH", "SZ"),
) -> list[ShardEntry]:
    """扫描 ``{tokens_dir}/{market}/{symbol}/{date}.parquet`` 为分片条目。"""
    tokens_dir = Path(tokens_dir)
    entries: list[ShardEntry] = []
    for market in markets:
        mkt_dir = tokens_dir / market
        if not mkt_dir.is_dir():
            continue
        for sym_dir in sorted(mkt_dir.iterdir()):
            if not sym_dir.is_dir():
                continue
            for shard in sorted(sym_dir.glob("*.parquet")):
                date = shard.stem
                rows = pq.ParquetFile(shard).metadata.num_rows
                entries.append(
                    ShardEntry(
                        market=market,
                        symbol=sym_dir.name,
                        date=date,
                        path=str(shard.resolve()),
                        rows=int(rows),
                        sha256=_sha256(shard),
                        data_contract_sha256=(
                            _sha256(token_contract_path(shard))
                            if token_contract_path(shard).is_file()
                            else None
                        ),
                    )
                )
    logger.info("scanned %d shards under %s", len(entries), tokens_dir)
    return entries


def _assign_split(date: str, train_end: str, val_end: str) -> str:
    """按日期边界将分片划入 train/val/test（训练集含边界）。"""
    # [导读] 字符串日期 "YYYY-MM-DD" 可直接用 <= 比较（字典序 = 时间序）
    if date <= train_end:
        return "train"
    if date <= val_end:
        return "val"
    return "test"


def build_manifest(
    tokens_dir: Path,
    *,
    train_end: str,
    val_end: str,
    markets: Sequence[str] = ("SH", "SZ"),
    purge_days: int = 5,
    embargo_days: int = 2,
    vocab_path: str | None = None,
) -> Manifest:
    """
    扫描 token 分片并分配基于时间的切分。

    参数
    ----------
    tokens_dir
        token 分片根目录。
    train_end, val_end
        ISO 日期；``date <= train_end`` 为训练，``<= val_end`` 为验证，否则为测试。
    markets
        包含的市场。
    purge_days, embargo_days
        记入清单的下游评估门控 CPCV 参数。
    vocab_path
        生成分片所用 ``vocab.json`` 的路径。

    返回
    -------
    Manifest
    """
    train_end = _canonical_date(train_end, field_name="train_end")
    val_end = _canonical_date(val_end, field_name="val_end")
    if train_end > val_end:
        msg = f"train_end must be <= val_end, got {train_end} > {val_end}"
        raise ValueError(msg)
    entries = scan_token_dir(tokens_dir, markets=markets)
    vocab = None
    if vocab_path is not None:
        artifact = Path(vocab_path)
        if not artifact.is_file():
            msg = f"vocab_path does not exist: {artifact}"
            raise FileNotFoundError(msg)
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("vocab_version") == "2.0":
            from quant_fm.tokenizer.vocab_v2 import VocabV2

            vocab = VocabV2.load(artifact)
        else:
            from quant_fm.tokenizer.vocab import Vocab

            vocab = Vocab.load(artifact)
        for entry in entries:
            assert_token_contract_matches(Path(entry.path), vocab)
    for e in entries:
        e.split = _assign_split(e.date, train_end, val_end)
    manifest = Manifest(
        shards=entries,
        train_end=train_end,
        val_end=val_end,
        purge_days=purge_days,
        embargo_days=embargo_days,
        vocab_path=vocab_path,
        vocab_sha256=(stable_vocab_sha256(vocab) if vocab is not None else None),
        schema_version=(vocab.schema_version if vocab is not None else "cn_l2_v1"),
        event_ordering_version=(
            vocab.event_ordering_version if vocab is not None else None
        ),
        feature_transform_version=(
            vocab.feature_transform_version if vocab is not None else None
        ),
    )
    manifest.validate_split_contract(context="built manifest")
    for name in ("train", "val", "test"):
        logger.info("split %s: %d shards", name, len(manifest.split(name)))
    return manifest
