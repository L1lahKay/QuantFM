"""Embedding 产线吞吐/延迟记录结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingBenchmark:
    """Embedding 生成和加载的统一性能结果。"""

    embedding_stock_days_per_second: float
    cold_checkpoint_load_seconds: float
    warm_inference_latency_p50_seconds: float
    warm_inference_latency_p95_seconds: float
    checkpoint_pause_seconds: float = 0.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            msg = "benchmark metrics must be non-negative"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, float]:
        """转换为实验 registry 可写的映射。"""
        return asdict(self)
