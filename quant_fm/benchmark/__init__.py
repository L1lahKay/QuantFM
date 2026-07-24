"""统一的训练、推理与 embedding 性能指标。"""

from quant_fm.benchmark.embedding_benchmark import EmbeddingBenchmark
from quant_fm.benchmark.model_benchmark import ModelBenchmark, benchmark_model

__all__ = ["EmbeddingBenchmark", "ModelBenchmark", "benchmark_model"]
