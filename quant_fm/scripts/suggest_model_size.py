r"""
根据预估事件量（清洗后 cn_l2_v1 行数）推荐 OrderFlow FM 规模与训练步数。

用法::

    python -m quant_fm.scripts.suggest_model_size --events 1.9e10
    python -m quant_fm.scripts.suggest_model_size \
        --dates-file quant_fm/data/medium_60_dates.txt \
        --symbols-per-day 5105 --events-per-symbol-day 37000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelSuggestion:
    """推荐超参与简要依据。"""

    events: int
    d_model: int
    n_layers: int
    n_heads: int
    max_steps: int
    label: str

    @property
    def approx_params_m(self) -> float:
        """粗算参数量（百万），与 pilot vocab 量级一致时误差 <10%。"""
        # 嵌入 + Transformer 主导项；与实测 512×16≈67M 对齐
        embed = 9 * 32 * self.d_model  # 连续字段 ~32 bin
        per_layer = 12 * self.d_model * self.d_model
        return (embed + self.n_layers * per_layer) / 1e6


def suggest(events: int) -> ModelSuggestion:
    """
    按全市场 OrderFlow FM 经验曲线推荐配置。

    参考：pilot 6M@15 symbol-day；全市场 190B 事件对应 67–114M 参数。
    """
    e = max(events, 1)
    if e < 50_000_000:
        return ModelSuggestion(e, 256, 6, 8, 20_000, "pilot")
    if e < 500_000_000:
        return ModelSuggestion(e, 384, 8, 8, 40_000, "small")
    if e < 2_000_000_000:
        return ModelSuggestion(e, 512, 10, 8, 60_000, "medium")
    if e < 10_000_000_000:
        return ModelSuggestion(e, 512, 16, 8, 100_000, "large")
    return ModelSuggestion(e, 768, 12, 12, 150_000, "xlarge")


def estimate_events(
    *,
    n_dates: int,
    symbols_per_day: int,
    events_per_symbol_day: int,
) -> int:
    """由日期数 × 标的数 × 日均事件数估算总事件量。"""
    return n_dates * symbols_per_day * events_per_symbol_day


def main() -> None:
    """Estimate data scale and print a recommended model configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=float,
        default=None,
        help="已知总事件数（科学计数法可写 1.9e10）",
    )
    parser.add_argument("--dates-file", type=Path, default=None)
    parser.add_argument("--symbols-per-day", type=int, default=5105)
    parser.add_argument(
        "--events-per-symbol-day",
        type=int,
        default=62_000,
        help="全市场单标的日均事件粗算（190B/603天/5105≈6.2万）",
    )
    args = parser.parse_args()

    if args.events is not None:
        total = int(args.events)
    elif args.dates_file is not None:
        dates = [
            ln.strip() for ln in args.dates_file.read_text().splitlines() if ln.strip()
        ]
        total = estimate_events(
            n_dates=len(dates),
            symbols_per_day=args.symbols_per_day,
            events_per_symbol_day=args.events_per_symbol_day,
        )
    else:
        parser.error("请指定 --events 或 --dates-file")

    s = suggest(total)
    tokens_gb = total * 9 * 4 / 1e9  # 9 个 int32 字段
    print(f"预估事件量:     {total:,} ({total / 1e9:.2f}B)")
    print(f"推荐档位:       {s.label}")
    print(f"model.d_model:  {s.d_model}")
    print(f"model.n_layers: {s.n_layers}")
    print(f"model.n_heads:  {s.n_heads}")
    print(f"optim.max_steps:{s.max_steps}")
    print(f"粗算参数量:     ~{s.approx_params_m:.0f}M")
    print(f"token 磁盘粗算: ~{tokens_gb:.0f} GB（仅 tok 列，不含 events）")
    print()
    print("# 可粘贴到 config_medium_8gpu.yaml 的 model / optim 段")
    print(f"d_model: {s.d_model}")
    print(f"n_layers: {s.n_layers}")
    print(f"n_heads: {s.n_heads}")
    print(f"max_steps: {s.max_steps}")


if __name__ == "__main__":
    main()
