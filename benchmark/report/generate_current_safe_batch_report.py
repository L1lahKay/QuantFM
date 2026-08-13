#!/usr/bin/env python3
"""
Generate the standalone 2026-08-06 current-safe N=3 experiment report.

Only run tokens declared by the two selected orchestration summaries are
accepted.  Historical rows in the aggregate benchmark CSV are deliberately
excluded so the generated tables and figures describe exactly 45 runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
DEFAULT_MATRIX = BENCHMARK_ROOT / "config" / "current-safe-matrix.json"
DEFAULT_RUNTIME = BENCHMARK_ROOT / "config" / "runtime.json"
DEFAULT_RESULTS = BENCHMARK_ROOT / "report" / "benchmark_results.csv"
DEFAULT_KV_SUMMARY = (
    BENCHMARK_ROOT
    / "results"
    / "orchestration"
    / "current-safe-kv-n3-20260806a"
    / "summary.json"
)
DEFAULT_K8S_SUMMARY = (
    BENCHMARK_ROOT
    / "results"
    / "orchestration"
    / "current-safe-k8s-n3-20260806b"
    / "summary.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "gpu-scheduler-reports" / "current-safe-20260806"

EXPECTED_MATRIX_SHA256 = (
    "eb9f41067008490d3894dfa619f759f5e086d08df5bc24f5d6f0a8f3506d70df"
)
EXPECTED_RUNTIME_SHA256 = (
    "7b3d662d2f84f2373e4a2e6d71932b83856840cc58c68b01c2e7a00534071e6b"
)
EXPECTED_IMAGE = (
    "registry.zs/gpu-dev/dylan-trainer@sha256:"
    "9e7f7f8dc3c15c522408d1e8da38401ac224b99ddfba363078f40403eb456574"
)
SCHEDULERS = ("K8s", "Kueue", "Volcano")
SCHEDULER_COLORS = {
    "K8s": "#2563eb",
    "Kueue": "#16a34a",
    "Volcano": "#dc2626",
}
SCENARIOS = (
    "nn-single-gpu1",
    "nn-multipod-gpu4",
    "transformer-single-gpu1",
    "transformer-single-gpu4",
    "transformer-multipod-gpu4",
)
SCENARIO_LABELS = {
    "nn-single-gpu1": "NN · 1 GPU · 单 Pod",
    "nn-multipod-gpu4": "NN · 4 GPU · 多 Pod",
    "transformer-single-gpu1": "Transformer · 1 GPU · 单 Pod",
    "transformer-single-gpu4": "Transformer · 4 GPU · 单 Pod",
    "transformer-multipod-gpu4": "Transformer · 4 GPU · 多 Pod",
}
NUMERIC_FIELDS = (
    "queue_time",
    "training_time",
    "wall_clock_time",
    "gpu_utilization",
    "throughput",
)


class BatchReportError(RuntimeError):
    """Raised when the exact-batch evidence contract is not satisfied."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchReportError(f"cannot read JSON {path}: {exc}") from exc


def normalize_scheduler(value: Any) -> str:
    text = str(value).strip().lower()
    if text == "k8s":
        return "K8s"
    if text == "kueue":
        return "Kueue"
    if text == "volcano":
        return "Volcano"
    raise BatchReportError(f"unexpected scheduler: {value!r}")


def load_task_contract(
    summary_paths: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    tasks: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for path in summary_paths:
        payload = load_json(path)
        if not isinstance(payload, Mapping) or payload.get("status") != "succeeded":
            raise BatchReportError(f"orchestration is not succeeded: {path}")
        matrix = payload.get("matrix")
        runtime = payload.get("runtime")
        if not isinstance(matrix, Mapping) or not isinstance(runtime, Mapping):
            raise BatchReportError(
                f"orchestration lacks matrix/runtime identity: {path}"
            )
        for name, value in (
            ("matrix", matrix.get("sha256")),
            ("runtime", runtime.get("sha256")),
        ):
            value = str(value or "")
            if name in digests and digests[name] != value:
                raise BatchReportError(f"orchestration {name} digests disagree")
            digests[name] = value
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or len(raw_tasks) != payload.get(
            "task_count"
        ):
            raise BatchReportError(f"orchestration task count is inconsistent: {path}")
        for task in raw_tasks:
            if not isinstance(task, Mapping):
                raise BatchReportError(f"invalid orchestration task: {path}")
            result = task.get("result")
            if not isinstance(result, Mapping) or result.get("status") != "succeeded":
                raise BatchReportError(f"task is not succeeded in {path}")
            token = str(task.get("run_token") or "")
            if not token or token in tasks:
                raise BatchReportError(f"missing or duplicate run token: {token!r}")
            tasks[token] = {
                "scenario_id": str(task.get("scenario_id") or ""),
                "scheduler": normalize_scheduler(task.get("scheduler")),
                "repetition": int(task.get("repetition")),
                "summary": path,
            }
    if len(tasks) != 45:
        raise BatchReportError(
            f"expected exactly 45 orchestration tasks, found {len(tasks)}"
        )
    if digests.get("matrix") != EXPECTED_MATRIX_SHA256:
        raise BatchReportError("selected batch does not use the reviewed matrix digest")
    if digests.get("runtime") != EXPECTED_RUNTIME_SHA256:
        raise BatchReportError(
            "selected batch does not use the reviewed runtime digest"
        )
    return tasks, digests


def load_exact_rows(
    csv_path: Path, tasks: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    matched: dict[str, dict[str, Any]] = {}
    try:
        stream = csv_path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise BatchReportError(
            f"cannot read aggregate result CSV {csv_path}: {exc}"
        ) from exc
    with stream:
        for raw in csv.DictReader(stream):
            run_id = raw.get("run_id", "")
            tokens = [token for token in tasks if token in run_id]
            if not tokens:
                continue
            if len(tokens) != 1:
                raise BatchReportError(
                    f"run ID ambiguously matches multiple tokens: {run_id}"
                )
            token = tokens[0]
            if token in matched:
                raise BatchReportError(
                    f"duplicate aggregate row for run token: {token}"
                )
            contract = tasks[token]
            if (
                raw.get("status") != "completed"
                or raw.get("execution_stage") != "completed"
            ):
                raise BatchReportError(f"selected row is not completed: {run_id}")
            if raw.get("scenario_id") != contract["scenario_id"]:
                raise BatchReportError(f"scenario mismatch for {run_id}")
            if raw.get("scheduler") != contract["scheduler"]:
                raise BatchReportError(f"scheduler mismatch for {run_id}")
            if raw.get("image_identity") != EXPECTED_IMAGE:
                raise BatchReportError(f"image mismatch for {run_id}")
            row: dict[str, Any] = dict(raw)
            row["run_token"] = token
            row["repetition"] = contract["repetition"]
            for field in NUMERIC_FIELDS:
                try:
                    row[field] = float(raw[field])
                except (KeyError, TypeError, ValueError) as exc:
                    raise BatchReportError(f"invalid {field} for {run_id}") from exc
            matched[token] = row
    missing = sorted(set(tasks) - set(matched))
    if missing:
        raise BatchReportError(f"aggregate CSV is missing {len(missing)} selected runs")
    return [matched[token] for token in sorted(matched)]


def summarize(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario_id"]), str(row["scheduler"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for scheduler in SCHEDULERS:
            cell = grouped.get((scenario, scheduler), [])
            repetitions = sorted(int(row["repetition"]) for row in cell)
            if len(cell) != 3 or repetitions != [1, 2, 3]:
                raise BatchReportError(
                    f"{scenario}/{scheduler} requires repetitions 1,2,3; found {repetitions}"
                )
            record: dict[str, Any] = {
                "scenario_id": scenario,
                "scenario": SCENARIO_LABELS[scenario],
                "scheduler": scheduler,
                "n": 3,
                "run_ids": sorted(str(row["run_id"]) for row in cell),
            }
            for field in NUMERIC_FIELDS:
                values = [float(row[field]) for row in cell]
                record[field] = {
                    "min": min(values),
                    "median": statistics.median(values),
                    "max": max(values),
                }
            summaries.append(record)
    if len(summaries) != 15:
        raise BatchReportError(
            f"expected exactly 15 completed cells, found {len(summaries)}"
        )
    return summaries


def svg_text(x: float, y: float, value: str, **attrs: str) -> str:
    attributes = " ".join(
        f'{name}="{html.escape(text)}"' for name, text in attrs.items()
    )
    return f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>{html.escape(value)}</text>'


def render_range_plot(
    summaries: Sequence[Mapping[str, Any]],
    field: str,
    title: str,
    subtitle: str,
    ticks: Sequence[float],
    output: Path,
    *,
    log_scale: bool = False,
    value_suffix: str = " s",
) -> None:
    width, height = 1280, 690
    left, right, top, bottom = 320, 90, 125, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = min(ticks), max(ticks)
    if log_scale and x_min <= 0:
        raise BatchReportError("log-scale ticks must be positive")

    def x_position(value: float) -> float:
        if log_scale:
            ratio = (math.log10(value) - math.log10(x_min)) / (
                math.log10(x_max) - math.log10(x_min)
            )
        else:
            ratio = (value - x_min) / (x_max - x_min)
        return left + max(0.0, min(1.0, ratio)) * plot_width

    rows_by_cell = {
        (str(item["scenario_id"]), str(item["scheduler"])): item for item in summaries
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#172033}.title{font-size:25px;font-weight:700}.subtitle{font-size:14px;fill:#526072}.axis{font-size:13px;fill:#526072}.scenario{font-size:14px;font-weight:600}.value{font-size:11px;font-weight:600}</style>",
        svg_text(36, 43, title, **{"class": "title"}),
        svg_text(36, 72, subtitle, **{"class": "subtitle"}),
    ]
    legend_x = 720
    for index, scheduler in enumerate(SCHEDULERS):
        x = legend_x + index * 150
        lines.append(
            f'<circle cx="{x}" cy="42" r="6" fill="{SCHEDULER_COLORS[scheduler]}"/>'
        )
        lines.append(svg_text(x + 12, 47, scheduler, **{"class": "axis"}))

    for tick in ticks:
        x = x_position(float(tick))
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" stroke="#dce2ea" stroke-width="1"/>'
        )
        label = f"{tick:g}"
        lines.append(
            svg_text(
                x, height - 39, label, **{"class": "axis", "text-anchor": "middle"}
            )
        )

    group_height = plot_height / len(SCENARIOS)
    scheduler_offsets = (-22.0, 0.0, 22.0)
    for scenario_index, scenario in enumerate(SCENARIOS):
        center = top + group_height * (scenario_index + 0.5)
        if scenario_index:
            separator = top + group_height * scenario_index
            lines.append(
                f'<line x1="36" y1="{separator:.2f}" x2="{width - right}" y2="{separator:.2f}" stroke="#eef1f5"/>'
            )
        lines.append(
            svg_text(
                left - 20,
                center + 5,
                SCENARIO_LABELS[scenario],
                **{"class": "scenario", "text-anchor": "end"},
            )
        )
        for scheduler, offset in zip(SCHEDULERS, scheduler_offsets):
            item = rows_by_cell[(scenario, scheduler)]
            stats = item[field]
            minimum = float(stats["min"])
            median = float(stats["median"])
            maximum = float(stats["max"])
            y = center + offset
            x1, xm, x2 = x_position(minimum), x_position(median), x_position(maximum)
            color = SCHEDULER_COLORS[scheduler]
            lines.extend(
                [
                    f'<line x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="4" stroke-linecap="round" opacity="0.70"/>',
                    f'<line x1="{x1:.2f}" y1="{y - 5:.2f}" x2="{x1:.2f}" y2="{y + 5:.2f}" stroke="{color}" stroke-width="2"/>',
                    f'<line x1="{x2:.2f}" y1="{y - 5:.2f}" x2="{x2:.2f}" y2="{y + 5:.2f}" stroke="{color}" stroke-width="2"/>',
                    f'<circle cx="{xm:.2f}" cy="{y:.2f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="2"/>',
                ]
            )
            label_x = min(width - right + 8, xm + 10)
            anchor = "start"
            if label_x > width - 125:
                label_x = xm - 10
                anchor = "end"
            lines.append(
                svg_text(
                    label_x,
                    y + 4,
                    f"{median:.3f}{value_suffix}",
                    **{"class": "value", "text-anchor": anchor, "fill": color},
                )
            )
    axis_note = (
        "对数刻度；横线：最小值–最大值；圆点：中位数；N=3"
        if log_scale
        else "横线：最小值–最大值；圆点：中位数；N=3"
    )
    lines.append(svg_text(left, height - 13, axis_note, **{"class": "axis"}))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def write_aggregate_csv(summaries: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = ["scenario_id", "scenario", "scheduler", "n"]
    for metric in NUMERIC_FIELDS:
        fields.extend((f"{metric}_min", f"{metric}_median", f"{metric}_max"))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            row = {
                key: item[key] for key in ("scenario_id", "scenario", "scheduler", "n")
            }
            for metric in NUMERIC_FIELDS:
                for stat in ("min", "median", "max"):
                    row[f"{metric}_{stat}"] = item[metric][stat]
            writer.writerow(row)


def write_markdown(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
    matrix: Mapping[str, Any],
    generated_at: str,
) -> None:
    by_cell = {
        (str(item["scenario_id"]), str(item["scheduler"])): item for item in summaries
    }
    scenario_config = {
        str(item["id"]): item
        for item in matrix.get("scenarios", [])
        if isinstance(item, Mapping)
    }

    def evidence_link(label: str, target: Path) -> str:
        relative = os.path.relpath(target.resolve(), output.parent.resolve())
        return f"[{label}]({Path(relative).as_posix()})"

    result_rows: list[str] = []
    for scenario in SCENARIOS:
        for scheduler in SCHEDULERS:
            item = by_cell[(scenario, scheduler)]
            result_rows.append(
                "| {scenario} | {scheduler} | 3 | {training} | {wall} | {queue} | {gpu} |".format(
                    scenario=SCENARIO_LABELS[scenario],
                    scheduler=scheduler,
                    training=fmt(float(item["training_time"]["median"]), 6),
                    wall=fmt(float(item["wall_clock_time"]["median"]), 3),
                    queue=fmt(float(item["queue_time"]["median"]), 6),
                    gpu=fmt(float(item["gpu_utilization"]["median"]), 3),
                )
            )
    config_rows: list[str] = []
    for scenario in SCENARIOS:
        item = scenario_config[scenario]
        parameters = item.get("parameters", {})
        if scenario.startswith("nn-"):
            shape = f"features={parameters.get('features')}; layers={parameters.get('layers')}"
            work = (
                f"global batch={parameters.get('global_batch_size')}; "
                f"steps={parameters.get('steps')}; warmup={parameters.get('warmup_steps')}"
            )
        else:
            shape = (
                f"L={parameters.get('layers')}; H={parameters.get('hidden')}; "
                f"heads={parameters.get('attention_heads')}; seq={parameters.get('sequence_length')}"
            )
            work = (
                f"effective batch={parameters.get('global_effective_batch_size')}; "
                f"steps={parameters.get('steps')}; warmup={parameters.get('warmup_steps')}"
            )
        config_rows.append(
            f"| {SCENARIO_LABELS[scenario]} | {item.get('replicas')} × {item.get('gpus_per_pod')} | "
            f"{item.get('cpu_per_pod')} / {item.get('memory_per_pod')} | {shape} | {work} |"
        )

    lines = [
        "# NN 与 Transformer 三调度器对比实验（2026-08-06）",
        "",
        "## 1. 结论",
        "",
        "本次实验共设置 5 个场景，分别在 Native Kubernetes、Kueue 和 Volcano 上",
        "重复运行 3 次，45/45 次运行完成。各调度器使用同一镜像、同一组参数和",
        "相同的确定性合成数据。",
        "",
        "主要结果如下：",
        "",
        "- NN 单卡训练时间中位数为 0.292–0.314 秒；Transformer 单卡为",
        "  7.949–8.201 秒；Transformer 单 Pod 四卡为 8.947–9.129 秒。",
        "- NN 多 Pod 四卡训练时间中位数为 1.122–1.143 秒。Transformer 多 Pod",
        "  四卡为 26.062–35.171 秒，三次重复的离散程度高于其他场景。",
        "- 端到端时间明显高于训练时间。对于这组短任务，Pod 创建、调度、容器启动和",
        "  分布式进程初始化占据了主要时间。",
        "- 三种调度器的单 Pod 训练时间接近。本实验重复次数为 3，且单次训练时间较短，",
        "  因此不据此进行调度器性能排名。",
        "- 本次运行限制在单节点、最多 4 张 GPU。8-GPU 和跨节点 NCCL/DDP 性能不在",
        "  本报告范围内。",
        "",
        "## 2. 实验设计",
        "",
        "### 2.1 对比原则",
        "",
        "实验只改变调度器和 Pod 布局。模型结构、固定工作量、镜像、运行参数和数据",
        "生成方式保持不变。每个“场景 × 调度器”组合运行 3 次，执行顺序使用种子",
        "`20260806` 随机打散，以减少运行先后顺序的影响。",
        "",
        "三个调度路径分别为：",
        "",
        "- Native Kubernetes：直接使用 `default-scheduler`。",
        "- Kueue：由实验 LocalQueue/ClusterQueue 完成准入，再交给默认调度器。",
        "- Volcano：使用实验 Queue 和 PodGroup，由 Volcano Scheduler 调度。",
        "",
        "### 2.2 运行环境",
        "",
        "| 项目 | 配置 |",
        "|---|---|",
        "| Namespace | `gpu-dev` |",
        "| 单次任务最大 GPU 数 | 4 |",
        "| 镜像 | `" + EXPECTED_IMAGE + "` |",
        "| 输入数据 | 进程内确定性合成数据 |",
        "| 临时存储 | 512Mi 内存型 `emptyDir` |",
        "| 持久存储 | 未挂载 PVC 或 hostPath |",
        "| Matrix SHA-256 | `" + EXPECTED_MATRIX_SHA256 + "` |",
        "| Runtime SHA-256 | `" + EXPECTED_RUNTIME_SHA256 + "` |",
        "",
        "### 2.3 实验矩阵",
        "",
        "| 场景 | Pod × GPU/Pod | CPU / 内存（每 Pod） | 模型结构 | 固定工作量 |",
        "|---|---:|---:|---|---|",
        *config_rows,
        "",
        "### 2.4 指标定义",
        "",
        "- **训练时间（Training Time）：** 训练程序记录的全局训练区间。多 Pod 运行需",
        "  各 rank 的结构化记录一致。",
        "- **端到端时间（Wall Clock）：** 从客户端提交到 Kubernetes Job Complete。",
        "- **排队时间（Queue Time）：** 从提交到最后一个预期 Pod 完成调度或准入。",
        "- **GPU 利用率：** 同一采样时刻先对各 GPU 求平均，再按时间求平均。",
        "",
        "完成记录必须同时具备 Job/Pod UID、Event、容器训练记录、GPU 样本、配额恢复",
        "和清理结果。dry-run 和 readiness 记录不计入 45 次正式运行。",
        "",
        "## 3. 实验结果",
        "",
        "### 3.1 中位数汇总",
        "",
        "| 场景 | 调度器 | N | 训练时间 s | 端到端时间 s | 排队时间 s | GPU 利用率 % |",
        "|---|---|---:|---:|---:|---:|---:|",
        *result_rows,
        "",
        "表中数值为三次运行的中位数。最小值、中位数和最大值可在",
        "[CSV 数据](current-safe-20260806-results.csv) 和",
        "[JSON 数据](current-safe-20260806-results.json) 中查询。",
        "",
        "### 3.2 训练时间",
        "",
        "横线表示三次运行的最小值和最大值，圆点表示中位数。由于 NN 和",
        "Transformer 的训练时间相差较大，横轴采用对数刻度。",
        "",
        "![训练时间对比](images/training-time-n3.svg)",
        "",
        "单 Pod 场景中，调度器之间的训练时间差异较小。Transformer 多 Pod 四卡场景",
        "在三种调度器下都出现了约 26 秒和约 35 秒两组结果，需要通过更长训练和更多",
        "重复次数进一步确认原因。",
        "",
        "### 3.3 端到端时间",
        "",
        "![端到端时间对比](images/wall-clock-n3.svg)",
        "",
        "所有场景的端到端时间都明显高于训练时间。短任务中，容器和分布式进程的启动",
        "成本会显著影响总耗时。",
        "",
        "### 3.4 排队时间",
        "",
        "![排队时间对比](images/queue-time-n3.svg)",
        "",
        "Volcano 的 Transformer 单卡场景有两个排队时间受 Kubernetes Event 整秒",
        "时间戳精度影响，按左删失规则记为 0。这里的 0 表示耗时低于可靠分辨范围，",
        "不表示调度过程没有开销。",
        "",
        "### 3.5 GPU 利用率",
        "",
        "![GPU 利用率对比](images/gpu-utilization-n3.svg)",
        "",
        "多 Pod 场景的 GPU 利用率高于短时单卡场景。由于任务持续时间短，这些数值",
        "主要用于核对本批次运行，不作为长时稳态训练的利用率估计。",
        "",
        "### 3.6 适用范围",
        "",
        "- 结果适用于本报告记录的镜像、参数、数据生成方式和四卡配额环境。",
        "- 合成数据不代表真实数据读取和存储吞吐已经验证。",
        "- 多 Pod 结果验证了调度和多进程启动链，但不是跨节点 NCCL/DDP 结果。",
        "- 持久训练仍需使用经确认的非根盘 PVC。",
        "",
        "## 4. 证据与清理记录",
        "",
        "- " + evidence_link("Kueue/Volcano 30-run orchestration", DEFAULT_KV_SUMMARY),
        "- " + evidence_link("Native K8s 15-run orchestration", DEFAULT_K8S_SUMMARY),
        "- " + evidence_link("current-safe matrix", DEFAULT_MATRIX),
        "- " + evidence_link("runtime configuration", DEFAULT_RUNTIME),
        "- "
        + evidence_link(
            "调度对象精确清理时间线",
            BENCHMARK_ROOT
            / "results"
            / "scheduler-setup"
            / "cleanup-20260806T062358Z"
            / "timeline.txt",
        ),
        "- "
        + evidence_link(
            "运行后集群与物理拓扑基线",
            BENCHMARK_ROOT
            / "results"
            / "follow-up-baseline"
            / "20260806T062821Z"
            / "metadata.tsv",
        ),
        "- "
        + evidence_link(
            "完整综合评估报告",
            BENCHMARK_ROOT / "report" / "Scheduler_Evaluation_Report.md",
        ),
        "",
        "运行结束后，实验 ResourceFlavor、ClusterQueue、LocalQueue、Volcano Queue、",
        "Workload、PodGroup、Job 和 Pod 均已按名称/UID 核验并清理；最终只读复核显示",
        "`gpu-dev` 的 `requests.nvidia.com/gpu` used=0。一次性 current-safe 授权已经用完。",
        "",
        f"文档生成时间：`{generated_at}`。",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def generate(
    output_dir: Path,
    result_csv: Path = DEFAULT_RESULTS,
    summary_paths: Sequence[Path] = (DEFAULT_KV_SUMMARY, DEFAULT_K8S_SUMMARY),
    matrix_path: Path = DEFAULT_MATRIX,
    runtime_path: Path = DEFAULT_RUNTIME,
) -> dict[str, Any]:
    if sha256(matrix_path) != EXPECTED_MATRIX_SHA256:
        raise BatchReportError("current-safe matrix file digest has changed")
    if sha256(runtime_path) != EXPECTED_RUNTIME_SHA256:
        raise BatchReportError("runtime file digest has changed")
    matrix = load_json(matrix_path)
    if not isinstance(matrix, Mapping) or matrix.get("matrix_name") != "current-safe":
        raise BatchReportError("matrix is not current-safe")
    tasks, digests = load_task_contract(summary_paths)
    rows = load_exact_rows(result_csv, tasks)
    summaries = summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = output_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    render_range_plot(
        summaries,
        "training_time",
        "各场景训练时间",
        "2026-08-06 current-safe 批次，共 45 次完成运行",
        (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 60.0),
        images / "training-time-n3.svg",
        log_scale=True,
    )
    render_range_plot(
        summaries,
        "wall_clock_time",
        "各场景端到端时间",
        "客户端提交 → Kubernetes Job Complete",
        tuple(float(value) for value in range(0, 71, 10)),
        images / "wall-clock-n3.svg",
    )
    render_range_plot(
        summaries,
        "queue_time",
        "各场景排队时间",
        "Event 时间戳精度不足的亚秒结果按规则左删失为 0",
        (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
        images / "queue-time-n3.svg",
    )
    render_range_plot(
        summaries,
        "gpu_utilization",
        "各场景平均 GPU 利用率",
        "同一时刻先对各 GPU 求均值，再按时间求均值",
        tuple(float(value) for value in range(0, 101, 20)),
        images / "gpu-utilization-n3.svg",
        value_suffix="%",
    )
    generated_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    result_payload = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "batch": "current-safe-20260806-n3",
        "matrix_sha256": digests["matrix"],
        "runtime_sha256": digests["runtime"],
        "image": EXPECTED_IMAGE,
        "run_count": len(rows),
        "cell_count": len(summaries),
        "required_repetitions": 3,
        "cells": summaries,
    }
    (output_dir / "current-safe-20260806-results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_aggregate_csv(summaries, output_dir / "current-safe-20260806-results.csv")
    document = output_dir / "CURRENT_SAFE_N3_EXPERIMENT_REPORT.md"
    write_markdown(document, summaries, matrix, generated_at)
    return {
        "document": str(document),
        "run_count": len(rows),
        "cell_count": len(summaries),
        "images": sorted(str(path) for path in images.glob("*.svg")),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--kv-summary", type=Path, default=DEFAULT_KV_SUMMARY)
    parser.add_argument("--k8s-summary", type=Path, default=DEFAULT_K8S_SUMMARY)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate(
        args.output_dir.resolve(),
        args.results.resolve(),
        (args.kv_summary.resolve(), args.k8s_summary.resolve()),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
