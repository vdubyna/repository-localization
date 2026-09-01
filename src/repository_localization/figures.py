"""Generate the eight fixed Chapter 4 figures from sealed cell features."""

from __future__ import annotations

import csv
import io
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from repository_localization.core import (  # noqa: E402
    IntegrityError,
    PipelineError,
    _publish,
    _read_file,
    canonical,
    digest,
    strict_json,
)

CONDITIONS = ("NO-DOC", "OPTIONAL", "DOC-FIRST")
CONDITION_ALIASES = {
    "NO_DOC_GUIDANCE": "NO-DOC",
    "FUNCTIONAL_OPTIONAL": "OPTIONAL",
    "FUNCTIONAL_REQUIRED_BEFORE_SOURCE": "DOC-FIRST",
    **{condition: condition for condition in CONDITIONS},
}
TASK_TYPES = {
    "EXPLICIT_LOCATOR_CLUE": "З файловою підказкою",
    "NO_EXPLICIT_LOCATOR_CLUE": "Без файлової підказки",
}
COLORS = {"NO-DOC": "#64748b", "OPTIONAL": "#d97706", "DOC-FIRST": "#2563eb"}
TYPE_COLORS = {
    "EXPLICIT_LOCATOR_CLUE": "#0f766e",
    "NO_EXPLICIT_LOCATOR_CLUE": "#7c3aed",
}
QUALITY = {
    "Recall@3": "recall_at_3",
    "Recall@5": "recall_at_5",
    "nDCG@3": "ndcg_at_3",
    "F1": "returned_set_f1",
}
TRAJECTORY = {
    "Шлях з’явився\nбудь-коли": "gold_seen_any",
    "Шлях з’явився у перших\nтрьох діях із кодом": "gold_seen_by_3_source_actions",
    "Агент безпосередньо\nзвернувся до файла": "gold_targeted_any",
}
REQUIRED_COLUMNS = {
    "cell_id",
    "safe_task_id",
    "task_type",
    "condition",
    "model",
    "reasoning_effort",
    "repeat",
    "provider_total_tokens",
    "elapsed_seconds",
    "agent_step_count",
    *QUALITY.values(),
    *TRAJECTORY.values(),
}
EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2, "xhigh": 3}


@dataclass(frozen=True)
class Cell:
    cell_id: str
    task_id: str
    task_type: str
    condition: str
    model: str
    reasoning_effort: str
    repeat: int
    recall_at_3: float
    recall_at_5: float
    ndcg_at_3: float
    returned_set_f1: float
    provider_total_tokens: float
    elapsed_seconds: float
    agent_step_count: float
    gold_seen_any: float
    gold_seen_by_3_source_actions: float
    gold_targeted_any: float


@dataclass(frozen=True)
class FigureInput:
    identity: dict[str, str]
    artifact_root: Path
    source_path: Path
    source_checksum: str
    cells: list[Cell]


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{label} must be a non-empty string")
    return value


def _number(value: str, label: str, *, minimum: float = 0) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise PipelineError(f"{label} must be numeric") from exc
    if not isfinite(result) or result < minimum:
        raise PipelineError(f"{label} is outside the supported range")
    return result


def _probability(value: str, label: str) -> float:
    result = _number(value, label)
    if result > 1:
        raise PipelineError(f"{label} must be between zero and one")
    return result


def _integer(value: str, label: str) -> int:
    result = _number(value, label)
    if not result.is_integer():
        raise PipelineError(f"{label} must be an integer")
    return int(result)


def _record(config_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path = config_path.expanduser().resolve()
    try:
        record = tomllib.loads(_read_file(path, "experiment config").decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PipelineError(f"invalid experiment config: {path}") from exc
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise PipelineError("experiment config must use schema_version = 1")
    experiment_id = _text(record.get("experiment_id"), "experiment_id")
    experiment_version = _text(record.get("experiment_version"), "experiment_version")
    artifact_value = _text(record.get("artifact_dir"), "artifact_dir")
    artifact_dir = Path(artifact_value).expanduser()
    if not artifact_dir.is_absolute():
        artifact_dir = path.parent / artifact_dir
    root = (artifact_dir / experiment_id / experiment_version).resolve()
    report = strict_json(_read_file(root / "report" / "data.json", "report data"), "report data")
    if not isinstance(report, dict):
        raise IntegrityError("report data must be an object")
    for key, expected in (
        ("experiment_id", experiment_id),
        ("experiment_version", experiment_version),
    ):
        if report.get(key) != expected:
            raise IntegrityError(f"report data has the wrong {key}")
    plan_id = _text(report.get("plan_id"), "report plan_id")
    declared_plan = record.get("source_plan_id", record.get("plan_id"))
    if declared_plan is not None and declared_plan != plan_id:
        raise IntegrityError("experiment config and report data have different plan_id values")
    return (
        record,
        root,
        {
            "experiment_id": experiment_id,
            "experiment_version": experiment_version,
            "plan_id": plan_id,
        },
    )


def _cells(payload: bytes) -> list[Cell]:
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    except UnicodeDecodeError as exc:
        raise PipelineError("cell feature table must be UTF-8") from exc
    if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        raise PipelineError(f"cell feature table is missing columns: {missing}")
    result: list[Cell] = []
    seen: set[str] = set()
    for number, row in enumerate(reader, 2):
        cell_id = _text(row.get("cell_id"), f"cell row {number} cell_id")
        if cell_id in seen:
            raise PipelineError(f"duplicate cell_id: {cell_id}")
        seen.add(cell_id)
        raw_condition = _text(row.get("condition"), f"cell row {number} condition")
        condition = CONDITION_ALIASES.get(raw_condition)
        if condition is None:
            raise PipelineError(f"cell row {number} has unsupported condition: {raw_condition}")
        task_type = _text(row.get("task_type"), f"cell row {number} task_type")
        if task_type not in TASK_TYPES:
            raise PipelineError(f"cell row {number} has unsupported task_type: {task_type}")
        result.append(
            Cell(
                cell_id=cell_id,
                task_id=_text(row.get("safe_task_id"), f"cell row {number} safe_task_id"),
                task_type=task_type,
                condition=condition,
                model=_text(row.get("model"), f"cell row {number} model"),
                reasoning_effort=_text(
                    row.get("reasoning_effort"), f"cell row {number} reasoning_effort"
                ),
                repeat=_integer(row["repeat"], f"cell row {number} repeat"),
                recall_at_3=_probability(row["recall_at_3"], f"cell row {number} recall_at_3"),
                recall_at_5=_probability(row["recall_at_5"], f"cell row {number} recall_at_5"),
                ndcg_at_3=_probability(row["ndcg_at_3"], f"cell row {number} ndcg_at_3"),
                returned_set_f1=_probability(
                    row["returned_set_f1"], f"cell row {number} returned_set_f1"
                ),
                provider_total_tokens=_number(
                    row["provider_total_tokens"], f"cell row {number} provider_total_tokens"
                ),
                elapsed_seconds=_number(
                    row["elapsed_seconds"], f"cell row {number} elapsed_seconds"
                ),
                agent_step_count=_number(
                    row["agent_step_count"], f"cell row {number} agent_step_count"
                ),
                gold_seen_any=_probability(
                    row["gold_seen_any"], f"cell row {number} gold_seen_any"
                ),
                gold_seen_by_3_source_actions=_probability(
                    row["gold_seen_by_3_source_actions"],
                    f"cell row {number} gold_seen_by_3_source_actions",
                ),
                gold_targeted_any=_probability(
                    row["gold_targeted_any"], f"cell row {number} gold_targeted_any"
                ),
            )
        )
    if not result:
        raise PipelineError("cell feature table is empty")
    return result


def _validate_design(record: dict[str, Any], cells: list[Cell]) -> None:
    if {cell.condition for cell in cells} != set(CONDITIONS):
        raise PipelineError("figures require NO-DOC, OPTIONAL, and DOC-FIRST observations")
    task_types: dict[str, str] = {}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for cell in cells:
        previous = task_types.setdefault(cell.task_id, cell.task_type)
        if previous != cell.task_type:
            raise IntegrityError(f"task_type changes within task {cell.task_id}")
        counts[cell.task_id, cell.condition] += 1
    for task_id in task_types:
        task_counts = {counts[task_id, condition] for condition in CONDITIONS}
        if len(task_counts) != 1 or 0 in task_counts:
            raise IntegrityError(f"unbalanced conditions for task {task_id}")
    expectations = {
        "task_count": len(task_types),
        "cell_count": len(cells),
        "profile_count": len({(cell.model, cell.reasoning_effort) for cell in cells}),
    }
    for key, actual in expectations.items():
        declared = record.get(key)
        if declared is not None and declared != actual:
            raise IntegrityError(f"{key} is {actual}, but experiment config declares {declared}")


def _input(config_path: Path) -> FigureInput:
    record, root, identity = _record(config_path)
    source = root / "features" / "cell_features.csv"
    payload = _read_file(source, "cell feature table")
    cells = _cells(payload)
    _validate_design(record, cells)
    return FigureInput(identity, root, source, digest(payload), cells)


def _mean(cells: list[Cell], condition: str, metric: str) -> float:
    return fmean(getattr(cell, metric) for cell in cells if cell.condition == condition)


def _task_means(cells: list[Cell], metric: str) -> tuple[list[str], dict[str, list[float]]]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cell in cells:
        buckets[cell.task_id, cell.condition].append(getattr(cell, metric))
    tasks = sorted({cell.task_id for cell in cells})
    return tasks, {
        condition: [fmean(buckets[task, condition]) for task in tasks] for condition in CONDITIONS
    }


def _task_contrasts_by_type(cells: list[Cell], metric: str) -> dict[str, float]:
    tasks, means = _task_means(cells, metric)
    task_types = {cell.task_id: cell.task_type for cell in cells}
    return {
        task_type: fmean(
            means["DOC-FIRST"][index] - means["OPTIONAL"][index]
            for index, task in enumerate(tasks)
            if task_types[task] == task_type
        )
        for task_type in TASK_TYPES
    }


def _comma(value: float, digits: int = 3, *, sign: bool = False) -> str:
    flag = "+" if sign else ""
    return f"{value:{flag}.{digits}f}".replace(".", ",")


def _thousands(value: float) -> str:
    return f"{round(value):,}".replace(",", " ")


def _axis_decimal(value: float, _: float) -> str:
    return f"{value:g}".replace(".", ",")


def _finish(figure: Figure, title: str, identity: dict[str, str]) -> None:
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.text(
        0.995,
        0.004,
        f"{identity['experiment_id']} · {identity['experiment_version']}",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#64748b",
    )


def _figure_4_1(source: FigureInput) -> tuple[Figure, str, dict[str, Any]]:
    title = "Рисунок 4.1 – Середні Recall@3 і nDCG@3 за документаційною умовою"
    metrics = (("Recall@3", "recall_at_3", "#0f766e"), ("nDCG@3", "ndcg_at_3", "#7c3aed"))
    values = {
        label: [_mean(source.cells, condition, key) for condition in CONDITIONS]
        for label, key, _ in metrics
    }
    figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    width = 0.34
    positions = list(range(len(CONDITIONS)))
    for offset, (label, _, color) in zip((-width / 2, width / 2), metrics, strict=True):
        bars = axis.bar(
            [position + offset for position in positions],
            values[label],
            width,
            label=label,
            color=color,
        )
        for bar, value in zip(bars, values[label], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                _comma(value),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    axis.set_xticks(positions, CONDITIONS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Середнє значення метрики")
    axis.yaxis.set_major_formatter(FuncFormatter(_axis_decimal))
    axis.legend(frameon=False, ncols=2, loc="upper left")
    axis.grid(axis="y", alpha=0.22)
    _finish(figure, title, source.identity)
    return figure, title, values


def _horizontal_points(
    source: FigureInput,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    x_label: str,
) -> Figure:
    figure, axis = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    positions = list(range(len(labels)))
    offsets = [0] if len(series) == 1 else [-0.13, 0.13]
    for offset, (name, values, color) in zip(offsets, series, strict=True):
        y = [position + offset for position in positions]
        axis.scatter(values, y, s=58, color=color, label=name, zorder=3)
        for value, vertical in zip(values, y, strict=True):
            axis.annotate(
                _comma(value, sign=True),
                (value, vertical),
                xytext=(5 if value >= 0 else -5, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8,
            )
    maximum = max(abs(value) for _, values, _ in series for value in values)
    margin = max(maximum * 0.38, 0.012)
    axis.set_xlim(-maximum - margin, maximum + margin)
    axis.axvline(0, color="#334155", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(x_label)
    axis.xaxis.set_major_formatter(FuncFormatter(_axis_decimal))
    axis.grid(axis="x", alpha=0.22)
    axis.legend(frameon=False, loc="lower right")
    _finish(figure, title, source.identity)
    return figure


def _figure_4_2(source: FigureInput) -> tuple[Figure, str, dict[str, Any]]:
    title = "Рисунок 4.2 – Середні різниці якості DOC-FIRST проти OPTIONAL і NO-DOC"
    labels = list(QUALITY)
    optional = [
        _mean(source.cells, "DOC-FIRST", key) - _mean(source.cells, "OPTIONAL", key)
        for key in QUALITY.values()
    ]
    no_doc = [
        _mean(source.cells, "DOC-FIRST", key) - _mean(source.cells, "NO-DOC", key)
        for key in QUALITY.values()
    ]
    series = [
        ("DOC-FIRST − OPTIONAL", optional, COLORS["OPTIONAL"]),
        ("DOC-FIRST − NO-DOC", no_doc, COLORS["NO-DOC"]),
    ]
    figure = _horizontal_points(source, title, labels, series, "Різниця середнього значення")
    summary = {name: dict(zip(labels, values, strict=True)) for name, values, _ in series}
    return figure, title, summary


def _figure_4_3(source: FigureInput) -> tuple[Figure, str, dict[str, Any]]:
    title = "Рисунок 4.3 – Різниці DOC-FIRST проти OPTIONAL за наявністю файлової підказки"
    selected = {
        "Recall@3": "recall_at_3",
        "nDCG@3": "ndcg_at_3",
        "F1": "returned_set_f1",
    }
    series = []
    summary: dict[str, Any] = {}
    for task_type, label in TASK_TYPES.items():
        contrasts = [
            _task_contrasts_by_type(source.cells, metric)[task_type] for metric in selected.values()
        ]
        series.append((label, contrasts, TYPE_COLORS[task_type]))
        summary[label] = dict(zip(selected, contrasts, strict=True))
    figure = _horizontal_points(
        source,
        title,
        list(selected),
        series,
        "Середня різниця DOC-FIRST − OPTIONAL",
    )
    return figure, title, summary


def _boxplot(
    source: FigureInput,
    title: str,
    metric: str,
    y_label: str,
    formatter: Callable[[float], str],
    *,
    logarithmic: bool = False,
) -> tuple[Figure, dict[str, float]]:
    _, values = _task_means(source.cells, metric)
    condition_values = [values[condition] for condition in CONDITIONS]
    means = {condition: _mean(source.cells, condition, metric) for condition in CONDITIONS}
    figure, axis = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    boxes = axis.boxplot(
        condition_values,
        tick_labels=CONDITIONS,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0f172a", "linewidth": 1.5},
        whiskerprops={"color": "#475569"},
        capprops={"color": "#475569"},
    )
    for patch, condition in zip(boxes["boxes"], CONDITIONS, strict=True):
        patch.set_facecolor(COLORS[condition])
        patch.set_alpha(0.24)
    for position, condition in enumerate(CONDITIONS, 1):
        task_values = values[condition]
        offsets = [((index % 13) - 6) * 0.012 for index in range(len(task_values))]
        axis.scatter(
            [position + offset for offset in offsets],
            task_values,
            s=12,
            color=COLORS[condition],
            alpha=0.45,
            edgecolors="none",
            zorder=2,
        )
        mean = means[condition]
        axis.scatter([position], [mean], marker="D", s=48, color="#111827", zorder=4)
        axis.annotate(
            f"середнє {formatter(mean)}",
            (position, mean),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )
    if logarithmic:
        if any(value <= 0 for group in condition_values for value in group):
            raise PipelineError(f"{metric} must be positive for a logarithmic figure")
        axis.set_yscale("log")
    axis.set_ylabel(y_label)
    axis.grid(axis="y", alpha=0.22)
    _finish(figure, title, source.identity)
    return figure, means


def _figure_4_4(source: FigureInput) -> tuple[Figure, str, dict[str, Any]]:
    title = "Рисунок 4.4 – Розподіл кількості кроків агента за документаційною умовою"
    figure, means = _boxplot(
        source,
        title,
        "agent_step_count",
        "Середня кількість кроків агента для задачі",
        lambda value: _comma(value),
    )
    return figure, title, means


def _figure_4_5(source: FigureInput) -> tuple[Figure, str, dict[str, Any]]:
    title = "Рисунок 4.5 – Розподіл загальної кількості токенів за документаційною умовою"
    figure, means = _boxplot(
        source,
        title,
        "provider_total_tokens",
        "Загальна кількість токенів (логарифмічна шкала)",
        _thousands,
        logarithmic=True,
    )
    return figure, title, means


def _figure_4_6(source: FigureInput) -> tuple[Figure, str, dict[str, Any]]:
    title = "Рисунок 4.6 – Розподіл повного часу виконання за документаційною умовою"
    figure, means = _boxplot(
        source,
        title,
        "elapsed_seconds",
        "Повний час виконання, с",
        lambda value: _comma(value, 2),
    )
    return figure, title, means


def _profile_label(profile: tuple[str, str]) -> str:
    model, effort = profile
    return f"{model.removeprefix('gpt-5.6-')} · {effort}"


def _figure_4_7(source: FigureInput) -> tuple[Figure, str, dict[str, Any]]:
    title = "Рисунок 4.7 – Середні витрати токенів і часу за моделлю та рівнем міркування"
    profiles = sorted(
        {(cell.model, cell.reasoning_effort) for cell in source.cells},
        key=lambda item: (item[0], EFFORT_ORDER.get(item[1], 99), item[1]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), sharey=True, constrained_layout=True)
    offsets = {"NO-DOC": -0.2, "OPTIONAL": 0, "DOC-FIRST": 0.2}
    summary: dict[str, Any] = {}
    for axis, metric, panel, x_label in (
        (axes[0], "provider_total_tokens", "А. Середня кількість токенів", "Токени"),
        (axes[1], "elapsed_seconds", "Б. Середній повний час", "Секунди"),
    ):
        for condition in CONDITIONS:
            values = []
            for profile in profiles:
                selected = [
                    getattr(cell, metric)
                    for cell in source.cells
                    if (cell.model, cell.reasoning_effort) == profile
                    and cell.condition == condition
                ]
                values.append(fmean(selected))
                summary.setdefault(_profile_label(profile), {}).setdefault(condition, {})[
                    metric
                ] = values[-1]
            axis.scatter(
                values,
                [index + offsets[condition] for index in range(len(profiles))],
                s=42,
                color=COLORS[condition],
                label=condition,
                zorder=3,
            )
        axis.set_title(panel, fontsize=11)
        axis.set_xlabel(x_label)
        axis.grid(axis="x", alpha=0.22)
        axis.xaxis.set_major_formatter(
            FuncFormatter(
                (lambda value, _: _thousands(value))
                if metric == "provider_total_tokens"
                else _axis_decimal
            )
        )
    axes[0].set_yticks(range(len(profiles)), [_profile_label(profile) for profile in profiles])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
    )
    _finish(figure, title, source.identity)
    return figure, title, summary


def _figure_4_8(source: FigureInput) -> tuple[Figure, str, dict[str, Any]]:
    title = "Рисунок 4.8 – Поява еталонного файла в ході пошуку за OPTIONAL і DOC-FIRST"
    categories = list(TRAJECTORY)
    values = {
        condition: [100 * _mean(source.cells, condition, metric) for metric in TRAJECTORY.values()]
        for condition in ("OPTIONAL", "DOC-FIRST")
    }
    figure, axis = plt.subplots(figsize=(10.2, 5.7), constrained_layout=True)
    positions = list(range(len(categories)))
    width = 0.36
    for offset, condition in ((-width / 2, "OPTIONAL"), (width / 2, "DOC-FIRST")):
        bars = axis.bar(
            [position + offset for position in positions],
            values[condition],
            width,
            color=COLORS[condition],
            label=condition,
        )
        for bar, value in zip(bars, values[condition], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{_comma(value, 2)} %",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    axis.set_xticks(positions, categories)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Частка запусків, %")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncols=2, loc="upper left")
    _finish(figure, title, source.identity)
    return figure, title, values


FIGURES = (
    ("figure-4-1-quality-by-condition", _figure_4_1),
    ("figure-4-2-doc-first-quality-differences", _figure_4_2),
    ("figure-4-3-differences-by-file-clue", _figure_4_3),
    ("figure-4-4-agent-steps", _figure_4_4),
    ("figure-4-5-total-tokens", _figure_4_5),
    ("figure-4-6-elapsed-time", _figure_4_6),
    ("figure-4-7-cost-by-profile", _figure_4_7),
    ("figure-4-8-early-gold-file-discovery", _figure_4_8),
)


def _render(figure: Figure, title: str, identity: dict[str, str], format_name: str) -> bytes:
    target = io.BytesIO()
    description = (
        f"{identity['experiment_id']} {identity['experiment_version']}; plan {identity['plan_id']}"
    )
    metadata: dict[str, Any]
    if format_name == "png":
        metadata = {
            "Title": title,
            "Description": description,
            "Software": "repository-localization",
        }
    else:
        metadata = {
            "Title": title,
            "Subject": description,
            "Creator": "repository-localization",
            "CreationDate": None,
            "ModDate": None,
        }
    figure.savefig(
        target,
        format=format_name,
        dpi=220 if format_name == "png" else None,
        metadata=metadata,
        bbox_inches="tight",
        facecolor="white",
    )
    return target.getvalue()


def figures(config_path: Path) -> tuple[dict[str, str], Path]:
    """Publish PNG/PDF figures and a checksum manifest without rerunning the provider."""
    source = _input(config_path)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": 220,
        }
    )
    files: dict[str, bytes] = {}
    entries = []
    for stem, factory in FIGURES:
        figure, title, summary = factory(source)
        try:
            png = _render(figure, title, source.identity, "png")
            pdf = _render(figure, title, source.identity, "pdf")
        finally:
            plt.close(figure)
        files[f"{stem}.png"] = png
        files[f"{stem}.pdf"] = pdf
        entries.append(
            {
                "figure": stem,
                "title": title,
                "summary": summary,
                "files": {
                    "png": {"name": f"{stem}.png", "checksum": digest(png)},
                    "pdf": {"name": f"{stem}.pdf", "checksum": digest(pdf)},
                },
            }
        )
    manifest = {
        "schema_version": 1,
        **source.identity,
        "source": "features/cell_features.csv",
        "source_checksum": source.source_checksum,
        "task_count": len({cell.task_id for cell in source.cells}),
        "cell_count": len(source.cells),
        "profile_count": len({(cell.model, cell.reasoning_effort) for cell in source.cells}),
        "figure_count": len(entries),
        "figures": entries,
    }
    files["manifest.json"] = canonical(manifest)
    root = source.artifact_root / "report" / "figures"
    _publish(root, files)
    return source.identity, root
