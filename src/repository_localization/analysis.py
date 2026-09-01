"""Extract public evidence, add gold-aware features, and publish analysis tables."""

from __future__ import annotations

import ast
import csv
import io
import re
import unicodedata
from collections import defaultdict
from math import log2
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import tiktoken

from repository_localization.core import (
    CONDITIONS,
    Config,
    IntegrityError,
    PipelineError,
    StateError,
    _absolute,
    _files,
    _jsonl,
    _paths,
    _publish,
    _read_file,
    _safe_id,
    _schema_one,
    _table,
    _write_once,
    canonical,
    digest,
    load_config,
    strict_json,
)
from repository_localization.execution import _runs
from repository_localization.planning import _current, _read_plan, _task_map, identity

_PATH = re.compile(r"(?<![A-Za-z0-9_./-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?![A-Za-z0-9_./-])")
_FILENAME = re.compile(
    r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_-]+\."
    r"(?:cfg|css|html?|ini|ipynb|js|json|md|mdx|pyi?|rst|sh|toml|ts|txt|ya?ml)"
    r"(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_BACKTICK_SYMBOL = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`")
_CALL_SYMBOL = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_WIKI_ENCODING = tiktoken.get_encoding("o200k_base")

TASK_TYPES = ("EXPLICIT_LOCATOR_CLUE", "NO_EXPLICIT_LOCATOR_CLUE")
CELL_COLUMNS = (
    "experiment_id",
    "experiment_version",
    "plan_id",
    "cell_id",
    "task_id",
    "repository",
    "task_type",
    "condition",
    "model",
    "reasoning_effort",
    "repeat",
    "status",
    "terminal_reason",
    "recall_at_3",
    "recall_at_5",
    "ndcg_at_3",
    "returned_set_f1",
    "provider_total_tokens",
    "elapsed_seconds",
    "agent_step_count",
    "wiki_read_count",
    "wiki_tokens",
    "unique_wiki_pages",
    "beyond_entry_reads",
    "gold_seen_any",
    "gold_seen_by_3_source_actions",
    "gold_targeted_any",
)
TASK_METRICS = (
    "recall_at_3",
    "recall_at_5",
    "ndcg_at_3",
    "returned_set_f1",
    "provider_total_tokens",
    "elapsed_seconds",
    "agent_step_count",
    "wiki_read_count",
    "wiki_tokens",
    "unique_wiki_pages",
    "beyond_entry_reads",
    "gold_seen_any",
    "gold_seen_by_3_source_actions",
    "gold_targeted_any",
)
_CONDITION_SLUGS = {"NO-DOC": "no_doc", "OPTIONAL": "optional", "DOC-FIRST": "doc_first"}
_CONTRASTS = (
    ("doc_first_minus_optional", "DOC-FIRST", "OPTIONAL"),
    ("doc_first_minus_no_doc", "DOC-FIRST", "NO-DOC"),
    ("optional_minus_no_doc", "OPTIONAL", "NO-DOC"),
)


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _prompt_locators(prompt: str) -> dict[str, set[str]]:
    paths = {_normalized(value) for value in _PATH.findall(prompt)}
    filenames = {_normalized(value) for value in _FILENAME.findall(prompt)}
    symbols: set[str] = set()
    for qualified in _BACKTICK_SYMBOL.findall(prompt):
        symbols.update(_normalized(part) for part in qualified.split("."))
    symbols.update(_normalized(value) for value in _CALL_SYMBOL.findall(prompt))
    for word in _WORD.findall(prompt):
        body = word.strip("_")
        if "_" in body or (
            any(character.islower() for character in body)
            and sum(character.isupper() for character in body) >= 2
        ):
            symbols.add(_normalized(word))
    return {"paths": paths, "filenames": filenames, "symbols": symbols}


def _prompt_features(prompt: str) -> dict[str, bool]:
    locators = _prompt_locators(prompt)
    return {
        "prompt_has_path": bool(locators["paths"]),
        "prompt_has_filename": bool(locators["filenames"]),
        "prompt_has_symbol": bool(locators["symbols"]),
    }


def _task_source_roots(config: Config) -> dict[str, Path]:
    _, rows = _jsonl(config.tasks, "tasks JSONL")
    result: dict[str, Path] = {}
    keys = {
        "task_id",
        "repository",
        "base_commit",
        "prompt",
        "source_root",
        "documentation_entry",
    }
    for number, row in enumerate(rows, 1):
        row = _table(row, f"tasks row {number}", keys)
        task_id = _safe_id(row["task_id"], f"tasks row {number} task_id")
        result[task_id] = _absolute(config.path.parent, row["source_root"], "source_root")
    return result


def _durable_runs(config: Config, plan: dict[str, Any]) -> list[dict[str, Any]]:
    runs = _runs(config.root, plan)
    ordered = [runs.get(cell["cell_id"]) for cell in plan["cells"]]
    if any(row is None for row in ordered):
        raise StateError("all planned cells must have a durable outcome before features")
    return ordered  # type: ignore[return-value]


def _manifest(plan: dict[str, Any], checksum: str) -> dict[str, Any]:
    return {"schema_version": 1, **identity(plan), "data_checksum": checksum}


def _command_actions(payload: bytes) -> list[tuple[str, str]]:
    actions: list[tuple[str, str]] = []
    for number, line in enumerate(payload.splitlines(), 1):
        event = strict_json(line, f"Codex event {number}")
        if not isinstance(event, dict) or event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        command = item.get("command")
        output = item.get("aggregated_output", "")
        if not isinstance(command, str) or not isinstance(output, str):
            raise IntegrityError("Codex command event has invalid command or output")
        actions.append((command, output))
    return actions


def _mentioned_paths(text: str, paths: Iterable[str]) -> set[str]:
    return {
        path
        for path in paths
        if re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(path)}(?![A-Za-z0-9_./-])", text)
    }


def _wiki_features(events: bytes, task: dict[str, Any]) -> dict[str, int]:
    documentation = set(task["documentation"]["paths"])
    source = set(task["source_files"]) - documentation
    entry = task["documentation"]["entry_path"]
    reads: list[tuple[set[str], str]] = []
    for command, output in _command_actions(events):
        documentation_paths = _mentioned_paths(command, documentation)
        source_paths = _mentioned_paths(command, source)
        if output and documentation_paths and not source_paths:
            reads.append((documentation_paths, output))
    pages = set().union(*(paths for paths, _ in reads)) if reads else set()
    return {
        "wiki_read_count": len(reads),
        "wiki_tokens": sum(len(_WIKI_ENCODING.encode(output)) for _, output in reads),
        "unique_wiki_pages": len(pages),
        "beyond_entry_reads": sum(bool(paths - {entry}) for paths, _ in reads),
    }


def _feature_rows(
    config: Config, plan: dict[str, Any], runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    tasks = _task_map(plan)
    task_features = {task_id: _prompt_features(task["prompt"]) for task_id, task in tasks.items()}
    rows: list[dict[str, Any]] = []
    for run in runs:
        task = tasks[run["task_id"]]
        events = _read_file(config.root / "runs" / run["cell_id"] / "events.jsonl", "Codex events")
        input_tokens = run.get("input_tokens")
        output_tokens = run.get("output_tokens")
        rows.append(
            {
                "schema_version": 1,
                **identity(plan),
                **task_features[run["task_id"]],
                **_wiki_features(events, task),
                "cell_id": run["cell_id"],
                "task_id": run["task_id"],
                "repository": task["repository"],
                "condition": run["condition"],
                "model": run["model"],
                "reasoning_effort": run["reasoning_effort"],
                "repeat": run["repeat"],
                "status": run["status"],
                "terminal_reason": run.get("terminal_reason"),
                "files": run["files"],
                "provider_total_tokens": (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else None
                ),
                "elapsed_seconds": run["duration_ms"] / 1000,
                "agent_step_count": run.get("tool_steps"),
            }
        )
    return rows


def features(config_path: Path) -> tuple[dict[str, str], Path]:
    config, plan = _current(config_path)
    rows = _feature_rows(config, plan, _durable_runs(config, plan))
    data = b"".join(canonical(row) for row in rows)
    root = config.root / "features"
    if root.exists():
        if _read_features(config, plan) != rows:
            raise IntegrityError("features do not match run evidence")
    else:
        _publish(
            root,
            {"manifest.json": canonical(_manifest(plan, digest(data))), "data.jsonl": data},
        )
    return identity(plan), root / "data.jsonl"


def _read_features(config: Config, plan: dict[str, Any]) -> list[dict[str, Any]]:
    payloads = _files(config.root / "features")
    required = {"manifest.json", "data.jsonl"}
    allowed = required | {"cell_features.csv", "task_features.csv"}
    if not required.issubset(payloads) or not set(payloads).issubset(allowed):
        raise IntegrityError("features artifact set is invalid")
    manifest = strict_json(payloads["manifest.json"], "features manifest")
    if manifest != _manifest(plan, digest(payloads["data.jsonl"])):
        raise IntegrityError("features identity is invalid")
    expected = _feature_rows(config, plan, _durable_runs(config, plan))
    if payloads["data.jsonl"] != b"".join(canonical(row) for row in expected):
        raise IntegrityError("features do not match run evidence")
    return expected


def _load_gold(config: Config, plan: dict[str, Any]) -> tuple[bytes, dict[str, list[str]]]:
    raw, rows = _jsonl(config.gold, "gold JSONL")
    source_files = {
        task["task_id"]: set(task["source_files"]) - set(task["documentation"]["paths"])
        for task in plan["tasks"]
    }
    result: dict[str, list[str]] = {}
    for number, row in enumerate(rows, 1):
        row = _table(row, f"gold row {number}", {"task_id", "files"})
        task_id = _safe_id(row["task_id"], f"gold row {number} task_id")
        if task_id in result:
            raise PipelineError(f"duplicate gold task_id: {task_id}")
        files = _paths(row["files"], f"gold row {number} files")
        if not files:
            raise PipelineError("gold files must not be empty")
        if task_id not in source_files:
            raise PipelineError(f"gold contains unknown task_id: {task_id}")
        unknown = sorted(set(files) - source_files[task_id])
        if unknown:
            raise PipelineError(f"gold contains files outside the frozen source tree: {unknown}")
        result[task_id] = files
    return raw, result


def _python_symbols(path: Path) -> set[str]:
    if path.suffix not in {".py", ".pyi"}:
        return set()
    payload = _read_file(path, "gold source file")
    try:
        tree = ast.parse(payload, filename=str(path))
    except (SyntaxError, ValueError):
        return set()
    return {
        _normalized(node.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _has_explicit_gold_locator(prompt: str, gold_files: list[str], source_root: Path) -> bool:
    locators = _prompt_locators(prompt)
    for relative in gold_files:
        path = Path(relative)
        if (
            _normalized(path.as_posix()) in locators["paths"]
            or _normalized(path.name) in locators["filenames"]
        ):
            return True
        if _python_symbols(source_root / path).intersection(locators["symbols"]):
            return True
    return False


def _trajectory(events: bytes, task: dict[str, Any], gold: list[str]) -> dict[str, int]:
    documentation = set(task["documentation"]["paths"])
    source = set(task["source_files"]) - documentation
    gold_set = set(gold)
    source_action = 0
    seen_any = False
    seen_by_three = False
    targeted = False
    for command, output in _command_actions(events):
        documentation_paths = _mentioned_paths(command, documentation)
        source_paths = _mentioned_paths(command, source)
        if documentation_paths and not source_paths:
            continue
        source_action += 1
        seen = bool(_mentioned_paths(output, gold_set))
        seen_any = seen_any or seen
        seen_by_three = seen_by_three or (source_action <= 3 and seen)
        targeted = targeted or bool(_mentioned_paths(command, gold_set))
    return {
        "gold_seen_any": int(seen_any),
        "gold_seen_by_3_source_actions": int(seen_by_three),
        "gold_targeted_any": int(targeted),
    }


def _score(feature: dict[str, Any], gold_files: list[str]) -> dict[str, Any]:
    if feature["status"] == "terminal":
        return {
            "recall_at_3": None,
            "recall_at_5": None,
            "ndcg_at_3": None,
            "returned_set_f1": None,
        }
    gold = set(gold_files)
    predictions = feature["files"]
    relevance = [path in gold for path in predictions[:3]]
    matched_at_3 = sum(relevance)
    matched_at_5 = len(gold.intersection(predictions))
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(3, len(gold)) + 1))
    actual = sum(int(value) / log2(rank + 1) for rank, value in enumerate(relevance, 1))
    return {
        "recall_at_3": matched_at_3 / len(gold),
        "recall_at_5": matched_at_5 / len(gold),
        "ndcg_at_3": actual / ideal,
        "returned_set_f1": 2 * matched_at_5 / (len(predictions) + len(gold)),
    }


def _csv_bytes(columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    fieldnames = list(columns)
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})
    return stream.getvalue().encode("utf-8")


def _task_rows(plan: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "succeeded":
            grouped[row["task_id"], row["condition"]].append(row)
    result: list[dict[str, Any]] = []
    for task in plan["tasks"]:
        task_id = task["task_id"]
        task_types = {row["task_type"] for row in rows if row["task_id"] == task_id}
        if len(task_types) != 1:
            raise IntegrityError(f"task_type is inconsistent for task {task_id}")
        row: dict[str, Any] = {
            **identity(plan),
            "task_id": task_id,
            "repository": task["repository"],
            "task_type": task_types.pop(),
        }
        means: dict[tuple[str, str], float | None] = {}
        for condition in CONDITIONS:
            selected = grouped[task_id, condition]
            for metric in TASK_METRICS:
                values = [value for item in selected if (value := item[metric]) is not None]
                mean = fmean(values) if values else None
                means[condition, metric] = mean
                row[f"{_CONDITION_SLUGS[condition]}_mean_{metric}"] = mean
        for prefix, left, right in _CONTRASTS:
            for metric in TASK_METRICS:
                left_value = means[left, metric]
                right_value = means[right, metric]
                row[f"{prefix}_{metric}"] = (
                    left_value - right_value
                    if left_value is not None and right_value is not None
                    else None
                )
        result.append(row)
    return result


def _task_columns() -> tuple[str, ...]:
    columns = [
        "experiment_id",
        "experiment_version",
        "plan_id",
        "task_id",
        "repository",
        "task_type",
    ]
    columns.extend(
        f"{_CONDITION_SLUGS[condition]}_mean_{metric}"
        for condition in CONDITIONS
        for metric in TASK_METRICS
    )
    columns.extend(f"{prefix}_{metric}" for prefix, _, _ in _CONTRASTS for metric in TASK_METRICS)
    return tuple(columns)


def _aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        succeeded = [row for row in selected if row["status"] == "succeeded"]
        result.append(
            {
                "condition": condition,
                "planned_observations": len(selected),
                "successful_observations": len(succeeded),
                "terminal_observations": len(selected) - len(succeeded),
                **{
                    f"mean_{metric}": (
                        fmean(row[metric] for row in succeeded) if succeeded else None
                    )
                    for metric in (
                        "recall_at_3",
                        "recall_at_5",
                        "ndcg_at_3",
                        "returned_set_f1",
                    )
                },
            }
        )
    return result


def analyze(config_path: Path) -> tuple[dict[str, str], Path]:
    config, plan = _current(config_path)
    features_rows = _read_features(config, plan)
    gold_raw, gold = _load_gold(config, plan)
    if set(gold) != {task["task_id"] for task in plan["tasks"]}:
        raise PipelineError("gold task coverage must exactly match public tasks")
    source_roots = _task_source_roots(config)
    tasks = _task_map(plan)
    task_types = {
        task_id: (
            "EXPLICIT_LOCATOR_CLUE"
            if _has_explicit_gold_locator(tasks[task_id]["prompt"], files, source_roots[task_id])
            else "NO_EXPLICIT_LOCATOR_CLUE"
        )
        for task_id, files in gold.items()
    }
    rows: list[dict[str, Any]] = []
    for feature in features_rows:
        task_id = feature["task_id"]
        events = _read_file(
            config.root / "runs" / feature["cell_id"] / "events.jsonl", "Codex events"
        )
        rows.append(
            {
                **identity(plan),
                "cell_id": feature["cell_id"],
                "task_id": task_id,
                "repository": feature["repository"],
                "task_type": task_types[task_id],
                "condition": feature["condition"],
                "model": feature["model"],
                "reasoning_effort": feature["reasoning_effort"],
                "repeat": feature["repeat"],
                "status": feature["status"],
                "terminal_reason": feature["terminal_reason"],
                **_score(feature, gold[task_id]),
                "provider_total_tokens": feature["provider_total_tokens"],
                "elapsed_seconds": feature["elapsed_seconds"],
                "agent_step_count": feature["agent_step_count"],
                "wiki_read_count": feature["wiki_read_count"],
                "wiki_tokens": feature["wiki_tokens"],
                "unique_wiki_pages": feature["unique_wiki_pages"],
                "beyond_entry_reads": feature["beyond_entry_reads"],
                **_trajectory(events, tasks[task_id], gold[task_id]),
            }
        )
    tasks_rows = _task_rows(plan, rows)
    cell_csv = _csv_bytes(CELL_COLUMNS, rows)
    task_csv = _csv_bytes(_task_columns(), tasks_rows)
    _write_once(config.root / "features" / "cell_features.csv", cell_csv)
    _write_once(config.root / "features" / "task_features.csv", task_csv)
    analysis = {
        "schema_version": 1,
        **identity(plan),
        "gold_checksum": digest(gold_raw),
        "features_checksum": digest(b"".join(canonical(row) for row in features_rows)),
        "cell_features_checksum": digest(cell_csv),
        "task_features_checksum": digest(task_csv),
        "rows": rows,
        "task_rows": tasks_rows,
        "aggregates": _aggregates(rows),
    }
    data = canonical(analysis)
    root = config.root / "analysis"
    _publish(root, {"manifest.json": canonical(_manifest(plan, digest(data))), "data.json": data})
    return identity(plan), root / "data.json"


def _read_analysis(config: Config, plan: dict[str, Any]) -> dict[str, Any]:
    payloads = _files(config.root / "analysis")
    if set(payloads) != {"manifest.json", "data.json"}:
        raise IntegrityError("analysis artifact set is invalid")
    manifest = strict_json(payloads["manifest.json"], "analysis manifest")
    analysis = strict_json(payloads["data.json"], "analysis")
    if manifest != _manifest(plan, digest(payloads["data.json"])) or not isinstance(analysis, dict):
        raise IntegrityError("analysis manifest is invalid")
    if not _schema_one(analysis.get("schema_version")) or any(
        analysis.get(key) != value for key, value in identity(plan).items()
    ):
        raise IntegrityError("analysis identity is invalid")
    rows = analysis.get("rows")
    task_rows = analysis.get("task_rows")
    if not isinstance(rows, list) or len(rows) != len(plan["cells"]):
        raise IntegrityError("analysis rows are invalid")
    if not isinstance(task_rows, list) or len(task_rows) != len(plan["tasks"]):
        raise IntegrityError("analysis task rows are invalid")
    for row, cell in zip(rows, plan["cells"], strict=True):
        if not isinstance(row, dict) or any(
            row.get(key) != cell[key]
            for key in (
                "cell_id",
                "task_id",
                "condition",
                "model",
                "reasoning_effort",
                "repeat",
            )
        ):
            raise IntegrityError("analysis row does not match the plan")
    cell_csv = _read_file(config.root / "features" / "cell_features.csv", "cell feature table")
    task_csv = _read_file(config.root / "features" / "task_features.csv", "task feature table")
    if (
        cell_csv != _csv_bytes(CELL_COLUMNS, rows)
        or task_csv != _csv_bytes(_task_columns(), task_rows)
        or task_rows != _task_rows(plan, rows)
        or analysis.get("cell_features_checksum") != digest(cell_csv)
        or analysis.get("task_features_checksum") != digest(task_csv)
        or analysis.get("aggregates") != _aggregates(rows)
    ):
        raise IntegrityError("analysis tables do not match analysis data")
    feature_rows = _read_features(config, plan)
    if analysis.get("features_checksum") != digest(
        b"".join(canonical(row) for row in feature_rows)
    ):
        raise IntegrityError("analysis does not match persisted features")
    return analysis


def report(config_path: Path) -> tuple[dict[str, str], Path]:
    config = load_config(config_path)
    plan = _read_plan(config)
    analysis = _read_analysis(config, plan)
    data = canonical(
        {
            "schema_version": 1,
            **identity(plan),
            "dataset": plan["dataset"],
            "source_checksum": digest(canonical(analysis)),
            "cell_features": "features/cell_features.csv",
            "task_features": "features/task_features.csv",
            "rows": analysis["rows"],
            "task_rows": analysis["task_rows"],
            "aggregates": analysis["aggregates"],
        }
    )
    root = config.root / "report"
    _publish(root, {"manifest.json": canonical(_manifest(plan, digest(data))), "data.json": data})
    return identity(plan), root / "data.json"
