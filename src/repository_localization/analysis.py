"""Derive features, score predictions, and publish EDA data."""

from __future__ import annotations

import ast
import re
import unicodedata
from math import log2
from pathlib import Path
from statistics import fmean
from typing import Any

from repository_localization.core import (
    _SHA256,
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
    canonical,
    digest,
    load_config,
    strict_json,
)
from repository_localization.execution import _runs
from repository_localization.planning import _current, _read_plan, _task_map, identity

# Feature extraction from durable run evidence

_PATH = re.compile(r"(?<![A-Za-z0-9_./-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?![A-Za-z0-9_./-])")
_FILENAME = re.compile(
    r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_-]+\."
    r"(?:cfg|css|html?|ini|ipynb|js|json|md|pyi?|rst|sh|toml|ts|txt|ya?ml)"
    r"(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_BACKTICK_SYMBOL = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`")
_CALL_SYMBOL = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


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


def _feature_rows(plan: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repositories = {task["task_id"]: task["repository"] for task in plan["tasks"]}
    task_features = {task["task_id"]: _prompt_features(task["prompt"]) for task in plan["tasks"]}
    return [
        {
            "schema_version": 1,
            **identity(plan),
            **task_features[row["task_id"]],
            "cell_id": row["cell_id"],
            "task_id": row["task_id"],
            "repository": repositories[row["task_id"]],
            "condition": row["condition"],
            "repeat": row["repeat"],
            "status": row["status"],
            "terminal_reason": row.get("terminal_reason"),
            "files": row["files"],
            "input_tokens": row.get("input_tokens"),
            "cached_input_tokens": row.get("cached_input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "reasoning_output_tokens": row.get("reasoning_output_tokens"),
            "tool_steps": row.get("tool_steps"),
            "duration_ms": row["duration_ms"],
        }
        for row in runs
    ]


def features(config_path: Path) -> tuple[dict[str, str], Path]:
    config, plan = _current(config_path)
    rows = _feature_rows(plan, _durable_runs(config, plan))
    data = b"".join(canonical(row) for row in rows)
    root = config.root / "features"
    _publish(root, {"manifest.json": canonical(_manifest(plan, digest(data))), "data.jsonl": data})
    return identity(plan), root / "data.jsonl"


def _read_features(config: Config, plan: dict[str, Any]) -> list[dict[str, Any]]:
    root = config.root / "features"
    payloads = _files(root)
    if set(payloads) != {"manifest.json", "data.jsonl"}:
        raise IntegrityError("features artifact set is invalid")
    manifest = strict_json(payloads["manifest.json"], "features manifest")
    expected_manifest = _manifest(plan, digest(payloads["data.jsonl"]))
    if (
        not isinstance(manifest, dict)
        or not _schema_one(manifest.get("schema_version"))
        or manifest != expected_manifest
    ):
        raise IntegrityError("features identity is invalid")
    expected = _feature_rows(plan, _durable_runs(config, plan))
    expected_data = b"".join(canonical(row) for row in expected)
    if payloads["data.jsonl"] != expected_data:
        raise IntegrityError("features do not match run evidence")
    return expected


def _load_gold(config: Config, plan: dict[str, Any]) -> tuple[bytes, dict[str, list[str]]]:
    raw, rows = _jsonl(config.gold, "gold JSONL")
    source_files = {
        task["task_id"]: set(task["source_files"]) - {task["documentation"]["entry_path"]}
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


def _gold_locator_mentioned(prompt: str, gold_files: list[str], source_root: Path) -> bool:
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


def _gold_locator_mentions(
    config: Config, plan: dict[str, Any], gold: dict[str, list[str]]
) -> dict[str, bool]:
    source_roots = _task_source_roots(config)
    return {
        task["task_id"]: _gold_locator_mentioned(
            task["prompt"], gold[task["task_id"]], source_roots[task["task_id"]]
        )
        for task in plan["tasks"]
    }


# Gold-aware metrics


def analyze(config_path: Path) -> tuple[dict[str, str], Path]:
    config, plan = _current(config_path)
    feature_rows = _read_features(config, plan)
    gold_raw, gold = _load_gold(config, plan)
    expected_tasks = {task["task_id"] for task in plan["tasks"]}
    if set(gold) != expected_tasks:
        raise PipelineError("gold task coverage must exactly match public tasks")
    gold_mentions = _gold_locator_mentions(config, plan, gold)
    rows: list[dict[str, Any]] = []
    for feature in feature_rows:
        gold_files = set(gold[feature["task_id"]])
        common = {
            **identity(plan),
            "prompt_has_path": feature["prompt_has_path"],
            "prompt_has_filename": feature["prompt_has_filename"],
            "prompt_has_symbol": feature["prompt_has_symbol"],
            "gold_locator_mentioned": gold_mentions[feature["task_id"]],
            "cell_id": feature["cell_id"],
            "task_id": feature["task_id"],
            "repository": feature["repository"],
            "condition": feature["condition"],
            "repeat": feature["repeat"],
            "status": feature["status"],
            "terminal_reason": feature["terminal_reason"],
            "gold_file_count": len(gold_files),
            "input_tokens": feature["input_tokens"],
            "output_tokens": feature["output_tokens"],
            "tool_steps": feature["tool_steps"],
            "duration_ms": feature["duration_ms"],
        }
        if feature["status"] == "terminal":
            rows.append(
                {
                    **common,
                    "relevance_at_3": [],
                    "recall_at_3": None,
                    "ndcg_at_3": None,
                    "returned_set_f1": None,
                    "recall_at_5": None,
                    "matched_gold_at_3": 0,
                    "matched_gold_at_5": 0,
                    "predicted_file_count": 0,
                }
            )
            continue
        predictions = feature["files"]
        relevance_at_3 = [path in gold_files for path in predictions[:3]]
        matched_at_3 = sum(relevance_at_3)
        matched_at_5 = len(gold_files.intersection(predictions))
        ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, min(3, len(gold_files)) + 1))
        dcg = sum(int(relevant) / log2(rank + 1) for rank, relevant in enumerate(relevance_at_3, 1))
        rows.append(
            {
                **common,
                "relevance_at_3": relevance_at_3,
                "recall_at_3": matched_at_3 / len(gold_files),
                "ndcg_at_3": dcg / ideal_dcg,
                "returned_set_f1": 2 * matched_at_5 / (len(predictions) + len(gold_files)),
                "recall_at_5": matched_at_5 / len(gold_files),
                "matched_gold_at_3": matched_at_3,
                "matched_gold_at_5": matched_at_5,
                "predicted_file_count": len(predictions),
            }
        )
    aggregates = []
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        succeeded = [row for row in condition_rows if row["status"] == "succeeded"]
        aggregates.append(
            {
                "condition": condition,
                "planned_observations": len(condition_rows),
                "successful_observations": len(succeeded),
                "terminal_observations": len(condition_rows) - len(succeeded),
                "mean_recall_at_3": (
                    fmean(row["recall_at_3"] for row in succeeded) if succeeded else None
                ),
                "mean_ndcg_at_3": (
                    fmean(row["ndcg_at_3"] for row in succeeded) if succeeded else None
                ),
                "mean_returned_set_f1": (
                    fmean(row["returned_set_f1"] for row in succeeded) if succeeded else None
                ),
                "mean_recall_at_5": (
                    fmean(row["recall_at_5"] for row in succeeded) if succeeded else None
                ),
            }
        )
    analysis = {
        "schema_version": 1,
        **identity(plan),
        "gold_checksum": digest(gold_raw),
        "features_checksum": digest(b"".join(canonical(row) for row in feature_rows)),
        "rows": rows,
        "aggregates": aggregates,
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
    if not isinstance(manifest, dict) or not isinstance(analysis, dict):
        raise IntegrityError("analysis is invalid")
    if not _schema_one(manifest.get("schema_version")) or manifest != _manifest(
        plan, digest(payloads["data.json"])
    ):
        raise IntegrityError("analysis manifest is invalid")
    expected_analysis_keys = {
        "schema_version",
        *identity(plan),
        "gold_checksum",
        "features_checksum",
        "rows",
        "aggregates",
    }
    gold_checksum = analysis.get("gold_checksum")
    features_checksum = analysis.get("features_checksum")
    if (
        set(analysis) != expected_analysis_keys
        or not _schema_one(analysis.get("schema_version"))
        or any(analysis.get(key) != value for key, value in identity(plan).items())
        or not isinstance(gold_checksum, str)
        or _SHA256.fullmatch(gold_checksum) is None
        or not isinstance(features_checksum, str)
        or _SHA256.fullmatch(features_checksum) is None
    ):
        raise IntegrityError("analysis identity is invalid")
    feature_rows = _read_features(config, plan)
    if features_checksum != digest(b"".join(canonical(row) for row in feature_rows)):
        raise IntegrityError("analysis does not match persisted features")
    rows = analysis.get("rows")
    aggregates = analysis.get("aggregates")
    if (
        not isinstance(rows, list)
        or not isinstance(aggregates, list)
        or len(rows) != len(plan["cells"])
        or len(aggregates) != len(CONDITIONS)
    ):
        raise IntegrityError("analysis rows are invalid")
    row_keys = {
        *identity(plan),
        "prompt_has_path",
        "prompt_has_filename",
        "prompt_has_symbol",
        "gold_locator_mentioned",
        "cell_id",
        "task_id",
        "repository",
        "condition",
        "repeat",
        "status",
        "terminal_reason",
        "relevance_at_3",
        "recall_at_3",
        "ndcg_at_3",
        "returned_set_f1",
        "recall_at_5",
        "gold_file_count",
        "matched_gold_at_3",
        "matched_gold_at_5",
        "predicted_file_count",
        "input_tokens",
        "output_tokens",
        "tool_steps",
        "duration_ms",
    }
    tasks = _task_map(plan)
    gold_counts: dict[str, int] = {}
    gold_mentions: dict[str, bool] = {}
    for row, cell, feature in zip(rows, plan["cells"], feature_rows, strict=True):
        if not isinstance(row, dict) or set(row) != row_keys:
            raise IntegrityError("analysis row shape is invalid")
        if (
            any(row.get(key) != value for key, value in identity(plan).items())
            or any(
                row.get(key) != cell[key] for key in ("cell_id", "task_id", "condition", "repeat")
            )
            or row.get("repository") != tasks[cell["task_id"]]["repository"]
            or row.get("status") != feature["status"]
            or row.get("terminal_reason") != feature["terminal_reason"]
        ):
            raise IntegrityError("analysis row does not match the plan")
        prompt_feature_keys = (
            "prompt_has_path",
            "prompt_has_filename",
            "prompt_has_symbol",
        )
        if (
            any(row[key] != feature[key] for key in prompt_feature_keys)
            or type(row["gold_locator_mentioned"]) is not bool
        ):
            raise IntegrityError("analysis locator features are invalid")
        gold_count = row["gold_file_count"]
        matched_at_3 = row["matched_gold_at_3"]
        matched_at_5 = row["matched_gold_at_5"]
        predicted_count = row["predicted_file_count"]
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (gold_count, matched_at_3, matched_at_5, predicted_count)
            )
            or gold_count < 1
        ):
            raise IntegrityError("analysis row counts are invalid")
        if predicted_count != len(feature["files"]) or any(
            row[key] != feature[key]
            for key in ("input_tokens", "output_tokens", "tool_steps", "duration_ms")
        ):
            raise IntegrityError("analysis row does not match persisted features")
        previous_gold_count = gold_counts.setdefault(row["task_id"], gold_count)
        if previous_gold_count != gold_count:
            raise IntegrityError("analysis gold count is inconsistent")
        previous_gold_mention = gold_mentions.setdefault(
            row["task_id"], row["gold_locator_mentioned"]
        )
        if previous_gold_mention != row["gold_locator_mentioned"]:
            raise IntegrityError("analysis gold locator feature is inconsistent")
        relevance = row["relevance_at_3"]
        if row["status"] == "terminal":
            if (
                not isinstance(row["terminal_reason"], str)
                or not row["terminal_reason"]
                or matched_at_3 != 0
                or matched_at_5 != 0
                or predicted_count != 0
                or relevance != []
                or any(
                    row[key] is not None
                    for key in (
                        "recall_at_3",
                        "ndcg_at_3",
                        "returned_set_f1",
                        "recall_at_5",
                        "input_tokens",
                        "output_tokens",
                        "tool_steps",
                    )
                )
            ):
                raise IntegrityError("terminal analysis row is invalid")
            duration = row["duration_ms"]
            if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
                raise IntegrityError("analysis duration is invalid")
            continue
        if row["status"] != "succeeded" or row["terminal_reason"] is not None:
            raise IntegrityError("analysis status is invalid")
        if not (
            0 <= matched_at_3 <= min(3, gold_count, predicted_count)
            and matched_at_3 <= matched_at_5 <= min(gold_count, predicted_count)
            and 1 <= predicted_count <= 5
        ):
            raise IntegrityError("successful analysis row counts are invalid")
        if (
            not isinstance(relevance, list)
            or len(relevance) != min(3, predicted_count)
            or any(type(value) is not bool for value in relevance)
            or sum(relevance) != matched_at_3
        ):
            raise IntegrityError("analysis rank relevance is invalid")
        ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, min(3, gold_count) + 1))
        dcg = sum(int(relevant) / log2(rank + 1) for rank, relevant in enumerate(relevance, 1))
        expected_metrics = {
            "recall_at_3": matched_at_3 / gold_count,
            "ndcg_at_3": dcg / ideal_dcg,
            "returned_set_f1": 2 * matched_at_5 / (predicted_count + gold_count),
            "recall_at_5": matched_at_5 / gold_count,
        }
        if any(
            isinstance(row[key], bool) or not isinstance(row[key], int | float) or row[key] != value
            for key, value in expected_metrics.items()
        ):
            raise IntegrityError("analysis row metric is invalid")
        for key in ("input_tokens", "output_tokens"):
            value = row[key]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise IntegrityError("analysis token count is invalid")
        for key in ("tool_steps", "duration_ms"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise IntegrityError("analysis resource count is invalid")
    expected_aggregates = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        succeeded = [row for row in selected if row["status"] == "succeeded"]
        expected_aggregates.append(
            {
                "condition": condition,
                "planned_observations": len(selected),
                "successful_observations": len(succeeded),
                "terminal_observations": len(selected) - len(succeeded),
                "mean_recall_at_3": (
                    fmean(row["recall_at_3"] for row in succeeded) if succeeded else None
                ),
                "mean_ndcg_at_3": (
                    fmean(row["ndcg_at_3"] for row in succeeded) if succeeded else None
                ),
                "mean_returned_set_f1": (
                    fmean(row["returned_set_f1"] for row in succeeded) if succeeded else None
                ),
                "mean_recall_at_5": (
                    fmean(row["recall_at_5"] for row in succeeded) if succeeded else None
                ),
            }
        )
    if aggregates != expected_aggregates:
        raise IntegrityError("analysis aggregates are invalid")
    return analysis


# EDA data


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
            "rows": analysis["rows"],
            "aggregates": analysis["aggregates"],
        }
    )
    root = config.root / "report"
    _publish(root, {"manifest.json": canonical(_manifest(plan, digest(data))), "data.json": data})
    return identity(plan), root / "data.json"
