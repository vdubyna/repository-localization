"""Validate inputs, freeze the experiment plan, and expose plan lookups."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from repository_localization.core import (
    _SHA256,
    CONDITIONS,
    DATASET,
    RUNNER_CONTRACT,
    WIKI_TOKENIZER,
    Config,
    IntegrityError,
    PipelineError,
    _absolute,
    _git_commit,
    _jsonl,
    _no_link_components,
    _publish,
    _read_executable,
    _read_file,
    _relative_path,
    _safe_id,
    _schema_one,
    _table,
    _text,
    canonical,
    digest,
    load_config,
    strict_json,
)

# Repository snapshot and task validation


def _inspect_tree(root: Path, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        _no_link_components(root, f"{label} root")
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise OSError("not directory")
    except OSError as exc:
        raise PipelineError(f"{label} root is missing, linked, or not a directory") from exc
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: PurePosixPath) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise PipelineError(f"cannot scan {label} root: {directory}") from exc
        for entry in children:
            child = relative / entry.name
            name = child.as_posix()
            if child.parts[0] in {".git", ".codex", ".agents", ".experiment"}:
                raise PipelineError(f"{label} root contains forbidden control path: {name}")
            if entry.name.startswith("AGENTS") and entry.name.endswith(".md"):
                raise PipelineError(f"{label} root contains competing instructions: {name}")
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise PipelineError(f"cannot inspect {label} entry: {name}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise PipelineError(f"{label} root contains symlink: {name}")
            if stat.S_ISDIR(info.st_mode):
                entries.append({"path": name, "type": "dir", "mode": "040000"})
                visit(Path(entry.path), child)
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise PipelineError(f"{label} entry is not a plain single-link file: {name}")
            try:
                payload = Path(entry.path).read_bytes()
            except OSError as exc:
                raise PipelineError(f"cannot read {label} file: {name}") from exc
            entries.append(
                {
                    "path": name,
                    "type": "file",
                    "mode": "100755" if info.st_mode & 0o111 else "100644",
                    "bytes": len(payload),
                    "sha256": digest(payload),
                }
            )

    visit(root, PurePosixPath())
    files = [entry for entry in entries if entry["type"] == "file"]
    if not files:
        raise PipelineError(f"{label} root contains no files")
    entries.sort(key=lambda entry: entry["path"])
    manifest = b"".join(canonical(entry) for entry in entries)
    binding = {
        "root": str(root),
        "tree_checksum": digest(manifest),
        "file_count": len(files),
        "byte_count": sum(entry["bytes"] for entry in files),
    }
    return binding, entries


def _guidance(condition: str, entry_path: str) -> str | None:
    if condition == "NO-DOC":
        return None
    header = (
        "## Repository documentation\n\n"
        f"The repository's native functional documentation starts at `{entry_path}`.\n"
    )
    if condition == "OPTIONAL":
        return header + (
            "You may consult it if it seems useful for locating the implementation files "
            "relevant to the task.\n"
        )
    if condition == "DOC-FIRST":
        return header + (
            "Before searching, listing, or reading source-code paths or contents, read this "
            "documentation entry. After that first documentation read, decide freely whether "
            "to continue with documentation or inspect source code.\n"
        )
    raise IntegrityError(f"unknown experiment condition: {condition}")


def _load_tasks(config: Config) -> tuple[bytes, list[dict[str, Any]]]:
    raw, rows = _jsonl(config.tasks, "tasks JSONL")
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, 1):
        row = _table(
            row,
            f"tasks row {number}",
            {
                "task_id",
                "repository",
                "base_commit",
                "prompt",
                "source_root",
                "documentation_entry",
            },
        )
        task_id = _safe_id(row["task_id"], f"tasks row {number} task_id")
        if task_id in seen:
            raise PipelineError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        source_root = _absolute(config.path.parent, row["source_root"], "source_root")
        if config.gold == source_root or config.gold.is_relative_to(source_root):
            raise PipelineError(f"{task_id}: gold must be outside source_root")
        if (
            config.artifact_dir == source_root
            or config.artifact_dir.is_relative_to(source_root)
            or source_root.is_relative_to(config.artifact_dir)
        ):
            raise PipelineError(f"{task_id}: artifact_dir must be outside source_root")
        source, entries = _inspect_tree(source_root, "source")
        entry_path = _relative_path(row["documentation_entry"], "documentation_entry")
        if any(character in entry_path for character in ("`", "\n", "\r")):
            raise PipelineError("documentation_entry contains unsafe Markdown characters")
        documentation_entry = next(
            (entry for entry in entries if entry["type"] == "file" and entry["path"] == entry_path),
            None,
        )
        if documentation_entry is None or documentation_entry["bytes"] == 0:
            raise PipelineError(f"{task_id}: documentation_entry is missing or empty")
        documentation_parent = PurePosixPath(entry_path).parent
        if documentation_parent == PurePosixPath("."):
            documentation_paths = [entry_path]
        else:
            prefix = f"{documentation_parent.as_posix()}/"
            documentation_paths = [
                entry["path"]
                for entry in entries
                if entry["type"] == "file" and entry["path"].startswith(prefix)
            ]
        tasks.append(
            {
                "task_id": task_id,
                "repository": _text(row["repository"], "repository", single_line=True),
                "base_commit": _git_commit(row["base_commit"], "base_commit"),
                "prompt": _text(row["prompt"], "prompt"),
                "prompt_checksum": digest(_text(row["prompt"], "prompt").encode()),
                "source": source,
                "source_files": [entry["path"] for entry in entries if entry["type"] == "file"],
                "documentation": {
                    "entry_path": entry_path,
                    "entry_checksum": documentation_entry["sha256"],
                    "entry_bytes": documentation_entry["bytes"],
                    "paths": documentation_paths,
                },
                "guidance": {
                    condition: _guidance(condition, entry_path) for condition in CONDITIONS
                },
            }
        )
    return raw, tasks


def build_plan(config_path: Path) -> tuple[Config, dict[str, Any], bytes]:
    config = load_config(config_path)
    tasks_raw, tasks = _load_tasks(config)
    binary, binary_runtime = _read_executable(config.binary)
    cells: list[dict[str, Any]] = []
    seed = {
        "runner_contract": RUNNER_CONTRACT,
        "wiki_tokenizer": WIKI_TOKENIZER,
        "experiment_id": config.experiment_id,
        "experiment_version": config.experiment_version,
        "dataset": {**DATASET, "revision": config.dataset_revision},
        "config_checksum": digest(config.raw),
        "tasks_checksum": digest(tasks_raw),
        "tasks": tasks,
        "runner": {
            "binary": str(config.binary),
            "binary_checksum": digest(binary),
            "binary_runtime": binary_runtime,
            "profiles": [profile.as_dict() for profile in config.profiles],
            "timeout_seconds": config.timeout_seconds,
        },
        "repeats": config.repeats,
    }
    seed_checksum = digest(canonical(seed))
    for task in tasks:
        for profile in config.profiles:
            for repeat in range(1, config.repeats + 1):
                for condition in CONDITIONS:
                    body = {
                        "task_id": task["task_id"],
                        "condition": condition,
                        **profile.as_dict(),
                        "repeat": repeat,
                    }
                    cells.append(
                        {
                            "cell_id": digest(canonical({"seed": seed_checksum, **body})),
                            **body,
                        }
                    )
    body = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "experiment_version": config.experiment_version,
        "dataset": seed["dataset"],
        "config_path": str(config.path),
        "config_checksum": digest(config.raw),
        "tasks_path": str(config.tasks),
        "tasks_checksum": digest(tasks_raw),
        "runner_contract": RUNNER_CONTRACT,
        "wiki_tokenizer": seed["wiki_tokenizer"],
        "runner": seed["runner"],
        "conditions": list(CONDITIONS),
        "repeats": config.repeats,
        "tasks": tasks,
        "cells": cells,
    }
    plan = {**body, "plan_id": digest(canonical(body))}
    return config, plan, canonical(plan)


# Frozen plan identity and lookup


def identity(plan: dict[str, Any]) -> dict[str, str]:
    return {
        "experiment_id": plan["experiment_id"],
        "experiment_version": plan["experiment_version"],
        "plan_id": plan["plan_id"],
    }


def prepare(config_path: Path) -> tuple[dict[str, str], Path]:
    config, plan, plan_bytes = build_plan(config_path)
    if config.root.exists() or config.root.is_symlink():
        if _read_file(config.root / "plan.json", "prepared plan") != plan_bytes:
            raise IntegrityError(
                "experiment_id/experiment_version already belongs to a different plan"
            )
    else:
        _publish(config.root, {"plan.json": plan_bytes})
    return identity(plan), config.root / "plan.json"


def _read_plan(config: Config) -> dict[str, Any]:
    payload = _read_file(config.root / "plan.json", "prepared plan")
    plan = strict_json(payload, "prepared plan")
    if (
        not isinstance(plan, dict)
        or not _schema_one(plan.get("schema_version"))
        or not isinstance(plan.get("plan_id"), str)
    ):
        raise IntegrityError("prepared plan is invalid")
    body = dict(plan)
    plan_id = body.pop("plan_id")
    if _SHA256.fullmatch(plan_id) is None or digest(canonical(body)) != plan_id:
        raise IntegrityError("prepared plan checksum is invalid")
    if (
        plan.get("experiment_id") != config.experiment_id
        or plan.get("experiment_version") != config.experiment_version
        or plan.get("config_path") != str(config.path)
        or plan.get("config_checksum") != digest(config.raw)
    ):
        raise IntegrityError("config does not match the frozen experiment version")
    return plan


def _current(config_path: Path) -> tuple[Config, dict[str, Any]]:
    config, current, current_bytes = build_plan(config_path)
    frozen = _read_plan(config)
    if canonical(frozen) != current_bytes:
        raise IntegrityError(
            "config, tasks, source snapshots, or runner changed under the frozen experiment_version"
        )
    return config, frozen


def _cell_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise IntegrityError("plan cells are invalid")
    result = {cell["cell_id"]: cell for cell in cells if isinstance(cell, dict)}
    if len(result) != len(cells):
        raise IntegrityError("plan cells are duplicated or invalid")
    return result


def _task_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise IntegrityError("plan tasks are invalid")
    result = {task["task_id"]: task for task in tasks if isinstance(task, dict)}
    if len(result) != len(tasks):
        raise IntegrityError("plan tasks are duplicated or invalid")
    return result
