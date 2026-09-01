"""Validate Git-backed inputs and freeze the readable experiment plan."""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from repository_localization.core import (
    CONDITIONS,
    DATASET,
    RUNNER_CONTRACT,
    WIKI_TOKENIZER,
    Config,
    IntegrityError,
    PipelineError,
    _executable_version,
    _git_commit,
    _jsonl,
    _no_link_components,
    _publish,
    _read_file,
    _relative_path,
    _safe_id,
    _schema_one,
    _table,
    _text,
    canonical,
    load_config,
    strict_json,
)

GIT = "/usr/bin/git"


def _git(repository: Path, *arguments: str, label: str) -> bytes:
    try:
        _no_link_components(repository, f"{label} repository")
        if not stat.S_ISDIR(repository.lstat().st_mode):
            raise OSError("not a directory")
        completed = subprocess.run(
            [GIT, "-C", str(repository), *arguments],
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PipelineError(f"{label}: cannot read Git repository {repository}") from exc
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PipelineError(f"{label}: Git rejected the repository or commit: {message}")
    return completed.stdout


def _git_blob(repository: Path, commit: str, path: str) -> bytes:
    return _git(repository, "show", f"{commit}:{path}", label=f"Git blob {path}")


def _tree(repository: Path, commit: str, label: str) -> list[dict[str, str]]:
    _git(repository, "cat-file", "-e", f"{commit}^{{commit}}", label=label)
    payload = _git(repository, "ls-tree", "-rz", "--full-tree", commit, label=label)
    entries: list[dict[str, str]] = []
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, kind, _object_id = metadata.decode("ascii").split(" ", 2)
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise PipelineError(f"{label}: unsupported Git tree entry") from exc
        normalized = _relative_path(path, f"{label} path")
        parts = PurePosixPath(normalized).parts
        if parts[0] in {".git", ".codex", ".agents", ".experiment"}:
            raise PipelineError(f"{label}: forbidden control path: {normalized}")
        if parts[-1].startswith("AGENTS") and parts[-1].endswith(".md"):
            raise PipelineError(f"{label}: competing instructions: {normalized}")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise PipelineError(f"{label}: unsupported linked or embedded entry: {normalized}")
        entries.append({"path": normalized, "mode": mode})
    if not entries:
        raise PipelineError(f"{label}: Git commit contains no files")
    return entries


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
            f"Your first tool call must read only `{entry_path}` and return its contents. "
            "Do not run pwd, ls, find, a path search, or any other tool first. After that "
            "documentation read, decide freely whether to continue with documentation or inspect "
            "source code.\n"
        )
    raise IntegrityError(f"unknown experiment condition: {condition}")


def _load_tasks(config: Config) -> list[dict[str, Any]]:
    _, rows = _jsonl(config.tasks, "tasks JSONL")
    repositories = {repository.name: repository.path for repository in config.repositories}
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    trees: dict[tuple[str, str], list[dict[str, str]]] = {}
    for number, row in enumerate(rows, 1):
        row = _table(
            row,
            f"tasks row {number}",
            {"task_id", "repository", "base_commit", "prompt", "documentation_entry"},
        )
        task_id = _safe_id(row["task_id"], f"tasks row {number} task_id")
        if task_id in seen:
            raise PipelineError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        repository_name = _text(row["repository"], "repository", single_line=True)
        repository_path = repositories.get(repository_name)
        if repository_path is None:
            raise PipelineError(f"{task_id}: repository is not configured: {repository_name}")
        base_commit = _git_commit(row["base_commit"], "base_commit")
        key = (repository_name, base_commit)
        if key not in trees:
            trees[key] = _tree(repository_path, base_commit, f"{task_id} source")
        entries = trees[key]
        source_files = [entry["path"] for entry in entries]
        entry_path = _relative_path(row["documentation_entry"], "documentation_entry")
        if entry_path not in source_files or not _git_blob(
            repository_path, base_commit, entry_path
        ):
            raise PipelineError(f"{task_id}: documentation_entry is missing or empty")
        documentation_parent = PurePosixPath(entry_path).parent
        if documentation_parent == PurePosixPath("."):
            documentation_paths = [entry_path]
        else:
            prefix = f"{documentation_parent.as_posix()}/"
            documentation_paths = [path for path in source_files if path.startswith(prefix)]
        prompt = _text(row["prompt"], "prompt")
        tasks.append(
            {
                "task_id": task_id,
                "repository": repository_name,
                "repository_path": str(repository_path),
                "base_commit": base_commit,
                "prompt": prompt,
                "source_files": source_files,
                "documentation": {"entry_path": entry_path, "paths": documentation_paths},
                "guidance": {
                    condition: _guidance(condition, entry_path) for condition in CONDITIONS
                },
            }
        )
    return tasks


def build_plan(config_path: Path) -> tuple[Config, dict[str, Any], bytes]:
    config = load_config(config_path)
    tasks = _load_tasks(config)
    cells: list[dict[str, Any]] = []
    ordinal = 0
    for task in tasks:
        for profile in config.profiles:
            for repeat in range(1, config.repeats + 1):
                for condition in CONDITIONS:
                    ordinal += 1
                    cells.append(
                        {
                            "cell_id": f"cell-{ordinal:06d}",
                            "task_id": task["task_id"],
                            "condition": condition,
                            **profile.as_dict(),
                            "repeat": repeat,
                        }
                    )
    plan = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "experiment_version": config.experiment_version,
        "dataset": {**DATASET, "revision": config.dataset_revision},
        "config_path": str(config.path),
        "tasks_path": str(config.tasks),
        "runner_contract": RUNNER_CONTRACT,
        "wiki_tokenizer": WIKI_TOKENIZER,
        "runner": {
            "binary": str(config.binary),
            "version": _executable_version(config.binary),
            "profiles": [profile.as_dict() for profile in config.profiles],
            "timeout_seconds": config.timeout_seconds,
        },
        "repositories": [repository.as_dict() for repository in config.repositories],
        "conditions": list(CONDITIONS),
        "repeats": config.repeats,
        "tasks": tasks,
        "cells": cells,
    }
    return config, plan, canonical(plan)


def identity(plan: dict[str, Any]) -> dict[str, str]:
    return {
        "experiment_id": plan["experiment_id"],
        "experiment_version": plan["experiment_version"],
    }


def prepare(config_path: Path) -> tuple[dict[str, str], Path]:
    config, plan, plan_bytes = build_plan(config_path)
    if config.root.exists() or config.root.is_symlink():
        if _read_file(config.root / "plan.json", "prepared plan") != plan_bytes:
            raise IntegrityError(
                "experiment_id/experiment_version already belongs to a different Git-backed plan"
            )
    else:
        _publish(config.root, {"plan.json": plan_bytes})
    return identity(plan), config.root / "plan.json"


def _read_plan(config: Config) -> dict[str, Any]:
    plan = strict_json(_read_file(config.root / "plan.json", "prepared plan"), "prepared plan")
    if (
        not isinstance(plan, dict)
        or not _schema_one(plan.get("schema_version"))
        or plan.get("experiment_id") != config.experiment_id
        or plan.get("experiment_version") != config.experiment_version
        or plan.get("config_path") != str(config.path)
    ):
        raise IntegrityError("prepared plan has the wrong experiment identity")
    return plan


def _current(config_path: Path) -> tuple[Config, dict[str, Any]]:
    config, current, current_bytes = build_plan(config_path)
    frozen = _read_plan(config)
    if canonical(frozen) != current_bytes:
        raise IntegrityError(
            "config, tasks, repository commits, or runner changed under this experiment_version"
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
