"""Minimal config-driven experiment pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass
from math import log2
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any
from uuid import uuid4

CONDITIONS = (
    "NO_DOC_GUIDANCE",
    "FUNCTIONAL_OPTIONAL",
    "FUNCTIONAL_REQUIRED_BEFORE_SOURCE",
)
RUNNER_CONTRACT = "repository-localization-runner-v3"
DATASET = {
    "name": "Contextbench/ContextBench",
    "config": "default",
    "split": "train",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PipelineError(RuntimeError):
    """Expected operator-facing pipeline failure."""

    exit_code = 2


class StateError(PipelineError):
    exit_code = 3


class IntegrityError(PipelineError):
    exit_code = 4


class ExecutionError(PipelineError):
    exit_code = 5


@dataclass(frozen=True, slots=True)
class Config:
    path: Path
    raw: bytes
    experiment_id: str
    experiment_version: str
    artifact_dir: Path
    tasks: Path
    gold: Path
    dataset_revision: str
    repeats: int
    binary: Path
    model: str
    reasoning_effort: str
    timeout_seconds: int

    @property
    def root(self) -> Path:
        return self.artifact_dir / self.experiment_id / self.experiment_version


@dataclass(frozen=True, slots=True)
class Evidence:
    observation: dict[str, Any]
    events: bytes
    stderr: bytes
    final_output: bytes


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _schema_one(value: object) -> bool:
    return type(value) is int and value == 1


def strict_json(payload: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise IntegrityError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise IntegrityError(f"{label}: invalid JSON number {value}")

    try:
        return json.loads(payload.decode(), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"{label}: invalid UTF-8 JSON") from exc


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PipelineError(f"{label} must be a filesystem-safe identifier")
    return value


def _git_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise PipelineError(f"{label} must be a full lowercase Git commit")
    return value


def _text(value: object, label: str, *, single_line: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{label} must be a non-empty string")
    result = value.strip()
    if single_line and ("\n" in result or "|" in result):
        raise PipelineError(f"{label} must be one line without pipes")
    return result


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise PipelineError(f"{label} must be an integer from {minimum} to {maximum}")
    return value


def _table(value: object, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise PipelineError(f"{label} must contain exactly: {', '.join(sorted(keys))}")
    return value


def _absolute(base: Path, value: object, label: str) -> Path:
    raw = _text(value, label)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(path))


def _no_link_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            current /= part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise OSError("symlink")
    except OSError as exc:
        raise IntegrityError(f"{label}: missing or linked path component: {current}") from exc


def _read_file(path: Path, label: str) -> bytes:
    try:
        _no_link_components(path, label)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise OSError("not regular")
        return path.read_bytes()
    except OSError as exc:
        raise PipelineError(f"{label}: missing, linked, or unreadable: {path}") from exc


def _read_executable(path: Path) -> tuple[bytes, dict[str, str]]:
    payload = _read_file(path, "Codex binary")
    if not os.access(path, os.X_OK):
        raise PipelineError(f"Codex binary is not executable: {path}")
    runtime = {"kind": "native"}
    if payload.startswith(b"#!"):
        try:
            shebang = payload.splitlines()[0][2:].decode().strip().split()
        except UnicodeDecodeError as exc:
            raise PipelineError("Codex binary has an invalid shebang") from exc
        if len(shebang) != 1 or not shebang[0].startswith("/"):
            raise PipelineError("Codex script must use one absolute interpreter without arguments")
        interpreter = Path(shebang[0])
        if interpreter.name == "env":
            raise PipelineError("Codex script must not depend on /usr/bin/env or ambient PATH")
        interpreter_bytes = _read_file(interpreter, "Codex script interpreter")
        if not os.access(interpreter, os.X_OK):
            raise PipelineError(f"Codex script interpreter is not executable: {interpreter}")
        runtime = {
            "kind": "script",
            "interpreter": str(interpreter),
            "interpreter_checksum": digest(interpreter_bytes),
        }
    return payload, runtime


def load_config(path: Path) -> Config:
    path = Path(os.path.abspath(path.expanduser()))
    raw = _read_file(path, "experiment TOML")
    try:
        decoded = tomllib.loads(raw.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PipelineError("experiment TOML is invalid") from exc
    top = _table(
        decoded,
        "experiment TOML",
        {
            "schema_version",
            "experiment_id",
            "experiment_version",
            "artifact_dir",
            "inputs",
            "design",
            "runner",
        },
    )
    if not _schema_one(top["schema_version"]):
        raise PipelineError("schema_version must be 1")
    inputs = _table(top["inputs"], "inputs", {"tasks", "gold", "dataset_revision"})
    design = _table(top["design"], "design", {"repeats"})
    runner = _table(
        top["runner"],
        "runner",
        {"binary", "model", "reasoning_effort", "timeout_seconds"},
    )
    effort = _text(runner["reasoning_effort"], "runner.reasoning_effort")
    if effort not in {"low", "medium", "high", "xhigh"}:
        raise PipelineError("runner.reasoning_effort must be low, medium, high, or xhigh")
    base = path.parent
    return Config(
        path=path,
        raw=raw,
        experiment_id=_safe_id(top["experiment_id"], "experiment_id"),
        experiment_version=_safe_id(top["experiment_version"], "experiment_version"),
        artifact_dir=_absolute(base, top["artifact_dir"], "artifact_dir"),
        tasks=_absolute(base, inputs["tasks"], "inputs.tasks"),
        gold=_absolute(base, inputs["gold"], "inputs.gold"),
        dataset_revision=_git_commit(inputs["dataset_revision"], "inputs.dataset_revision"),
        repeats=_integer(design["repeats"], "design.repeats", 1, 20),
        binary=_absolute(base, runner["binary"], "runner.binary"),
        model=_text(runner["model"], "runner.model"),
        reasoning_effort=effort,
        timeout_seconds=_integer(runner["timeout_seconds"], "runner.timeout_seconds", 1, 7200),
    )


def _jsonl(path: Path, label: str) -> tuple[bytes, list[dict[str, Any]]]:
    raw = _read_file(path, label)
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise PipelineError(f"{label}: blank row {number}")
        value = strict_json(line, f"{label} row {number}")
        if not isinstance(value, dict):
            raise PipelineError(f"{label}: row {number} must be an object")
        rows.append(value)
    if not rows:
        raise PipelineError(f"{label} must not be empty")
    return raw, rows


def _relative_path(value: object, label: str) -> str:
    raw = _text(value, label)
    if "\\" in raw:
        raise PipelineError(f"{label} must use forward slashes")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise PipelineError(f"{label} must be a normalized repository-relative path")
    return path.as_posix()


def _paths(value: object, label: str, *, maximum: int | None = None) -> list[str]:
    if not isinstance(value, list) or (maximum is not None and len(value) > maximum):
        limit = f"at most {maximum} paths" if maximum is not None else "paths"
        raise PipelineError(f"{label} must be a list of {limit}")
    paths = [_relative_path(item, label) for item in value]
    if len(paths) != len(set(paths)):
        raise PipelineError(f"{label} paths must be unique")
    return paths


def _prediction(value: object, label: str) -> list[str]:
    paths = _paths(value, label, maximum=5)
    if not paths:
        raise PipelineError(f"{label} must contain at least one path")
    return paths


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
    if condition == "NO_DOC_GUIDANCE":
        return None
    header = (
        "## Repository documentation\n\n"
        f"The repository's native functional documentation starts at `{entry_path}`.\n"
    )
    if condition == "FUNCTIONAL_OPTIONAL":
        return header + (
            "You may consult it if it seems useful for locating the implementation files "
            "relevant to the task.\n"
        )
    if condition == "FUNCTIONAL_REQUIRED_BEFORE_SOURCE":
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
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "timeout_seconds": config.timeout_seconds,
        },
        "repeats": config.repeats,
    }
    seed_checksum = digest(canonical(seed))
    for task in tasks:
        for repeat in range(1, config.repeats + 1):
            for condition in CONDITIONS:
                cells.append(
                    {
                        "cell_id": digest(
                            canonical(
                                {
                                    "seed": seed_checksum,
                                    "task_id": task["task_id"],
                                    "condition": condition,
                                    "repeat": repeat,
                                }
                            )
                        ),
                        "task_id": task["task_id"],
                        "condition": condition,
                        "repeat": repeat,
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
        "runner": seed["runner"],
        "conditions": list(CONDITIONS),
        "repeats": config.repeats,
        "tasks": tasks,
        "cells": cells,
    }
    plan = {**body, "plan_id": digest(canonical(body))}
    return config, plan, canonical(plan)


def identity(plan: dict[str, Any]) -> dict[str, str]:
    return {
        "experiment_id": plan["experiment_id"],
        "experiment_version": plan["experiment_version"],
        "plan_id": plan["plan_id"],
    }


def _ensure_directory(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists() and not current.is_symlink():
        missing.append(current)
        current = current.parent
    _no_link_components(current, "artifact parent")
    if not current.is_dir():
        raise IntegrityError(f"artifact parent is not a directory: {current}")
    try:
        for directory in reversed(missing):
            directory.mkdir(mode=0o700)
            if directory.is_symlink() or not directory.is_dir():
                raise OSError("unsafe directory")
            _fsync_directory(directory.parent)
    except OSError as exc:
        raise IntegrityError(f"unsafe artifact directory: {directory}") from exc


def _files(path: Path) -> dict[str, bytes]:
    if path.is_symlink() or not path.is_dir():
        raise IntegrityError(f"unsafe artifact directory: {path}")
    result: dict[str, bytes] = {}
    for child in path.iterdir():
        if child.is_symlink() or not child.is_file():
            raise IntegrityError(f"unexpected artifact entry: {child}")
        result[child.name] = _read_file(child, "artifact")
    return result


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise IntegrityError(f"cannot fsync artifact directory: {path}") from exc


def _publish(path: Path, files: dict[str, bytes]) -> None:
    _ensure_directory(path.parent)
    if path.exists() or path.is_symlink():
        if _files(path) != files:
            raise IntegrityError(f"immutable artifact conflicts with existing bytes: {path}")
        return
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        for name, payload in files.items():
            target = temporary / name
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            target.chmod(0o444)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(temporary, path)
        path.chmod(0o700)
        _fsync_directory(path.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_once(path: Path, payload: bytes) -> bool:
    _ensure_directory(path.parent)
    if path.exists() or path.is_symlink():
        if _read_file(path, "immutable artifact") != payload:
            raise IntegrityError(f"immutable artifact conflicts with existing bytes: {path}")
        return False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise IntegrityError(f"cannot publish immutable artifact: {path}") from exc
    _fsync_directory(path.parent)
    return True


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


def _claims(root: Path, plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    directory = root / "claims"
    if not directory.exists():
        return {}
    cells = _cell_map(plan)
    if directory.is_symlink() or not directory.is_dir():
        raise IntegrityError("claims directory is unsafe")
    result: dict[str, dict[str, Any]] = {}
    expected_identity = identity(plan)
    for path in sorted(directory.iterdir()):
        claim = strict_json(_read_file(path, "cell claim"), "cell claim")
        if not isinstance(claim, dict) or path.name != f"{claim.get('cell_id')}.json":
            raise IntegrityError("cell claim is invalid")
        cell = cells.get(claim["cell_id"])
        if cell is None or any(claim.get(key) != value for key, value in expected_identity.items()):
            raise IntegrityError("cell claim has the wrong experiment identity")
        if any(claim.get(key) != cell[key] for key in ("task_id", "condition", "repeat")):
            raise IntegrityError("cell claim does not match the plan")
        if not _schema_one(claim.get("schema_version")) or claim != {
            "schema_version": 1,
            **expected_identity,
            **cell,
        }:
            raise IntegrityError("cell claim shape is invalid")
        result[claim["cell_id"]] = claim
    return result


def _runs(root: Path, plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    directory = root / "runs"
    if not directory.exists():
        return {}
    if directory.is_symlink() or not directory.is_dir():
        raise IntegrityError("runs directory is unsafe")
    cells = _cell_map(plan)
    expected_identity = identity(plan)
    result: dict[str, dict[str, Any]] = {}
    names = {
        "manifest.json",
        "observation.json",
        "events.jsonl",
        "stderr.log",
        "final-output.json",
    }
    for run_root in sorted(directory.iterdir()):
        if run_root.is_symlink() or not run_root.is_dir() or set(_files(run_root)) != names:
            raise IntegrityError(f"run artifact set is invalid: {run_root}")
        payloads = _files(run_root)
        observation = strict_json(payloads["observation.json"], "observation")
        manifest = strict_json(payloads["manifest.json"], "run manifest")
        if not isinstance(observation, dict) or run_root.name != observation.get("cell_id"):
            raise IntegrityError("observation identity is invalid")
        if not _schema_one(observation.get("schema_version")):
            raise IntegrityError("observation schema version is invalid")
        cell = cells.get(observation["cell_id"])
        if cell is None or any(
            observation.get(key) != value for key, value in expected_identity.items()
        ):
            raise IntegrityError("observation has the wrong experiment identity")
        if any(observation.get(key) != cell[key] for key in ("task_id", "condition", "repeat")):
            raise IntegrityError("observation does not match the plan")
        expected_manifest = {
            "schema_version": 1,
            **identity(plan),
            "cell_id": observation["cell_id"],
            "observation_checksum": digest(payloads["observation.json"]),
        }
        if (
            not isinstance(manifest, dict)
            or not _schema_one(manifest.get("schema_version"))
            or manifest != expected_manifest
        ):
            raise IntegrityError("observation checksum is invalid")
        checksums = observation.get("checksums")
        raw_names = {"events.jsonl", "stderr.log", "final-output.json"}
        if (
            not isinstance(checksums, dict)
            or set(checksums) != raw_names
            or any(checksums.get(name) != digest(payloads[name]) for name in raw_names)
        ):
            raise IntegrityError("run raw evidence checksum is invalid")
        duration = observation.get("duration_ms")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
            raise IntegrityError("run duration is invalid")
        if observation.get("status") == "succeeded":
            expected_keys = {
                "schema_version",
                *expected_identity,
                *cell,
                "status",
                "files",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "tool_steps",
                "duration_ms",
                "checksums",
            }
            if set(observation) != expected_keys:
                raise IntegrityError("successful observation shape is invalid")
            final = strict_json(payloads["final-output.json"], "final output")
            if not isinstance(final, dict) or set(final) != {"files"}:
                raise IntegrityError("final output is invalid")
            if _prediction(final["files"], "final output files") != observation.get("files"):
                raise IntegrityError("final output differs from observation")
            _validate_events(payloads["events.jsonl"])
            expected_usage = _usage(payloads["events.jsonl"])
            if any(observation.get(key) != value for key, value in expected_usage.items()):
                raise IntegrityError("observation usage differs from native events")
        elif observation.get("status") == "terminal":
            expected_keys = {
                "schema_version",
                *expected_identity,
                *cell,
                "status",
                "files",
                "terminal_reason",
                "duration_ms",
                "checksums",
            }
            if (
                set(observation) != expected_keys
                or observation.get("files") != []
                or not isinstance(observation.get("terminal_reason"), str)
                or not observation["terminal_reason"]
            ):
                raise IntegrityError("terminal observation shape is invalid")
        else:
            raise IntegrityError("observation status is invalid")
        result[observation["cell_id"]] = observation
    return result


def _credentials() -> tuple[dict[str, str], bytes | None]:
    for key in ("CODEX_API_KEY", "OPENAI_API_KEY"):
        if value := os.environ.get(key):
            return {key: value}, None
    configured_home = os.environ.get("CODEX_HOME")
    if configured_home:
        source_home = Path(os.path.abspath(Path(configured_home).expanduser()))
    else:
        source_home = Path(os.path.abspath(Path.home() / ".codex"))
    auth = source_home / "auth.json"
    if auth.exists() or auth.is_symlink():
        payload = _read_file(auth, "Codex auth")
        if len(payload) > 65_536:
            raise IntegrityError("Codex auth is oversized")
        return {}, payload
    raise PipelineError("Codex credentials are unavailable")


def _isolated_environment(temporary_root: Path) -> dict[str, str]:
    home = temporary_root / "home"
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True, mode=0o700)
    environment = {
        "HOME": str(home),
        "CODEX_HOME": str(codex_home),
        "TMPDIR": str(temporary_root),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "SHELL": "/bin/sh",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    credential_environment, payload = _credentials()
    environment.update(credential_environment)
    if payload is not None:
        target = codex_home / "auth.json"
        target.write_bytes(payload)
        target.chmod(0o600)
    return environment


def _bounded_process(
    command: list[str],
    prompt: bytes,
    cwd: Path,
    environment: dict[str, str],
    temporary: Path,
    timeout: int,
) -> tuple[int, bytes, bytes, bool, bool, bool]:
    limit = 16 * 1024 * 1024
    stdout_path = temporary / "events.jsonl"
    stderr_path = temporary / "stderr.log"

    def child_limit() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))

    timed_out = False
    network_unavailable = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=child_limit,
        )

        def terminate() -> None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

        try:
            if process.stdin is None:
                raise RuntimeError("Codex stdin is unavailable")
            process.stdin.write(prompt)
            process.stdin.close()
            process.stdin = None
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    terminate()
                    break
                stdout.flush()
                if events := stdout_path.read_bytes():
                    network_unavailable = events.count(b"Reconnecting... waiting for network") >= 3
                    if network_unavailable:
                        terminate()
                        break
                time.sleep(0.1)
        except BaseException:
            terminate()
            raise
    events = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes()
    overflowed = (
        len(events) >= limit or len(stderr) >= limit or process.returncode == -signal.SIGXFSZ
    )
    return process.returncode, events, stderr, timed_out, overflowed, network_unavailable


def _validate_events(events: bytes) -> None:
    types: set[str] = set()
    for number, line in enumerate(events.splitlines(), 1):
        value = strict_json(line, f"Codex event {number}")
        if not isinstance(value, dict) or not isinstance(value.get("type"), str):
            raise IntegrityError("Codex event is not a typed object")
        types.add(value["type"])
    if not {"thread.started", "turn.completed"}.issubset(types):
        raise IntegrityError("Codex lifecycle is incomplete")


def _usage(events: bytes) -> dict[str, int | None]:
    usage: dict[str, int | None] = {
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_output_tokens": None,
        "tool_steps": 0,
    }

    completed_usage: dict[str, Any] | None = None
    for line in events.splitlines():
        value = strict_json(line, "Codex event")
        if not isinstance(value, dict):
            continue
        if value.get("type") == "item.completed":
            item = value.get("item")
            if isinstance(item, dict) and item.get("type") in {
                "command_execution",
                "mcp_tool_call",
            }:
                usage["tool_steps"] = int(usage["tool_steps"] or 0) + 1
        elif value.get("type") == "turn.completed":
            candidate = value.get("usage")
            if completed_usage is not None or not isinstance(candidate, dict):
                raise IntegrityError("Codex turn usage is missing or duplicated")
            completed_usage = candidate
    if completed_usage is None:
        raise IntegrityError("Codex turn usage is missing or duplicated")
    for key in ("input_tokens", "output_tokens"):
        value = completed_usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise IntegrityError(f"Codex {key} is invalid")
        usage[key] = value
    for key in ("cached_input_tokens", "reasoning_output_tokens"):
        value = completed_usage.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise IntegrityError(f"Codex {key} is invalid")
        usage[key] = value
    return usage


def _prompt(task: dict[str, Any]) -> str:
    return (
        "Locate the repository files relevant to the task. Work read-only. Return JSON with one "
        "to five unique repository-relative source file paths under the key files, ordered from "
        "most to least likely. Do not pad the list.\n\n"
        f"Task:\n{task['prompt']}\n"
    )


def _execute(plan: dict[str, Any], task: dict[str, Any], cell: dict[str, Any]) -> Evidence:
    runner = plan["runner"]
    binary = Path(runner["binary"])
    binary_bytes, binary_runtime = _read_executable(binary)
    if (
        digest(binary_bytes) != runner["binary_checksum"]
        or binary_runtime != runner["binary_runtime"]
    ):
        raise IntegrityError("Codex binary changed after prepare")
    binding = task["source"]
    started = time.monotonic()
    events = b""
    stderr = b""
    final_output = b""
    terminal_reason: str | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="repository-localization-cell-") as raw_temporary:
            temporary = Path(raw_temporary).resolve()
            repository = temporary / "repository"
            shutil.copytree(Path(binding["root"]), repository, symlinks=True)
            copied, _ = _inspect_tree(repository, "source")
            if any(
                copied[key] != binding[key] for key in ("tree_checksum", "file_count", "byte_count")
            ):
                raise IntegrityError("copied source root differs from frozen plan")
            guidance = task["guidance"][cell["condition"]]
            if guidance is not None:
                repository.chmod(0o755)
                (repository / "AGENTS.md").write_text(guidance, encoding="utf-8")
            for path in sorted(repository.rglob("*"), reverse=True):
                executable = path.stat().st_mode & 0o111
                path.chmod(0o555 if path.is_dir() or executable else 0o444)
            repository.chmod(0o555)
            schema = temporary / "prediction-schema.json"
            output = temporary / "final-output.json"
            schema.write_bytes(
                canonical(
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 5,
                            }
                        },
                        "required": ["files"],
                    }
                )
            )
            command = [
                str(binary),
                "exec",
                "--skip-git-repo-check",
                "--strict-config",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--json",
                "--color",
                "never",
                "--cd",
                str(repository),
                "--model",
                runner["model"],
                "--output-schema",
                str(schema),
                "--output-last-message",
                str(output),
                "-c",
                f'model_reasoning_effort="{runner["reasoning_effort"]}"',
                "-c",
                "project_doc_max_bytes=4096",
                "-c",
                "project_doc_fallback_filenames=[]",
                "-",
            ]
            code, events, stderr, timed_out, overflowed, network_unavailable = _bounded_process(
                command,
                _prompt(task).encode(),
                repository,
                _isolated_environment(temporary),
                temporary,
                runner["timeout_seconds"],
            )
            if output.exists():
                info = output.lstat()
                if stat.S_ISREG(info.st_mode) and info.st_size <= 1_048_576:
                    final_output = output.read_bytes()
                else:
                    terminal_reason = "invalid_output_file"
            if timed_out:
                terminal_reason = "timeout"
            elif network_unavailable:
                terminal_reason = "network_unavailable"
            elif overflowed:
                terminal_reason = "output_overflow"
            elif code != 0:
                terminal_reason = "process_nonzero"
            elif terminal_reason is None:
                try:
                    final = strict_json(final_output, "Codex final output")
                    if not isinstance(final, dict) or set(final) != {"files"}:
                        raise IntegrityError("Codex final output has unexpected keys")
                    files = _prediction(final["files"], "Codex final output files")
                    for selected in files:
                        if (
                            selected not in task["source_files"]
                            or selected == task["documentation"]["entry_path"]
                        ):
                            raise IntegrityError("Codex selected an ineligible implementation file")
                        target = repository / selected
                        if target.is_symlink() or not target.is_file():
                            raise IntegrityError(
                                f"Codex selected a missing source file: {selected}"
                            )
                    _validate_events(events)
                    observed_usage = _usage(events)
                except PipelineError:
                    terminal_reason = "invalid_output"
                else:
                    observation = {
                        "schema_version": 1,
                        **identity(plan),
                        **cell,
                        "status": "succeeded",
                        "files": files,
                        **observed_usage,
                        "duration_ms": int((time.monotonic() - started) * 1000),
                        "checksums": {
                            "events.jsonl": digest(events),
                            "stderr.log": digest(stderr),
                            "final-output.json": digest(final_output),
                        },
                    }
                    return Evidence(observation, events, stderr, final_output)
    except OSError:
        terminal_reason = "launch_failed"
    observation = {
        "schema_version": 1,
        **identity(plan),
        **cell,
        "status": "terminal",
        "files": [],
        "terminal_reason": terminal_reason or "execution_failed",
        "duration_ms": int((time.monotonic() - started) * 1000),
        "checksums": {
            "events.jsonl": digest(events),
            "stderr.log": digest(stderr),
            "final-output.json": digest(final_output),
        },
    }
    return Evidence(observation, events, stderr, final_output)


def run(config_path: Path, *, resume: bool) -> tuple[dict[str, str], Path]:
    config, plan = _current(config_path)
    claims = _claims(config.root, plan)
    runs = _runs(config.root, plan)
    if not set(runs).issubset(claims):
        raise IntegrityError("run evidence exists without a claim")
    unknown = sorted(set(claims) - set(runs))
    if unknown:
        raise StateError(
            f"claimed cells have no durable outcome and will not be retried: {unknown}"
        )
    if runs and not resume:
        raise StateError("run already contains completed cells; use run --resume")
    tasks = _task_map(plan)
    if len(runs) < len(plan["cells"]):
        _credentials()
    for ordinal, cell in enumerate(plan["cells"], 1):
        if cell["cell_id"] in runs:
            continue
        print(
            f"run {ordinal}/{len(plan['cells'])}: {cell['task_id']} {cell['condition']}",
            flush=True,
        )
        claim = {"schema_version": 1, **identity(plan), **cell}
        claimed = _write_once(config.root / "claims" / f"{cell['cell_id']}.json", canonical(claim))
        if not claimed:
            raise StateError(f"cell {cell['cell_id']} was claimed by another runner")
        evidence = _execute(plan, tasks[cell["task_id"]], cell)
        _, after, _ = build_plan(config_path)
        if after != plan:
            raise IntegrityError("experiment inputs changed during Codex execution")
        observation = canonical(evidence.observation)
        _publish(
            config.root / "runs" / cell["cell_id"],
            {
                "manifest.json": canonical(
                    {
                        "schema_version": 1,
                        **identity(plan),
                        "cell_id": cell["cell_id"],
                        "observation_checksum": digest(observation),
                    }
                ),
                "observation.json": observation,
                "events.jsonl": evidence.events,
                "stderr.log": evidence.stderr,
                "final-output.json": evidence.final_output,
            },
        )
        runs[cell["cell_id"]] = evidence.observation
    terminals = [
        f"{cell['cell_id']}:{runs[cell['cell_id']]['terminal_reason']}"
        for cell in plan["cells"]
        if runs[cell["cell_id"]]["status"] == "terminal"
    ]
    if terminals:
        raise ExecutionError(f"{len(terminals)} terminal cell(s): {', '.join(terminals)}")
    return identity(plan), config.root / "runs"


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
    return [
        {
            "schema_version": 1,
            **identity(plan),
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


def analyze(config_path: Path) -> tuple[dict[str, str], Path]:
    config, plan = _current(config_path)
    feature_rows = _read_features(config, plan)
    gold_raw, gold = _load_gold(config, plan)
    expected_tasks = {task["task_id"] for task in plan["tasks"]}
    if set(gold) != expected_tasks:
        raise PipelineError("gold task coverage must exactly match public tasks")
    rows: list[dict[str, Any]] = []
    for feature in feature_rows:
        gold_files = set(gold[feature["task_id"]])
        common = {
            **identity(plan),
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


def report(config_path: Path) -> tuple[dict[str, str], Path]:
    config = load_config(config_path)
    plan = _read_plan(config)
    analysis = _read_analysis(config, plan)

    def metric(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    summary = [
        f"# Experiment {plan['experiment_id']} {plan['experiment_version']}",
        "",
        f"Plan: `{plan['plan_id']}`",
        f"Dataset: `{plan['dataset']['name']}@{plan['dataset']['revision']}` "
        f"(`{plan['dataset']['config']}/{plan['dataset']['split']}`)",
        "",
        "| Condition | Planned | Succeeded | Terminal | Recall@3 | nDCG@3 | "
        "Returned-set F1 | Recall@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["aggregates"]:
        summary.append(
            f"| {row['condition']} | {row['planned_observations']} | "
            f"{row['successful_observations']} | {row['terminal_observations']} | "
            f"{metric(row['mean_recall_at_3'])} | {metric(row['mean_ndcg_at_3'])} | "
            f"{metric(row['mean_returned_set_f1'])} | {metric(row['mean_recall_at_5'])} |"
        )
    summary.append("")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"{line}\n" for line in summary],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "from pathlib import Path\n",
                    "\n",
                    "analysis = json.loads(Path('../analysis/data.json').read_text())\n",
                    "analysis['aggregates']\n",
                ],
            },
        ],
        "metadata": {
            "experiment": {**identity(plan), "dataset": plan["dataset"]},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    data = canonical(notebook)
    root = config.root / "report"
    _publish(
        root, {"manifest.json": canonical(_manifest(plan, digest(data))), "report.ipynb": data}
    )
    return identity(plan), root / "report.ipynb"


__all__ = [
    "ExecutionError",
    "IntegrityError",
    "PipelineError",
    "StateError",
    "analyze",
    "features",
    "prepare",
    "report",
    "run",
]
