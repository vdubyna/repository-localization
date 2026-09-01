"""Shared experiment contract, validation, and durable artifact I/O."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

CONDITIONS = (
    "NO-DOC",
    "OPTIONAL",
    "DOC-FIRST",
)
RUNNER_CONTRACT = "repository-localization-runner-v4"
DATASET = {
    "name": "Contextbench/ContextBench",
    "config": "default",
    "split": "train",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


# Operator-facing errors and shared records


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


# Canonical serialization and input validation


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


# Immutable artifact storage


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
