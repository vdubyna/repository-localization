"""Execute frozen experiment cells and validate raw Codex evidence."""

from __future__ import annotations

import os
import resource
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from repository_localization.core import (
    Evidence,
    ExecutionError,
    IntegrityError,
    PipelineError,
    StateError,
    _files,
    _prediction,
    _publish,
    _read_executable,
    _read_file,
    _schema_one,
    _write_once,
    canonical,
    digest,
    strict_json,
)
from repository_localization.planning import (
    _cell_map,
    _current,
    _inspect_tree,
    _task_map,
    build_plan,
    identity,
)

# Persisted claim and run validation


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
        if any(
            claim.get(key) != cell[key]
            for key in ("task_id", "condition", "model", "reasoning_effort", "repeat")
        ):
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
        if any(
            observation.get(key) != cell[key]
            for key in ("task_id", "condition", "model", "reasoning_effort", "repeat")
        ):
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


# Isolated Codex process


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
                cell["model"],
                "--output-schema",
                str(schema),
                "--output-last-message",
                str(output),
                "-c",
                f'model_reasoning_effort="{cell["reasoning_effort"]}"',
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
                            or selected in task["documentation"]["paths"]
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


# Public run stage


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
            f"run {ordinal}/{len(plan['cells'])}: {cell['task_id']} {cell['condition']} "
            f"{cell['model']}/{cell['reasoning_effort']}",
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
