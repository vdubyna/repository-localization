"""Public CLI for repository-localization experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from repository_localization.pipeline import (
    PipelineError,
    analyze,
    features,
    figures,
    prepare,
    report,
    run,
)

Command = Callable[[Path], tuple[dict[str, str], Path]]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="repository-localization",
        description="Run one versioned, config-driven Codex localization experiment.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("config", nargs="?", type=Path, default=Path("experiment.toml"))
    execute = commands.add_parser("run")
    execute.add_argument("config", nargs="?", type=Path, default=Path("experiment.toml"))
    execute.add_argument(
        "--resume",
        action="store_true",
        help="Run untouched cells only; claimed or terminal cells are never retried.",
    )
    for name in ("features", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("config", nargs="?", type=Path, default=Path("experiment.toml"))
    report_command = commands.add_parser("report")
    report_command.add_argument("config", nargs="?", type=Path, default=Path("experiment.toml"))
    report_command.add_argument(
        "--figures",
        action="store_true",
        help="Generate the eight Chapter 4 figures from persisted cell features.",
    )
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    operations: dict[str, Command] = {
        "prepare": prepare,
        "features": features,
        "analyze": analyze,
        "report": report,
    }
    try:
        if arguments.command == "run":
            experiment, artifact = run(arguments.config, resume=arguments.resume)
        elif arguments.command == "report" and arguments.figures:
            experiment, artifact = figures(arguments.config)
        else:
            experiment, artifact = operations[arguments.command](arguments.config)
    except PipelineError as exc:
        print(f"error: {exc}")
        return exc.exit_code
    print(f"experiment: {experiment['experiment_id']}")
    print(f"version: {experiment['experiment_version']}")
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
