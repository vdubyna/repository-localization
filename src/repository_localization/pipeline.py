"""Stable public API for the five experiment stages."""

from pathlib import Path

from repository_localization.analysis import analyze, features, report
from repository_localization.core import ExecutionError, IntegrityError, PipelineError, StateError
from repository_localization.execution import run
from repository_localization.planning import prepare


def figures(config_path: Path) -> tuple[dict[str, str], Path]:
    from repository_localization.figures import figures as build_figures

    return build_figures(config_path)


__all__ = [
    "ExecutionError",
    "IntegrityError",
    "PipelineError",
    "StateError",
    "analyze",
    "features",
    "figures",
    "prepare",
    "report",
    "run",
]
