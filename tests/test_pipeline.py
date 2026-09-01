from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from repository_localization.analysis import _has_explicit_gold_locator, _prompt_features

FIXTURE = Path(__file__).parent / "fixtures" / "experiment"


def fake_codex(path: Path, *, exit_code: int = 0, network_failure: bool = False) -> Path:
    path.write_text(
        f"""#!/bin/sh
output=""
schema=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output-last-message" ]; then
    shift
    output="$1"
  elif [ "$1" = "--output-schema" ]; then
    shift
    schema="$1"
  fi
  shift
done
if [ {exit_code} -ne 0 ]; then
  exit {exit_code}
fi
if [ {int(network_failure)} -eq 1 ]; then
  printf '%s\n' '{{"type":"thread.started"}}'
  printf '%s\n' '{{"type":"error","message":"Reconnecting... waiting for network"}}'
  printf '%s\n' '{{"type":"error","message":"Reconnecting... waiting for network"}}'
  printf '%s\n' '{{"type":"error","message":"Reconnecting... waiting for network"}}'
  sleep 30
  exit 8
fi
if grep -q 'uniqueItems' "$schema"; then
  exit 10
fi
if [ ! -x pkg/service.py ]; then
  exit 9
fi
if [ -f AGENTS.md ]; then
  files='["pkg/service.py"]'
  event='{{"type":"item.completed",'\
'"item":{{"type":"command_execution","command":"docs/index.md",'\
'"aggregated_output":"documentation\\n"}}}}'
  printf '%s\n' "$event"
  event='{{"type":"item.completed",'\
'"item":{{"type":"command_execution","command":"docs/guide.md",'\
'"aggregated_output":"service guide\\n"}}}}'
  printf '%s\n' "$event"
  event='{{"type":"item.completed",'\
'"item":{{"type":"command_execution","command":"rg greeting pkg",'\
'"aggregated_output":"pkg/service.py:1:def greeting():\\n"}}}}'
  printf '%s\n' "$event"
  event='{{"type":"item.completed",'\
'"item":{{"type":"command_execution","command":"pkg/service.py",'\
'"aggregated_output":"def greeting():\\n"}}}}'
  printf '%s\n' "$event"
else
  files='["pkg/readme.py"]'
  event='{{"type":"item.completed",'\
'"item":{{"type":"command_execution","command":"pkg/readme.py",'\
'"aggregated_output":"pkg/readme.py\\n"}}}}'
  printf '%s\n' "$event"
fi
printf '{{"files":%s}}\n' "$files" > "$output"
printf '%s\n' '{{"type":"thread.started"}}'
printf '%s\n' '{{"type":"turn.completed","usage":{{"input_tokens":10,"output_tokens":2}}}}'
""",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def setup(
    tmp_path: Path, *, exit_code: int = 0, network_failure: bool = False
) -> tuple[Path, Path]:
    fixture = tmp_path / "experiment"
    shutil.copytree(FIXTURE, fixture)
    (fixture / "source" / "pkg" / "service.py").chmod(0o755)
    (fixture / "source").chmod(0o555)
    binary = fake_codex(
        tmp_path / "fake-codex", exit_code=exit_code, network_failure=network_failure
    )
    config = fixture / "experiment.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("/replaced/by/test", str(binary)),
        encoding="utf-8",
    )
    return fixture, config


def invoke(config: Path, command: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "repository_localization.cli", command, str(config), *arguments],
        capture_output=True,
        check=False,
        env={**os.environ, "CODEX_API_KEY": "fixture-key"},
        text=True,
    )


def test_task_locator_features_are_explicit_and_gold_aware(tmp_path: Path) -> None:
    assert _prompt_features("Tested on 1.8 and 1.6.2.") == {
        "prompt_has_path": False,
        "prompt_has_filename": False,
        "prompt_has_symbol": False,
    }
    assert _prompt_features("Edit `pkg/service.py` and ServiceHandler.run().") == {
        "prompt_has_path": True,
        "prompt_has_filename": True,
        "prompt_has_symbol": True,
    }
    source = tmp_path / "pkg"
    source.mkdir()
    (source / "service.py").write_text("def greeting():\n    return 'hello'\n")
    assert _has_explicit_gold_locator("Fix `greeting`.", ["pkg/service.py"], tmp_path)
    assert not _has_explicit_gold_locator("Fix unrelated behavior.", ["pkg/service.py"], tmp_path)


def test_eight_profiles_expand_the_frozen_plan(tmp_path: Path) -> None:
    fixture, config = setup(tmp_path)
    profiles = "".join(
        f'\n[[runner.profiles]]\nmodel = "fixture-{number}"\nreasoning_effort = "high"\n'
        for number in range(3, 9)
    )
    config.write_text(config.read_text(encoding="utf-8") + profiles, encoding="utf-8")

    prepared = invoke(config, "prepare")

    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    plan = json.loads(
        (fixture / "artifacts" / "fixture-localization" / "v1" / "plan.json").read_text()
    )
    assert len(plan["runner"]["profiles"]) == 8
    assert len(plan["cells"]) == 24


def test_five_stage_versioned_pipeline_and_gold_boundary(tmp_path: Path) -> None:
    fixture, config = setup(tmp_path)
    gold = fixture / "gold.jsonl"
    hidden_gold = fixture / "gold.hidden"
    gold.rename(hidden_gold)

    for command in ("prepare", "run"):
        result = invoke(config, command)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "version: v1" in result.stdout

    gold_payload = hidden_gold.read_bytes()
    root = fixture / "artifacts" / "fixture-localization" / "v1"
    first_observation = next((root / "runs").glob("*/observation.json"))
    observation_payload = first_observation.read_bytes()
    observation = json.loads(observation_payload)
    observation["input_tokens"] = 999
    first_observation.chmod(0o600)
    first_observation.write_text(json.dumps(observation, sort_keys=True) + "\n")
    run_manifest = first_observation.parent / "manifest.json"
    manifest_payload = run_manifest.read_bytes()
    manifest = json.loads(manifest_payload)
    manifest["observation_checksum"] = hashlib.sha256(first_observation.read_bytes()).hexdigest()
    run_manifest.chmod(0o600)
    run_manifest.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    invalid_features = invoke(config, "features")
    assert invalid_features.returncode == 4
    assert "usage differs" in invalid_features.stdout
    first_observation.write_bytes(observation_payload)
    first_observation.chmod(0o444)
    run_manifest.write_bytes(manifest_payload)
    run_manifest.chmod(0o444)

    featured = invoke(config, "features")
    assert featured.returncode == 0, featured.stdout + featured.stderr
    feature_rows = [
        json.loads(line) for line in (root / "features" / "data.jsonl").read_text().splitlines()
    ]
    assert all(row["prompt_has_path"] for row in feature_rows)
    assert all(row["prompt_has_filename"] for row in feature_rows)
    assert all(row["prompt_has_symbol"] for row in feature_rows)
    assert all("task_type" not in row for row in feature_rows)

    hidden_gold.rename(gold)

    gold.write_text('{"task_id":"ContextBench__fixture-task","files":["pkg/missing.py"]}\n')
    invalid_gold = invoke(config, "analyze")
    assert invalid_gold.returncode == 2
    assert "outside the frozen source tree" in invalid_gold.stdout
    gold.write_bytes(gold_payload)

    analyzed = invoke(config, "analyze")
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    plan = json.loads((root / "plan.json").read_text())
    identity = {
        "experiment_id": "fixture-localization",
        "experiment_version": "v1",
        "plan_id": plan["plan_id"],
    }
    analysis = json.loads((root / "analysis" / "data.json").read_text())
    assert {key: analysis[key] for key in identity} == identity
    assert plan["conditions"] == [
        "NO-DOC",
        "OPTIONAL",
        "DOC-FIRST",
    ]
    assert plan["dataset"] == {
        "name": "Contextbench/ContextBench",
        "config": "default",
        "split": "train",
        "revision": "c2855792b006af41c67202d33883fb9d46362853",
    }
    assert plan["tasks"][0]["guidance"]["NO-DOC"] is None
    assert plan["tasks"][0]["documentation"]["paths"] == ["docs/guide.md", "docs/index.md"]
    assert plan["tasks"][0]["base_commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert plan["runner"]["profiles"] == [
        {"model": "fixture-model", "reasoning_effort": "low"},
        {"model": "fixture-model", "reasoning_effort": "medium"},
    ]
    assert "You may consult" in plan["tasks"][0]["guidance"]["OPTIONAL"]
    assert "Before searching" in plan["tasks"][0]["guidance"]["DOC-FIRST"]
    assert [row["mean_recall_at_5"] for row in analysis["aggregates"]] == [0.0, 1.0, 1.0]
    assert [row["mean_recall_at_3"] for row in analysis["aggregates"]] == [0.0, 1.0, 1.0]
    assert [row["mean_ndcg_at_3"] for row in analysis["aggregates"]] == [0.0, 1.0, 1.0]
    assert [row["mean_returned_set_f1"] for row in analysis["aggregates"]] == [0.0, 1.0, 1.0]
    assert [row["successful_observations"] for row in analysis["aggregates"]] == [2, 2, 2]
    assert [row["terminal_observations"] for row in analysis["aggregates"]] == [0, 0, 0]
    assert all(row["task_type"] == "EXPLICIT_LOCATOR_CLUE" for row in analysis["rows"])
    assert [row["wiki_read_count"] for row in analysis["rows"]] == [0, 2, 2, 0, 2, 2]
    assert [row["unique_wiki_pages"] for row in analysis["rows"]] == [0, 2, 2, 0, 2, 2]
    assert [row["beyond_entry_reads"] for row in analysis["rows"]] == [0, 1, 1, 0, 1, 1]
    assert all(row["wiki_tokens"] > 0 for row in analysis["rows"] if row["condition"] != "NO-DOC")
    assert [row["gold_seen_any"] for row in analysis["rows"]] == [0, 1, 1, 0, 1, 1]
    assert [row["gold_seen_by_3_source_actions"] for row in analysis["rows"]] == [0, 1, 1, 0, 1, 1]
    assert [row["gold_targeted_any"] for row in analysis["rows"]] == [0, 1, 1, 0, 1, 1]
    with (root / "features" / "cell_features.csv").open(newline="") as handle:
        cell_rows = list(csv.DictReader(handle))
    with (root / "features" / "task_features.csv").open(newline="") as handle:
        task_rows = list(csv.DictReader(handle))
    assert len(cell_rows) == 6
    assert len(task_rows) == 1
    assert task_rows[0]["doc_first_minus_optional_recall_at_3"] == "0.0"
    json_artifacts = [
        *sorted((root / "claims").glob("*.json")),
        *sorted((root / "runs").glob("*/manifest.json")),
        *sorted((root / "runs").glob("*/observation.json")),
        root / "features" / "manifest.json",
        root / "analysis" / "manifest.json",
    ]
    for path in json_artifacts:
        payload = json.loads(path.read_text())
        assert {key: payload[key] for key in identity} == identity
    for line in (root / "features" / "data.jsonl").read_text().splitlines():
        payload = json.loads(line)
        assert {key: payload[key] for key in identity} == identity
    for row in analysis["rows"]:
        assert {key: row[key] for key in identity} == identity
    for run_root in (root / "runs").iterdir():
        observation = json.loads((run_root / "observation.json").read_text())
        for name, checksum in observation["checksums"].items():
            assert hashlib.sha256((run_root / name).read_bytes()).hexdigest() == checksum

    gold.rename(hidden_gold)
    (fixture / "tasks.jsonl").rename(fixture / "tasks.hidden")
    (fixture / "source").rename(fixture / "source.hidden")
    reported = invoke(config, "report")
    assert reported.returncode == 0, reported.stdout + reported.stderr
    report_data = json.loads((root / "report" / "data.json").read_text())
    assert {key: report_data[key] for key in identity} == identity
    assert report_data["dataset"] == plan["dataset"]
    assert report_data["rows"] == analysis["rows"]
    assert report_data["aggregates"] == analysis["aggregates"]
    assert len(report_data["source_checksum"]) == 64
    report_manifest = json.loads((root / "report" / "manifest.json").read_text())
    assert {key: report_manifest[key] for key in identity} == identity
    shared_eda = json.loads(Path("analysis/eda.ipynb").read_text())
    notebook_source = "".join(
        line for cell in shared_eda["cells"] for line in cell.get("source", [])
    )
    assert "EXPERIMENT_CONFIG" in notebook_source
    assert "features/cell_features.csv" in notebook_source
    assert "features/task_features.csv" in notebook_source
    assert "report/data.json" not in notebook_source
    assert "build_focused_research_report" not in notebook_source


def test_report_generates_versioned_research_figures(tmp_path: Path) -> None:
    experiment_id = "figure-fixture"
    experiment_version = "v1"
    plan_id = "a" * 64
    artifact_root = tmp_path / "results" / experiment_id / experiment_version
    feature_root = artifact_root / "features"
    report_root = artifact_root / "report"
    feature_root.mkdir(parents=True)
    report_root.mkdir()
    config = tmp_path / "experiment-record.toml"
    config.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'experiment_id = "{experiment_id}"',
                f'experiment_version = "{experiment_version}"',
                f'source_plan_id = "{plan_id}"',
                'artifact_dir = "results"',
                "task_count = 2",
                "cell_count = 12",
                "profile_count = 2",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (report_root / "data.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "experiment_version": experiment_version,
                "plan_id": plan_id,
                "rows": [],
                "aggregates": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "experiment_id",
        "experiment_version",
        "plan_id",
        "cell_id",
        "task_id",
        "task_type",
        "condition",
        "model",
        "reasoning_effort",
        "repeat",
        "status",
        "recall_at_3",
        "recall_at_5",
        "ndcg_at_3",
        "returned_set_f1",
        "provider_total_tokens",
        "elapsed_seconds",
        "agent_step_count",
        "gold_seen_any",
        "gold_seen_by_3_source_actions",
        "gold_targeted_any",
    ]
    conditions = ["NO-DOC", "OPTIONAL", "DOC-FIRST"]
    with (feature_root / "cell_features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task_index, task_type in enumerate(
            ("EXPLICIT_LOCATOR_CLUE", "NO_EXPLICIT_LOCATOR_CLUE")
        ):
            for profile_index, (model, effort) in enumerate(
                (("gpt-5.6-terra", "low"), ("gpt-5.6-sol", "medium"))
            ):
                for condition_index, condition in enumerate(conditions):
                    quality = 0.45 + 0.05 * task_index + 0.02 * condition_index
                    writer.writerow(
                        {
                            "experiment_id": experiment_id,
                            "experiment_version": experiment_version,
                            "plan_id": plan_id,
                            "cell_id": f"cell-{task_index}-{profile_index}-{condition_index}",
                            "task_id": f"task-{task_index}",
                            "task_type": task_type,
                            "condition": condition,
                            "model": model,
                            "reasoning_effort": effort,
                            "repeat": 1,
                            "status": "succeeded",
                            "recall_at_3": quality,
                            "recall_at_5": quality + 0.02,
                            "ndcg_at_3": quality + 0.01,
                            "returned_set_f1": quality - 0.1,
                            "provider_total_tokens": 1000 + 100 * condition_index,
                            "elapsed_seconds": 10 + condition_index,
                            "agent_step_count": 4 + condition_index,
                            "gold_seen_any": 0.7 + 0.1 * condition_index,
                            "gold_seen_by_3_source_actions": 0.5 + 0.1 * condition_index,
                            "gold_targeted_any": 0.6 + 0.1 * condition_index,
                        }
                    )

    result = invoke(config, "report", "--figures")
    assert result.returncode == 0, result.stdout + result.stderr
    figure_root = report_root / "figures"
    manifest = json.loads((figure_root / "manifest.json").read_text())
    assert manifest["experiment_id"] == experiment_id
    assert manifest["experiment_version"] == experiment_version
    assert manifest["plan_id"] == plan_id
    assert manifest["figure_count"] == 8
    assert manifest["task_count"] == 2
    assert manifest["cell_count"] == 12
    assert manifest["profile_count"] == 2
    assert len(list(figure_root.glob("*.png"))) == 8
    assert len(list(figure_root.glob("*.pdf"))) == 8
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in figure_root.glob("*.png"))
    assert all(path.read_bytes().startswith(b"%PDF") for path in figure_root.glob("*.pdf"))
    repeated = invoke(config, "report", "--figures")
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr


def test_multiple_tasks_in_one_repository_are_isolated(tmp_path: Path) -> None:
    fixture, config = setup(tmp_path)
    tasks_path = fixture / "tasks.jsonl"
    first_task = json.loads(tasks_path.read_text())
    second_task = {
        **first_task,
        "task_id": "ContextBench__fixture-task-2",
        "prompt": "Locate the files relevant to a second independent issue.",
    }
    tasks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in (first_task, second_task))
    )
    gold_path = fixture / "gold.jsonl"
    first_gold = json.loads(gold_path.read_text())
    second_gold = {**first_gold, "task_id": second_task["task_id"]}
    gold_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in (first_gold, second_gold))
    )

    for command in ("prepare", "run", "features", "analyze", "report"):
        result = invoke(config, command)
        assert result.returncode == 0, result.stdout + result.stderr

    root = fixture / "artifacts" / "fixture-localization" / "v1"
    plan = json.loads((root / "plan.json").read_text())
    features = [
        json.loads(line) for line in (root / "features" / "data.jsonl").read_text().splitlines()
    ]
    assert len(plan["tasks"]) == 2
    assert len(plan["cells"]) == 12
    assert len({cell["cell_id"] for cell in plan["cells"]}) == 12
    assert len(list((root / "claims").glob("*.json"))) == 12
    assert len(list((root / "runs").glob("*/observation.json"))) == 12
    assert len(features) == 12
    assert all(
        sum(row["task_id"] == task["task_id"] for row in features) == 6 for task in plan["tasks"]
    )
    analysis = json.loads((root / "analysis" / "data.json").read_text())
    by_task = {
        task["task_id"]: [row for row in analysis["rows"] if row["task_id"] == task["task_id"]]
        for task in plan["tasks"]
    }
    assert all(
        row["task_type"] == "EXPLICIT_LOCATOR_CLUE" for row in by_task[first_task["task_id"]]
    )
    assert all(
        row["task_type"] == "NO_EXPLICIT_LOCATOR_CLUE" for row in by_task[second_task["task_id"]]
    )


def test_version_drift_resume_terminal_and_help(tmp_path: Path) -> None:
    fixture, config = setup(tmp_path / "invalid-provenance")
    tasks = fixture / "tasks.jsonl"
    tasks.write_text(tasks.read_text().replace("0123456789abcdef0123456789abcdef01234567", "main"))
    invalid_provenance = invoke(config, "prepare")
    assert invalid_provenance.returncode == 2
    assert "base_commit must be a full lowercase Git commit" in invalid_provenance.stdout

    fixture, config = setup(tmp_path / "drift")
    assert invoke(config, "prepare").returncode == 0
    config.write_text(
        config.read_text(encoding="utf-8").replace("fixture-model", "changed-model"),
        encoding="utf-8",
    )
    drift = invoke(config, "run")
    assert drift.returncode == 4
    assert "frozen experiment version" in drift.stdout

    fixture, config = setup(tmp_path / "resume")
    assert invoke(config, "prepare").returncode == 0
    assert invoke(config, "run").returncode == 0
    root = fixture / "artifacts" / "fixture-localization" / "v1"
    cells = json.loads((root / "plan.json").read_text())["cells"]
    shutil.rmtree(root / "runs" / cells[1]["cell_id"])
    (root / "claims" / f"{cells[1]['cell_id']}.json").unlink()
    assert invoke(config, "run").returncode == 3
    assert invoke(config, "run", "--resume").returncode == 0

    fixture, config = setup(tmp_path / "terminal", exit_code=7)
    assert invoke(config, "prepare").returncode == 0
    terminal = invoke(config, "run")
    assert terminal.returncode == 5
    root = fixture / "artifacts" / "fixture-localization" / "v1"
    assert len(list((root / "runs").glob("*/observation.json"))) == 6
    retried = invoke(config, "run", "--resume")
    assert retried.returncode == 5
    assert "6 terminal cell(s)" in retried.stdout
    assert invoke(config, "features").returncode == 0
    assert invoke(config, "analyze").returncode == 0
    analysis = json.loads((root / "analysis" / "data.json").read_text())
    assert [row["successful_observations"] for row in analysis["aggregates"]] == [0, 0, 0]
    assert [row["terminal_observations"] for row in analysis["aggregates"]] == [2, 2, 2]
    assert all(row["mean_recall_at_3"] is None for row in analysis["aggregates"])
    assert invoke(config, "report").returncode == 0

    fixture, config = setup(tmp_path / "network", network_failure=True)
    assert invoke(config, "prepare").returncode == 0
    started = time.monotonic()
    terminal = invoke(config, "run")
    assert time.monotonic() - started < 5
    assert terminal.returncode == 5
    root = fixture / "artifacts" / "fixture-localization" / "v1"
    observation = json.loads(next((root / "runs").glob("*/observation.json")).read_text())
    assert observation["terminal_reason"] == "network_unavailable"

    help_result = subprocess.run(
        [sys.executable, "-m", "repository_localization.cli", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert help_result.returncode == 0
    assert "{prepare,run,features,analyze,report}" in help_result.stdout
    for command in ("prepare", "run", "features", "analyze", "report"):
        assert command in help_result.stdout
    for legacy in ("corpus", "index", "query", "recovery", "continuation"):
        assert legacy not in help_result.stdout
