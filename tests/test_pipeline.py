from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

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
else
  files='["pkg/readme.py"]'
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


def test_five_stage_versioned_pipeline_and_gold_boundary(tmp_path: Path) -> None:
    fixture, config = setup(tmp_path)
    gold = fixture / "gold.jsonl"
    hidden_gold = fixture / "gold.hidden"
    gold.rename(hidden_gold)

    for command in ("prepare", "run"):
        result = invoke(config, command)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "version: v1" in result.stdout

    hidden_gold.rename(gold)
    gold_payload = gold.read_bytes()
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
    assert plan["tasks"][0]["base_commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert "You may consult" in plan["tasks"][0]["guidance"]["OPTIONAL"]
    assert "Before searching" in plan["tasks"][0]["guidance"]["DOC-FIRST"]
    assert [row["mean_recall_at_5"] for row in analysis["aggregates"]] == [0.0, 1.0, 1.0]
    assert [row["mean_recall_at_3"] for row in analysis["aggregates"]] == [0.0, 1.0, 1.0]
    assert [row["mean_ndcg_at_3"] for row in analysis["aggregates"]] == [0.0, 1.0, 1.0]
    assert [row["mean_returned_set_f1"] for row in analysis["aggregates"]] == [0.0, 1.0, 1.0]
    assert [row["successful_observations"] for row in analysis["aggregates"]] == [1, 1, 1]
    assert [row["terminal_observations"] for row in analysis["aggregates"]] == [0, 0, 0]
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
    assert "EDA_CONFIG" in notebook_source
    assert "build_report" not in notebook_source


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
    assert len(plan["cells"]) == 6
    assert len({cell["cell_id"] for cell in plan["cells"]}) == 6
    assert len(list((root / "claims").glob("*.json"))) == 6
    assert len(list((root / "runs").glob("*/observation.json"))) == 6
    assert len(features) == 6
    assert all(
        sum(row["task_id"] == task["task_id"] for row in features) == 3 for task in plan["tasks"]
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
    assert len(list((root / "runs").glob("*/observation.json"))) == 3
    retried = invoke(config, "run", "--resume")
    assert retried.returncode == 5
    assert "3 terminal cell(s)" in retried.stdout
    assert invoke(config, "features").returncode == 0
    assert invoke(config, "analyze").returncode == 0
    analysis = json.loads((root / "analysis" / "data.json").read_text())
    assert [row["successful_observations"] for row in analysis["aggregates"]] == [0, 0, 0]
    assert [row["terminal_observations"] for row in analysis["aggregates"]] == [1, 1, 1]
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
