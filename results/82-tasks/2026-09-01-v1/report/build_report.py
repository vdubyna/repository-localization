# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nbconvert.preprocessors import ExecutePreprocessor

REPORT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = REPORT_DIR.parent
DATA_DIR = BUNDLE_ROOT / "features"
TASK_CSV = DATA_DIR / "task_features.csv"
CELL_CSV = DATA_DIR / "cell_features.csv"
OUTPUT_HTML = REPORT_DIR / "focused-research.html"
OUTPUT_NOTEBOOK = REPORT_DIR / "focused-research.ipynb"
OUTPUT_SUMMARY = REPORT_DIR / "focused-research-summary.json"

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260831

CONDITION_LABELS = {
    "no_doc": "Без документації",
    "optional": "Документація доступна",
    "required": "Документація спочатку",
}
CONDITION_CODES = {
    "NO_DOC_GUIDANCE": "Без документації",
    "FUNCTIONAL_OPTIONAL": "Документація доступна",
    "FUNCTIONAL_REQUIRED_BEFORE_SOURCE": "Документація спочатку",
}
METRIC_LABELS = {
    "recall_at_3": "Recall@3",
    "recall_at_5": "Recall@5",
    "ndcg_at_3": "NDCG@3",
    "returned_set_f1": "Returned-set F1",
}
TASK_TYPE_LABELS = {
    "EXPLICIT_LOCATOR_CLUE": "Є явна підказка",
    "NO_EXPLICIT_LOCATOR_CLUE": "Немає явної підказки",
}
PAGE_TYPE_LABELS = {
    "short_toctree_index": "Короткий toctree-індекс",
    "curated_navigation_hub": "Навігаційний hub",
    "detailed_topical_map": "Детальна тематична карта",
    "large_reference_catalog": "Великий reference-каталог",
}

PALETTE = {
    "Без документації": "#64748b",
    "Документація доступна": "#f59e0b",
    "Документація спочатку": "#2563eb",
    "Є явна підказка": "#7c3aed",
    "Немає явної підказки": "#0f766e",
    "Recall@3": "#2563eb",
    "NDCG@3": "#0f766e",
    "Returned-set F1": "#f59e0b",
}


def _bootstrap_mean(
    frame: pd.DataFrame,
    value_col: str,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    clean = frame[["repository", value_col]].dropna()
    arrays = [group[value_col].to_numpy(dtype=float) for _, group in clean.groupby("repository")]
    rng = np.random.default_rng(seed)
    sums = np.zeros(replicates, dtype=float)
    total = 0
    for values in arrays:
        indices = rng.integers(0, len(values), size=(replicates, len(values)))
        sums += values[indices].sum(axis=1)
        total += len(values)
    samples = sums / total
    return {
        "estimate": float(clean[value_col].mean()),
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "n": int(len(clean)),
    }


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank().corr(right.rank()))


def _base_layout(fig: go.Figure, *, height: int = 470) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=64, r=32, t=72, b=64),
        font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif", size=13, color="#172033"),
        title=dict(font=dict(size=20, color="#0f172a"), x=0.02),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font_size=13),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8edf4", zerolinecolor="#94a3b8")
    fig.update_yaxes(showgrid=True, gridcolor="#e8edf4", zerolinecolor="#94a3b8")
    return fig


def prepare_analysis(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    tasks = pd.read_csv(data_dir / TASK_CSV.name)
    cells = pd.read_csv(data_dir / CELL_CSV.name)

    quality_rows: list[dict[str, Any]] = []
    for metric in ("recall_at_3", "ndcg_at_3", "returned_set_f1"):
        for condition in ("no_doc", "optional", "required"):
            values = tasks[f"{condition}_mean_{metric}"]
            quality_rows.append(
                {
                    "metric": METRIC_LABELS[metric],
                    "condition": CONDITION_LABELS[condition],
                    "mean": float(values.mean()),
                }
            )
    quality = pd.DataFrame(quality_rows)

    contrast_rows: list[dict[str, Any]] = []
    for comparator, comparator_label in (("optional", "проти доступної"), ("no_doc", "проти без документації")):
        for offset, metric in enumerate(("recall_at_3", "recall_at_5", "ndcg_at_3", "returned_set_f1")):
            col = f"required_minus_{comparator}_{metric}"
            result = _bootstrap_mean(tasks, col, seed=BOOTSTRAP_SEED + offset)
            contrast_rows.append(
                {
                    "comparison": comparator_label,
                    "metric": METRIC_LABELS[metric],
                    **result,
                }
            )
    contrasts = pd.DataFrame(contrast_rows)

    subgroup_rows: list[dict[str, Any]] = []
    for task_type, task_label in TASK_TYPE_LABELS.items():
        subset = tasks[tasks["task_type"] == task_type]
        for offset, metric in enumerate(("recall_at_3", "ndcg_at_3", "returned_set_f1")):
            result = _bootstrap_mean(
                subset,
                f"required_minus_optional_{metric}",
                seed=BOOTSTRAP_SEED + 10 + offset,
            )
            subgroup_rows.append(
                {
                    "task_type": task_label,
                    "metric": METRIC_LABELS[metric],
                    **result,
                }
            )
    subgroups = pd.DataFrame(subgroup_rows)

    mechanism_rows: list[dict[str, Any]] = []
    mechanism_metrics = {
        "gold_seen_any": "Gold-файл зʼявився у траєкторії",
        "gold_seen_by_3_source_actions": "Gold до третьої source-дії",
        "gold_targeted_any": "Gold явно таргетувався",
    }
    for metric, label in mechanism_metrics.items():
        for condition in ("optional", "required"):
            mechanism_rows.append(
                {
                    "mechanism": label,
                    "condition": CONDITION_LABELS[condition],
                    "rate": float(tasks[f"{condition}_mean_{metric}"].mean()),
                }
            )
    mechanism = pd.DataFrame(mechanism_rows)

    token_task_long = pd.concat(
        [
            tasks[
                ["safe_task_id", "repository", "task_type", "page_type", f"{condition}_mean_provider_total_tokens"]
            ]
            .rename(columns={f"{condition}_mean_provider_total_tokens": "provider_total_tokens"})
            .assign(condition=CONDITION_LABELS[condition])
            for condition in ("no_doc", "optional", "required")
        ],
        ignore_index=True,
    )
    time_task_long = pd.concat(
        [
            tasks[["safe_task_id", "repository", "task_type", "page_type", f"{condition}_mean_elapsed_seconds"]]
            .rename(columns={f"{condition}_mean_elapsed_seconds": "elapsed_seconds"})
            .assign(condition=CONDITION_LABELS[condition])
            for condition in ("no_doc", "optional", "required")
        ],
        ignore_index=True,
    )

    token_composition_rows: list[dict[str, Any]] = []
    for condition_code, condition_label in CONDITION_CODES.items():
        group = cells[cells["condition"] == condition_code]
        input_tokens = float(group["provider_input_tokens"].mean())
        cached_input = float(group["provider_cached_input_tokens"].mean())
        output_tokens = float(group["provider_output_tokens"].mean())
        reasoning_tokens = float(group["provider_reasoning_tokens"].mean())
        for component, tokens in (
            ("Cached input", cached_input),
            ("Non-cached input", input_tokens - cached_input),
            ("Reasoning output", reasoning_tokens),
            ("Visible output", output_tokens - reasoning_tokens),
        ):
            token_composition_rows.append(
                {
                    "condition": condition_label,
                    "component": component,
                    "tokens": tokens,
                }
            )
    token_composition = pd.DataFrame(token_composition_rows)

    profile_cells = cells.copy()
    profile_cells["profile"] = profile_cells["model"].astype(str) + " · " + profile_cells["reasoning_effort"].astype(str)
    profile_token_quality = (
        profile_cells.groupby(["profile", "condition"], as_index=False)[
            ["provider_total_tokens", "elapsed_seconds", "ndcg_at_3", "recall_at_3"]
        ]
        .mean()
        .assign(condition_label=lambda frame: frame["condition"].map(CONDITION_CODES))
    )
    required_profile = profile_token_quality[
        profile_token_quality["condition"] == "FUNCTIONAL_REQUIRED_BEFORE_SOURCE"
    ].set_index("profile")
    optional_profile = profile_token_quality[
        profile_token_quality["condition"] == "FUNCTIONAL_OPTIONAL"
    ].set_index("profile")
    profile_token_delta = pd.DataFrame(
        {
            "profile": required_profile.index,
            "optional_provider_total_tokens": optional_profile.loc[
                required_profile.index, "provider_total_tokens"
            ],
            "required_provider_total_tokens": required_profile["provider_total_tokens"],
            "optional_elapsed_seconds": optional_profile.loc[required_profile.index, "elapsed_seconds"],
            "required_elapsed_seconds": required_profile["elapsed_seconds"],
            "token_overhead_pct": 100
            * (
                required_profile["provider_total_tokens"]
                / optional_profile.loc[required_profile.index, "provider_total_tokens"]
                - 1
            ),
            "ndcg_delta": required_profile["ndcg_at_3"]
            - optional_profile.loc[required_profile.index, "ndcg_at_3"],
            "recall_delta": required_profile["recall_at_3"]
            - optional_profile.loc[required_profile.index, "recall_at_3"],
            "time_overhead_pct": 100
            * (
                required_profile["elapsed_seconds"]
                / optional_profile.loc[required_profile.index, "elapsed_seconds"]
                - 1
            ),
        }
    ).reset_index(drop=True)

    token_overhead_tasks = tasks.assign(
        token_overhead_pct=100
        * (
            tasks["required_mean_provider_total_tokens"]
            / tasks["optional_mean_provider_total_tokens"]
            - 1
        )
    )
    token_overhead_type = pd.concat(
        [
            token_overhead_tasks.assign(
                family="Тип задачі",
                category=token_overhead_tasks["task_type"].map(TASK_TYPE_LABELS),
            )
            .groupby(["family", "category"], as_index=False)["token_overhead_pct"]
            .mean(),
            token_overhead_tasks.assign(
                family="Тип стартової сторінки",
                category=token_overhead_tasks["page_type"].map(PAGE_TYPE_LABELS),
            )
            .groupby(["family", "category"], as_index=False)["token_overhead_pct"]
            .mean(),
        ],
        ignore_index=True,
    )

    page_rows: list[dict[str, Any]] = []
    for page_type, group in tasks.groupby("page_type"):
        for metric in ("recall_at_3", "ndcg_at_3"):
            page_rows.append(
                {
                    "page_type": PAGE_TYPE_LABELS.get(page_type, page_type),
                    "metric": METRIC_LABELS[metric],
                    "delta": float(group[f"required_minus_optional_{metric}"].mean()),
                    "n": int(len(group)),
                }
            )
    pages = pd.DataFrame(page_rows)

    grouped = (
        profile_cells.groupby(["safe_task_id", "repository", "profile", "condition"], as_index=False)[
            ["recall_at_3", "ndcg_at_3"]
        ]
        .mean()
    )
    profile_rows: list[dict[str, Any]] = []
    for metric_offset, metric in enumerate(("recall_at_3", "ndcg_at_3")):
        pivot = grouped.pivot(
            index=["safe_task_id", "repository", "profile"],
            columns="condition",
            values=metric,
        ).reset_index()
        pivot["delta"] = (
            pivot["FUNCTIONAL_REQUIRED_BEFORE_SOURCE"] - pivot["FUNCTIONAL_OPTIONAL"]
        )
        for profile_offset, (profile, group) in enumerate(pivot.groupby("profile")):
            result = _bootstrap_mean(
                group,
                "delta",
                seed=BOOTSTRAP_SEED + 100 + metric_offset * 20 + profile_offset,
            )
            profile_rows.append(
                {
                    "profile": profile,
                    "metric": METRIC_LABELS[metric],
                    **result,
                }
            )
    profiles = pd.DataFrame(profile_rows)

    loro_rows: list[dict[str, Any]] = []
    for repository in sorted(tasks["repository"].unique()):
        subset = tasks[tasks["repository"] != repository]
        for metric in ("recall_at_3", "ndcg_at_3"):
            loro_rows.append(
                {
                    "excluded": repository,
                    "metric": METRIC_LABELS[metric],
                    "estimate": float(subset[f"required_minus_optional_{metric}"].mean()),
                    "n": int(len(subset)),
                }
            )
    loro = pd.DataFrame(loro_rows)

    slot_tokens = np.array(
        [tasks[f"required_mean_page_{slot}_tokens"].sum() for slot in range(1, 6)],
        dtype=float,
    )
    slot_share = slot_tokens / slot_tokens.sum()
    wiki_slots = pd.DataFrame(
        {
            "slot": [f"Wiki-read {slot}" for slot in range(1, 6)],
            "token_share": slot_share,
            "tokens": slot_tokens,
        }
    )

    log_dose = np.log10(tasks["required_mean_wiki_tokens"].clip(lower=1))
    ndcg_delta = tasks["required_minus_optional_ndcg_at_3"]
    raw_spearman = _spearman(log_dose, ndcg_delta)
    centered_log_dose = log_dose - tasks.groupby("repository")["required_mean_wiki_tokens"].transform(
        lambda values: np.log10(values.clip(lower=1)).mean()
    )
    centered_delta = ndcg_delta - tasks.groupby("repository")["required_minus_optional_ndcg_at_3"].transform("mean")
    centered_spearman = _spearman(centered_log_dose, centered_delta)

    summary = {
        "task_rows": int(len(tasks)),
        "cell_rows": int(len(cells)),
        "repositories": int(tasks["repository"].nunique()),
        "profiles": int(profile_cells["profile"].nunique()),
        "required_minus_optional": {
            metric: float(tasks[f"required_minus_optional_{metric}"].mean())
            for metric in ("recall_at_3", "recall_at_5", "ndcg_at_3", "returned_set_f1")
        },
        "efficiency": {
            "elapsed_seconds_delta": float(tasks["required_minus_optional_elapsed_seconds"].mean()),
            "agent_steps_delta": float(tasks["required_minus_optional_agent_step_count"].mean()),
            "provider_tokens_delta": float(tasks["required_minus_optional_provider_total_tokens"].mean()),
        },
        "tokens": {
            "no_doc_mean_total": float(tasks["no_doc_mean_provider_total_tokens"].mean()),
            "optional_mean_total": float(tasks["optional_mean_provider_total_tokens"].mean()),
            "required_mean_total": float(tasks["required_mean_provider_total_tokens"].mean()),
            "absolute_delta": float(tasks["required_minus_optional_provider_total_tokens"].mean()),
            "grand_mean_overhead_pct": float(
                100
                * (
                    tasks["required_mean_provider_total_tokens"].mean()
                    / tasks["optional_mean_provider_total_tokens"].mean()
                    - 1
                )
            ),
            "mean_task_relative_overhead_pct": float(token_overhead_tasks["token_overhead_pct"].mean()),
            "median_task_relative_overhead_pct": float(token_overhead_tasks["token_overhead_pct"].median()),
        },
        "time": {
            "no_doc_mean_seconds": float(tasks["no_doc_mean_elapsed_seconds"].mean()),
            "optional_mean_seconds": float(tasks["optional_mean_elapsed_seconds"].mean()),
            "required_mean_seconds": float(tasks["required_mean_elapsed_seconds"].mean()),
            "optional_median_task_seconds": float(tasks["optional_mean_elapsed_seconds"].median()),
            "required_median_task_seconds": float(tasks["required_mean_elapsed_seconds"].median()),
            "required_p95_task_seconds": float(tasks["required_mean_elapsed_seconds"].quantile(0.95)),
            "absolute_delta_seconds": float(tasks["required_minus_optional_elapsed_seconds"].mean()),
            "grand_mean_overhead_pct": float(
                100
                * (
                    tasks["required_mean_elapsed_seconds"].mean()
                    / tasks["optional_mean_elapsed_seconds"].mean()
                    - 1
                )
            ),
        },
        "profile_economics": [
            {
                "profile": str(row["profile"]),
                "optional_provider_total_tokens": float(row["optional_provider_total_tokens"]),
                "required_provider_total_tokens": float(row["required_provider_total_tokens"]),
                "token_overhead_pct": float(row["token_overhead_pct"]),
                "optional_elapsed_seconds": float(row["optional_elapsed_seconds"]),
                "required_elapsed_seconds": float(row["required_elapsed_seconds"]),
                "time_overhead_pct": float(row["time_overhead_pct"]),
                "ndcg_delta": float(row["ndcg_delta"]),
                "recall_delta": float(row["recall_delta"]),
            }
            for _, row in profile_token_delta.sort_values("profile").iterrows()
        ],
        "mechanism": {
            "gold_seen_any_delta_pp": float(
                100 * (tasks["required_mean_gold_seen_any"] - tasks["optional_mean_gold_seen_any"]).mean()
            ),
            "gold_seen_by_3_delta_pp": float(
                100
                * (
                    tasks["required_mean_gold_seen_by_3_source_actions"]
                    - tasks["optional_mean_gold_seen_by_3_source_actions"]
                ).mean()
            ),
            "gold_targeted_delta_pp": float(
                100
                * (tasks["required_mean_gold_targeted_any"] - tasks["optional_mean_gold_targeted_any"]).mean()
            ),
        },
        "wiki": {
            "first_read_token_share": float(slot_share[0]),
            "raw_dose_ndcg_spearman": raw_spearman,
            "repo_centered_dose_ndcg_spearman": centered_spearman,
        },
        "bootstrap_replicates_for_report": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "provider_accessed": False,
        "model_executed": False,
    }

    return {
        "tasks": tasks,
        "cells": cells,
        "quality": quality,
        "contrasts": contrasts,
        "subgroups": subgroups,
        "mechanism": mechanism,
        "token_task_long": token_task_long,
        "time_task_long": time_task_long,
        "token_composition": token_composition,
        "profile_token_quality": profile_token_quality,
        "profile_token_delta": profile_token_delta,
        "token_overhead_type": token_overhead_type,
        "pages": pages,
        "profiles": profiles,
        "loro": loro,
        "wiki_slots": wiki_slots,
        "summary": summary,
    }


def build_figures(analysis: dict[str, Any]) -> dict[str, go.Figure]:
    tasks = analysis["tasks"]
    quality = analysis["quality"]
    contrasts = analysis["contrasts"]
    subgroups = analysis["subgroups"]
    mechanism = analysis["mechanism"]
    token_task_long = analysis["token_task_long"]
    time_task_long = analysis["time_task_long"]
    token_composition = analysis["token_composition"]
    profile_token_quality = analysis["profile_token_quality"]
    profile_token_delta = analysis["profile_token_delta"]
    token_overhead_type = analysis["token_overhead_type"]
    pages = analysis["pages"]
    profiles = analysis["profiles"]
    loro = analysis["loro"]
    wiki_slots = analysis["wiki_slots"]
    summary = analysis["summary"]

    quality_fig = px.line(
        quality,
        x="metric",
        y="mean",
        color="condition",
        markers=True,
        color_discrete_map=PALETTE,
        category_orders={"condition": list(CONDITION_LABELS.values())},
        labels={"metric": "Метрика", "mean": "Середня якість", "condition": "Умова"},
        title="Якість локалізації за умовою",
    )
    quality_fig.update_traces(line=dict(width=3), marker=dict(size=10))
    quality_fig.update_yaxes(range=[0.48, 0.87], tickformat=".2f")
    _base_layout(quality_fig, height=460)

    contrast_order = [
        f"{metric} · {comparison}"
        for comparison in ("проти доступної", "проти без документації")
        for metric in ("Recall@3", "Recall@5", "NDCG@3", "Returned-set F1")
    ]
    contrasts = contrasts.assign(label=contrasts["metric"] + " · " + contrasts["comparison"])
    contrast_fig = go.Figure()
    for comparison, color in (("проти доступної", "#2563eb"), ("проти без документації", "#0f766e")):
        subset = contrasts[contrasts["comparison"] == comparison]
        contrast_fig.add_trace(
            go.Scatter(
                x=subset["estimate"],
                y=subset["label"],
                mode="markers",
                name=comparison,
                marker=dict(size=11, color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=subset["high"] - subset["estimate"],
                    arrayminus=subset["estimate"] - subset["low"],
                    thickness=1.8,
                    width=6,
                ),
                customdata=np.stack([subset["low"], subset["high"], subset["n"]], axis=-1),
                hovertemplate="Δ=%{x:.4f}<br>95% CI [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>n=%{customdata[2]}<extra></extra>",
            )
        )
    contrast_fig.add_vline(x=0, line_color="#94a3b8", line_dash="dash")
    contrast_fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(contrast_order)))
    contrast_fig.update_xaxes(title="Різниця якості", tickformat="+.3f")
    contrast_fig.update_layout(title="Парні контрасти документаційної орієнтації")
    _base_layout(contrast_fig, height=570)

    subgroup_fig = go.Figure()
    for task_type, color in (
        ("Є явна підказка", PALETTE["Є явна підказка"]),
        ("Немає явної підказки", PALETTE["Немає явної підказки"]),
    ):
        subset = subgroups[subgroups["task_type"] == task_type]
        subgroup_fig.add_trace(
            go.Scatter(
                x=subset["estimate"],
                y=subset["metric"],
                mode="markers",
                name=task_type,
                marker=dict(size=12, color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=subset["high"] - subset["estimate"],
                    arrayminus=subset["estimate"] - subset["low"],
                    thickness=1.8,
                    width=6,
                ),
                customdata=np.stack([subset["low"], subset["high"], subset["n"]], axis=-1),
                hovertemplate="Δ=%{x:.4f}<br>95% CI [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>задач=%{customdata[2]}<extra></extra>",
            )
        )
    subgroup_fig.add_vline(x=0, line_color="#94a3b8", line_dash="dash")
    subgroup_fig.update_xaxes(title="Документація спочатку − доступна", tickformat="+.3f")
    subgroup_fig.update_layout(title="Ефект більший для задач без явного locator clue")
    _base_layout(subgroup_fig, height=430)

    mechanism_fig = px.bar(
        mechanism,
        x="mechanism",
        y="rate",
        color="condition",
        barmode="group",
        text=mechanism["rate"].map(lambda value: f"{100 * value:.1f}%"),
        color_discrete_map=PALETTE,
        category_orders={"condition": ["Документація доступна", "Документація спочатку"]},
        labels={"mechanism": "", "rate": "Частка клітинок", "condition": "Умова"},
        title="Після wiki агент частіше знаходить і таргетує правильний файл",
    )
    mechanism_fig.update_yaxes(range=[0.68, 1.0], tickformat=".0%")
    mechanism_fig.update_traces(textposition="outside")
    _base_layout(mechanism_fig, height=490)

    condition_order = list(CONDITION_LABELS.values())
    token_distribution_fig = px.box(
        token_task_long,
        x="condition",
        y="provider_total_tokens",
        color="condition",
        points="all",
        log_y=True,
        color_discrete_map=PALETTE,
        category_orders={"condition": condition_order},
        labels={"condition": "Умова", "provider_total_tokens": "Total provider tokens, log scale"},
        title="Provider tokens на задачу",
    )
    token_distribution_fig.update_traces(jitter=0.32, pointpos=0, marker=dict(size=5, opacity=0.42))
    token_distribution_fig.update_yaxes(dtick=1, tickformat="~s")
    token_distribution_fig.update_layout(showlegend=False)
    _base_layout(token_distribution_fig, height=500)

    time_distribution_fig = px.box(
        time_task_long,
        x="condition",
        y="elapsed_seconds",
        color="condition",
        points="all",
        color_discrete_map=PALETTE,
        category_orders={"condition": condition_order},
        labels={"condition": "Умова", "elapsed_seconds": "Provider-process wall-clock, секунд"},
        title="Повний wall-clock agent run",
    )
    time_distribution_fig.update_traces(jitter=0.32, pointpos=0, marker=dict(size=5, opacity=0.42))
    time_distribution_fig.update_layout(showlegend=False)
    _base_layout(time_distribution_fig, height=500)

    profile_cost_bars = profile_token_quality.assign(
        time_label=profile_token_quality["elapsed_seconds"].map(lambda value: f"{value:.1f} с"),
        token_label=profile_token_quality["provider_total_tokens"].map(lambda value: f"{value / 1000:.0f}k"),
    )
    profile_order = [
        f"{model} · {reasoning}"
        for model in ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra")
        for reasoning in ("low", "medium", "high")
        if f"{model} · {reasoning}" in set(profile_cost_bars["profile"])
    ]
    time_by_profile_fig = px.bar(
        profile_cost_bars,
        x="elapsed_seconds",
        y="profile",
        color="condition_label",
        barmode="group",
        orientation="h",
        text="time_label",
        color_discrete_map=PALETTE,
        category_orders={"profile": list(reversed(profile_order)), "condition_label": condition_order},
        labels={
            "elapsed_seconds": "Середній wall-clock, секунд",
            "profile": "Model · reasoning",
            "condition_label": "Умова",
        },
        title="Час за model/reasoning profile",
    )
    time_by_profile_fig.update_traces(textposition="outside")
    _base_layout(time_by_profile_fig, height=650)

    tokens_by_profile_fig = px.bar(
        profile_cost_bars,
        x="provider_total_tokens",
        y="profile",
        color="condition_label",
        barmode="group",
        orientation="h",
        text="token_label",
        color_discrete_map=PALETTE,
        category_orders={"profile": list(reversed(profile_order)), "condition_label": condition_order},
        labels={
            "provider_total_tokens": "Середні provider tokens",
            "profile": "Model · reasoning",
            "condition_label": "Умова",
        },
        title="Токени за model/reasoning profile",
    )
    tokens_by_profile_fig.update_traces(textposition="outside")
    _base_layout(tokens_by_profile_fig, height=650)

    component_colors = {
        "Cached input": "#60a5fa",
        "Non-cached input": "#1d4ed8",
        "Reasoning output": "#7c3aed",
        "Visible output": "#c4b5fd",
    }
    token_composition_fig = px.bar(
        token_composition,
        x="condition",
        y="tokens",
        color="component",
        barmode="stack",
        color_discrete_map=component_colors,
        category_orders={"condition": condition_order, "component": list(component_colors)},
        labels={"condition": "Умова", "tokens": "Середні токени на клітинку", "component": "Компонент"},
        title="Склад token usage без подвійного рахунку",
    )
    token_composition_fig.update_layout(barmode="stack")
    _base_layout(token_composition_fig, height=500)

    def profile_frontier(*, x_col: str, x_title: str, title: str) -> go.Figure:
        fig = go.Figure()
        optional = profile_token_quality[
            profile_token_quality["condition"] == "FUNCTIONAL_OPTIONAL"
        ].set_index("profile")
        required = profile_token_quality[
            profile_token_quality["condition"] == "FUNCTIONAL_REQUIRED_BEFORE_SOURCE"
        ].set_index("profile")
        line_x: list[float | None] = []
        line_y: list[float | None] = []
        for profile in optional.index:
            line_x.extend([float(optional.loc[profile, x_col]), float(required.loc[profile, x_col]), None])
            line_y.extend([float(optional.loc[profile, "ndcg_at_3"]), float(required.loc[profile, "ndcg_at_3"]), None])
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color="#cbd5e1", width=1.8),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        for code, label in (
            ("FUNCTIONAL_OPTIONAL", "Документація доступна"),
            ("FUNCTIONAL_REQUIRED_BEFORE_SOURCE", "Документація спочатку"),
        ):
            subset = profile_token_quality[profile_token_quality["condition"] == code]
            fig.add_trace(
                go.Scatter(
                    x=subset[x_col],
                    y=subset["ndcg_at_3"],
                    mode="markers",
                    name=label,
                    marker=dict(size=11, color=PALETTE[label], line=dict(width=1, color="white")),
                    customdata=subset[["profile", "recall_at_3"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        + x_title
                        + ": %{x:,.1f}<br>NDCG@3: %{y:.3f}<br>Recall@3: %{customdata[1]:.3f}<extra></extra>"
                    ),
                )
            )
        fig.update_xaxes(title=x_title)
        fig.update_yaxes(title="Середній NDCG@3")
        fig.update_layout(title=title)
        _base_layout(fig, height=520)
        return fig

    token_frontier_fig = profile_frontier(
        x_col="provider_total_tokens",
        x_title="Середні provider tokens",
        title="Profile-level trade-off: NDCG@3 проти token cost",
    )
    time_frontier_fig = profile_frontier(
        x_col="elapsed_seconds",
        x_title="Середній час, секунд",
        title="Profile-level trade-off: NDCG@3 проти wall-clock time",
    )

    profile_cost_delta = profile_token_delta.assign(
        time_size=profile_token_delta["time_overhead_pct"].abs().clip(lower=2)
    )
    profile_cost_delta_fig = px.scatter(
        profile_cost_delta,
        x="token_overhead_pct",
        y="ndcg_delta",
        color="time_overhead_pct",
        size="time_size",
        text="profile",
        color_continuous_scale="RdBu_r",
        labels={
            "token_overhead_pct": "Token overhead, %",
            "ndcg_delta": "NDCG@3 Δ",
            "time_overhead_pct": "Time overhead, %",
            "time_size": "|Time overhead|",
        },
        title="Чи окупається додаткова вартість у кожному profile?",
    )
    profile_cost_delta_fig.add_hline(y=0, line_color="#94a3b8", line_dash="dash")
    profile_cost_delta_fig.add_vline(x=0, line_color="#94a3b8", line_dash="dash")
    profile_cost_delta_fig.update_traces(textposition="top center", marker=dict(line=dict(width=0.8, color="white")))
    _base_layout(profile_cost_delta_fig, height=560)

    token_overhead_type_fig = px.bar(
        token_overhead_type,
        x="category",
        y="token_overhead_pct",
        color="family",
        facet_col="family",
        facet_col_wrap=2,
        text=token_overhead_type["token_overhead_pct"].map(lambda value: f"{value:+.1f}%"),
        labels={"category": "", "token_overhead_pct": "Середній task-relative token overhead", "family": "Зріз"},
        title="Де token overhead найбільший? Описові task/page-type зрізи",
    )
    token_overhead_type_fig.update_traces(textposition="outside", showlegend=False)
    token_overhead_type_fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    token_overhead_type_fig.update_xaxes(tickangle=-20)
    _base_layout(token_overhead_type_fig, height=560)

    efficiency = tasks.assign(
        task_label=tasks["task_type"].map(TASK_TYPE_LABELS),
        wiki_size=np.sqrt(tasks["required_mean_wiki_tokens"].clip(lower=1)),
        short_id=tasks["safe_task_id"].str[:10],
    )
    efficiency_fig = px.scatter(
        efficiency,
        x="required_minus_optional_elapsed_seconds",
        y="required_minus_optional_ndcg_at_3",
        color="task_label",
        size="wiki_size",
        hover_name="repository",
        hover_data={
            "short_id": True,
            "required_mean_wiki_tokens": ":.0f",
            "required_minus_optional_agent_step_count": ":+.2f",
            "wiki_size": False,
            "task_label": False,
        },
        color_discrete_map=PALETTE,
        labels={
            "required_minus_optional_elapsed_seconds": "Додатковий час, секунд",
            "required_minus_optional_ndcg_at_3": "NDCG@3 Δ",
            "task_label": "Тип задачі",
            "short_id": "Task ID",
            "required_mean_wiki_tokens": "Wiki tokens",
            "required_minus_optional_agent_step_count": "Кроки Δ",
        },
        title="Якість–вартість: користь неоднорідна між задачами",
    )
    efficiency_fig.add_hline(y=0, line_color="#94a3b8", line_dash="dash")
    efficiency_fig.add_vline(x=0, line_color="#94a3b8", line_dash="dash")
    efficiency_fig.update_traces(marker=dict(opacity=0.78, line=dict(width=0.7, color="white")))
    _base_layout(efficiency_fig, height=520)

    dose = tasks.assign(
        task_label=tasks["task_type"].map(TASK_TYPE_LABELS),
        short_id=tasks["safe_task_id"].str[:10],
    )
    dose_fig = px.scatter(
        dose,
        x="required_mean_wiki_tokens",
        y="required_minus_optional_ndcg_at_3",
        color="repository",
        symbol="task_label",
        hover_name="repository",
        hover_data={"short_id": True, "required_mean_wiki_tokens": ":.0f", "task_label": True},
        labels={
            "required_mean_wiki_tokens": "Wiki tokens, log scale",
            "required_minus_optional_ndcg_at_3": "NDCG@3 Δ",
            "task_label": "Тип задачі",
            "short_id": "Task ID",
        },
        title="Більший wiki-обсяг не означає більший ефект",
    )
    dose_fig.update_xaxes(type="log")
    dose_fig.add_hline(y=0, line_color="#94a3b8", line_dash="dash")
    dose_fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=(
            f"Spearman raw: {summary['wiki']['raw_dose_ndcg_spearman']:+.2f}<br>"
            f"після repo-centering: {summary['wiki']['repo_centered_dose_ndcg_spearman']:+.2f}"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="#cbd5e1",
        borderpad=7,
    )
    dose_fig.update_traces(marker=dict(size=10, opacity=0.78, line=dict(width=0.6, color="white")))
    _base_layout(dose_fig, height=520)

    page_fig = px.bar(
        pages,
        x="page_type",
        y="delta",
        color="metric",
        barmode="group",
        text=pages.apply(lambda row: f"{row['delta']:+.3f}<br>n={row['n']}", axis=1),
        color_discrete_map=PALETTE,
        labels={"page_type": "Тип стартової сторінки", "delta": "Різниця якості", "metric": "Метрика"},
        title="Тип стартової сторінки: описовий зріз, не причинний ефект",
    )
    page_fig.add_hline(y=0, line_color="#94a3b8", line_dash="dash")
    page_fig.update_traces(textposition="outside")
    page_fig.update_xaxes(tickangle=-18)
    _base_layout(page_fig, height=540)

    wiki_depth_fig = px.bar(
        wiki_slots[wiki_slots["token_share"] > 0],
        x="slot",
        y="token_share",
        text=wiki_slots.loc[wiki_slots["token_share"] > 0, "token_share"].map(lambda value: f"{100 * value:.2f}%"),
        labels={"slot": "Послідовне wiki-читання", "token_share": "Частка wiki-токенів"},
        title="Майже весь wiki-контекст надходить із першого читання",
        color_discrete_sequence=["#2563eb"],
    )
    wiki_depth_fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
    wiki_depth_fig.update_traces(textposition="outside")
    _base_layout(wiki_depth_fig, height=420)

    profile_fig = go.Figure()
    for metric, color in (("Recall@3", "#2563eb"), ("NDCG@3", "#0f766e")):
        subset = profiles[profiles["metric"] == metric].sort_values("estimate")
        profile_fig.add_trace(
            go.Scatter(
                x=subset["estimate"],
                y=subset["profile"],
                mode="markers",
                name=metric,
                marker=dict(size=10, color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=subset["high"] - subset["estimate"],
                    arrayminus=subset["estimate"] - subset["low"],
                    thickness=1.5,
                    width=5,
                ),
                customdata=np.stack([subset["low"], subset["high"]], axis=-1),
                hovertemplate="Δ=%{x:.4f}<br>95% CI [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<extra></extra>",
            )
        )
    profile_fig.add_vline(x=0, line_color="#94a3b8", line_dash="dash")
    profile_fig.update_xaxes(title="Документація спочатку − доступна", tickformat="+.3f")
    profile_fig.update_layout(title="Чутливість за model/reasoning profile")
    _base_layout(profile_fig, height=max(500, 60 * profiles["profile"].nunique()))

    loro_fig = go.Figure()
    for metric, color in (("Recall@3", "#2563eb"), ("NDCG@3", "#0f766e")):
        subset = loro[loro["metric"] == metric]
        loro_fig.add_trace(
            go.Scatter(
                x=subset["estimate"],
                y=subset["excluded"],
                mode="markers",
                name=metric,
                marker=dict(size=11, color=color),
                hovertemplate="Δ=%{x:.4f}<br>Виключено: %{y}<extra></extra>",
            )
        )
    loro_fig.add_vline(x=0, line_color="#94a3b8", line_dash="dash")
    loro_fig.update_xaxes(title="Середній ефект після виключення репозиторію", tickformat="+.3f")
    loro_fig.update_yaxes(title="Виключений репозиторій")
    loro_fig.update_layout(title="Leave-one-repository-out sensitivity")
    _base_layout(loro_fig, height=490)

    sorted_effects = tasks.sort_values("required_minus_optional_ndcg_at_3").copy()
    sorted_effects["task_order"] = np.arange(1, len(sorted_effects) + 1)
    sorted_effects["task_label"] = sorted_effects["task_type"].map(TASK_TYPE_LABELS)
    sorted_effects["short_id"] = sorted_effects["safe_task_id"].str[:10]
    distribution_fig = px.bar(
        sorted_effects,
        x="task_order",
        y="required_minus_optional_ndcg_at_3",
        color="task_label",
        hover_name="repository",
        hover_data={"short_id": True, "task_order": False, "task_label": False},
        color_discrete_map=PALETTE,
        labels={
            "task_order": "Задачі, впорядковані за ефектом",
            "required_minus_optional_ndcg_at_3": "NDCG@3 Δ",
            "task_label": "Тип задачі",
            "short_id": "Task ID",
        },
        title="Середній позитивний ефект приховує сильну неоднорідність задач",
    )
    distribution_fig.add_hline(y=0, line_color="#0f172a", line_width=1)
    distribution_fig.update_traces(marker_line_width=0)
    _base_layout(distribution_fig, height=460)

    return {
        "quality": quality_fig,
        "contrasts": contrast_fig,
        "subgroups": subgroup_fig,
        "mechanism": mechanism_fig,
        "token_distribution": token_distribution_fig,
        "time_distribution": time_distribution_fig,
        "time_by_profile": time_by_profile_fig,
        "tokens_by_profile": tokens_by_profile_fig,
        "token_composition": token_composition_fig,
        "token_frontier": token_frontier_fig,
        "time_frontier": time_frontier_fig,
        "profile_cost_delta": profile_cost_delta_fig,
        "token_overhead_type": token_overhead_type_fig,
        "efficiency": efficiency_fig,
        "dose": dose_fig,
        "pages": page_fig,
        "wiki_depth": wiki_depth_fig,
        "profiles": profile_fig,
        "loro": loro_fig,
        "distribution": distribution_fig,
    }


def _figure_html(fig: go.Figure, *, include_plotlyjs: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="inline" if include_plotlyjs else False,
        config={"displaylogo": False, "responsive": True, "scrollZoom": False},
    )


def build_html(analysis: dict[str, Any], figures: dict[str, go.Figure]) -> str:
    summary = analysis["summary"]
    delta = summary["required_minus_optional"]
    efficiency = summary["efficiency"]
    mechanism = summary["mechanism"]
    first_share = summary["wiki"]["first_read_token_share"]
    tokens = summary["tokens"]
    time = summary["time"]

    figure_order = [
        "quality",
        "contrasts",
        "subgroups",
        "distribution",
        "mechanism",
        "efficiency",
        "token_distribution",
        "time_distribution",
        "time_by_profile",
        "tokens_by_profile",
        "token_composition",
        "token_frontier",
        "time_frontier",
        "profile_cost_delta",
        "token_overhead_type",
        "wiki_depth",
        "dose",
        "pages",
        "profiles",
        "loro",
    ]
    rendered = {
        name: _figure_html(figures[name], include_plotlyjs=index == 0)
        for index, name in enumerate(figure_order)
    }

    return f"""<!doctype html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Документаційна орієнтація для локалізації файлів — дослідницький звіт</title>
  <style>
    :root {{ --ink:#142033; --muted:#5e6b7f; --line:#dfe6ef; --blue:#2563eb; --teal:#0f766e; --amber:#f59e0b; --paper:#fff; --wash:#f3f6fa; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:var(--wash); font:16px/1.6 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    a {{ color:var(--blue); }}
    .hero {{ background:linear-gradient(120deg,#0b1730 0%,#12306c 62%,#0f766e 130%); color:#fff; padding:72px 24px 58px; }}
    .hero-inner,.page {{ max-width:1180px; margin:0 auto; }}
    .eyebrow {{ letter-spacing:.12em; text-transform:uppercase; font-weight:700; font-size:.78rem; color:#93c5fd; }}
    h1 {{ margin:.35rem 0 1rem; max-width:920px; font-size:clamp(2.2rem,5vw,4.6rem); line-height:1.05; letter-spacing:-.045em; }}
    .hero p {{ max-width:850px; margin:0; color:#d9e6fb; font-size:1.18rem; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:10px 20px; margin-top:30px; color:#c5d8f5; font-size:.92rem; }}
    .page {{ padding:36px 24px 80px; }}
    nav {{ position:sticky; top:0; z-index:20; background:rgba(255,255,255,.94); backdrop-filter:blur(12px); border-bottom:1px solid var(--line); }}
    nav .inner {{ max-width:1180px; margin:auto; padding:12px 24px; display:flex; gap:18px; overflow:auto; white-space:nowrap; }}
    nav a {{ color:#334155; text-decoration:none; font-size:.9rem; font-weight:650; }}
    section {{ scroll-margin-top:66px; margin:28px 0; background:var(--paper); border:1px solid var(--line); border-radius:18px; padding:clamp(22px,4vw,44px); box-shadow:0 16px 42px rgba(15,23,42,.05); }}
    h2 {{ margin:0 0 8px; font-size:clamp(1.55rem,3vw,2.35rem); letter-spacing:-.025em; line-height:1.15; }}
    h3 {{ margin:28px 0 8px; font-size:1.25rem; }}
    .lede {{ color:var(--muted); max-width:880px; margin:0 0 20px; }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:24px 0; }}
    .card {{ border:1px solid var(--line); border-radius:14px; padding:18px; background:#fbfdff; }}
    .card strong {{ display:block; font-size:1.8rem; line-height:1.1; letter-spacing:-.03em; color:#0f172a; }}
    .card span {{ display:block; margin-top:7px; color:var(--muted); font-size:.88rem; }}
    .finding {{ border-left:4px solid var(--blue); padding:4px 0 4px 18px; margin:20px 0; color:#334155; }}
    .finding.teal {{ border-color:var(--teal); }}
    .finding.amber {{ border-color:var(--amber); }}
    .chart {{ margin:12px -16px 2px; min-height:380px; }}
    .two {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
    .note {{ background:#fff8e7; border:1px solid #f9d98a; border-radius:12px; padding:14px 16px; color:#713f12; font-size:.94rem; }}
    .method {{ background:#eef6ff; border:1px solid #bfdbfe; border-radius:12px; padding:16px; color:#1e3a5f; }}
    ul {{ padding-left:1.25rem; }}
    footer {{ color:#64748b; font-size:.88rem; padding:10px 4px 0; }}
    code {{ background:#eef2f7; padding:.12rem .35rem; border-radius:5px; }}
    @media (max-width:900px) {{ .cards,.two {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width:620px) {{ .cards,.two {{ grid-template-columns:1fr; }} .hero {{ padding-top:48px; }} section {{ border-radius:12px; padding:20px 15px; }} .chart {{ margin-left:-12px; margin-right:-12px; }} }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-inner">
      <div class="eyebrow">File-localization experiment · focused research report</div>
      <h1>Чи допомагає документація агенту локалізувати файли?</h1>
      <p>Сфокусований аналіз 1&nbsp;968 уже виконаних клітинок. Усі графіки побудовані з запечатаних CSV; нових модельних запусків або provider-викликів немає.</p>
      <div class="meta"><span>82 задачі</span><span>6 репозиторіїв</span><span>8 model/reasoning profiles</span><span>3 умови</span><span>10&nbsp;000 bootstrap replicates для нових sensitivity-зрізів</span></div>
    </div>
  </header>
  <nav><div class="inner"><a href="#answer">Відповідь</a><a href="#effect">Ефект</a><a href="#ambiguity">Неоднозначні задачі</a><a href="#mechanism">Механізм</a><a href="#cost">Токени й час</a><a href="#wiki">Wiki</a><a href="#sensitivity">Стійкість</a><a href="#limits">Обмеження</a></div></nav>
  <main class="page">
    <section id="answer">
      <h2>Коротка відповідь</h2>
      <p class="lede">Документаційна орієнтація перед source-пошуком помірно покращує ранжування файлів, особливо для задач без явної locator-підказки. Це не універсальне домінування: середня користь невелика, неоднорідна між задачами й має вимірювану вартість.</p>
      <div class="cards">
        <div class="card"><strong>{100 * delta['recall_at_3']:+.2f} п.п.</strong><span>Recall@3 проти optional</span></div>
        <div class="card"><strong>{100 * delta['ndcg_at_3']:+.2f} п.п.</strong><span>NDCG@3 проти optional</span></div>
        <div class="card"><strong>{efficiency['elapsed_seconds_delta']:+.1f} с</strong><span>середній додатковий час</span></div>
        <div class="card"><strong>{efficiency['agent_steps_delta']:+.2f}</strong><span>додаткових кроків агента</span></div>
      </div>
      <div class="finding"><strong>Головна теза:</strong> optional-доступність документації майже нічого не змінює, але короткий обовʼязковий orientation checkpoint змінює подальший source-пошук.</div>
      <div class="chart">{rendered['quality']}</div>
    </section>

    <section id="effect">
      <h2>1. Який середній ефект?</h2>
      <p class="lede">Найпослідовніший сигнал — Recall і NDCG. Returned-set F1 не має переконливого покращення, отже документаційна орієнтація допомагає насамперед включити й краще ранжувати правильні файли, а не зробити весь returned set чистішим.</p>
      <div class="chart">{rendered['contrasts']}</div>
      <div class="finding amber"><strong>Інтерпретація:</strong> це trade-off ранжування проти витрат, а не доказ того, що документація завжди покращує кожен аспект відповіді.</div>
    </section>

    <section id="ambiguity">
      <h2>2. Коли документація корисніша?</h2>
      <p class="lede">Для задач без явного імені файла, символу або locator clue середній NDCG-сигнал більший. Водночас інтервали для Recall ширші: підгрупа містить лише 43 незалежні задачі.</p>
      <div class="chart">{rendered['subgroups']}</div>
      <div class="chart">{rendered['distribution']}</div>
      <div class="finding"><strong>Практичний наслідок:</strong> селективна policy для неоднозначних задач виглядає перспективніше за blanket-вимогу для всіх задач.</div>
    </section>

    <section id="mechanism">
      <h2>3. Що змінюється після читання wiki?</h2>
      <p class="lede">Час переходу до source tools майже однаковий для явних і неоднозначних задач. Відмінність проявляється пізніше: правильний файл частіше входить у траєкторію, зʼявляється раніше та явно таргетується.</p>
      <div class="cards">
        <div class="card"><strong>{mechanism['gold_seen_any_delta_pp']:+.2f} п.п.</strong><span>gold побачено взагалі</span></div>
        <div class="card"><strong>{mechanism['gold_seen_by_3_delta_pp']:+.2f} п.п.</strong><span>gold до третьої source-дії</span></div>
        <div class="card"><strong>{mechanism['gold_targeted_delta_pp']:+.2f} п.п.</strong><span>gold явно таргетовано</span></div>
        <div class="card"><strong>{efficiency['agent_steps_delta']:+.2f}</strong><span>кроків за цю ширшу coverage</span></div>
      </div>
      <div class="chart">{rendered['mechanism']}</div>
      <div class="chart">{rendered['efficiency']}</div>
      <div class="finding teal"><strong>Найкраще пояснення поточних даних:</strong> mechanism — coverage і path selection, а не довше перебування у документації.</div>
    </section>

    <section id="cost">
      <h2>4. Скільки коштує кожна умова?</h2>
      <p class="lede">За однакової задачі й profile документаційна орієнтація має дві різні ціни: provider tokens та повний wall-clock agent run. <code>elapsed_seconds</code> вимірює час від старту provider-процесу до завершення всього запуску — model reasoning, tool calls і фінальної відповіді; це не окремий час читання wiki. Менше й лівіше на cost-quality графіках — дешевше; вище — кращий NDCG@3.</p>
      <div class="cards">
        <div class="card"><strong>{tokens['required_mean_total']:,.0f}</strong><span>provider tokens: документація спочатку</span></div>
        <div class="card"><strong>{tokens['optional_mean_total']:,.0f}</strong><span>provider tokens: документація доступна</span></div>
        <div class="card"><strong>{tokens['grand_mean_overhead_pct']:+.1f}%</strong><span>token overhead за відношенням загальних середніх</span></div>
        <div class="card"><strong>{time['required_mean_seconds']:.1f} с</strong><span>doc-first mean; task median {time['required_median_task_seconds']:.1f} с</span></div>
        <div class="card"><strong>{time['optional_mean_seconds']:.1f} с</strong><span>optional mean; task median {time['optional_median_task_seconds']:.1f} с</span></div>
        <div class="card"><strong>{time['grand_mean_overhead_pct']:+.1f}%</strong><span>wall-clock overhead; +{time['absolute_delta_seconds']:.1f} с</span></div>
      </div>
      <div class="two">
        <div class="chart">{rendered['token_distribution']}</div>
        <div class="chart">{rendered['time_distribution']}</div>
      </div>
      <div class="note"><strong>Чому середнє часу високе:</strong> розподіл має довгий хвіст і змішує швидкі та high-reasoning profiles. Для doc-first task-level медіана становить {time['required_median_task_seconds']:.1f} с, середнє {time['required_mean_seconds']:.1f} с, а 95-й перцентиль {time['required_p95_task_seconds']:.1f} с. Основний condition-overhead — не 71 с, а парна різниця +{time['absolute_delta_seconds']:.1f} с відносно optional.</div>
      <h3>Model × reasoning breakdown</h3>
      <p class="lede">Час і токени суттєво залежать від конкретного profile, тому нижче не змішуємо reasoning-рівні в одне середнє «по моделі». Для кожного profile показані всі три умови; відповідний quality-effect наведений у sensitivity-блоці та на cost-quality frontiers нижче.</p>
      <div class="chart">{rendered['time_by_profile']}</div>
      <div class="chart">{rendered['tokens_by_profile']}</div>
      <div class="finding teal"><strong>Видима структура:</strong> Sol-high і Luna-high мають найбільший абсолютний wall-clock, але найбільший відносний doc-first overhead у швидких Terra-low/medium profiles. Саме Terra-low/medium водночас дають найвиразніший NDCG gain, тому для них додаткова вартість має зміст; для profiles із майже нульовим quality gain вона виглядає менш виправданою.</div>
      <div class="finding amber"><strong>Дві коректні token-оцінки:</strong> відношення загальних середніх дає {tokens['grand_mean_overhead_pct']:+.2f}%; середнє від task-relative відношень — {tokens['mean_task_relative_overhead_pct']:+.2f}% (медіана {tokens['median_task_relative_overhead_pct']:+.2f}%). Перша відповідає на «скільки токенів у середньому», друга дає кожній задачі однакову вагу у відносному overhead.</div>
      <div class="chart">{rendered['token_composition']}</div>
      <div class="note"><strong>Як читати stack:</strong> total = cached input + non-cached input + reasoning output + visible output. Cached input уже є частиною input, а reasoning — частиною output, тому компоненти розкладено без подвійного рахунку. Це usage, не доларова ціна: cache може тарифікуватися інакше.</div>
      <div class="two">
        <div class="chart">{rendered['token_frontier']}</div>
        <div class="chart">{rendered['time_frontier']}</div>
      </div>
      <div class="chart">{rendered['profile_cost_delta']}</div>
      <div class="finding"><strong>Decision view:</strong> перехід від optional до doc-first корисний лише тоді, коли сіра лінія рухається достатньо вгору, щоб виправдати рух праворуч. У поточних даних найвиразніший quality gain зосереджений у gpt-5.6-terra low/medium; для більшості інших profiles token/time overhead є, а NDCG gain близький до нуля.</div>
      <div class="chart">{rendered['token_overhead_type']}</div>
      <div class="note"><strong>Обережно:</strong> task/page-type cost-зрізи описові. Тип сторінки змішаний із repository та snapshot, тому графік корисний для планування наступного експерименту, але не доводить причинність.</div>
    </section>

    <section id="wiki">
      <h2>5. Чи важливий обсяг і тип wiki?</h2>
      <p class="lede">{100 * first_share:.2f}% виміряних wiki-токенів припадає на перше читання. Тому цей експеримент переважно перевіряє стартову сторінку як orientation checkpoint, а не глибоку wiki-навігацію.</p>
      <div class="two">
        <div class="chart">{rendered['wiki_depth']}</div>
        <div class="chart">{rendered['dose']}</div>
      </div>
      <div class="chart">{rendered['pages']}</div>
      <div class="note"><strong>Обережно:</strong> тип і розмір сторінки сильно змішані з репозиторієм та snapshot-версією. Цей зріз не доводить, що короткий формат причинно кращий.</div>
    </section>

    <section id="sensitivity">
      <h2>6. Наскільки результат стійкий?</h2>
      <p class="lede">Два sensitivity-зрізи перевіряють, чи не пояснюється весь ефект одним model/reasoning profile або одним репозиторієм. Це вже сильніша перевірка, ніж автоматична таблиця кореляцій.</p>
      <div class="chart">{rendered['profiles']}</div>
      <div class="chart">{rendered['loro']}</div>
      <div class="finding amber"><strong>Важлива неоднорідність:</strong> більша частина
      середнього покращення зосереджена у профілях gpt-5.6-terra low/medium; для більшості
      інших профілів середній ефект близький до нуля. Після почергового виключення кожного
      репозиторію знак Recall@3 і NDCG@3 лишається позитивним, але без SymPy ефект суттєво
      слабшає. Отже результат не є універсальним для всіх agent profiles.</div>
      <div class="method">Інтервали для нових profile/subgroup зрізів отримані з 10&nbsp;000 repository-stratified whole-task bootstrap replicates. Основні раніше запечатані контрасти залишаються незмінними; ці додаткові оцінки є sensitivity-аналізом.</div>
    </section>

    <section id="limits">
      <h2>Що цей експеримент не доводить</h2>
      <ul>
        <li>Він не ізолює причинний ефект конкретного типу або обсягу стартової сторінки.</li>
        <li>Він майже не тестує глибоку навігацію між wiki-сторінками.</li>
        <li>Post-treatment trajectory-фічі описують механізм, але не є незалежно рандомізованими факторами.</li>
        <li>Середні ефекти приховують значні позитивні й негативні task-level результати.</li>
        <li>Автоматичні кореляції з широкого EDA слід використовувати для генерації гіпотез, а не як основні докази.</li>
      </ul>
      <h3>Рекомендоване формулювання</h3>
      <p>Обовʼязкова документаційна орієнтація перед source-пошуком помірно покращила file-ranking quality у вже виконаному наборі задач, із сильнішим сигналом для задач без явної locator-підказки. Ефект повʼязаний із кращою coverage правильних шляхів, має додаткову часову й token-вартість і не встановлює користь глибокого wiki retrieval.</p>
    </section>
    <footer>Джерела: <code>task_features.csv</code> і <code>cell_features.csv</code>. Task є незалежною одиницею; profile/repeat спершу агрегуються всередині task. Generated deterministically with seed {BOOTSTRAP_SEED}. Provider accessed: false. Model executed: false.</footer>
  </main>
</body>
</html>
"""


def write_notebook(data_dir: Path = DATA_DIR) -> None:
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Focused research analysis: documentation-guided file localization\n\n"
            "Відтворюваний notebook для найважливіших дослідницьких зрізів. "
            "Він читає вже запечатані CSV і не виконує моделей або provider-викликів."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n\n"
            "HERE = Path.cwd()\n"
            "if (HERE / 'task_features.csv').exists():\n"
            "    DATA_DIR = HERE\n"
            "    REPORT_DIR = HERE.parent / 'report'\n"
            "elif (HERE.parent / 'features' / 'task_features.csv').exists():\n"
            "    REPORT_DIR = HERE\n"
            "    DATA_DIR = HERE.parent / 'features'\n"
            "else:\n"
            "    REPORT_DIR = HERE / 'thesis' / 'final-results' / '2026-09-01-plan-0153a6e8-01' / 'report'\n"
            "    DATA_DIR = REPORT_DIR.parent / 'features'\n"
            "sys.path.insert(0, str(REPORT_DIR))\n"
            "import build_focused_research_report as report\n\n"
            "analysis = report.prepare_analysis(DATA_DIR)\n"
            "figures = report.build_figures(analysis)\n"
            "analysis['summary']"
        ),
        nbformat.v4.new_markdown_cell(
            "## 1. Основний ефект\n\n"
            "Спочатку дивимось на condition means та парні task-level контрасти."
        ),
        nbformat.v4.new_code_cell("figures['quality'].show(renderer='plotly_mimetype')\nfigures['contrasts'].show(renderer='plotly_mimetype')"),
        nbformat.v4.new_markdown_cell(
            "## 2. Неоднозначність задачі та неоднорідність ефекту\n\n"
            "Підгрупа без явної locator-підказки була зафіксована до оцінювання результатів."
        ),
        nbformat.v4.new_code_cell("figures['subgroups'].show(renderer='plotly_mimetype')\nfigures['distribution'].show(renderer='plotly_mimetype')"),
        nbformat.v4.new_markdown_cell(
            "## 3. Механізм і вартість\n\n"
            "Trajectory-фічі є post-treatment діагностикою: вони пояснюють шлях, але не є окремими причинними втручаннями."
        ),
        nbformat.v4.new_code_cell("figures['mechanism'].show(renderer='plotly_mimetype')\nfigures['efficiency'].show(renderer='plotly_mimetype')"),
        nbformat.v4.new_markdown_cell(
            "## 4. Token і time economics\n\n"
            "Порівнюємо розподіли повної вартості, token composition та profile-level quality–cost frontiers. "
            "Cached input і reasoning не додаються вдруге: вони розкладені як підмножини input/output."
        ),
        nbformat.v4.new_code_cell(
            "figures['token_distribution'].show(renderer='plotly_mimetype')\n"
            "figures['time_distribution'].show(renderer='plotly_mimetype')\n"
            "figures['time_by_profile'].show(renderer='plotly_mimetype')\n"
            "figures['tokens_by_profile'].show(renderer='plotly_mimetype')\n"
            "figures['token_composition'].show(renderer='plotly_mimetype')\n"
            "figures['token_frontier'].show(renderer='plotly_mimetype')\n"
            "figures['time_frontier'].show(renderer='plotly_mimetype')\n"
            "figures['profile_cost_delta'].show(renderer='plotly_mimetype')\n"
            "figures['token_overhead_type'].show(renderer='plotly_mimetype')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 5. Обсяг і структура wiki\n\n"
            "Перевіряємо глибину читання, дозу та тип стартової сторінки. Page type змішаний із repository/version."
        ),
        nbformat.v4.new_code_cell("figures['wiki_depth'].show(renderer='plotly_mimetype')\nfigures['dose'].show(renderer='plotly_mimetype')\nfigures['pages'].show(renderer='plotly_mimetype')"),
        nbformat.v4.new_markdown_cell(
            "## 6. Sensitivity\n\n"
            "Перевіряємо model/reasoning profiles і leave-one-repository-out оцінки."
        ),
        nbformat.v4.new_code_cell("figures['profiles'].show(renderer='plotly_mimetype')\nfigures['loro'].show(renderer='plotly_mimetype')"),
        nbformat.v4.new_markdown_cell(
            "## Висновок\n\n"
            "Документаційна орієнтація дає невелике покращення ranking quality, найбільш цікаве для неоднозначних задач. "
            "Механізм узгоджується з кращою coverage/path selection, а не з довшим читанням wiki. "
            "Doc-first використовує більше provider tokens і wall-clock time; для decision-making це слід оцінювати разом із profile-level quality gain. "
            "Експеримент не встановлює причинну перевагу певного типу сторінки або глибокого wiki retrieval."
        ),
    ]
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(notebook, {"metadata": {"path": str(data_dir)}})
    nbformat.write(notebook, OUTPUT_NOTEBOOK)


def main() -> None:
    analysis = prepare_analysis(DATA_DIR)
    figures = build_figures(analysis)
    OUTPUT_HTML.write_text(build_html(analysis, figures), encoding="utf-8")
    OUTPUT_SUMMARY.write_text(
        json.dumps(analysis["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_notebook(DATA_DIR)
    print(
        json.dumps(
            {
                "html": str(OUTPUT_HTML),
                "notebook": str(OUTPUT_NOTEBOOK),
                "summary": str(OUTPUT_SUMMARY),
                "task_rows": analysis["summary"]["task_rows"],
                "cell_rows": analysis["summary"]["cell_rows"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
