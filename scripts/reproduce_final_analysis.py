#!/usr/bin/env python3
"""Reproduce every published PatchPilot final-analysis table from raw runs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CONDITIONS = [
    "one-shot",
    "fixed-workflow",
    "tool-agent-no-reflection",
    "full-reflective-agent",
]

LABELS = {
    "one-shot": "One-shot",
    "fixed-workflow": "Fixed workflow",
    "tool-agent-no-reflection": "Tool agent, no reflection",
    "full-reflective-agent": "Full reflective agent",
}

BOOLEAN_COLUMNS = [
    "succeeded",
    "hidden_verified_success",
    "visible_hidden_disagreement",
    "policy_failure",
    "budget_exhausted",
]

NUMERIC_COLUMNS = [
    "elapsed_seconds",
    "model_calls",
    "decision_parse_failures",
    "steps",
    "tool_calls",
    "invalid_patch_count",
    "no_progress_rejection_count",
    "reflection_count",
    "hypothesis_revision_count",
]


def parse_bool(value: object) -> bool:
    """Parse common CSV boolean encodings."""
    return str(value).strip().lower() in {"1", "true", "yes"}


def wilson(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval."""
    if total == 0:
        return 0.0, 0.0
    z = float(stats.norm.ppf(0.975))
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return centre - half, centre + half


def holm(values: list[float]) -> list[float]:
    """Apply Holm family-wise error correction."""
    order = np.argsort(values)
    adjusted = [1.0] * len(values)
    previous = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * values[index])
        value = max(previous, value)
        adjusted[index] = value
        previous = value
    return adjusted


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write deterministic LF-only CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def load_runs(path: Path) -> pd.DataFrame:
    """Load and validate the exact 53 x 4 final run matrix."""
    frame = pd.read_csv(path)

    required = {
        "task_id",
        "condition",
        "origin_type",
        "difficulty",
        "defect_category",
        "succeeded",
        "hidden_verified_success",
        "visible_hidden_disagreement",
        "hidden_suite_status",
        "terminal_failure_category",
        "elapsed_seconds",
        "model_calls",
        "decision_parse_failures",
        "steps",
        "tool_calls",
        "policy_failure",
        "budget_exhausted",
        "invalid_patch_count",
        "no_progress_rejection_count",
        "reflection_count",
        "hypothesis_revision_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"runs.csv is missing columns: {missing}")

    for column in BOOLEAN_COLUMNS:
        frame[column] = frame[column].map(parse_bool)

    for column in NUMERIC_COLUMNS:
        frame[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(0)
        )

    if (
        len(frame) != 212
        or frame["task_id"].nunique() != 53
        or set(frame["condition"]) != set(CONDITIONS)
        or frame.duplicated(["task_id", "condition"]).any()
    ):
        raise SystemExit(
            "The runs CSV is not the complete paired 53 x 4 matrix."
        )

    counts = frame.groupby("condition").size().to_dict()
    if any(counts.get(condition) != 53 for condition in CONDITIONS):
        raise SystemExit(f"Unexpected condition counts: {counts}")

    return frame


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the full published condition summary."""
    rows: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        group = frame[frame["condition"] == condition]
        runs = len(group)
        visible = int(group["succeeded"].sum())
        hidden = int(group["hidden_verified_success"].sum())
        low, high = wilson(hidden, runs)
        total_seconds = float(group["elapsed_seconds"].sum())

        rows.append(
            {
                "condition": condition,
                "label": LABELS[condition],
                "runs": runs,
                "visible_successes": visible,
                "visible_rate": visible / runs,
                "hidden_verified_successes": hidden,
                "hidden_verified_rate": hidden / runs,
                "hidden_ci95_low": low,
                "hidden_ci95_high": high,
                "visible_hidden_disagreements": int(
                    group["visible_hidden_disagreement"].sum()
                ),
                "mean_elapsed_seconds": float(
                    group["elapsed_seconds"].mean()
                ),
                "median_elapsed_seconds": float(
                    group["elapsed_seconds"].median()
                ),
                "p95_elapsed_seconds": float(
                    group["elapsed_seconds"].quantile(0.95)
                ),
                "total_elapsed_seconds": total_seconds,
                "seconds_per_hidden_verified_success": (
                    total_seconds / hidden if hidden else math.inf
                ),
                "mean_model_calls": float(
                    group["model_calls"].mean()
                ),
                "mean_steps": float(group["steps"].mean()),
                "mean_tool_calls": float(
                    group["tool_calls"].mean()
                ),
                "parse_failure_runs": int(
                    (group["decision_parse_failures"] > 0).sum()
                ),
                "mean_decision_parse_failures": float(
                    group["decision_parse_failures"].mean()
                ),
                "policy_failures": int(
                    group["policy_failure"].sum()
                ),
                "budget_exhaustions": int(
                    group["budget_exhausted"].sum()
                ),
                "invalid_patch_runs": int(
                    (group["invalid_patch_count"] > 0).sum()
                ),
                "no_progress_runs": int(
                    (
                        group["no_progress_rejection_count"] > 0
                    ).sum()
                ),
                "reflection_count": int(
                    group["reflection_count"].sum()
                ),
                "hypothesis_revision_count": int(
                    group["hypothesis_revision_count"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def task_level_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one row per task with all four hidden outcomes."""
    outcomes = (
        frame.pivot(
            index="task_id",
            columns="condition",
            values="hidden_verified_success",
        )[CONDITIONS]
        .astype(int)
        .reset_index()
    )

    metadata = (
        frame[
            [
                "task_id",
                "origin_type",
                "difficulty",
                "defect_category",
            ]
        ]
        .drop_duplicates("task_id")
        .sort_values("task_id")
    )

    return (
        outcomes.merge(metadata, on="task_id", validate="one_to_one")
        .sort_values("task_id")
        .reset_index(drop=True)
    )


def disagreement_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return visible-success/hidden-failure disagreements."""
    columns = [
        "task_id",
        "condition",
        "origin_type",
        "difficulty",
        "succeeded",
        "hidden_suite_status",
        "hidden_verified_success",
        "elapsed_seconds",
    ]
    return frame.loc[
        frame["visible_hidden_disagreement"], columns
    ].reset_index(drop=True)


def failure_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    """Count terminal failure categories by condition."""
    rows: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        group = frame[frame["condition"] == condition].copy()
        category = (
            group["terminal_failure_category"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace(
                "",
                "none_success_or_hidden_disagreement",
            )
        )
        for name, count in category.value_counts().items():
            rows.append(
                {
                    "condition": condition,
                    "terminal_failure_category": name,
                    "runs": int(count),
                }
            )

    return pd.DataFrame(rows)


def parse_failure_association(frame: pd.DataFrame) -> pd.DataFrame:
    """Associate decision-parse failures with hidden success."""
    rows: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        group = frame[frame["condition"] == condition]
        has_failure = group["decision_parse_failures"] > 0
        success = group["hidden_verified_success"]

        success_with = int((has_failure & success).sum())
        failure_with = int((has_failure & ~success).sum())
        success_without = int((~has_failure & success).sum())
        failure_without = int((~has_failure & ~success).sum())

        odds_ratio, p_value = stats.fisher_exact(
            [
                [success_with, failure_with],
                [success_without, failure_without],
            ],
            alternative="two-sided",
        )

        rows.append(
            {
                "condition": condition,
                "parse_failure_runs": int(has_failure.sum()),
                "success_rate_with_parse_failure": (
                    success_with / int(has_failure.sum())
                    if has_failure.any()
                    else 0.0
                ),
                "success_rate_without_parse_failure": (
                    success_without / int((~has_failure).sum())
                    if (~has_failure).any()
                    else 0.0
                ),
                "fisher_exact_odds_ratio": float(odds_ratio),
                "fisher_exact_p": float(p_value),
            }
        )

    return pd.DataFrame(rows)


def stratified_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Build exploratory origin, difficulty, and defect summaries."""
    rows: list[dict[str, Any]] = []

    for grouping in (
        "origin_type",
        "difficulty",
        "defect_category",
    ):
        for level in sorted(frame[grouping].dropna().unique()):
            subset = frame[frame[grouping] == level]
            for condition in sorted(CONDITIONS):
                group = subset[subset["condition"] == condition]
                runs = len(group)
                hidden = int(
                    group["hidden_verified_success"].sum()
                )
                low, high = wilson(hidden, runs)
                rows.append(
                    {
                        "grouping": grouping,
                        "level": level,
                        "condition": condition,
                        "runs": runs,
                        "hidden_verified_successes": hidden,
                        "hidden_verified_rate": (
                            hidden / runs if runs else 0.0
                        ),
                        "hidden_ci95_low": low,
                        "hidden_ci95_high": high,
                        "visible_successes": int(
                            group["succeeded"].sum()
                        ),
                        "mean_elapsed_seconds": float(
                            group["elapsed_seconds"].mean()
                        ),
                        "parse_failure_runs": int(
                            (
                                group["decision_parse_failures"] > 0
                            ).sum()
                        ),
                    }
                )

    return pd.DataFrame(rows)


def paired_binary_statistics(
    frame: pd.DataFrame,
) -> tuple[float, float, pd.DataFrame]:
    """Compute Cochran Q and pairwise exact McNemar tests."""
    outcomes = frame.pivot(
        index="task_id",
        columns="condition",
        values="hidden_verified_success",
    )[CONDITIONS].astype(int)

    matrix = outcomes.to_numpy()
    k = matrix.shape[1]
    column_sums = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)
    total = matrix.sum()
    q_value = float(
        (k - 1)
        * (k * np.sum(column_sums**2) - total**2)
        / (k * total - np.sum(row_sums**2))
    )
    q_p = float(stats.chi2.sf(q_value, k - 1))

    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(CONDITIONS, 2):
        first_values = outcomes[first].to_numpy()
        second_values = outcomes[second].to_numpy()
        first_only = int(
            (
                (first_values == 1)
                & (second_values == 0)
            ).sum()
        )
        second_only = int(
            (
                (first_values == 0)
                & (second_values == 1)
            ).sum()
        )
        discordant = first_only + second_only
        p_value = (
            1.0
            if discordant == 0
            else float(
                stats.binomtest(
                    min(first_only, second_only),
                    discordant,
                    0.5,
                ).pvalue
            )
        )
        rows.append(
            {
                "condition_a": first,
                "condition_b": second,
                "a_only_successes": first_only,
                "b_only_successes": second_only,
                "discordant_pairs": discordant,
                "paired_risk_difference": float(
                    first_values.mean()
                    - second_values.mean()
                ),
                "exact_mcnemar_p_raw": p_value,
            }
        )

    adjusted = holm(
        [row["exact_mcnemar_p_raw"] for row in rows]
    )
    for row, value in zip(rows, adjusted, strict=True):
        row["exact_mcnemar_p_holm"] = value
        row["significant_after_holm_0_05"] = value < 0.05

    return q_value, q_p, pd.DataFrame(rows)


def paired_latency_statistics(
    frame: pd.DataFrame,
) -> tuple[float, float, pd.DataFrame]:
    """Compute Friedman and paired Wilcoxon latency tests."""
    latency = frame.pivot(
        index="task_id",
        columns="condition",
        values="elapsed_seconds",
    )[CONDITIONS]

    friedman = stats.friedmanchisquare(
        *(latency[condition].to_numpy() for condition in CONDITIONS)
    )

    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(CONDITIONS, 2):
        first_values = latency[first].to_numpy()
        second_values = latency[second].to_numpy()
        differences = first_values - second_values
        result = stats.wilcoxon(
            first_values,
            second_values,
            alternative="two-sided",
            method="auto",
        )
        rows.append(
            {
                "condition_a": first,
                "condition_b": second,
                "mean_difference_seconds_a_minus_b": float(
                    differences.mean()
                ),
                "median_difference_seconds_a_minus_b": float(
                    np.median(differences)
                ),
                "wilcoxon_statistic": float(result.statistic),
                "wilcoxon_p_raw": float(result.pvalue),
            }
        )

    adjusted = holm([row["wilcoxon_p_raw"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["wilcoxon_p_holm"] = value
        row["significant_after_holm_0_05"] = value < 0.05

    return (
        float(friedman.statistic),
        float(friedman.pvalue),
        pd.DataFrame(rows),
    )


def save_figures(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    parse_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate the five report-relevant evaluation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [LABELS[value] for value in summary["condition"]]
    rates = summary["hidden_verified_rate"].to_numpy() * 100
    lower = (
        summary["hidden_verified_rate"]
        - summary["hidden_ci95_low"]
    ).to_numpy() * 100
    upper = (
        summary["hidden_ci95_high"]
        - summary["hidden_verified_rate"]
    ).to_numpy() * 100

    plt.figure(figsize=(9, 5))
    positions = np.arange(len(labels))
    plt.bar(positions, rates)
    plt.errorbar(
        positions,
        rates,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        capsize=5,
    )
    plt.xticks(positions, labels, rotation=15, ha="right")
    plt.ylabel("Hidden-verified repair rate (%)")
    plt.ylim(0, 100)
    plt.title("Hidden-verified repair rates with 95% Wilson intervals")
    plt.tight_layout()
    plt.savefig(
        output_dir / "hidden_verified_rates_ci95.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, summary["mean_elapsed_seconds"])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Mean elapsed time (seconds)")
    plt.title("Mean latency by condition")
    plt.tight_layout()
    plt.savefig(
        output_dir / "mean_latency_by_condition.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    mutmut = frame[frame["origin_type"] == "mutmut"]
    mutmut_rows: list[tuple[str, float]] = []
    for condition in CONDITIONS:
        group = mutmut[mutmut["condition"] == condition]
        mutmut_rows.append(
            (
                LABELS[condition],
                float(group["hidden_verified_success"].mean()) * 100,
            )
        )

    plt.figure(figsize=(9, 5))
    plt.bar(
        [row[0] for row in mutmut_rows],
        [row[1] for row in mutmut_rows],
    )
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Hidden-verified repair rate (%)")
    plt.ylim(0, 100)
    plt.title("Mutation-generated task performance")
    plt.tight_layout()
    plt.savefig(
        output_dir / "mutmut_hidden_verified_rates.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    x_positions = np.arange(len(parse_table))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(
        x_positions - width / 2,
        parse_table[
            "success_rate_with_parse_failure"
        ].to_numpy()
        * 100,
        width=width,
        label="With parse failure",
    )
    plt.bar(
        x_positions + width / 2,
        parse_table[
            "success_rate_without_parse_failure"
        ].to_numpy()
        * 100,
        width=width,
        label="Without parse failure",
    )
    plt.xticks(
        x_positions,
        [LABELS[value] for value in parse_table["condition"]],
        rotation=15,
        ha="right",
    )
    plt.ylabel("Hidden-verified repair rate (%)")
    plt.ylim(0, 100)
    plt.title("Repair success conditional on decision parsing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "parse_failure_success_rates.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    origin_counts = (
        frame[["task_id", "origin_type"]]
        .drop_duplicates()
        ["origin_type"]
        .value_counts()
        .sort_index()
    )
    plt.figure(figsize=(7, 4))
    plt.bar(origin_counts.index.tolist(), origin_counts.values)
    plt.ylabel("Task count")
    plt.title("Primary benchmark composition")
    plt.tight_layout()
    plt.savefig(
        output_dir / "benchmark_composition.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def write_markdown(
    summary: pd.DataFrame,
    q_value: float,
    q_p: float,
    friedman_value: float,
    friedman_p: float,
    output_path: Path,
) -> None:
    """Write a concise human-readable statistical report."""
    lines = [
        "# PatchPilot Final Statistical Analysis",
        "",
        "This report is regenerated directly from the committed raw "
        "212-run matrix.",
        "",
        "## Primary outcome",
        "",
        "| Condition | Hidden verified | Rate | Visible | Mean time |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['label']} | "
            f"{row['hidden_verified_successes']}/{row['runs']} | "
            f"{row['hidden_verified_rate'] * 100:.1f}% | "
            f"{row['visible_successes']}/{row['runs']} | "
            f"{row['mean_elapsed_seconds']:.2f} s |"
        )

    lines.extend(
        [
            "",
            f"Cochran's Q: Q(3)={q_value:.3f}, p={q_p:.12g}.",
            "",
            "No claim of causal reflection benefit is made because "
            "the recorded reflective condition completed zero "
            "reflection and hypothesis-revision events.",
            "",
            "## Latency",
            "",
            f"Friedman test: chi-square(3)={friedman_value:.3f}, "
            f"p={friedman_p:.12g}.",
            "",
            "The separate targeted-only versus full-suite runtime "
            "verification ablation proposed during planning was not "
            "executed in the final 212-run matrix. Visible-versus-hidden "
            "verification disagreement is reported instead and should "
            "be interpreted descriptively, not as a causal ablation.",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path(
            "results/final-research-evaluation-6524a37/"
            "raw/runs.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    """Generate every published final-analysis artifact."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_runs(args.runs_csv)
    summary = condition_summary(frame)
    outcomes = task_level_outcomes(frame)
    disagreements = disagreement_rows(frame)
    failures = failure_taxonomy(frame)
    parse_table = parse_failure_association(frame)
    stratified = stratified_results(frame)
    q_value, q_p, mcnemar = paired_binary_statistics(frame)
    friedman_value, friedman_p, latency = (
        paired_latency_statistics(frame)
    )

    write_csv(summary, output_dir / "condition_summary.csv")
    write_csv(
        outcomes,
        output_dir / "task_level_hidden_outcomes.csv",
    )
    write_csv(
        disagreements,
        output_dir / "visible_hidden_disagreements.csv",
    )
    write_csv(failures, output_dir / "failure_taxonomy.csv")
    write_csv(
        parse_table,
        output_dir / "parse_failure_association.csv",
    )
    write_csv(
        stratified,
        output_dir / "stratified_results.csv",
    )
    write_csv(
        mcnemar,
        output_dir / "pairwise_exact_mcnemar_holm.csv",
    )
    write_csv(
        latency,
        output_dir / "paired_latency_wilcoxon_holm.csv",
    )

    result = {
        "schema_version": "2.0",
        "source_runs_csv": str(args.runs_csv),
        "source_runs_sha256": sha256(args.runs_csv),
        "total_runs": len(frame),
        "tasks": int(frame["task_id"].nunique()),
        "conditions": len(CONDITIONS),
        "cochran_q": q_value,
        "cochran_q_df": len(CONDITIONS) - 1,
        "cochran_q_p": q_p,
        "friedman_latency_chi_square": friedman_value,
        "friedman_latency_df": len(CONDITIONS) - 1,
        "friedman_latency_p": friedman_p,
        "condition_summary": summary.to_dict(
            orient="records"
        ),
        "pairwise_mcnemar": mcnemar.to_dict(
            orient="records"
        ),
        "paired_latency_wilcoxon": latency.to_dict(
            orient="records"
        ),
        "scope": {
            "reflection_events_observed": 0,
            "hypothesis_revisions_observed": 0,
            "runtime_verification_ablation_executed": False,
            "visible_hidden_analysis_executed": True,
        },
    }
    (
        output_dir / "final_statistical_analysis.json"
    ).write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    write_markdown(
        summary,
        q_value,
        q_p,
        friedman_value,
        friedman_p,
        output_dir / "PatchPilot_Final_Statistical_Analysis.md",
    )
    save_figures(
        frame,
        summary,
        parse_table,
        output_dir / "figures",
    )

    generated = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    manifest = {
        "source_runs_csv": str(args.runs_csv),
        "source_runs_sha256": sha256(args.runs_csv),
        "generated_file_count": len(generated),
        "generated_files": generated,
    }
    (
        output_dir / "reproduction_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("FINAL_ANALYSIS_REPRODUCED=1")
    print("RUNS=212")
    print("TASKS=53")
    print("CONDITIONS=4")
    print(f"COCHRAN_Q={q_value:.6f}")
    print(f"COCHRAN_P={q_p:.12f}")
    print(f"FRIEDMAN_CHI_SQUARE={friedman_value:.6f}")
    print(f"FRIEDMAN_P={friedman_p:.12g}")
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
