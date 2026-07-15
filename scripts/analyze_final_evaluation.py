#!/usr/bin/env python3
"""Generate PatchPilot's final paired statistical analysis from runs.csv."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONDITIONS = [
    "one-shot",
    "fixed-workflow",
    "tool-agent-no-reflection",
    "full-reflective-agent",
]


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = stats.norm.ppf(0.975)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return centre - half, centre + half


def holm(values: list[float]) -> list[float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.runs_csv)

    for column in [
        "succeeded",
        "hidden_verified_success",
        "visible_hidden_disagreement",
        "policy_failure",
        "budget_exhausted",
    ]:
        frame[column] = frame[column].map(parse_bool)

    for column in [
        "elapsed_seconds",
        "model_calls",
        "decision_parse_failures",
        "steps",
        "tool_calls",
        "invalid_patch_count",
        "no_progress_rejection_count",
        "reflection_count",
        "hypothesis_revision_count",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    if (
        len(frame) != 212
        or frame["task_id"].nunique() != 53
        or frame["condition"].nunique() != 4
        or frame.duplicated(["task_id", "condition"]).any()
    ):
        raise SystemExit("The runs CSV is not the complete 53 × 4 matrix.")

    summaries = []
    for condition in CONDITIONS:
        group = frame[frame["condition"] == condition]
        successes = int(group["hidden_verified_success"].sum())
        low, high = wilson(successes, len(group))
        summaries.append(
            {
                "condition": condition,
                "runs": len(group),
                "hidden_verified_successes": successes,
                "hidden_verified_rate": successes / len(group),
                "hidden_ci95_low": low,
                "hidden_ci95_high": high,
                "visible_successes": int(group["succeeded"].sum()),
                "mean_elapsed_seconds": float(group["elapsed_seconds"].mean()),
                "median_elapsed_seconds": float(group["elapsed_seconds"].median()),
                "mean_model_calls": float(group["model_calls"].mean()),
                "parse_failure_runs": int((group["decision_parse_failures"] > 0).sum()),
                "policy_failures": int(group["policy_failure"].sum()),
                "reflections": int(group["reflection_count"].sum()),
                "hypothesis_revisions": int(group["hypothesis_revision_count"].sum()),
            }
        )
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "condition_summary.csv",
        index=False,
    )

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
    q = (
        (k - 1)
        * (k * np.sum(column_sums**2) - total**2)
        / (k * total - np.sum(row_sums**2))
    )
    q_p = float(stats.chi2.sf(q, k - 1))

    pairwise = []
    for first, second in itertools.combinations(CONDITIONS, 2):
        a = outcomes[first].to_numpy()
        b = outcomes[second].to_numpy()
        a_only = int(((a == 1) & (b == 0)).sum())
        b_only = int(((a == 0) & (b == 1)).sum())
        discordant = a_only + b_only
        p_value = (
            1.0
            if discordant == 0
            else float(
                stats.binomtest(
                    min(a_only, b_only),
                    discordant,
                    0.5,
                ).pvalue
            )
        )
        pairwise.append(
            {
                "condition_a": first,
                "condition_b": second,
                "a_only_successes": a_only,
                "b_only_successes": b_only,
                "discordant_pairs": discordant,
                "paired_risk_difference": float(a.mean() - b.mean()),
                "exact_mcnemar_p_raw": p_value,
            }
        )
    adjusted = holm([row["exact_mcnemar_p_raw"] for row in pairwise])
    for row, value in zip(pairwise, adjusted, strict=True):
        row["exact_mcnemar_p_holm"] = value
    pd.DataFrame(pairwise).to_csv(
        args.output_dir / "pairwise_exact_mcnemar_holm.csv",
        index=False,
    )

    result = {
        "total_runs": len(frame),
        "tasks": int(frame["task_id"].nunique()),
        "conditions": int(frame["condition"].nunique()),
        "cochran_q": float(q),
        "cochran_q_df": k - 1,
        "cochran_q_p": q_p,
        "condition_summary": summaries,
        "pairwise_mcnemar": pairwise,
    }
    (args.output_dir / "final_statistical_analysis.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("ANALYSIS_COMPLETE=1")
    print(f"OUTPUT_DIR={args.output_dir}")
    print(f"COCHRAN_Q={q:.6f}")
    print(f"COCHRAN_P={q_p:.12f}")


if __name__ == "__main__":
    main()
