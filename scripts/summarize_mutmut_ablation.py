"""Summarize paired mutmut ablation results for PatchPilot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def exact_mcnemar(merged: pd.DataFrame, a: str, b: str) -> tuple[int, int, int, float]:
    """Return exact two-sided McNemar/binomial sign-test values."""
    a_only = int(((merged[f"{a}_succeeded"]) & (~merged[f"{b}_succeeded"])).sum())
    b_only = int(((~merged[f"{a}_succeeded"]) & (merged[f"{b}_succeeded"])).sum())
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, n, 1.0

    k = min(a_only, b_only)
    p_value = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))
    return a_only, b_only, n, p_value


def load_runs(path: Path, name: str) -> pd.DataFrame:
    """Load one condition's run CSV using task_id as the pairing key."""
    df = pd.read_csv(path)
    df = df[["task_id", "succeeded", "status", "patch_attempts", "steps", "tool_calls"]]
    return df.rename(
        columns={
            "succeeded": f"{name}_succeeded",
            "status": f"{name}_status",
            "patch_attempts": f"{name}_patches",
            "steps": f"{name}_steps",
            "tool_calls": f"{name}_tool_calls",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--one-shot", required=True, type=Path)
    parser.add_argument("--no-retry", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    merged = (
        load_runs(args.full, "full")
        .merge(load_runs(args.one_shot, "one_shot"), on="task_id")
        .merge(load_runs(args.no_retry, "no_retry"), on="task_id")
    )

    if len(merged) != 20:
        raise SystemExit(f"Expected 20 paired tasks, got {len(merged)}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    print("SUMMARY")
    for name in ["full", "one_shot", "no_retry"]:
        successes = int(merged[f"{name}_succeeded"].sum())
        print(name, successes, "/", len(merged), f"{successes / len(merged):.2%}")

    print("\nPAIRWISE EXACT MCNEMAR")
    for a, b in [("full", "one_shot"), ("full", "no_retry"), ("one_shot", "no_retry")]:
        a_only, b_only, n, p_value = exact_mcnemar(merged, a, b)
        print(
            f"{a} vs {b}: "
            f"{a}_only={a_only}, {b}_only={b_only}, "
            f"discordant={n}, p={p_value:.4f}"
        )

    print(f"\nPAIRED_CSV={args.output_csv}")


if __name__ == "__main__":
    main()
