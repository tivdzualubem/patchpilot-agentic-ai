"""Generate a reproducible statistical comparison report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from patchpilot.evaluation.statistics import exact_mcnemar_test


def read_successes(path: Path) -> dict[str, bool]:
    """Read task success outcomes from a PatchPilot runs CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    outcomes: dict[str, bool] = {}
    for row in rows:
        outcomes[row["task_id"]] = row["succeeded"] == "True"

    return outcomes


def pct(numerator: int, denominator: int) -> str:
    """Format a percentage."""
    if denominator == 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"


def markdown_lines(
    *,
    full_runs_csv: Path,
    baseline_runs_csv: Path,
    full_label: str,
    baseline_label: str,
    task_count: int,
    full_successes: int,
    baseline_successes: int,
    both_success: int,
    full_only_success: int,
    baseline_only_success: int,
    both_failure: int,
    discordant: int,
    p_value: float,
) -> list[str]:
    """Build a Markdown report from computed statistics."""
    return [
        "# PatchPilot Statistical Analysis",
        "",
        "This document records the reproducible statistical comparison between "
        f"`{full_label}` and `{baseline_label}` on PatchPilot-Bench v0.",
        "",
        "## Inputs",
        "",
        f"- Full-agent runs CSV: `{full_runs_csv}`",
        f"- Baseline/ablation runs CSV: `{baseline_runs_csv}`",
        f"- Paired tasks: {task_count}",
        "",
        "## Repair Success",
        "",
        "| Condition | Successes | Repair Rate |",
        "| --- | ---: | ---: |",
        (
            f"| `{full_label}` | {full_successes}/{task_count} | "
            f"{pct(full_successes, task_count)} |"
        ),
        (
            f"| `{baseline_label}` | {baseline_successes}/{task_count} | "
            f"{pct(baseline_successes, task_count)} |"
        ),
        "",
        "## Paired Success Table",
        "",
        "| Outcome | Count |",
        "| --- | ---: |",
        f"| Both succeeded | {both_success} |",
        f"| Full agent only succeeded | {full_only_success} |",
        f"| Baseline/ablation only succeeded | {baseline_only_success} |",
        f"| Both failed | {both_failure} |",
        "",
        "## Exact McNemar Test",
        "",
        (
            "Because both conditions ran on the same benchmark tasks, repair "
            "success is compared as paired binary outcomes."
        ),
        "",
        "| Test | Value |",
        "| --- | ---: |",
        f"| Discordant pairs | {discordant} |",
        f"| Exact McNemar two-sided p-value | {p_value:.4f} |",
        "",
        "## Interpretation",
        "",
        (
            f"The full agent repaired {full_successes}/{task_count} tasks, "
            f"while the baseline/ablation repaired {baseline_successes}/"
            f"{task_count} tasks."
        ),
        (
            f"The paired effect size is {full_only_success} tasks repaired "
            "only by the full agent versus "
            f"{baseline_only_success} tasks repaired only by the "
            "baseline/ablation."
        ),
        "",
        (
            f"Because the benchmark currently contains {task_count} tasks, "
            "p-values should be interpreted cautiously."
        ),
        (
            "The strongest evidence is the paired success difference and the "
            "fact that the full agent succeeds on tasks where the reduced "
            "condition exhausts its budget."
        ),
        "",
    ]


def generate_report(
    *,
    full_runs_csv: Path,
    baseline_runs_csv: Path,
    output_path: Path,
    full_label: str,
    baseline_label: str,
) -> None:
    """Generate a Markdown statistical report from paired run CSV files."""
    full = read_successes(full_runs_csv)
    baseline = read_successes(baseline_runs_csv)

    tasks = sorted(set(full) & set(baseline))
    if not tasks:
        raise ValueError("No paired tasks found in both CSV files.")

    result = exact_mcnemar_test(
        full_success=[full[task] for task in tasks],
        baseline_success=[baseline[task] for task in tasks],
    )

    full_successes = sum(1 for task in tasks if full[task])
    baseline_successes = sum(1 for task in tasks if baseline[task])

    lines = markdown_lines(
        full_runs_csv=full_runs_csv,
        baseline_runs_csv=baseline_runs_csv,
        full_label=full_label,
        baseline_label=baseline_label,
        task_count=len(tasks),
        full_successes=full_successes,
        baseline_successes=baseline_successes,
        both_success=result.both_success,
        full_only_success=result.full_only_success,
        baseline_only_success=result.baseline_only_success,
        both_failure=result.both_failure,
        discordant=result.discordant,
        p_value=result.p_value,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate PatchPilot statistical analysis report."
    )
    parser.add_argument("--full-runs-csv", required=True)
    parser.add_argument("--baseline-runs-csv", required=True)
    parser.add_argument(
        "--output",
        default="docs/statistical_analysis.md",
    )
    parser.add_argument(
        "--full-label",
        default="full-agent-live-qwen",
    )
    parser.add_argument(
        "--baseline-label",
        default="no-retry-live-qwen",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    generate_report(
        full_runs_csv=Path(args.full_runs_csv),
        baseline_runs_csv=Path(args.baseline_runs_csv),
        output_path=Path(args.output),
        full_label=args.full_label,
        baseline_label=args.baseline_label,
    )


if __name__ == "__main__":
    main()
