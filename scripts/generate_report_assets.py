"""Generate final PatchPilot report tables and figures."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "report_assets"

LABELS = {
    "full-live-qwen": "Full agent",
    "one-shot-live-qwen": "One-shot",
    "no-retry-live-qwen": "No-retry",
    "official-harness-smoke": "Official harness smoke",
    "controlled": "Controlled",
    "mutmut": "Mutmut",
    "quixbugs": "QuixBugs",
    "swebench-lite": "SWE-bench Lite",
}


def label(value: str) -> str:
    """Return a display label."""
    return LABELS.get(value, value)


def pct(value: object) -> str:
    """Format a unit rate as percent."""
    if value == "" or pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def num(value: object) -> str:
    """Format a numeric value or blank."""
    if value == "" or pd.isna(value):
        return ""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def short_task(task_id: str) -> str:
    """Shorten a mutmut task id for figures."""
    marker = "core-x-"
    if marker not in task_id:
        return task_id
    return task_id.split(marker, 1)[1].rsplit("-mutmut-", 1)[0]



def save_table(name: str, frame: pd.DataFrame) -> None:
    """Save a table to results and report assets."""
    frame.to_csv(RESULTS / f"{name}.csv", index=False)
    frame.to_csv(OUT / f"table_{name}.csv", index=False)


def save_fig(name: str) -> None:
    """Save the current matplotlib figure."""
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close()


def mcnemar(frame: pd.DataFrame, a: str, b: str) -> dict[str, object]:
    """Exact paired McNemar/binomial sign test."""
    a_col = f"{a}_succeeded"
    b_col = f"{b}_succeeded"
    a_only = int((frame[a_col] & ~frame[b_col]).sum())
    b_only = int((~frame[a_col] & frame[b_col]).sum())
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(math.comb(discordant, i) for i in range(k + 1))
        p_value = min(1.0, 2 * tail / (2**discordant))

    return {
        "comparison": f"{a.replace('_', ' ')} vs {b.replace('_', ' ')}",
        "first_only_successes": a_only,
        "second_only_successes": b_only,
        "discordant_pairs": discordant,
        "exact_p_value": f"{p_value:.4f}",
    }


def build_tables(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    """Build all CSV and markdown tables."""
    final = summary.copy()
    final["benchmark"] = final["benchmark"].map(label)
    final["condition"] = final["condition"].map(label)
    final["repair_rate"] = final["repair_rate"].map(pct)
    final["full_suite_pass_rate"] = final["full_suite_pass_rate"].map(pct)
    final["invalid_patch_rate"] = final["invalid_patch_rate"].map(pct)
    final["budget_exhaustions"] = final["budget_exhaustions"].map(num)
    final["escalations"] = final["escalations"].map(num)
    save_table("final_evaluation_summary", final)

    suite = pd.DataFrame(
        [
            {
                "benchmark": "Controlled",
                "role": "End-to-end sanity benchmark",
                "tasks": 12,
                "main_result": "12/12 for all variants",
            },
            {
                "benchmark": "Mutmut-generated",
                "role": "Primary ablation benchmark",
                "tasks": 20,
                "main_result": "Full 8/20; one-shot 6/20; no-retry 5/20",
            },
            {
                "benchmark": "QuixBugs smoke",
                "role": "External generalization check",
                "tasks": 8,
                "main_result": "3/8 repaired",
            },
            {
                "benchmark": "SWE-bench Lite",
                "role": "Official harness feasibility check",
                "tasks": 1,
                "main_result": "Blocked by local WSL/Docker I/O instability",
            },
        ]
    )
    save_table("benchmark_suite_summary", suite)

    outcomes = pd.DataFrame(
        {
            "task": [short_task(v) for v in paired["task_id"]],
            "full_agent": paired["full_succeeded"].map(
                {True: "pass", False: "fail"}
            ),
            "one_shot": paired["one_shot_succeeded"].map(
                {True: "pass", False: "fail"}
            ),
            "no_retry": paired["no_retry_succeeded"].map(
                {True: "pass", False: "fail"}
            ),
            "full_status": paired["full_status"],
            "one_shot_status": paired["one_shot_status"],
            "no_retry_status": paired["no_retry_status"],
        }
    )
    save_table("mutmut_task_outcomes_summary", outcomes)

    tests = pd.DataFrame(
        [
            mcnemar(paired, "full", "one_shot"),
            mcnemar(paired, "full", "no_retry"),
            mcnemar(paired, "one_shot", "no_retry"),
        ]
    )
    save_table("statistical_tests", tests)

    status_rows = []
    for prefix, name in [
        ("full", "Full agent"),
        ("one_shot", "One-shot"),
        ("no_retry", "No-retry"),
    ]:
        counts = Counter(paired[f"{prefix}_status"])
        for status, count in sorted(counts.items()):
            status_rows.append(
                {"condition": name, "status": status, "count": count}
            )
    save_table("mutmut_status_breakdown", pd.DataFrame(status_rows))

    deliverables = pd.DataFrame(
        [
            {
                "deliverable": "Tool-using repair agent",
                "repo_evidence": "src/patchpilot/agent and src/patchpilot/tools",
            },
            {
                "deliverable": "Controlled benchmark",
                "repo_evidence": "benchmarks/",
            },
            {
                "deliverable": "Real mutmut benchmark",
                "repo_evidence": "generated_benchmarks/mutmut_algorithms/",
            },
            {
                "deliverable": "Evaluation scripts",
                "repo_evidence": "scripts/run_evaluation.py",
            },
            {
                "deliverable": "Live demo",
                "repo_evidence": "demo/streamlit_app.py",
            },
        ]
    )
    save_table("project_deliverables", deliverables)


def plot_repair_rates(summary: pd.DataFrame) -> None:
    """Plot repair rates."""
    keep = summary["benchmark"].isin(["controlled", "mutmut", "quixbugs"])
    frame = summary[keep].copy()
    frame["name"] = [
        f"{label(b)} · {label(c)}"
        for b, c in zip(frame["benchmark"], frame["condition"], strict=True)
    ]
    frame["rate"] = frame["repair_rate"].astype(float) * 100

    plt.figure(figsize=(9, 5))
    plt.barh(frame["name"][::-1], frame["rate"][::-1])
    plt.xlabel("Repair rate (%)")
    plt.xlim(0, 105)
    plt.title("Repair rate across evaluated benchmarks")
    for i, value in enumerate(frame["rate"][::-1]):
        plt.text(value + 1, i, f"{value:.1f}%", va="center")
    save_fig("fig_repair_rate_comparison")


def plot_mutmut_ablation(summary: pd.DataFrame) -> None:
    """Plot mutmut ablation repair rates."""
    frame = summary[summary["benchmark"] == "mutmut"].copy()
    frame["name"] = frame["condition"].map(label)
    frame["rate"] = frame["repair_rate"].astype(float) * 100

    plt.figure(figsize=(7, 4))
    plt.bar(frame["name"], frame["rate"])
    plt.ylabel("Repair rate (%)")
    plt.ylim(0, 55)
    plt.title("Mutmut ablation comparison")
    for i, value in enumerate(frame["rate"]):
        plt.text(i, value + 1, f"{value:.1f}%", ha="center")
    save_fig("fig_mutmut_ablation_repair_rate")


def plot_invalid_patch(summary: pd.DataFrame) -> None:
    """Plot invalid patch rate."""
    keep = summary["benchmark"].isin(["controlled", "mutmut", "quixbugs"])
    frame = summary[keep].copy()
    frame["name"] = [
        f"{label(b)} · {label(c)}"
        for b, c in zip(frame["benchmark"], frame["condition"], strict=True)
    ]
    frame["rate"] = frame["invalid_patch_rate"].astype(float) * 100

    plt.figure(figsize=(9, 5))
    plt.barh(frame["name"][::-1], frame["rate"][::-1])
    plt.xlabel("Invalid patch rate (%)")
    plt.xlim(0, 5)
    plt.title("Safety metric: invalid patch rate")
    for i, value in enumerate(frame["rate"][::-1]):
        plt.text(value + 0.05, i, f"{value:.1f}%", va="center")
    save_fig("fig_invalid_patch_rate")


def plot_status_breakdown(paired: pd.DataFrame) -> None:
    """Plot mutmut statuses."""
    rows = []
    for prefix, name in [
        ("full", "Full agent"),
        ("one_shot", "One-shot"),
        ("no_retry", "No-retry"),
    ]:
        row = {"condition": name}
        row.update(Counter(paired[f"{prefix}_status"]))
        rows.append(row)

    frame = pd.DataFrame(rows).fillna(0).set_index("condition")
    ax = frame.plot(kind="bar", stacked=True, figsize=(7, 4))
    ax.set_ylabel("Task count")
    ax.set_title("Mutmut status breakdown")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig("fig_mutmut_status_breakdown")


def plot_task_matrix(paired: pd.DataFrame) -> None:
    """Plot task success matrix."""
    cols = ["full_succeeded", "one_shot_succeeded", "no_retry_succeeded"]
    matrix = paired[cols].astype(int).to_numpy()
    tasks = [short_task(v) for v in paired["task_id"]]

    fig, ax = plt.subplots(figsize=(6, 8))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Full", "One-shot", "No-retry"])
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=7)
    ax.set_title("Mutmut task-level repair outcomes")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "✓" if matrix[i, j] else "×", ha="center")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    save_fig("fig_mutmut_task_success_matrix")


def plot_workflow() -> None:
    """Plot a simple PatchPilot workflow figure."""
    steps = [
        "Buggy task",
        "Run tests",
        "Inspect code",
        "Apply bounded patch",
        "Verify tests",
        "Trace result",
    ]

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    x_positions = list(range(len(steps)))
    for x_pos, step in zip(x_positions, steps, strict=True):
        ax.text(
            x_pos,
            0.5,
            step,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "fill": False},
        )
        if x_pos < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_pos + 0.78, 0.5),
                xytext=(x_pos + 0.22, 0.5),
                arrowprops={"arrowstyle": "->"},
            )
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(0, 1)
    save_fig("fig_agent_workflow_loop")


def write_manifest() -> None:
    """Write an asset manifest."""
    assets = sorted(path.name for path in OUT.iterdir() if path.is_file())
    payload = {
        "asset_count": len(assets),
        "asset_directory": str(OUT.relative_to(ROOT)),
        "assets": assets,
    }
    path = OUT / "asset_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def main() -> None:
    """Generate all assets."""
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    for path in OUT.glob("*"):
        if path.is_file():
            path.unlink()

    summary = pd.read_csv(RESULTS / "evaluation_summary.csv")
    paired = pd.read_csv(RESULTS / "mutmut_paired_outcomes.csv")

    build_tables(summary.fillna(""), paired)
    plot_repair_rates(summary)
    plot_mutmut_ablation(summary)
    plot_invalid_patch(summary)
    plot_status_breakdown(paired)
    plot_task_matrix(paired)
    plot_workflow()
    write_manifest()

    print(f"ASSETS={OUT}")
    print(f"RESULTS={RESULTS}")
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            print(path.relative_to(ROOT))

def normalize_text_outputs() -> None:
    """Normalize generated text assets so repository whitespace checks pass."""
    for pattern in ("*.csv", "*.json", "*.svg"):
        for file_path in OUT.glob(pattern):
            text = file_path.read_text(encoding="utf-8")
            lines = [line.rstrip() for line in text.splitlines()]
            file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
    normalize_text_outputs()
