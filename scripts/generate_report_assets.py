"""Generate PatchPilot report figures, plots, and tables from project data."""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report_assets"

FULL_DIR = ROOT / "artifacts" / "evaluation" / "20260704-091749"
NO_RETRY_DIR = ROOT / "artifacts" / "evaluation" / "20260704-094853"
ONE_SHOT_DIR = ROOT / "artifacts" / "evaluation" / "20260704-135312"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_table_png(path: Path, rows: list[dict[str, object]], title: str) -> None:
    columns = list(rows[0])
    cell_text = []
    for row in rows:
        cell_text.append(
            ["\n".join(textwrap.wrap(str(row[col]), width=28)) for col in columns]
        )

    fig_height = max(2.8, 0.42 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)

    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("0.75")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_benchmark_catalog() -> list[dict[str, object]]:
    rows = []
    for task_path in sorted((ROOT / "benchmarks").glob("*/task.json")):
        data = json.loads(task_path.read_text(encoding="utf-8"))
        metadata = data.get("metadata", {})
        rows.append(
            {
                "Task": data["task_id"],
                "Category": metadata.get("category", "repair"),
                "Difficulty": metadata.get("difficulty", "unknown"),
                "Initial failures": metadata.get("initial_failures", ""),
                "Editable paths": ", ".join(data.get("allowed_paths", [])),
            }
        )
    return rows


def load_eval_summary() -> list[dict[str, object]]:
    dirs = [
        ("Full agent", FULL_DIR),
        ("One-shot baseline", ONE_SHOT_DIR),
        ("No-retry ablation", NO_RETRY_DIR),
    ]
    rows = []
    for label, directory in dirs:
        summary = read_csv(directory / "summary.csv")[0]
        rows.append(
            {
                "Condition": label,
                "Runs": summary["runs"],
                "Successes": summary["successes"],
                "Repair rate": f"{float(summary['repair_rate']) * 100:.1f}%",
                "Full-suite pass rate": (
                    f"{float(summary['full_suite_pass_rate']) * 100:.1f}%"
                ),
                "Invalid patch rate": (
                    f"{float(summary['invalid_patch_rate']) * 100:.1f}%"
                ),
                "Budget exhaustions": summary["budget_exhaustions"],
                "Escalations": summary["escalations"],
                "Mean patch attempts": f"{float(summary['mean_patch_attempts']):.2f}",
            }
        )
    return rows


def load_full_runs() -> list[dict[str, str]]:
    return read_csv(FULL_DIR / "runs.csv")


def plot_repair_rates(rows: list[dict[str, object]]) -> None:
    labels = [str(row["Condition"]) for row in rows]
    rates = [float(str(row["Repair rate"]).rstrip("%")) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, rates)
    ax.set_ylabel("Repair rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Repair Rate by Evaluation Condition", fontweight="bold")
    for index, value in enumerate(rates):
        ax.text(index, value + 2, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "plot_repair_rate_comparison.png", dpi=220)
    plt.close(fig)


def plot_failure_modes(rows: list[dict[str, object]]) -> None:
    labels = [str(row["Condition"]) for row in rows]
    budget = [int(row["Budget exhaustions"]) for row in rows]
    escalations = [int(row["Escalations"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], budget, width=0.36, label="Budget exhaustion")
    ax.bar([i + 0.18 for i in x], escalations, width=0.36, label="Escalation")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Runs")
    ax.set_title("Failure Modes by Condition", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "plot_failure_modes.png", dpi=220)
    plt.close(fig)


def plot_patch_attempts() -> None:
    runs = load_full_runs()
    labels = [row["task_id"] for row in runs]
    attempts = [float(row["patch_attempts"]) for row in runs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, attempts)
    ax.set_ylabel("Patch attempts")
    ax.set_title("Full-Agent Patch Attempts by Task", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "plot_patch_attempts_by_task.png", dpi=220)
    plt.close(fig)


def plot_paired_success() -> None:
    rows = [
        {"Outcome": "Both succeeded", "Count": 8},
        {"Outcome": "Full agent only", "Count": 4},
        {"Outcome": "One-shot only", "Count": 0},
        {"Outcome": "Both failed", "Count": 0},
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([r["Outcome"] for r in rows], [r["Count"] for r in rows])
    ax.set_ylabel("Tasks")
    ax.set_title("Paired Success Outcomes", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / "plot_paired_task_success.png", dpi=220)
    plt.close(fig)


def draw_architecture() -> None:
    boxes = [
        ("Benchmark task", 0.06, 0.65),
        ("Agent state\nbudget + trace", 0.28, 0.65),
        ("Policy\nQwen via Ollama", 0.50, 0.65),
        ("Restricted tools", 0.72, 0.65),
        ("Isolated workspace", 0.28, 0.25),
        ("Patch manager\nrollback + diff", 0.50, 0.25),
        ("Pytest verification", 0.72, 0.25),
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    for text, x, y in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.18, 0.18, fill=False, linewidth=1.8))
        ax.text(x + 0.09, y + 0.09, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.24, 0.74), (0.28, 0.74)),
        ((0.46, 0.74), (0.50, 0.74)),
        ((0.68, 0.74), (0.72, 0.74)),
        ((0.81, 0.65), (0.81, 0.43)),
        ((0.72, 0.34), (0.68, 0.34)),
        ((0.50, 0.34), (0.46, 0.34)),
        ((0.37, 0.43), (0.37, 0.65)),
    ]
    for start, end in arrows:
        ax.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5}
        )
    ax.set_title("PatchPilot System Architecture", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_architecture_diagram.png", dpi=220)
    plt.close(fig)


def draw_workflow() -> None:
    steps = ["Plan", "Act", "Observe", "Reflect", "Verify"]
    angles = [90, 18, -54, -126, 162]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis("off")
    import math

    coords = []
    for step, angle in zip(steps, angles, strict=True):
        rad = math.radians(angle)
        x = 0.5 + 0.33 * math.cos(rad)
        y = 0.5 + 0.33 * math.sin(rad)
        coords.append((x, y))
        ax.add_patch(plt.Circle((x, y), 0.10, fill=False, linewidth=2))
        ax.text(x, y, step, ha="center", va="center", fontsize=11, fontweight="bold")
    for i, start in enumerate(coords):
        end = coords[(i + 1) % len(coords)]
        ax.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5}
        )
    ax.text(0.5, 0.5, "bounded\nrepair loop", ha="center", va="center", fontsize=12)
    ax.set_title(
        "Plan–Act–Observe–Reflect–Verify Workflow", fontsize=15, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_agent_workflow_loop.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    benchmark_rows = load_benchmark_catalog()
    summary_rows = load_eval_summary()

    compliance_rows = [
        {
            "Proposal item": "Tool-using debugging agent",
            "Status": "Met",
            "Evidence": "Restricted repository tools, patching, pytest verification",
        },
        {
            "Proposal item": "Plan-Act-Observe-Reflect-Verify loop",
            "Status": "Met",
            "Evidence": "Trace records test, search, read, patch, verify, finish steps",
        },
        {
            "Proposal item": "Isolated workspace and safety boundaries",
            "Status": "Met",
            "Evidence": (
                "Disposable workspaces, allowed paths, forbidden tests, rollback"
            ),
        },
        {
            "Proposal item": "Open local model backend",
            "Status": "Met",
            "Evidence": "Ollama with Qwen2.5-Coder 1.5B",
        },
        {
            "Proposal item": "12-18 repair tasks",
            "Status": "Met",
            "Evidence": "PatchPilot-Bench v0 contains 12 tasks",
        },
        {
            "Proposal item": "Baseline/ablation comparison",
            "Status": "Met",
            "Evidence": (
                "One-shot baseline and no-retry ablation compared to full agent"
            ),
        },
        {
            "Proposal item": "Metrics and statistical analysis",
            "Status": "Met",
            "Evidence": (
                "Repair rate, pass rate, invalid patches, budgets, McNemar test"
            ),
        },
        {
            "Proposal item": "Demo/repo/report deliverables",
            "Status": "Met",
            "Evidence": "GitHub repo, Dockerized Streamlit demo, Hugging Face Space",
        },
    ]

    stats_rows = [
        {"Metric": "Full agent successes", "Value": "12/12"},
        {"Metric": "One-shot baseline successes", "Value": "8/12"},
        {"Metric": "Both succeeded", "Value": "8"},
        {"Metric": "Full agent only succeeded", "Value": "4"},
        {"Metric": "One-shot only succeeded", "Value": "0"},
        {"Metric": "Both failed", "Value": "0"},
        {"Metric": "Exact McNemar p-value", "Value": "0.1250"},
    ]

    write_csv(OUT / "table_benchmark_catalog.csv", benchmark_rows)
    write_csv(OUT / "table_evaluation_summary.csv", summary_rows)
    write_csv(OUT / "table_proposal_compliance.csv", compliance_rows)
    write_csv(OUT / "table_statistical_analysis.csv", stats_rows)

    save_table_png(
        OUT / "table_benchmark_catalog.png",
        benchmark_rows,
        "PatchPilot-Bench v0 Catalog",
    )
    save_table_png(
        OUT / "table_evaluation_summary.png", summary_rows, "Evaluation Summary"
    )
    save_table_png(
        OUT / "table_proposal_compliance.png",
        compliance_rows,
        "Proposal Compliance Audit",
    )
    save_table_png(
        OUT / "table_statistical_analysis.png",
        stats_rows,
        "Statistical Analysis Summary",
    )

    plot_repair_rates(summary_rows)
    plot_failure_modes(summary_rows)
    plot_patch_attempts()
    plot_paired_success()
    draw_architecture()
    draw_workflow()

    (OUT / "README.md").write_text(
        "# PatchPilot Report Assets\n\n"
        "Generated from local benchmark manifests and evaluation CSV files.\n",
        encoding="utf-8",
    )

    print(f"Generated report assets in {OUT}")


if __name__ == "__main__":
    main()
