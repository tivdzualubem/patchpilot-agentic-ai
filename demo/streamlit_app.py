"""Interactive Streamlit demo for PatchPilot."""

from __future__ import annotations

import csv
import difflib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOTS = (
    ("Controlled benchmark", PROJECT_ROOT / "benchmarks"),
    (
        "Mutmut · algorithms",
        PROJECT_ROOT / "generated_benchmarks" / "mutmut_algorithms",
    ),
    (
        "Mutmut · collections",
        PROJECT_ROOT / "generated_benchmarks" / "mutmut_collections",
    ),
    (
        "Mutmut · text data",
        PROJECT_ROOT / "generated_benchmarks" / "mutmut_textdata",
    ),
)
FINAL_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "final-research-evaluation-6524a37"
    / "condition_summary.csv"
)
FINAL_EXPERIMENT = "final-research-evaluation-6524a37-20260714-090253"
FINAL_COMMIT = "6524a37"


@st.cache_data(show_spinner=False)
def manifests() -> list[dict[str, Any]]:
    """Load benchmark manifests suitable for the live demo."""
    tasks: list[dict[str, Any]] = []
    for family, root in BENCHMARK_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/task.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            data["benchmark_family"] = family
            data["manifest_path"] = str(path.relative_to(PROJECT_ROOT))
            tasks.append(data)
    return tasks


@st.cache_data(show_spinner=False)
def final_result_rows() -> list[dict[str, str]]:
    """Load final-evaluation evidence when the result assets exist."""
    if not FINAL_RESULTS.is_file():
        return []
    with FINAL_RESULTS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_files(task: dict[str, Any]) -> list[Path]:
    repository = PROJECT_ROOT / str(task["repository_root"])
    files = sorted((repository / "src").rglob("*.py"))
    return [path for path in files if path.name != "__init__.py"]


def test_files(task: dict[str, Any]) -> list[Path]:
    repository = PROJECT_ROOT / str(task["repository_root"])
    return sorted((repository / "tests").rglob("test_*.py"))


def file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def render_info_cards(task: dict[str, Any]) -> None:
    cards = (
        ("Benchmark", str(task["benchmark_family"]).replace("Mutmut · ", "")),
        ("Defect category", str(task["defect_category"]).replace("_", " ")),
        ("Initial failures", str(task["expected_initial_failures"])),
        ("Editable paths", ", ".join(task["allowed_paths"])),
    )
    for column, (label, value) in zip(st.columns(4), cards, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="info-card">
                  <div class="info-label">{label}</div>
                  <div class="info-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_code_viewer(title: str, files: list[Path], key: str) -> None:
    st.subheader(title)
    if not files:
        st.info("No matching files were found.")
        return
    labels = {str(path.relative_to(PROJECT_ROOT)): path for path in files}
    selected = st.selectbox(
        f"{title} file",
        list(labels),
        key=key,
        label_visibility="collapsed",
    )
    st.caption(selected)
    with st.expander("View file", expanded=True):
        st.code(file_text(labels[selected]), language="python", line_numbers=True)


def load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload.get("state", payload)
    return state if isinstance(state, dict) else {}


def render_timeline(trace_path: Path) -> None:
    state = load_trace(trace_path)
    actions = state.get("actions", [])
    observations = state.get("observations", [])
    st.subheader("Agent execution timeline")
    if not actions:
        st.warning("No actions were recorded in this trace.")
        return

    for index, (action, observation) in enumerate(
        zip(actions, observations, strict=False),
        start=1,
    ):
        tool = action.get("tool", "unknown")
        status = observation.get("status", "unknown")
        expanded = index == 1 or status not in {"ok", "passed"}
        with st.expander(
            f"Step {index}: {tool} → {status}",
            expanded=expanded,
        ):
            summary = observation.get("summary")
            if summary:
                st.write(summary)
            rationale = action.get("rationale")
            if rationale:
                st.caption(f"Rationale: {rationale}")
            output = observation.get("output")
            if output:
                st.code(str(output)[:6000], language="text")


def render_diff(
    *,
    original_root: Path,
    repaired_root: Path,
    changed_files: list[str],
) -> None:
    st.subheader("Generated patch")
    if not changed_files:
        st.info("No changed files were recorded.")
        return

    for relative in changed_files:
        diff = difflib.unified_diff(
            file_text(original_root / relative).splitlines(keepends=True),
            file_text(repaired_root / relative).splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
        with st.expander(relative, expanded=True):
            st.code("".join(diff), language="diff")


def run_live_repair(task: dict[str, Any], model: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_demo_task.py",
            "--task-id",
            str(task["task_id"]),
            "--manifest-path",
            str(task["manifest_path"]),
            "--model",
            model,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    payload["ok"] = True
    return payload


def render_evaluation_snapshot() -> None:
    rows = final_result_rows()
    with st.expander("Research evidence · final 212-run evaluation"):
        st.caption(
            f"Commit {FINAL_COMMIT} · {FINAL_EXPERIMENT} · "
            "53 tasks × 4 conditions · hidden verification"
        )
        if not rows:
            st.info("Final result assets are not installed in this checkout yet.")
            return

        by_condition = {row.get("condition", ""): row for row in rows}
        labels = (
            ("one-shot", "One-shot"),
            ("fixed-workflow", "Fixed workflow"),
            ("tool-agent-no-reflection", "Tool agent"),
            ("full-reflective-agent", "Reflective agent"),
        )
        for column, (condition, label) in zip(st.columns(4), labels, strict=True):
            row = by_condition.get(condition, {})
            successes = row.get("hidden_verified_successes", "—")
            rate = row.get("hidden_verified_rate", "")
            delta = f"{100 * float(rate):.1f}% hidden verified" if rate else None
            column.metric(label, f"{successes}/53", delta=delta, delta_color="off")

        st.caption(
            "Supporting evidence only—the main experience remains the live repair demo."
        )


def render_task_selector() -> dict[str, Any]:
    all_tasks = manifests()
    if not all_tasks:
        st.error("No demo-capable benchmark manifests were found.")
        st.stop()

    families = sorted({str(task["benchmark_family"]) for task in all_tasks})
    family = st.selectbox("Benchmark family", families)
    family_tasks = [task for task in all_tasks if task["benchmark_family"] == family]
    labels = {f"{task['title']} · {task['task_id']}": task for task in family_tasks}
    task = labels[st.selectbox("Prepared repair task", list(labels))]

    st.markdown(f"**Repair objective:** {task['goal']}")
    render_info_cards(task)
    with st.expander("Task metadata"):
        st.json(
            {
                "task_id": task["task_id"],
                "title": task["title"],
                "benchmark_family": task["benchmark_family"],
                "manifest_path": task["manifest_path"],
                "defect_category": task["defect_category"],
                "allowed_paths": task["allowed_paths"],
                "forbidden_paths": task["forbidden_paths"],
                "test_command": task["test_command"],
            }
        )
    return task


def render_run_result(result: dict[str, Any], task: dict[str, Any]) -> None:
    verified = result.get("status") == "succeeded" and bool(
        result.get("full_suite_passed")
    )
    if verified:
        st.success("Repair verified: the full pytest suite passed.")
    else:
        st.error("The run completed without a verified repair.")

    metrics = st.columns(4)
    metrics[0].metric("Status", str(result.get("status", "unknown")))
    metrics[1].metric("Steps", str(result.get("steps", "—")))
    metrics[2].metric("Patch attempts", str(result.get("patch_attempts", "—")))
    metrics[3].metric("Changed files", str(len(result.get("changed_files", []))))

    if result.get("final_message"):
        st.info(str(result["final_message"]))

    verification_tab, trajectory_tab, patch_tab = st.tabs(
        ["Verification", "Agent trajectory", "Patch diff"]
    )
    with verification_tab:
        state_class = "verified" if verified else "failed"
        st.markdown(
            f"""
            <div class="verification-card {state_class}">
              <strong>Executable verification</strong><br>
              status={result.get("status", "unknown")}<br>
              full_suite_passed={result.get("full_suite_passed", False)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Run artifacts"):
            st.caption(f"Workspace: `{result.get('workspace', '')}`")
            st.caption(f"Trace: `{result.get('trace', '')}`")

    with trajectory_tab:
        trace_path = Path(str(result.get("trace", "")))
        if trace_path.is_file():
            render_timeline(trace_path)
        else:
            st.warning("The trace artifact was not found.")

    with patch_tab:
        render_diff(
            original_root=PROJECT_ROOT / str(task["repository_root"]),
            repaired_root=Path(str(result.get("repository", ""))),
            changed_files=list(result.get("changed_files", [])),
        )


def render_interactive_demo() -> None:
    st.header("Live repair demonstration")
    st.write(
        "Choose a prepared defective Python task, inspect its source and tests, "
        "then run the actual PatchPilot repair pipeline."
    )
    task = render_task_selector()

    source_tab, tests_tab = st.tabs(["Broken source before repair", "Regression tests"])
    with source_tab:
        render_code_viewer(
            "Broken source before repair",
            source_files(task),
            f"source_{task['task_id']}",
        )
    with tests_tab:
        render_code_viewer(
            "Regression tests",
            test_files(task),
            f"tests_{task['task_id']}",
        )

    st.divider()
    st.subheader("Run PatchPilot")
    model_options = {
        "qwen2.5-coder:3b": "Validated final-evaluation model",
        "qwen2.5-coder:1.5b": "Faster legacy demo option",
    }
    model = st.selectbox(
        "Ollama model",
        list(model_options),
        format_func=lambda value: f"{value} — {model_options[value]}",
    )
    st.info(
        "PatchPilot runs failing tests, inspects bounded source paths, generates "
        "a patch, checks syntax, and reports success only after pytest verification."
    )

    if st.button("Run PatchPilot on this task", type="primary"):
        with st.spinner("PatchPilot is repairing and verifying the selected task..."):
            result = run_live_repair(task, model)
        if not result.get("ok"):
            st.error(
                "The live run failed before a result artifact was produced. "
                "Check Ollama availability and the diagnostic output."
            )
            if result.get("stdout"):
                st.code(str(result["stdout"]), language="text")
            if result.get("stderr"):
                st.code(str(result["stderr"]), language="text")
            return
        render_run_result(result, task)


def main() -> None:
    st.set_page_config(
        page_title="PatchPilot · Live Repair Demo",
        page_icon="🛠️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1240px;
        }
        .hero {
            padding: 2rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 1rem;
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: white;
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 3rem;
            margin: 0 0 0.4rem 0;
        }
        .hero p {
            max-width: 900px;
            font-size: 1.08rem;
            color: #dbeafe;
        }
        .badge {
            display: inline-block;
            margin: 0.25rem 0.3rem 0 0;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #dbeafe;
            color: #0f172a;
            font-weight: 700;
        }
        .info-card {
            min-height: 108px;
            padding: 0.9rem 1rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 0.8rem;
            background: rgba(30, 41, 59, 0.35);
            overflow-wrap: anywhere;
        }
        .info-label {
            margin-bottom: 0.45rem;
            color: #94a3b8;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .info-value {
            color: #f8fafc;
            font-size: 1.05rem;
            font-weight: 650;
            line-height: 1.3;
        }
        .verification-card {
            padding: 1rem 1.1rem;
            border-radius: 0.8rem;
            line-height: 1.7;
        }
        .verification-card.verified {
            border: 1px solid rgba(34, 197, 94, 0.45);
            background: rgba(22, 101, 52, 0.24);
        }
        .verification-card.failed {
            border: 1px solid rgba(239, 68, 68, 0.45);
            background: rgba(127, 29, 29, 0.24);
        }
        div[data-testid="stCodeBlock"] {
            max-height: 560px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
          <h1>PatchPilot</h1>
          <p>
            A bounded agentic Python repair prototype. Inspect a defective
            benchmark, run the live repair pipeline, and review the verified
            outcome, tool trajectory, and generated patch.
          </p>
          <span class="badge">Live repair</span>
          <span class="badge">Bounded tools</span>
          <span class="badge">Agent trace</span>
          <span class="badge">Pytest verification</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_evaluation_snapshot()
    with st.sidebar:
        st.header("Demo flow")
        st.write("1. Choose a prepared defective task.")
        st.write("2. Inspect source and regression tests.")
        st.write("3. Run PatchPilot with the local model.")
        st.write("4. Review verification, trajectory, and patch.")
        st.divider()
        st.caption("This is a live repair demonstration, not a chat interface.")

    render_interactive_demo()
    st.divider()
    st.caption(
        "PatchPilot is a bounded research prototype. It reports success only "
        "after executable test verification."
    )


if __name__ == "__main__":
    main()
