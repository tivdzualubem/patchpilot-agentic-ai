"""Interactive Streamlit demo for PatchPilot."""

from __future__ import annotations

import difflib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = PROJECT_ROOT / "benchmarks"
DEMO_DATA = PROJECT_ROOT / "demo" / "data"
DOCS = PROJECT_ROOT / "docs"


def read_markdown(path: Path) -> str:
    """Read Markdown content when available."""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file when available."""
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def manifests() -> list[dict[str, Any]]:
    """Load benchmark manifests."""
    tasks: list[dict[str, Any]] = []
    for path in sorted(BENCHMARKS.glob("*/task.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["manifest_path"] = str(path)
        data["task_dir"] = path.parent.name
        tasks.append(data)
    return tasks


def task_by_id(task_id: str) -> dict[str, Any]:
    """Return one benchmark manifest by task id."""
    for task in manifests():
        if task["task_id"] == task_id:
            return task
    raise KeyError(task_id)


def source_files(task: dict[str, Any]) -> list[Path]:
    """Return source files for a benchmark task."""
    repo = PROJECT_ROOT / str(task["repository_root"])
    files = sorted((repo / "src").glob("*.py"))
    return [path for path in files if path.name != "__init__.py"]


def test_files(task: dict[str, Any]) -> list[Path]:
    """Return test files for a benchmark task."""
    repo = PROJECT_ROOT / str(task["repository_root"])
    return sorted((repo / "tests").glob("test_*.py"))


def file_text(path: Path) -> str:
    """Read text safely for display."""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def render_file_panel(title: str, path: Path) -> None:
    """Render one code file."""
    st.subheader(title)
    st.caption(str(path.relative_to(PROJECT_ROOT)))
    st.code(file_text(path), language="python")


def load_trace(path: Path) -> dict[str, Any]:
    """Load a PatchPilot trace JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload.get("state", payload)
    if not isinstance(state, dict):
        return {}
    return state


def render_timeline(trace_path: Path) -> None:
    """Render a trace timeline."""
    state = load_trace(trace_path)
    actions = state.get("actions", [])
    observations = state.get("observations", [])

    if not actions:
        st.warning("No actions found in this trace.")
        return

    st.subheader("Agent execution timeline")
    for index, (action, observation) in enumerate(
        zip(actions, observations, strict=False),
        start=1,
    ):
        tool = action.get("tool", "unknown")
        status = observation.get("status", "unknown")
        summary = observation.get("summary", "")
        with st.expander(f"Step {index}: {tool} → {status}", expanded=True):
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
    """Render diffs between original and repaired files."""
    st.subheader("Patch diff")
    if not changed_files:
        st.info("No changed files were recorded.")
        return

    for relative in changed_files:
        before = original_root / relative
        after = repaired_root / relative
        before_lines = file_text(before).splitlines(keepends=True)
        after_lines = file_text(after).splitlines(keepends=True)
        diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
        with st.expander(relative, expanded=True):
            st.code("".join(diff), language="diff")


def run_live_repair(task_id: str, model: str) -> dict[str, Any]:
    """Run PatchPilot against one selected benchmark task."""
    result = subprocess.run(
        [
            "python",
            "scripts/run_demo_task.py",
            "--task-id",
            task_id,
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


def render_task_selector() -> dict[str, Any]:
    """Render benchmark task selector."""
    all_tasks = manifests()
    task_ids = [task["task_id"] for task in all_tasks]

    selected_id = st.selectbox(
        "Choose a benchmark repair task",
        task_ids,
        index=0,
    )
    task = task_by_id(selected_id)

    st.write(task["goal"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Difficulty", str(task["difficulty"]))
    c2.metric("Category", str(task["defect_category"]))
    c3.metric("Initial failures", str(task["expected_initial_failures"]))
    c4.metric("Editable paths", ", ".join(task["allowed_paths"]))

    with st.expander("Task metadata", expanded=False):
        st.json(
            {
                "task_id": task["task_id"],
                "title": task["title"],
                "defect_category": task["defect_category"],
                "allowed_paths": task["allowed_paths"],
                "forbidden_paths": task["forbidden_paths"],
                "test_command": task["test_command"],
            }
        )

    return task


def render_interactive_demo() -> None:
    """Render the main interactive repair prototype."""
    st.header("Interactive repair prototype")
    st.write(
        "Select a broken benchmark task, inspect the failing code and tests, "
        "then run PatchPilot locally. The UI displays the agent trace, patch "
        "diff, changed files, and verification result."
    )

    task = render_task_selector()

    left, right = st.columns(2)
    src_candidates = source_files(task)
    test_candidates = test_files(task)
    if src_candidates:
        with left:
            render_file_panel("Broken source before repair", src_candidates[0])
    if test_candidates:
        with right:
            render_file_panel("Regression tests", test_candidates[0])

    st.divider()
    st.subheader("Run the agent")

    model_options = {
        "qwen2.5-coder:1.5b": "Validated local model used in reported results",
        "qwen2.5-coder:3b": "Experimental stronger local model",
    }
    model = st.selectbox(
        "Ollama model",
        options=list(model_options),
        format_func=lambda value: f"{value} — {model_options[value]}",
        index=0,
        help=(
            "Only qwen2.5-coder:1.5b is part of the reported evaluation. "
            "Other models are optional local experiments."
        ),
    )
    st.info(
        "This runs the actual PatchPilot repair pipeline: tests fail, source "
        "is inspected, a patch is applied, tests rerun, and success is shown "
        "only after pytest verification."
    )

    if st.button("Run PatchPilot on this task", type="primary"):
        with st.spinner("PatchPilot is repairing the selected benchmark..."):
            result = run_live_repair(str(task["task_id"]), model)

        if not result.get("ok"):
            st.error("Live run failed. Check Ollama and the terminal output.")
            st.code(str(result.get("stdout", "")), language="text")
            st.code(str(result.get("stderr", "")), language="text")
            return

        st.success(
            "Run completed: "
            f"{result['status']} | "
            f"full_suite_passed={result['full_suite_passed']}"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", str(result["status"]))
        m2.metric("Steps", str(result["steps"]))
        m3.metric("Patch attempts", str(result["patch_attempts"]))
        m4.metric("Changed files", str(len(result["changed_files"])))

        st.caption(f"Workspace: `{result['workspace']}`")
        st.caption(f"Trace: `{result['trace']}`")

        trace_path = Path(str(result["trace"]))
        if trace_path.is_file():
            render_timeline(trace_path)

        original_root = PROJECT_ROOT / str(task["repository_root"])
        repaired_root = Path(str(result["repository"]))
        render_diff(
            original_root=original_root,
            repaired_root=repaired_root,
            changed_files=list(result["changed_files"]),
        )


def render_evaluation() -> None:
    """Render evaluation dashboard."""
    st.header("Evaluation evidence")
    summary = read_csv(DEMO_DATA / "summary.csv")
    full_runs = read_csv(DEMO_DATA / "full_agent_runs.csv")
    no_retry = read_csv(DEMO_DATA / "no_retry_runs.csv")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Full-agent repair rate", "100.0%", "12/12 tasks")
    c2.metric("No-retry repair rate", "66.7%", "8/12 tasks")
    c3.metric("Invalid patch rate", "0.0%", "Full agent")
    c4.metric("Benchmark tasks", "12", "PatchPilot-Bench v0")

    chart = summary[
        [
            "condition",
            "repair_rate",
            "full_suite_pass_rate",
            "invalid_patch_rate",
        ]
    ].set_index("condition")
    st.bar_chart(chart)

    st.subheader("Condition summary")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Full-agent per-task results")
    st.dataframe(full_runs, use_container_width=True)

    st.subheader("No-retry ablation per-task results")
    st.dataframe(no_retry, use_container_width=True)

    st.markdown(read_markdown(DOCS / "evaluation_comparison.md"))


def render_catalog() -> None:
    """Render benchmark catalog page."""
    st.header("PatchPilot-Bench v0 catalog")
    rows = []
    for task in manifests():
        rows.append(
            {
                "task_id": task["task_id"],
                "title": task["title"],
                "category": task["defect_category"],
                "difficulty": task["difficulty"],
                "initial_failures": task["expected_initial_failures"],
                "allowed_paths": ", ".join(task["allowed_paths"]),
                "forbidden_paths": ", ".join(task["forbidden_paths"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_architecture() -> None:
    """Render architecture explanation page."""
    st.header("Architecture")
    st.code(
        """
Benchmark task
  ↓
Agent state
  ↓
Structured LLM policy
  ↓
Validated tool action
  ↓
Restricted tool executor
  ├─ run_tests
  ├─ search_code
  ├─ read_file
  ├─ apply_patch
  ├─ restore_file
  └─ finish
  ↓
Tool observation + trace
  ↓
Verified success / bounded failure / escalation
""",
        language="text",
    )
    st.markdown(read_markdown(PROJECT_ROOT / "README.md"))


def render_statistics() -> None:
    """Render statistical analysis page."""
    st.header("Statistical analysis")
    st.markdown(read_markdown(DOCS / "statistical_analysis.md"))


def render_deployment() -> None:
    """Render deployment page."""
    st.header("Docker and deployment")
    st.write(
        "The same frontend can be used as a public results demo or as a "
        "local live prototype. Docker packages the app and repository. Live "
        "repair requires Ollama access from the running environment."
    )
    st.code("docker compose up --build", language="bash")
    st.code("http://localhost:8501", language="text")


def main() -> None:
    """Streamlit entrypoint."""
    st.set_page_config(
        page_title="PatchPilot",
        page_icon="🛠️",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; max-width: 1280px;}
        .hero {
            padding: 2rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: white;
            margin-bottom: 1.2rem;
        }
        .hero h1 {font-size: 3rem; margin-bottom: 0.3rem;}
        .hero p {font-size: 1.1rem; color: #dbeafe;}
        .badge {
            display: inline-block;
            background: #dbeafe;
            color: #0f172a;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-weight: 700;
            margin-right: 0.35rem;
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
            Interactive prototype for bounded Python debugging and repair.
            Select a defective benchmark, run the agent, inspect the trace,
            view the patch diff, and verify the final test result.
          </p>
          <span class="badge">Live repair</span>
          <span class="badge">Agent trace</span>
          <span class="badge">Patch diff</span>
          <span class="badge">Pytest verification</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Interactive repair demo",
            "Evaluation evidence",
            "Benchmark catalog",
            "Architecture",
            "Statistical analysis",
            "Docker/deployment",
        ],
    )

    if page == "Interactive repair demo":
        render_interactive_demo()
    elif page == "Evaluation evidence":
        render_evaluation()
    elif page == "Benchmark catalog":
        render_catalog()
    elif page == "Architecture":
        render_architecture()
    elif page == "Statistical analysis":
        render_statistics()
    else:
        render_deployment()

    st.divider()
    st.caption(
        "PatchPilot is a bounded research prototype. It only reports success "
        "after executable test verification."
    )


if __name__ == "__main__":
    main()
