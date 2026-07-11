from pathlib import Path

from patchpilot.agent import AgentDecision
from patchpilot.benchmark import BenchmarkRunner
from patchpilot.schemas import (
    AgentState,
    AgentStatus,
    ToolAction,
    ToolName,
)


class RepairPolicy:
    def decide(self, state: AgentState) -> AgentDecision:
        step = len(state.actions)

        actions = [
            ToolAction(
                tool=ToolName.RUN_TESTS,
                rationale="Reproduce the defect.",
            ),
            ToolAction(
                tool=ToolName.READ_FILE,
                arguments={"relative_path": "src/calculator.py"},
                rationale="Inspect the failing implementation.",
            ),
            ToolAction(
                tool=ToolName.APPLY_PATCH,
                arguments={"patch_text": repair_patch()},
                rationale="Replace subtraction with addition.",
            ),
            ToolAction(
                tool=ToolName.CHECK_SYNTAX,
                rationale="Validate changed Python syntax.",
            ),
            ToolAction(
                tool=ToolName.RUN_TESTS,
                rationale="Verify the repaired repository.",
            ),
            ToolAction(
                tool=ToolName.FINISH,
                arguments={
                    "status": "succeeded",
                    "message": "All regression tests pass.",
                },
                rationale="Finish after full verification.",
            ),
        ]

        return AgentDecision(
            reasoning_summary="Execute the next repair step.",
            action=actions[step],
        )


def repair_patch() -> str:
    return (
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -4,3 +4,3 @@\n"
        " def add(left: int, right: int) -> int:\n"
        '     """Return the sum of two integers."""\n'
        "-    return left - right\n"
        "+    return left + right\n"
    )


def test_complete_benchmark_repair(tmp_path: Path) -> None:
    runner = BenchmarkRunner(Path("."), tmp_path / "outputs")

    run = runner.run(
        Path("benchmarks/calculator-001/task.json"),
        RepairPolicy(),
        run_id="calculator-run-001",
    )

    repaired = run.prepared.repository_root / "src/calculator.py"

    assert run.state.status is AgentStatus.SUCCEEDED
    assert run.state.syntax_verified_revision == run.state.repository_revision
    assert run.state.full_suite_passed is True
    assert "left + right" in repaired.read_text(encoding="utf-8")
    assert run.trace_path.is_file()
