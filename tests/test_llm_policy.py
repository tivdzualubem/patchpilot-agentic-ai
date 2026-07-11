import pytest

from patchpilot.agent import (
    AgentDecision,
    PolicyResponseError,
    StructuredLLMPolicy,
)
from patchpilot.schemas import (
    AgentState,
    ObservationStatus,
    RepairTask,
    ToolAction,
    ToolName,
    ToolObservation,
)


class FakeModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.last_response_schema: dict[str, object] | None = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        assert "PatchPilot" in system_prompt
        self.calls += 1
        self.last_response_schema = response_schema
        return self.response


class NoCallModel:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        raise AssertionError("model should not be called")


def make_task() -> RepairTask:
    return RepairTask(
        task_id="llm-policy-001",
        goal="Repair the defective add function.",
        repository_root="benchmarks/example",
    )


def failed_test_state() -> AgentState:
    state = AgentState(task=make_task())
    state.actions.append(
        ToolAction(
            tool=ToolName.RUN_TESTS,
            arguments={},
            rationale="Run tests.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.RUN_TESTS,
            status=ObservationStatus.ERROR,
            summary="Tests failed.",
            output="E assert -1 == 5\nE where -1 = add(2, 3)",
        )
    )
    return state


def searched_state() -> AgentState:
    state = failed_test_state()
    state.actions.append(
        ToolAction(
            tool=ToolName.SEARCH_CODE,
            arguments={"query": "add", "relative_path": "src"},
            rationale="Search source.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.SEARCH_CODE,
            status=ObservationStatus.OK,
            summary="Found matches.",
            output="src/calculator.py:4:def add(left: int, right: int) -> int:",
        )
    )
    return state


def read_state() -> AgentState:
    state = searched_state()
    state.actions.append(
        ToolAction(
            tool=ToolName.READ_FILE,
            arguments={"relative_path": "src/calculator.py"},
            rationale="Read source.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.READ_FILE,
            status=ObservationStatus.OK,
            summary="Read source.",
            output=(
                "4: def add(left: int, right: int) -> int:\n5:     return left - right"
            ),
        )
    )
    return state


def valid_diff() -> str:
    return "\n".join(
        [
            "diff --git a/src/calculator.py b/src/calculator.py",
            "--- a/src/calculator.py",
            "+++ b/src/calculator.py",
            "@@ -1,2 +1,2 @@",
            " def add(left: int, right: int) -> int:",
            "-    return left - right",
            "+    return left + right",
            "",
        ]
    )


def test_first_decision_runs_full_suite_without_model_call() -> None:
    policy = StructuredLLMPolicy(NoCallModel())

    decision = policy.decide(AgentState(task=make_task()))

    assert isinstance(decision, AgentDecision)
    assert decision.action.tool is ToolName.RUN_TESTS
    assert decision.action.arguments == {}


def test_failed_tests_search_for_failing_symbol() -> None:
    policy = StructuredLLMPolicy(NoCallModel())

    decision = policy.decide(failed_test_state())

    assert decision.action.tool is ToolName.SEARCH_CODE
    assert decision.action.arguments == {"query": "add", "relative_path": "src"}


def test_search_result_reads_source_file() -> None:
    policy = StructuredLLMPolicy(NoCallModel())

    decision = policy.decide(searched_state())

    assert decision.action.tool is ToolName.READ_FILE
    assert decision.action.arguments == {"relative_path": "src/calculator.py"}


def test_read_file_generates_diff_only_patch_decision() -> None:
    model = FakeModel(valid_diff())
    policy = StructuredLLMPolicy(model)

    decision = policy.decide(read_state())

    assert model.calls == 1
    assert model.last_response_schema is None
    assert decision.action.tool is ToolName.APPLY_PATCH
    assert "return left + right" in decision.action.arguments["patch_text"]


def test_apply_patch_success_forces_syntax_check() -> None:
    state = read_state()
    state.actions.append(
        ToolAction(
            tool=ToolName.APPLY_PATCH,
            arguments={"patch_text": valid_diff()},
            rationale="Patch source.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.APPLY_PATCH,
            status=ObservationStatus.OK,
            summary="Patch applied.",
        )
    )
    policy = StructuredLLMPolicy(NoCallModel())

    decision = policy.decide(state)

    assert decision.action.tool is ToolName.CHECK_SYNTAX


def test_successful_syntax_check_forces_test_verification() -> None:
    state = read_state()
    state.changed_files.append("src/calculator.py")
    state.repository_revision = 1
    state.syntax_verified_revision = 1
    state.actions.append(
        ToolAction(
            tool=ToolName.CHECK_SYNTAX,
            arguments={},
            rationale="Check syntax.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.CHECK_SYNTAX,
            status=ObservationStatus.OK,
            summary="Syntax passed.",
        )
    )

    decision = StructuredLLMPolicy(NoCallModel()).decide(state)

    assert decision.action.tool is ToolName.RUN_TESTS


def test_failed_syntax_check_waits_for_runtime_rollback() -> None:
    state = read_state()
    state.changed_files.append("src/calculator.py")
    state.repository_revision = 1
    state.current_attempt_id = 1
    state.current_attempt_files = ["src/calculator.py"]
    state.rollback_required = True
    state.actions.append(
        ToolAction(
            tool=ToolName.CHECK_SYNTAX,
            arguments={},
            rationale="Check syntax.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.CHECK_SYNTAX,
            status=ObservationStatus.ERROR,
            summary="Syntax failed.",
            output="src/calculator.py:5:12: invalid syntax",
        )
    )

    with pytest.raises(
        PolicyResponseError,
        match="transactional rollback",
    ):
        StructuredLLMPolicy(NoCallModel()).decide(state)


def test_invalid_diff_fails_safely() -> None:
    policy = StructuredLLMPolicy(FakeModel("not a diff"))

    with pytest.raises(PolicyResponseError):
        policy.decide(read_state())


def test_malformed_diff_with_replacement_line_is_repaired() -> None:
    malformed = "\n".join(
        [
            "diff --git a/src/calculator.py b/src/calculator.py",
            "--- a/src/calculator.py",
            "+++ b/src/calculator.py",
            "@@ -5,7 +5,7 @@",
            " def add(left: int, right: int) -> int:",
            "     return left + right",
            "",
        ]
    )
    policy = StructuredLLMPolicy(FakeModel(malformed))

    decision = policy.decide(read_state())

    patch_text = decision.action.arguments["patch_text"]
    assert "-    return left - right" in patch_text
    assert "+    return left + right" in patch_text


def test_corrupt_model_hunk_header_is_rebuilt() -> None:
    corrupt = "\n".join(
        [
            "diff --git a/src/calculator.py b/src/calculator.py",
            "--- a/src/calculator.py",
            "+++ b/src/calculator.py",
            "@@ -4,7 +4,7 @@",
            "",
            " def add(left: int, right: int) -> int:",
            '     """Return the sum of two integers."""',
            "-    return left - right",
            "+    return left + right",
            "",
        ]
    )
    policy = StructuredLLMPolicy(FakeModel(corrupt))

    decision = policy.decide(read_state())

    patch_text = decision.action.arguments["patch_text"]
    assert "@@" in patch_text
    assert "-    return left - right" in patch_text
    assert "+    return left + right" in patch_text


def test_search_uses_task_allowed_source_root() -> None:
    task = RepairTask(
        task_id="quixbugs-gcd",
        goal="Repair gcd.",
        repository_root="repository",
        allowed_paths=["python_programs"],
    )
    state = AgentState(task=task)
    state.actions.append(
        ToolAction(tool=ToolName.RUN_TESTS, arguments={}, rationale="Run tests.")
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.RUN_TESTS,
            status=ObservationStatus.ERROR,
            summary="Tests failed.",
            output="python_programs/gcd.py:5: RecursionError",
        )
    )

    decision = StructuredLLMPolicy(NoCallModel()).decide(state)

    assert decision.action.tool is ToolName.SEARCH_CODE
    assert decision.action.arguments["relative_path"] == "python_programs"
    assert decision.action.arguments["query"]


def test_search_result_reads_file_under_allowed_source_root() -> None:
    state = failed_test_state()
    state.task.allowed_paths = ["python_programs"]
    state.actions.append(
        ToolAction(
            tool=ToolName.SEARCH_CODE,
            arguments={"query": "gcd", "relative_path": "python_programs"},
            rationale="Search source.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.SEARCH_CODE,
            status=ObservationStatus.OK,
            summary="Found matches.",
            output="python_programs/gcd.py:1:def gcd(a, b):",
        )
    )

    decision = StructuredLLMPolicy(NoCallModel()).decide(state)

    assert decision.action.tool is ToolName.READ_FILE
    assert decision.action.arguments == {"relative_path": "python_programs/gcd.py"}


def test_failed_verification_waits_for_runtime_rollback() -> None:
    state = read_state()
    state.changed_files.append("src/calculator.py")
    state.current_attempt_id = 1
    state.current_attempt_files = ["src/calculator.py"]
    state.rollback_required = True
    state.actions.append(
        ToolAction(tool=ToolName.RUN_TESTS, arguments={}, rationale="Run tests.")
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.RUN_TESTS,
            status=ObservationStatus.ERROR,
            summary="Tests still failed.",
        )
    )

    with pytest.raises(
        PolicyResponseError,
        match="transactional rollback",
    ):
        StructuredLLMPolicy(NoCallModel()).decide(state)


def test_restore_success_reads_clean_file_before_retry() -> None:
    state = read_state()
    state.actions.append(
        ToolAction(
            tool=ToolName.RESTORE_FILE,
            arguments={"relative_path": "src/calculator.py"},
            rationale="Restore failed patch.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.RESTORE_FILE,
            status=ObservationStatus.OK,
            summary="Restored.",
        )
    )

    decision = StructuredLLMPolicy(NoCallModel()).decide(state)

    assert decision.action.tool is ToolName.READ_FILE
    assert decision.action.arguments == {"relative_path": "src/calculator.py"}


def test_policy_error_keeps_raw_response() -> None:
    policy = StructuredLLMPolicy(FakeModel("```text\nno source change\n```"))

    with pytest.raises(PolicyResponseError) as exc_info:
        policy.decide(read_state())

    assert exc_info.value.raw_response is not None
    assert "no source change" in exc_info.value.raw_response


def test_runtime_attempt_rollback_reads_restored_attempt_file() -> None:
    state = read_state()
    state.last_rolled_back_attempt_id = 1
    state.last_rolled_back_attempt_files = ["src/calculator.py"]
    state.actions.append(
        ToolAction(
            tool=ToolName.RESTORE_FILE,
            arguments={"scope": "failed_attempt", "attempt_id": 1},
            rationale="Runtime-enforced transactional rollback.",
        )
    )
    state.observations.append(
        ToolObservation(
            tool=ToolName.RESTORE_FILE,
            status=ObservationStatus.OK,
            summary="Rolled back patch attempt 1 across 1 file(s).",
            output="src/calculator.py",
        )
    )

    decision = StructuredLLMPolicy(NoCallModel()).decide(state)

    assert decision.action.tool is ToolName.READ_FILE
    assert decision.action.arguments == {"relative_path": "src/calculator.py"}


def test_scaffolded_model_call_accounting() -> None:
    state = read_state()

    decision = StructuredLLMPolicy(FakeModel(valid_diff())).decide(state)

    assert decision.action.tool is ToolName.APPLY_PATCH
    assert state.model_calls == 1
    assert state.decision_parse_failures == 0


def test_scaffolded_patch_parse_failures_are_counted() -> None:
    state = read_state()

    with pytest.raises(PolicyResponseError):
        StructuredLLMPolicy(FakeModel("not a source replacement")).decide(state)

    assert state.model_calls == 2
    assert state.decision_parse_failures == 2


def test_scaffolded_model_call_trace_is_complete() -> None:
    state = read_state()

    StructuredLLMPolicy(FakeModel(valid_diff())).decide(state)

    assert len(state.model_call_records) == 1
    record = state.model_call_records[0]
    assert record.call_index == 1
    assert record.policy == "StructuredLLMPolicy"
    assert record.purpose == "patch_generation"
    assert record.attempt == 1
    assert record.backend.endswith(".FakeModel")
    assert record.system_prompt.startswith("You are PatchPilot")
    assert "FILE: src/calculator.py" in record.user_prompt
    assert record.response_schema is None
    assert record.raw_response == valid_diff()
    assert record.generation_succeeded is True
    assert record.parse_succeeded is True
    assert record.error_type is None


def test_scaffolded_parse_retry_records_each_response() -> None:
    state = read_state()

    with pytest.raises(PolicyResponseError):
        StructuredLLMPolicy(FakeModel("not a source replacement")).decide(state)

    assert len(state.model_call_records) == 2
    assert [record.attempt for record in state.model_call_records] == [1, 2]
    assert all(record.generation_succeeded for record in state.model_call_records)
    assert all(record.parse_succeeded is False for record in state.model_call_records)
    assert all(
        record.error_type == "PolicyResponseError"
        for record in state.model_call_records
    )


def test_model_generation_error_is_preserved_in_call_trace() -> None:
    class ErrorModel:
        def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            response_schema: dict[str, object] | None = None,
        ) -> str:
            del system_prompt, user_prompt, response_schema
            raise RuntimeError("backend unavailable")

    state = read_state()

    with pytest.raises(RuntimeError, match="backend unavailable"):
        StructuredLLMPolicy(ErrorModel()).decide(state)

    assert state.model_calls == 1
    assert len(state.model_call_records) == 1
    record = state.model_call_records[0]
    assert record.generation_succeeded is False
    assert record.parse_succeeded is None
    assert record.error_type == "RuntimeError"
    assert record.error_message == "backend unavailable"
