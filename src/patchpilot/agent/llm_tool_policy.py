"""Model-directed tool-selection policy for PatchPilot."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from patchpilot.agent.executor import (
    tool_argument_schema,
    validate_tool_arguments,
)
from patchpilot.agent.llm_policy import (
    PolicyResponseError,
    StructuredLLMPolicy,
    TextGenerationModel,
    _generate_with_trace,
    _mark_model_call_parse,
)
from patchpilot.agent.policy import AgentDecision
from patchpilot.schemas import (
    AgentState,
    ObservationStatus,
    ToolAction,
    ToolName,
)


class LLMToolPolicy:
    """Ask the model to choose one bounded tool action per control-loop step."""

    _MAX_HISTORY_ITEMS = 2
    _MAX_OBSERVATION_OUTPUT = 700
    _MAX_RAW_RESPONSE = 500

    _TOOL_CONTRACT: dict[str, dict[str, object]] = {
        "list_files": {
            "arguments": {"relative_path": "optional relative directory"},
            "purpose": "List repository files inside the restricted workspace.",
        },
        "read_file": {
            "arguments": {
                "relative_path": "required relative file path",
                "start_line": "optional integer >= 1",
                "end_line": "optional integer >= 1",
            },
            "purpose": "Read a bounded source-file range.",
        },
        "search_code": {
            "arguments": {
                "query": "required search text",
                "relative_path": "optional relative directory",
                "max_hits": "optional integer from 1 to 100",
            },
            "purpose": "Search source code inside the restricted workspace.",
        },
        "run_tests": {
            "arguments": {
                "target": (
                    "optional pytest target; omit it for required full-suite "
                    "verification"
                )
            },
            "purpose": "Run an allowed targeted test or the full test suite.",
        },
        "check_syntax": {
            "arguments": {},
            "purpose": (
                "Parse all currently changed Python files. This must be the "
                "verification step after every successful patch and before "
                "tests or another patch."
            ),
        },
        "apply_patch": {
            "arguments": {"patch_text": "required unified diff"},
            "purpose": (
                "Apply a minimal source-only unified diff affecting at most "
                "2 existing files and 20 added/removed lines. Never modify "
                "protected tests, create files, delete files, rename files, "
                "or use absolute paths. Changed Python files require a "
                "passing syntax check before tests or another patch."
            ),
        },
        "view_diff": {
            "arguments": {},
            "purpose": "Inspect the current workspace diff.",
        },
        "restore_file": {
            "arguments": {
                "relative_path": (
                    "optional changed file; omit it to restore all changed files"
                )
            },
            "purpose": (
                "Manually restore a changed file or all run changes. Failed "
                "verification is rolled back transactionally by the runtime."
            ),
        },
        "finish": {
            "arguments": {
                "status": "one of: succeeded, failed, escalated",
                "message": "required explanation",
            },
            "purpose": (
                "Stop the run. Success is valid only after a passing full suite "
                "for the current repository revision."
            ),
        },
    }

    def __init__(
        self,
        model: TextGenerationModel,
        max_parse_attempts: int = 2,
    ) -> None:
        if not 1 <= max_parse_attempts <= 3:
            raise ValueError("max_parse_attempts must be between 1 and 3.")

        self.model = model
        self.max_parse_attempts = max_parse_attempts

    @classmethod
    def _decision_response_schema(
        cls,
        state: AgentState | None = None,
    ) -> dict[str, object]:
        """Return legal tools with executor-derived argument schemas."""
        legal_values = (
            cls._legal_actions(state)
            if state is not None
            else [tool.value for tool in ToolName]
        )
        if not legal_values:
            legal_values = [tool.value for tool in ToolName]

        argument_schemas = [
            tool_argument_schema(ToolName(value)) for value in legal_values
        ]
        if len(argument_schemas) == 1:
            arguments_schema: dict[str, object] = argument_schemas[0]
        else:
            arguments_schema = {
                "type": "object",
                "description": (
                    "Arguments are canonicalized and validated against the "
                    "executor schema for the selected legal tool."
                ),
            }

        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reasoning_summary": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 1000,
                },
                "plan": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                },
                "hypothesis": {
                    "anyOf": [
                        {"type": "string", "maxLength": 1000},
                        {"type": "null"},
                    ]
                },
                "reflection": {
                    "anyOf": [
                        {"type": "string", "maxLength": 1000},
                        {"type": "null"},
                    ]
                },
                "action": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": legal_values,
                        },
                        "arguments": arguments_schema,
                        "rationale": {
                            "type": "string",
                            "minLength": 3,
                            "maxLength": 1000,
                        },
                    },
                    "required": ["tool", "arguments", "rationale"],
                },
            },
            "required": ["reasoning_summary", "plan", "action"],
        }

    @classmethod
    def _system_prompt(cls) -> str:
        return (
            "You are a bounded Python repair tool-selection policy. The JSON "
            "schema restricts action.tool to legal choices for the current state. "
            "Available tools include list_files, read_file, search_code, run_tests, "
            "apply_patch, check_syntax, view_diff, restore_file, and finish. "
            "Never invent repository evidence, edit tests, or use forbidden paths. "
            "After every successful patch choose check_syntax before tests. "
            "Use apply_patch only after reading relevant source. Never claim "
            "success until the current revision passes the full suite. "
            "Set reflection to null. Return JSON only."
        )

    @classmethod
    def _trajectory(cls, state: AgentState) -> str:
        """Return a compact recent action-observation trajectory."""
        pairs = list(
            zip(
                state.actions,
                state.observations,
                strict=False,
            )
        )
        recent = pairs[-cls._MAX_HISTORY_ITEMS :]
        if not recent:
            return "none"

        records: list[dict[str, Any]] = []
        start_index = len(pairs) - len(recent) + 1
        for offset, (action, observation) in enumerate(recent):
            output = observation.output
            if len(output) > cls._MAX_OBSERVATION_OUTPUT:
                output = (
                    output[: cls._MAX_OBSERVATION_OUTPUT]
                    + "\n...[observation output truncated]"
                )

            records.append(
                {
                    "step": start_index + offset,
                    "tool": action.tool.value,
                    "arguments": action.arguments,
                    "status": observation.status.value,
                    "summary": observation.summary,
                    "output": output,
                }
            )

        return json.dumps(
            records,
            sort_keys=True,
        )

    @classmethod
    def _legal_actions(
        cls,
        state: AgentState | None,
    ) -> list[str]:
        """Return a bounded action set that converges from evidence to repair."""
        if state is None:
            return [tool.value for tool in ToolName]
        if state.rollback_required:
            return []
        if state.syntax_check_required:
            return [ToolName.CHECK_SYNTAX.value]
        if not state.actions or not state.observations:
            return [ToolName.RUN_TESTS.value]

        paired_count = min(
            len(state.actions),
            len(state.observations),
        )
        last_test_index = -1
        last_patch_index = -1
        last_successful_read_index = -1

        for index in range(paired_count):
            action = state.actions[index]
            observation = state.observations[index]

            if action.tool is ToolName.RUN_TESTS:
                last_test_index = index
            elif action.tool is ToolName.APPLY_PATCH:
                last_patch_index = index
            elif (
                action.tool is ToolName.READ_FILE
                and observation.status is ObservationStatus.OK
            ):
                last_successful_read_index = index

        if last_test_index < 0 and state.last_failed_verification_tool is None:
            return [ToolName.RUN_TESTS.value]

        if (
            last_successful_read_index > last_test_index
            and last_successful_read_index > last_patch_index
        ):
            return [ToolName.APPLY_PATCH.value]

        last_action = state.actions[-1]
        last_observation = state.observations[-1]

        if last_action.tool is ToolName.RUN_TESTS:
            if last_observation.status is ObservationStatus.OK:
                candidates = [ToolName.FINISH]
            else:
                candidates = [
                    ToolName.SEARCH_CODE,
                    ToolName.LIST_FILES,
                    ToolName.READ_FILE,
                ]
        elif last_action.tool is ToolName.SEARCH_CODE:
            candidates = (
                [ToolName.READ_FILE]
                if last_observation.status is ObservationStatus.OK
                else [ToolName.LIST_FILES, ToolName.READ_FILE]
            )
        elif last_action.tool is ToolName.LIST_FILES:
            candidates = (
                [ToolName.READ_FILE]
                if last_observation.status is ObservationStatus.OK
                else [ToolName.SEARCH_CODE]
            )
        elif last_action.tool is ToolName.READ_FILE:
            candidates = (
                [ToolName.APPLY_PATCH]
                if last_observation.status is ObservationStatus.OK
                else [ToolName.SEARCH_CODE, ToolName.LIST_FILES]
            )
        elif last_action.tool is ToolName.APPLY_PATCH:
            candidates = (
                [ToolName.CHECK_SYNTAX]
                if last_observation.status is ObservationStatus.OK
                else [ToolName.READ_FILE, ToolName.VIEW_DIFF]
            )
        elif last_action.tool is ToolName.CHECK_SYNTAX:
            candidates = (
                [ToolName.RUN_TESTS]
                if last_observation.status is ObservationStatus.OK
                else [ToolName.READ_FILE, ToolName.VIEW_DIFF]
            )
        elif last_action.tool is ToolName.RESTORE_FILE:
            candidates = [ToolName.READ_FILE]
        elif last_action.tool is ToolName.VIEW_DIFF:
            candidates = (
                [ToolName.CHECK_SYNTAX]
                if state.syntax_check_required
                else [ToolName.APPLY_PATCH]
            )
        else:
            candidates = [ToolName.FINISH]

        if not state.changed_files and state.current_attempt_id is None:
            candidates = [
                tool for tool in candidates if tool is not ToolName.RESTORE_FILE
            ]

        previous = state.actions[-1]
        non_repeating = [tool for tool in candidates if tool is not previous.tool]
        if non_repeating:
            candidates = non_repeating

        return [tool.value for tool in candidates]

    @classmethod
    def _state_prompt(cls, state: AgentState) -> str:
        """Return compact state while preserving diagnostic trace fields."""
        remaining = {
            "steps": max(
                state.budget.max_steps - state.usage.steps,
                0,
            ),
            "tools": max(
                state.budget.max_tool_calls - state.usage.tool_calls,
                0,
            ),
            "patches": max(
                state.budget.max_patch_attempts - state.usage.patch_attempts,
                0,
            ),
        }
        state_summary = {
            "task_id": state.task.task_id,
            "goal": state.task.goal,
            "allowed_paths": state.task.allowed_paths,
            "forbidden_paths": state.task.forbidden_paths,
            "status": state.status.value,
            "legal_actions": cls._legal_actions(state),
            "remaining": remaining,
            "changed_files": state.changed_files,
            "repository_revision": state.repository_revision,
            "syntax_verified_revision": state.syntax_verified_revision,
            "syntax_check_required": state.syntax_check_required,
            "current_attempt_id": state.current_attempt_id,
            "current_attempt_files": state.current_attempt_files,
            "rollback_required": state.rollback_required,
            "last_failed_attempt_id": state.last_failed_attempt_id,
            "last_failed_attempt_files": state.last_failed_attempt_files,
            "last_rolled_back_attempt_id": state.last_rolled_back_attempt_id,
            "last_rolled_back_attempt_files": (state.last_rolled_back_attempt_files),
            "current_hypothesis": state.current_hypothesis,
            "last_failed_verification_tool": (
                state.last_failed_verification_tool.value
                if state.last_failed_verification_tool is not None
                else None
            ),
            "reflection_required": state.reflection_required,
            "verified_revision": state.verified_revision,
            "full_suite_passed": state.full_suite_passed,
        }
        return (
            "CURRENT REPAIR STATE:\n"
            + json.dumps(
                state_summary,
                sort_keys=True,
            )
            + "\nRECENT ACTION-OBSERVATION TRAJECTORY:\n"
            + cls._trajectory(state)
            + "\nChoose one action from legal_actions. Use only observed "
            "evidence and valid tool arguments. " + cls._reflection_instruction(state)
        )

    @classmethod
    def _reflection_instruction(cls, state: AgentState) -> str:
        """Return the reflection rule included in the model prompt."""
        del state
        return "Set the reflection field to null for this no-reflection policy."

    @staticmethod
    def _candidate_json(raw_response: str) -> str:
        candidate = raw_response.strip()
        if not candidate:
            raise PolicyResponseError(
                "Model returned an empty decision.",
                raw_response=raw_response,
            )

        if candidate.startswith("```") and candidate.endswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()

        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                nested = candidate[start : end + 1]
                try:
                    json.loads(nested)
                    return nested
                except json.JSONDecodeError:
                    pass

        raise PolicyResponseError(
            "Model response is not valid JSON.",
            raw_response=raw_response,
        )

    @classmethod
    def _parse_decision(cls, raw_response: str) -> AgentDecision:
        candidate = cls._candidate_json(raw_response)

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise PolicyResponseError(
                f"Model decision failed schema validation: {exc}",
                raw_response=raw_response,
            ) from exc

        if not isinstance(payload, dict):
            raise PolicyResponseError(
                "Model decision must be a JSON object.",
                raw_response=raw_response,
            )

        plan = payload.get("plan")
        if not isinstance(plan, list) or not plan:
            raise PolicyResponseError(
                "Model decision must contain at least one plan step.",
                raw_response=raw_response,
            )

        try:
            decision = AgentDecision.model_validate(payload)
        except ValidationError as exc:
            raise PolicyResponseError(
                f"Model decision failed schema validation: {exc}",
                raw_response=raw_response,
            ) from exc

        return decision

    @classmethod
    def _validate_common_decision(
        cls,
        state: AgentState,
        decision: AgentDecision,
        raw_response: str,
    ) -> AgentDecision:
        """Reject actions outside the current evidence-sensitive legal set."""
        if state.rollback_required:
            raise PolicyResponseError(
                "Runtime transactional rollback must complete before another "
                "model-directed decision.",
                raw_response=raw_response,
            )

        action = decision.action
        tool = action.tool

        if (
            not state.actions
            and not state.syntax_check_required
            and tool is not ToolName.RUN_TESTS
        ):
            raise PolicyResponseError(
                "The first action must be run_tests to establish executable "
                "failure evidence.",
                raw_response=raw_response,
            )

        if state.actions:
            previous = state.actions[-1]
            if previous.tool is tool and previous.arguments == action.arguments:
                raise PolicyResponseError(
                    "The proposed action exactly repeats the latest action "
                    "without new repository or verification evidence.",
                    raw_response=raw_response,
                )

        legal = cls._legal_actions(state)
        if tool.value not in legal:
            rendered = ", ".join(legal) or "<none>"
            raise PolicyResponseError(
                f"Tool {tool.value} is not currently legal. Choose one of: {rendered}.",
                raw_response=raw_response,
            )

        if tool is ToolName.FINISH:
            status = action.arguments.get("status")
            verified_current_revision = (
                state.full_suite_passed
                and state.verified_revision == state.repository_revision
            )
            if status == "succeeded" and not verified_current_revision:
                raise PolicyResponseError(
                    "finish status succeeded requires a passing full suite "
                    "for the current repository revision.",
                    raw_response=raw_response,
                )

        return decision

    def _validate_decision(
        self,
        state: AgentState,
        decision: AgentDecision,
        raw_response: str,
    ) -> AgentDecision:
        """Apply no-reflection rules before shared semantic validation."""
        if decision.reflection is not None:
            raise PolicyResponseError(
                "The no-reflection policy requires reflection to be null.",
                raw_response=raw_response,
            )

        return self._validate_common_decision(
            state,
            decision,
            raw_response,
        )

    @staticmethod
    def _allowed_source_path(
        state: AgentState,
        candidate: object,
    ) -> str | None:
        """Return a safe source path candidate inside an allowed root."""
        if not isinstance(candidate, str):
            return None

        path = candidate.strip().replace("\\", "/")
        if not path or path.startswith("/") or ".." in path.split("/"):
            return None

        for root in state.task.allowed_paths:
            clean_root = root.strip("/")
            if path.endswith(".py") and (
                path == clean_root or path.startswith(f"{clean_root}/")
            ):
                return path
        return None

    @classmethod
    def _source_path_from_state(
        cls,
        state: AgentState,
        raw_arguments: dict[str, object] | None = None,
    ) -> str:
        """Resolve a source path from trusted evidence before model aliases."""
        if state.last_rolled_back_attempt_files:
            safe = cls._allowed_source_path(
                state,
                state.last_rolled_back_attempt_files[0],
            )
            if safe is not None:
                return safe

        pairs = list(
            zip(
                state.actions,
                state.observations,
                strict=False,
            )
        )
        for action, observation in reversed(pairs):
            if (
                action.tool is ToolName.READ_FILE
                and observation.status is ObservationStatus.OK
            ):
                safe = cls._allowed_source_path(
                    state,
                    action.arguments.get("relative_path"),
                )
                if safe is not None:
                    return safe

            if observation.status is ObservationStatus.OK:
                try:
                    return StructuredLLMPolicy._source_path_from_search(
                        observation.output,
                        state.task.allowed_paths,
                    )
                except PolicyResponseError:
                    pass

        for observation in reversed(state.observations):
            try:
                return StructuredLLMPolicy._source_path_from_search(
                    observation.output,
                    state.task.allowed_paths,
                )
            except PolicyResponseError:
                pass

        arguments = raw_arguments or {}
        candidates: list[object] = [
            arguments.get("relative_path"),
            arguments.get("file_path"),
            arguments.get("path"),
            arguments.get("filename"),
            arguments.get("target"),
        ]
        files = arguments.get("files")
        if isinstance(files, list):
            candidates.extend(files)

        for candidate in candidates:
            safe = cls._allowed_source_path(state, candidate)
            if safe is not None:
                return safe

        raise PolicyResponseError(
            "No evidence-backed source path is available. Search or list "
            "the allowed source root before reading a file."
        )

    @classmethod
    def _canonicalize_decision(
        cls,
        state: AgentState,
        decision: AgentDecision,
        raw_response: str,
    ) -> AgentDecision:
        """Canonicalize model arguments and validate them before execution."""
        tool = decision.action.tool
        raw_arguments = dict(decision.action.arguments)

        if tool in {
            ToolName.RUN_TESTS,
            ToolName.CHECK_SYNTAX,
            ToolName.VIEW_DIFF,
        }:
            arguments: dict[str, object] = {}
        elif tool is ToolName.SEARCH_CODE:
            arguments = {
                "query": StructuredLLMPolicy._failure_query(state),
                "relative_path": state.task.allowed_paths[0],
            }
        elif tool is ToolName.LIST_FILES:
            arguments = {
                "relative_path": state.task.allowed_paths[0],
            }
        elif tool is ToolName.READ_FILE:
            arguments = {
                "relative_path": cls._source_path_from_state(
                    state,
                    raw_arguments,
                )
            }
        elif tool is ToolName.RESTORE_FILE:
            safe = cls._allowed_source_path(
                state,
                raw_arguments.get("relative_path"),
            )
            if safe is None and state.changed_files:
                safe = cls._allowed_source_path(
                    state,
                    state.changed_files[0],
                )
            arguments = {"relative_path": safe} if safe is not None else {}
        elif tool is ToolName.FINISH and (
            state.full_suite_passed
            and state.verified_revision == state.repository_revision
        ):
            arguments = {
                "status": "succeeded",
                "message": ("The current repository revision passed the full suite."),
            }
        else:
            arguments = raw_arguments

        try:
            validated_arguments = validate_tool_arguments(
                tool,
                arguments,
            )
        except ValidationError as exc:
            raise PolicyResponseError(
                f"Invalid arguments for {tool.value}: {exc}",
                raw_response=raw_response,
            ) from exc

        return decision.model_copy(
            update={
                "action": decision.action.model_copy(
                    update={"arguments": validated_arguments}
                )
            }
        )

    @staticmethod
    def _programmatic_decision(
        *,
        summary: str,
        plan: str,
        tool: ToolName,
        rationale: str,
        arguments: dict[str, object] | None = None,
    ) -> AgentDecision:
        """Create one runtime-owned mandatory transition."""
        return AgentDecision(
            reasoning_summary=summary,
            plan=[plan],
            hypothesis=None,
            reflection=None,
            action=ToolAction(
                tool=tool,
                arguments=arguments or {},
                rationale=rationale,
            ),
        )

    def _reflection_required_now(self, state: AgentState) -> bool:
        """Return whether this concrete policy must reflect now."""
        checker = getattr(self, "_reflection_required", None)
        return bool(checker(state)) if callable(checker) else False

    def _mandatory_transition(
        self,
        state: AgentState,
        legal: list[str],
    ) -> AgentDecision | None:
        """Execute mandatory plumbing without asking the model to retype it."""
        if self._reflection_required_now(state):
            return None

        if legal == [ToolName.READ_FILE.value]:
            return self._programmatic_decision(
                summary="Read the source path established by repository evidence.",
                plan="Inspect the implementation before generating a repair.",
                tool=ToolName.READ_FILE,
                arguments={"relative_path": self._source_path_from_state(state)},
                rationale="Use the evidence-backed source path.",
            )

        if legal == [ToolName.APPLY_PATCH.value]:
            return StructuredLLMPolicy(self.model).decide(state)

        if (
            legal == [ToolName.CHECK_SYNTAX.value]
            and state.actions
            and state.actions[-1].tool is ToolName.APPLY_PATCH
        ):
            return self._programmatic_decision(
                summary="Validate Python syntax after the source patch.",
                plan="Pass the syntax gate before running tests.",
                tool=ToolName.CHECK_SYNTAX,
                rationale="Every successful patch requires syntax validation.",
            )

        if (
            legal == [ToolName.RUN_TESTS.value]
            and state.actions
            and state.actions[-1].tool is ToolName.CHECK_SYNTAX
        ):
            return self._programmatic_decision(
                summary="Run the full suite after syntax validation.",
                plan="Verify the current repository revision.",
                tool=ToolName.RUN_TESTS,
                rationale="Full-suite verification is required for success.",
            )

        if legal == [ToolName.FINISH.value]:
            return self._programmatic_decision(
                summary="Finish after verified full-suite success.",
                plan="End the bounded repair run.",
                tool=ToolName.FINISH,
                arguments={
                    "status": "succeeded",
                    "message": (
                        "The current repository revision passed the full suite."
                    ),
                },
                rationale="Current-revision verification permits success.",
            )

        return None

    def decide(self, state: AgentState) -> AgentDecision:
        """Choose a legal tool and guarantee executor-valid arguments."""
        legal = self._legal_actions(state)
        mandatory = self._mandatory_transition(state, legal)
        if mandatory is not None:
            return mandatory

        system_prompt = self._system_prompt()
        base_prompt = self._state_prompt(state)
        response_schema = self._decision_response_schema(state)
        user_prompt = base_prompt

        for attempt in range(1, self.max_parse_attempts + 1):
            raw_response, record_index = _generate_with_trace(
                state=state,
                model=self.model,
                policy_name=type(self).__name__,
                purpose="tool_decision",
                attempt=attempt,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=response_schema,
            )

            try:
                parsed = self._parse_decision(raw_response)
                canonical = self._canonicalize_decision(
                    state,
                    parsed,
                    raw_response,
                )
                validated = self._validate_decision(
                    state,
                    canonical,
                    raw_response,
                )
                _mark_model_call_parse(
                    state,
                    record_index,
                    succeeded=True,
                )
                return validated
            except PolicyResponseError as exc:
                _mark_model_call_parse(
                    state,
                    record_index,
                    succeeded=False,
                    error=exc,
                )
                state.decision_parse_failures += 1
                if attempt >= self.max_parse_attempts:
                    raise PolicyResponseError(
                        "Model did not return a valid tool decision after "
                        f"{self.max_parse_attempts} attempt(s): {exc}",
                        raw_response=raw_response,
                    ) from exc

                legal_now = ", ".join(self._legal_actions(state))
                user_prompt = (
                    base_prompt
                    + "\nCORRECTION REQUIRED:"
                    + str(exc)
                    + "\nLEGAL ACTIONS NOW:"
                    + legal_now
                    + "\nUse the exact argument names in the response schema. "
                    "Return corrected JSON only. Previous invalid response:"
                    + raw_response[: self._MAX_RAW_RESPONSE]
                )

        raise AssertionError("Unreachable model-decision loop.")
