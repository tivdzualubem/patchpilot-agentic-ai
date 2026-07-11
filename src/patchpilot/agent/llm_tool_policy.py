"""Model-directed tool-selection policy for PatchPilot."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from patchpilot.agent.llm_policy import PolicyResponseError, TextGenerationModel
from patchpilot.agent.policy import AgentDecision
from patchpilot.schemas import AgentState


class LLMToolPolicy:
    """Ask the model to choose one bounded tool action per control-loop step."""

    _MAX_HISTORY_ITEMS = 8
    _MAX_OBSERVATION_OUTPUT = 3000
    _MAX_RAW_RESPONSE = 2000

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
                "Parse all currently changed Python files and report syntax "
                "errors before test execution."
            ),
        },
        "apply_patch": {
            "arguments": {"patch_text": "required unified diff"},
            "purpose": (
                "Apply a minimal source-only patch. Never modify protected tests, "
                "create files, delete files, rename files, or use absolute paths."
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
            "purpose": "Rollback a failed source modification.",
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
    def _system_prompt(cls) -> str:
        tool_contract = json.dumps(
            cls._TOOL_CONTRACT,
            indent=2,
            sort_keys=True,
        )
        return (
            "You are the no-reflection tool-selection policy for a bounded Python "
            "repair agent. Choose exactly one next action from the restricted tool "
            "contract. The runtime, not you, validates and executes the action. "
            "Never edit tests or forbidden paths. Prefer evidence-gathering before "
            "patching. Use a minimal unified diff for apply_patch. Do not report "
            "success until a full test run has passed for the current revision. "
            "The reflection field must be null because this policy is the "
            "no-reflection ablation. Return only one JSON object matching the "
            "provided schema. Do not use markdown.\n\n"
            f"RESTRICTED TOOL CONTRACT:\n{tool_contract}"
        )

    @classmethod
    def _trajectory(cls, state: AgentState) -> str:
        pairs = list(
            zip(
                state.actions,
                state.observations,
                strict=False,
            )
        )
        recent = pairs[-cls._MAX_HISTORY_ITEMS :]
        if not recent:
            return "No actions have been executed yet."

        records: list[dict[str, Any]] = []
        start_index = len(pairs) - len(recent) + 1
        for offset, (action, observation) in enumerate(recent):
            observation_data = observation.model_dump(mode="json")
            output = str(observation_data.get("output", ""))
            if len(output) > cls._MAX_OBSERVATION_OUTPUT:
                observation_data["output"] = (
                    output[: cls._MAX_OBSERVATION_OUTPUT]
                    + "\n...[observation output truncated]"
                )

            records.append(
                {
                    "step": start_index + offset,
                    "action": action.model_dump(mode="json"),
                    "observation": observation_data,
                }
            )

        return json.dumps(records, indent=2, sort_keys=True)

    @classmethod
    def _state_prompt(cls, state: AgentState) -> str:
        remaining = {
            "steps": max(
                state.budget.max_steps - state.usage.steps,
                0,
            ),
            "tool_calls": max(
                state.budget.max_tool_calls - state.usage.tool_calls,
                0,
            ),
            "patch_attempts": max(
                state.budget.max_patch_attempts - state.usage.patch_attempts,
                0,
            ),
            "seconds": max(
                state.budget.max_seconds - state.usage.elapsed_seconds,
                0.0,
            ),
        }
        state_summary = {
            "task_id": state.task.task_id,
            "goal": state.task.goal,
            "test_command": state.task.test_command,
            "allowed_paths": state.task.allowed_paths,
            "forbidden_paths": state.task.forbidden_paths,
            "status": state.status.value,
            "remaining_budget": remaining,
            "current_plan": state.plan,
            "current_hypothesis": state.current_hypothesis,
            "rejected_hypotheses": state.rejected_hypotheses[-5:],
            "changed_files": state.changed_files,
            "repository_revision": state.repository_revision,
            "verified_revision": state.verified_revision,
            "full_suite_passed": state.full_suite_passed,
        }
        return (
            "CURRENT REPAIR STATE:\n"
            f"{json.dumps(state_summary, indent=2, sort_keys=True)}\n\n"
            "RECENT ACTION-OBSERVATION TRAJECTORY:\n"
            f"{cls._trajectory(state)}\n\n"
            "Choose the single best next tool action. Include a concise plan, "
            "reasoning summary, optional current hypothesis, and one action "
            "with valid arguments.\n"
            f"{cls._reflection_instruction(state)}"
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
            decision = AgentDecision.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise PolicyResponseError(
                f"Model decision failed schema validation: {exc}",
                raw_response=raw_response,
            ) from exc

        if not decision.plan:
            raise PolicyResponseError(
                "Model decision must contain at least one plan step.",
                raw_response=raw_response,
            )

        return decision

    def _validate_decision(
        self,
        state: AgentState,
        decision: AgentDecision,
        raw_response: str,
    ) -> AgentDecision:
        """Apply policy-mode rules after shared schema validation."""
        del state

        if decision.reflection is not None:
            raise PolicyResponseError(
                "The no-reflection policy requires reflection to be null.",
                raw_response=raw_response,
            )

        return decision

    def decide(self, state: AgentState) -> AgentDecision:
        """Ask the model to select and justify one restricted tool action."""
        system_prompt = self._system_prompt()
        base_prompt = self._state_prompt(state)
        response_schema = AgentDecision.model_json_schema()
        user_prompt = base_prompt

        for attempt in range(1, self.max_parse_attempts + 1):
            raw_response = self.model.generate(
                system_prompt,
                user_prompt,
                response_schema=response_schema,
            )

            try:
                decision = self._parse_decision(raw_response)
                return self._validate_decision(
                    state,
                    decision,
                    raw_response,
                )
            except PolicyResponseError as exc:
                if attempt >= self.max_parse_attempts:
                    raise PolicyResponseError(
                        "Model did not return a valid tool decision after "
                        f"{self.max_parse_attempts} attempt(s): {exc}",
                        raw_response=raw_response,
                    ) from exc

                user_prompt = (
                    base_prompt
                    + "\n\nCORRECTION REQUIRED:\n"
                    + str(exc)
                    + "\nReturn a corrected JSON decision only. "
                    + "Previous invalid response:\n"
                    + raw_response[: self._MAX_RAW_RESPONSE]
                )

        raise AssertionError("Unreachable model-decision loop.")
