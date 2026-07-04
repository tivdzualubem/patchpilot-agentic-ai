"""LLM-backed decision policy for PatchPilot."""

from __future__ import annotations

import difflib
import json
import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from patchpilot.schemas import AgentState, ObservationStatus, ToolAction, ToolName


class PolicyResponseError(RuntimeError):
    """Raised when the model response cannot be converted into a decision."""


class AgentDecision(BaseModel):
    """One validated decision produced by the policy."""

    model_config = ConfigDict(extra="forbid")

    reasoning_summary: str = Field(min_length=3, max_length=2000)
    plan: list[str] = Field(min_length=1, max_length=20)
    hypothesis: str | None = Field(default=None, max_length=2000)
    reflection: str | None = Field(default=None, max_length=2000)
    action: ToolAction


class TextGenerationModel(Protocol):
    """Model interface required by the structured decision policy."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        """Generate one model response."""


class StructuredLLMPolicy:
    """Convert model output into validated PatchPilot decisions."""

    def __init__(self, model: TextGenerationModel) -> None:
        self.model = model

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are PatchPilot, a bounded Python repair agent. "
            "Return JSON only with top-level keys: reasoning_summary, "
            "plan, hypothesis, reflection, action. The action object must "
            "contain tool, arguments, rationale. Never modify tests. "
            "Use repository tools to inspect source, apply the smallest "
            "source patch, then run the full test suite. Finish only after "
            "the current source revision passes the full suite."
        )

    @staticmethod
    def _compact_prompt_value(value: object, max_string: int = 2000) -> object:
        """Bound observation data sent back to small local models."""
        if isinstance(value, str):
            if len(value) <= max_string:
                return value
            return (
                f"{value[:1000]}\n"
                "...[truncated for prompt budget]...\n"
                f"{value[-1000:]}"
            )

        if isinstance(value, list):
            return [
                StructuredLLMPolicy._compact_prompt_value(item, max_string)
                for item in value[-20:]
            ]

        if isinstance(value, dict):
            return {
                str(key): StructuredLLMPolicy._compact_prompt_value(item, max_string)
                for key, item in value.items()
            }

        return value

    @staticmethod
    def _user_prompt(state: AgentState) -> str:
        payload = {
            "task": state.task.model_dump(mode="json"),
            "status": state.status.value,
            "budget": state.budget.model_dump(mode="json"),
            "usage": state.usage.model_dump(mode="json"),
            "changed_files": state.changed_files,
            "repository_revision": state.repository_revision,
            "verified_revision": state.verified_revision,
            "last_action": (
                state.actions[-1].model_dump(mode="json")
                if state.actions
                else None
            ),
            "last_observation": (
                StructuredLLMPolicy._compact_prompt_value(
                    state.observations[-1].model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                )
                if state.observations
                else None
            ),
        }
        return (
            "Choose the next Plan-Act-Observe-Reflect-Verify step.\n"
            "Current validated state compact view below.\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _strip_code_fence(response: str) -> str:
        text = response.strip()

        if not text.startswith("```"):
            return text

        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise PolicyResponseError(
                "The model returned an incomplete JSON code fence."
            )

        return "\n".join(lines[1:-1]).strip()

    @staticmethod
    def _extract_json_object(response: str) -> str:
        text = StructuredLLMPolicy._strip_code_fence(response)
        decoder = json.JSONDecoder()

        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                _, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            return text[index : index + end]

        raise PolicyResponseError("The model response did not contain JSON.")

    @staticmethod
    def _string_or_none(value: object) -> str | None:
        if isinstance(value, str) and value.strip():
            return value
        return None

    @staticmethod
    def _plan(value: object) -> list[str]:
        if isinstance(value, str) and value.strip():
            return [value]
        if isinstance(value, list):
            items = [str(item) for item in value if str(item).strip()]
            if items:
                return items
        return ["Continue the bounded repair workflow."]

    @staticmethod
    def _normalise_tool(tool: str) -> str:
        value = tool.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "test": "run_tests",
            "pytest": "run_tests",
            "run_pytest": "run_tests",
            "read": "read_file",
            "read_source": "read_file",
            "search": "search_code",
            "grep": "search_code",
            "list": "list_files",
            "ls": "list_files",
            "patch": "apply_patch",
            "apply": "apply_patch",
            "edit": "apply_patch",
            "edit_file": "apply_patch",
            "write_file": "apply_patch",
            "modify_file": "apply_patch",
        }
        return aliases.get(value, value)

    @staticmethod
    def _normalise_payload(payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise PolicyResponseError("The model JSON was not an object.")

        action_value = payload.get("action")
        if isinstance(action_value, str):
            action_value = {"tool": action_value}
        elif not isinstance(action_value, dict):
            if "tool" in payload or "name" in payload:
                action_value = payload
            else:
                raise PolicyResponseError("The model JSON did not contain action.")

        tool_value = action_value.get("tool") or action_value.get("name")
        if not isinstance(tool_value, str) or not tool_value.strip():
            raise PolicyResponseError("The model action did not contain tool.")

        tool = StructuredLLMPolicy._normalise_tool(tool_value)

        arguments = action_value.get("arguments")
        if arguments is None:
            arguments = action_value.get("args")
        if arguments is None:
            arguments = action_value.get("parameters")
        if arguments is None:
            arguments = {}

        if isinstance(arguments, str) and tool == "apply_patch":
            arguments = {"patch_text": arguments}

        if not isinstance(arguments, dict):
            raise PolicyResponseError("The model action arguments were invalid.")

        for source in (payload, action_value):
            for key in (
                "patch_text",
                "relative_path",
                "query",
                "start_line",
                "end_line",
            ):
                if key in source and key not in arguments:
                    arguments[key] = source[key]

        rationale = (
            action_value.get("rationale")
            or action_value.get("reason")
            or action_value.get("reasoning")
            or "Take the next bounded repair action."
        )

        reasoning_summary = (
            payload.get("reasoning_summary")
            or payload.get("reasoning")
            or payload.get("summary")
            or "Select the next repair action."
        )

        return {
            "reasoning_summary": str(reasoning_summary),
            "plan": StructuredLLMPolicy._plan(payload.get("plan")),
            "hypothesis": StructuredLLMPolicy._string_or_none(
                payload.get("hypothesis")
            ),
            "reflection": StructuredLLMPolicy._string_or_none(
                payload.get("reflection")
            ),
            "action": {
                "tool": tool,
                "arguments": arguments,
                "rationale": str(rationale),
            },
        }

    @staticmethod
    def _decision_from_response(response: str) -> AgentDecision:
        try:
            payload = json.loads(StructuredLLMPolicy._extract_json_object(response))
            normalised = StructuredLLMPolicy._normalise_payload(payload)
            return AgentDecision.model_validate(normalised)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise PolicyResponseError(
                f"The model returned an invalid AgentDecision: {response[:500]}"
            ) from exc

    @staticmethod
    def _make_decision(
        *,
        summary: str,
        plan: str,
        tool: ToolName,
        arguments: dict[str, object] | None = None,
        rationale: str,
    ) -> AgentDecision:
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

    @staticmethod
    def _latest_failed_test_output(state: AgentState) -> str:
        for observation in reversed(state.observations):
            if (
                observation.tool is ToolName.RUN_TESTS
                and observation.status is ObservationStatus.ERROR
            ):
                return observation.output
        return ""

    @staticmethod
    def _failure_query(state: AgentState) -> str:
        output = StructuredLLMPolicy._latest_failed_test_output(state)
        patterns = (
            r"where .*?=\s*([A-Za-z_]\w*)\(",
            r"assert\s+([A-Za-z_]\w*)\(",
            r"\b([A-Za-z_]\w*)\s+function\b",
        )

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1)

        goal_match = re.search(r"\b([A-Za-z_]\w*)\s+function\b", state.task.goal)
        if goal_match:
            return goal_match.group(1)

        return state.task.goal.split()[0]

    @staticmethod
    def _source_path_from_search(output: str) -> str:
        match = re.search(r"src/[A-Za-z0-9_./-]+\.py", output)
        if not match:
            raise PolicyResponseError(
                "Could not locate a source file in search output."
            )
        return match.group(0)

    @staticmethod
    def _last_read_file(state: AgentState) -> tuple[str, str]:
        for action, observation in zip(
            reversed(state.actions),
            reversed(state.observations),
            strict=False,
        ):
            if (
                action.tool is ToolName.READ_FILE
                and observation.tool is ToolName.READ_FILE
                and observation.status is ObservationStatus.OK
            ):
                path = action.arguments.get("relative_path")
                if isinstance(path, str) and path:
                    return path, observation.output

        raise PolicyResponseError("No successful source read is available.")

    @staticmethod
    def _source_lines(read_output: str) -> list[tuple[int, str]]:
        lines: list[tuple[int, str]] = []
        for fallback, raw_line in enumerate(read_output.splitlines(), start=1):
            match = re.match(r"^(\d+):\s?(.*)$", raw_line)
            if match:
                lines.append((int(match.group(1)), match.group(2)))
            else:
                lines.append((fallback, raw_line))
        return lines

    @staticmethod
    def _has_changed_lines(diff_text: str) -> bool:
        has_removed = any(
            line.startswith("-") and not line.startswith("---")
            for line in diff_text.splitlines()
        )
        has_added = any(
            line.startswith("+") and not line.startswith("+++")
            for line in diff_text.splitlines()
        )
        return has_removed and has_added

    @staticmethod
    def _synthesise_single_line_diff(
        raw_response: str,
        path: str,
        read_output: str,
    ) -> str:
        source_lines = StructuredLLMPolicy._source_lines(read_output)
        original = [line for _, line in source_lines]

        best: tuple[float, int, str] | None = None
        for raw_line in raw_response.splitlines():
            if raw_line.startswith(("diff --git", "---", "+++", "@@", "index ")):
                continue

            candidate = raw_line[1:] if raw_line.startswith("+") else raw_line
            if raw_line.startswith(" ") and not raw_line.startswith("  "):
                candidate = raw_line[1:]
            if not candidate.strip() or candidate in original:
                continue

            for index, existing in enumerate(original):
                score = difflib.SequenceMatcher(
                    None,
                    existing.strip(),
                    candidate.strip(),
                ).ratio()
                if score < 0.70 or existing.strip() == candidate.strip():
                    continue

                indentation = existing[: len(existing) - len(existing.lstrip())]
                candidate = indentation + candidate.lstrip()

                if best is None or score > best[0]:
                    best = (score, index, candidate)

        if best is None:
            raise PolicyResponseError(
                f"No valid changed lines in model diff: {raw_response[:500]}"
            )

        _, index, replacement = best
        updated = list(original)
        updated[index] = replacement

        diff_lines = difflib.unified_diff(
            original,
            updated,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
            lineterm="",
        )
        return (
            f"diff --git a/{path} b/{path}\n"
            + "\n".join(diff_lines)
            + "\n"
        )

    @staticmethod
    def _extract_diff(raw_response: str, path: str, read_output: str) -> str:
        text = raw_response.strip()
        fence = re.search(r"```(?:diff)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        return StructuredLLMPolicy._synthesise_single_line_diff(
            text,
            path,
            read_output,
        )

    def _generate_patch_decision(self, state: AgentState) -> AgentDecision:
        path, content = self._last_read_file(state)
        test_output = self._latest_failed_test_output(state)

        raw_diff = self.model.generate(
            "You are PatchPilot. Return ONLY a unified diff. No JSON. No markdown.",
            (
                "Fix the Python source file using the failing tests.\n"
                f"FILE: {path}\n"
                f"CONTENT:\n{content}\n\n"
                f"FAILING TEST OUTPUT:\n{test_output[:2000]}\n\n"
                "Return ONLY a unified diff with diff --git, ---/+++, and @@ lines."
            ),
            response_schema=None,
        )
        patch_text = self._extract_diff(raw_diff, path, content)

        return self._make_decision(
            summary="Generate and apply a model-proposed source patch.",
            plan="Apply the smallest source patch.",
            tool=ToolName.APPLY_PATCH,
            arguments={"patch_text": patch_text},
            rationale="Apply the model-proposed unified diff.",
        )

    def decide(self, state: AgentState) -> AgentDecision:
        """Generate the next staged repair decision."""
        if not state.actions or not state.observations:
            return self._make_decision(
                summary="Run the full test suite before inspecting code.",
                plan="Reproduce the failure with the full test suite.",
                tool=ToolName.RUN_TESTS,
                rationale="Establish the failing test signal first.",
            )

        last_action = state.actions[-1]
        last_observation = state.observations[-1]

        if last_action.tool is ToolName.RUN_TESTS:
            if last_observation.status is ObservationStatus.OK:
                return self._make_decision(
                    summary="Finish after verified full-suite success.",
                    plan="End the repair run after successful verification.",
                    tool=ToolName.FINISH,
                    arguments={
                        "status": "succeeded",
                        "message": "Full test suite passed after repair.",
                    },
                    rationale="The current repository revision is verified.",
                )

            if state.changed_files:
                return self._make_decision(
                    summary="Read the changed source after failing verification.",
                    plan="Inspect the changed file before another repair attempt.",
                    tool=ToolName.READ_FILE,
                    arguments={"relative_path": state.changed_files[0]},
                    rationale="Inspect the patched source after failing tests.",
                )

            query = self._failure_query(state)
            return self._make_decision(
                summary="Search source code for the failing symbol.",
                plan="Locate the implementation connected to the failing tests.",
                tool=ToolName.SEARCH_CODE,
                arguments={"query": query, "relative_path": "src"},
                rationale="Find the source implementation referenced by failures.",
            )

        if (
            last_action.tool is ToolName.SEARCH_CODE
            and last_observation.status is ObservationStatus.OK
        ):
            return self._make_decision(
                summary="Read the source file found by search.",
                plan="Inspect the likely faulty implementation.",
                tool=ToolName.READ_FILE,
                arguments={
                    "relative_path": self._source_path_from_search(
                        last_observation.output
                    )
                },
                rationale="Inspect the source file containing the failing symbol.",
            )

        if (
            last_action.tool is ToolName.READ_FILE
            and last_observation.status is ObservationStatus.OK
        ):
            return self._generate_patch_decision(state)

        if (
            last_action.tool is ToolName.APPLY_PATCH
            and last_observation.status is ObservationStatus.OK
        ):
            return self._make_decision(
                summary="Run the full test suite after applying a patch.",
                plan="Verify the patched repository.",
                tool=ToolName.RUN_TESTS,
                rationale="Verify the current source revision.",
            )

        raise PolicyResponseError(
            f"No staged policy transition for {last_action.tool.value}/"
            f"{last_observation.status.value}."
        )
