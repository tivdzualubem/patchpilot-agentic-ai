"""LLM-backed decision policy for PatchPilot."""

from __future__ import annotations

import difflib
import re
from typing import Protocol

from patchpilot.agent.policy import AgentDecision
from patchpilot.schemas import AgentState, ObservationStatus, ToolAction, ToolName


class PolicyResponseError(RuntimeError):
    """Raised when the model response cannot be converted into a decision."""

    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response


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
    def _latest_failed_test_output(state: AgentState) -> str:
        for observation in reversed(state.observations):
            if (
                observation.tool is ToolName.RUN_TESTS
                and observation.status is not ObservationStatus.OK
            ):
                return observation.output
        return ""

    @staticmethod
    def _make_decision(
        summary: str,
        plan: str,
        tool: ToolName,
        rationale: str,
        arguments: dict[str, object] | None = None,
        hypothesis: str | None = None,
        reflection: str | None = None,
    ) -> AgentDecision:
        return AgentDecision(
            reasoning_summary=summary,
            plan=[plan],
            hypothesis=hypothesis,
            reflection=reflection,
            action=ToolAction(
                tool=tool,
                arguments=arguments or {},
                rationale=rationale,
            ),
        )

    @staticmethod
    def _target_function_from_failure(state: AgentState) -> str | None:
        """Infer the source function most directly implicated by test failure."""
        output = StructuredLLMPolicy._latest_failed_test_output(state)
        patterns = (
            r"where .*?=\s*([A-Za-z_]\w*)\(",
            r">\s*assert\s+([A-Za-z_]\w*)\(",
            r"\n\s{4}def\s+([A-Za-z_]\w*)\(",
            r"assert\s+([A-Za-z_]\w*)\(",
        )

        for pattern in patterns:
            for match in re.finditer(pattern, output):
                name = match.group(1)
                if not name.startswith("test_"):
                    return name

        return None

    @staticmethod
    def _failure_query(state: AgentState) -> str:
        target = StructuredLLMPolicy._target_function_from_failure(state)
        if target:
            return target

        goal_match = re.search(r"\b([A-Za-z_]\w*)\s+function\b", state.task.goal)
        if goal_match:
            return goal_match.group(1)

        return state.task.goal.split()[0]

    @staticmethod
    def _source_path_from_search(
        output: str,
        allowed_paths: list[str] | None = None,
    ) -> str:
        roots = allowed_paths or ["src"]
        escaped_roots = "|".join(re.escape(root.strip("/")) for root in roots)
        match = re.search(rf"(?:{escaped_roots})/[A-Za-z0-9_./-]+\.py", output)
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
            if raw_line.startswith("-"):
                continue

            candidate = raw_line[1:] if raw_line.startswith("+") else raw_line
            if candidate.lstrip().startswith(("+", "-")):
                continue
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
                f"No valid changed lines in model diff: {raw_response[:500]}",
                raw_response=raw_response,
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
        return f"diff --git a/{path} b/{path}\n" + "\n".join(diff_lines) + "\n"

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

    @staticmethod
    def _focused_source_context(state: AgentState, read_output: str) -> str:
        """Return the failing function block when it can be inferred."""
        target = StructuredLLMPolicy._target_function_from_failure(state)
        if not target:
            return read_output

        lines = StructuredLLMPolicy._source_lines(read_output)
        start_index: int | None = None
        for index, (_, line) in enumerate(lines):
            if re.match(rf"^(async\s+def|def)\s+{re.escape(target)}\(", line):
                start_index = index
                break

        if start_index is None:
            return read_output

        end_index = len(lines)
        for index in range(start_index + 1, len(lines)):
            _, line = lines[index]
            if re.match(r"^(async\s+def|def)\s+[A-Za-z_]\w*\(", line):
                end_index = index
                break

        context_start = max(0, start_index - 2)
        selected = lines[context_start:end_index]
        return "\n".join(f"{number}: {line}" for number, line in selected)

    def _generate_patch_decision(self, state: AgentState) -> AgentDecision:
        path, content = self._last_read_file(state)
        test_output = self._latest_failed_test_output(state)
        focused_content = self._focused_source_context(state, content)
        target = self._target_function_from_failure(state) or "the failing function"

        system_prompt = (
            "You are PatchPilot. Return only one corrected Python source line."
        )
        user_prompt = (
            "Fix the Python source file using the failing tests.\n"
            f"FILE: {path}\n"
            f"TARGET FUNCTION: {target}\n"
            f"RELEVANT SOURCE CONTEXT:\n{focused_content}\n\n"
            f"FAILING TEST OUTPUT:\n{test_output[:2000]}\n\n"
            "Return exactly one corrected replacement line from the source file. "
            "Do not return an assert line. Do not return test code. "
            "No diff markers, no markdown, no explanation."
        )

        raw_diff = self.model.generate(
            system_prompt,
            user_prompt,
            response_schema=None,
        )
        try:
            patch_text = self._extract_diff(raw_diff, path, content)
        except PolicyResponseError as first_error:
            retry_prompt = (
                user_prompt
                + "\n\nYour previous answer was not a valid source replacement "
                "line. Return only the corrected line that belongs in the source "
                f"function {target}. Previous invalid answer:\n"
                f"{raw_diff[:500]}"
            )
            retry_diff = self.model.generate(
                system_prompt,
                retry_prompt,
                response_schema=None,
            )
            try:
                patch_text = self._extract_diff(retry_diff, path, content)
            except PolicyResponseError as retry_error:
                raise PolicyResponseError(
                    f"{retry_error} First invalid response: {raw_diff[:300]}",
                    raw_response=retry_diff,
                ) from first_error

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
                    summary=(
                        "Restore the last changed source after failed verification."
                    ),
                    plan="Rollback the failed patch before trying another repair.",
                    tool=ToolName.RESTORE_FILE,
                    arguments={"relative_path": state.changed_files[0]},
                    rationale="Avoid stacking repairs on top of a failed patch.",
                )

            query = self._failure_query(state)
            return self._make_decision(
                summary="Search source code for the failing symbol.",
                plan="Locate the implementation connected to the failing tests.",
                tool=ToolName.SEARCH_CODE,
                arguments={
                    "query": query,
                    "relative_path": state.task.allowed_paths[0],
                },
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
                        last_observation.output,
                        state.task.allowed_paths,
                    )
                },
                rationale="Inspect the source file containing the failing symbol.",
            )

        if (
            last_action.tool is ToolName.RESTORE_FILE
            and last_observation.status is ObservationStatus.OK
        ):
            return self._make_decision(
                summary="Read restored source before retrying repair.",
                plan="Inspect the clean source before another patch attempt.",
                tool=ToolName.READ_FILE,
                arguments={
                    "relative_path": str(last_action.arguments["relative_path"])
                },
                rationale="Retry from a clean source snapshot.",
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
                summary="Check Python syntax immediately after applying a patch.",
                plan="Validate changed Python files before test execution.",
                tool=ToolName.CHECK_SYNTAX,
                rationale="Establish syntax evidence for the current revision.",
            )

        if last_action.tool is ToolName.CHECK_SYNTAX:
            if last_observation.status is ObservationStatus.OK:
                return self._make_decision(
                    summary="Run the full test suite after syntax validation.",
                    plan="Verify the syntax-checked repository.",
                    tool=ToolName.RUN_TESTS,
                    rationale="Verify the current source revision.",
                )

            if state.changed_files:
                return self._make_decision(
                    summary="Restore source after failed syntax validation.",
                    plan="Rollback the invalid patch before another repair attempt.",
                    tool=ToolName.RESTORE_FILE,
                    arguments={"relative_path": state.changed_files[0]},
                    rationale="Do not test or extend a syntactically invalid patch.",
                )

        raise PolicyResponseError(
            f"No staged policy transition for {last_action.tool.value}/"
            f"{last_observation.status.value}."
        )
