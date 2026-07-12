"""LLM-backed decision policy for PatchPilot."""

from __future__ import annotations

import difflib
import re
from time import perf_counter
from typing import Any, Protocol

from patchpilot.agent.policy import AgentDecision
from patchpilot.schemas import AgentState, ObservationStatus, ToolAction, ToolName
from patchpilot.schemas.models import ModelCallRecord


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


def _model_trace_metadata(
    model: TextGenerationModel,
) -> tuple[str, str | None, dict[str, Any]]:
    """Return stable model identity and generation configuration."""
    backend = f"{type(model).__module__}.{type(model).__qualname__}"
    model_name: str | None = None
    config: dict[str, Any] = {}

    provider = getattr(model, "trace_metadata", None)
    if not callable(provider):
        return backend, model_name, config

    metadata = provider()
    if not isinstance(metadata, dict):
        raise TypeError("trace_metadata() must return a dictionary.")

    copied = dict(metadata)
    raw_backend = copied.pop("backend", backend)
    raw_model_name = copied.pop("model", None)
    backend = str(raw_backend)

    if raw_model_name is not None:
        model_name = str(raw_model_name)

    return backend, model_name, copied


def _generate_with_trace(
    *,
    state: AgentState,
    model: TextGenerationModel,
    policy_name: str,
    purpose: str,
    attempt: int,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, object] | None,
) -> tuple[str, int]:
    """Generate once and preserve prompts, response, identity, and errors."""
    backend, model_name, model_config = _model_trace_metadata(model)
    state.model_calls += 1
    call_index = state.model_calls
    started_at = perf_counter()

    try:
        raw_response = model.generate(
            system_prompt,
            user_prompt,
            response_schema=response_schema,
        )
    except Exception as exc:
        state.model_call_records.append(
            ModelCallRecord(
                call_index=call_index,
                policy=policy_name,
                purpose=purpose,
                attempt=attempt,
                backend=backend,
                model_name=model_name,
                generation_config=model_config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=response_schema,
                generation_succeeded=False,
                duration_seconds=perf_counter() - started_at,
                error_type=type(exc).__name__,
                error_message=str(exc)[:2000],
            )
        )
        raise

    state.model_call_records.append(
        ModelCallRecord(
            call_index=call_index,
            policy=policy_name,
            purpose=purpose,
            attempt=attempt,
            backend=backend,
            model_name=model_name,
            generation_config=model_config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
            raw_response=raw_response,
            generation_succeeded=True,
            duration_seconds=perf_counter() - started_at,
        )
    )
    return raw_response, len(state.model_call_records) - 1


def _mark_model_call_parse(
    state: AgentState,
    record_index: int,
    *,
    succeeded: bool,
    error: Exception | None = None,
) -> None:
    """Attach parse or policy-validation outcome to one model call."""
    record = state.model_call_records[record_index]
    record.parse_succeeded = succeeded

    if error is not None:
        record.error_type = type(error).__name__
        record.error_message = str(error)[:2000]


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
    def _function_span(
        lines: list[str],
        target: str,
    ) -> tuple[int, int, str] | None:
        """Locate a Python function block by indentation."""
        pattern = re.compile(
            rf"^(?P<indent>[ \t]*)(?:async[ \t]+def|def)"
            rf"[ \t]+{re.escape(target)}[ \t]*\("
        )

        for start, line in enumerate(lines):
            match = pattern.match(line)
            if match is None:
                continue

            indentation = match.group("indent")
            indentation_width = len(indentation.expandtabs(4))
            end = len(lines)

            for index in range(start + 1, len(lines)):
                following = lines[index]
                if not following.strip():
                    continue

                leading = following[: len(following) - len(following.lstrip())]
                if len(leading.expandtabs(4)) <= indentation_width:
                    end = index
                    break

            return start, end, indentation

        return None

    @staticmethod
    def _unfenced_response(raw_response: str) -> str:
        """Strip one optional Python code fence from a model response."""
        text = raw_response.strip()
        fence = re.search(
            r"```(?:python|py)?\s*(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fence:
            return fence.group(1).strip()
        return text

    @classmethod
    def _synthesise_function_diff(
        cls,
        raw_response: str,
        path: str,
        read_output: str,
        target: str,
    ) -> str:
        """Build a bounded unified diff from one corrected function."""
        original = [line for _, line in cls._source_lines(read_output)]
        original_span = cls._function_span(original, target)
        if original_span is None:
            raise PolicyResponseError(
                f"Could not locate function {target} in the read source.",
                raw_response=raw_response,
            )

        candidate_text = cls._unfenced_response(raw_response)
        candidate_lines = candidate_text.splitlines()
        if any(
            line.startswith(("diff --git ", "--- ", "+++ ", "@@ "))
            for line in candidate_lines
        ):
            raise PolicyResponseError(
                "Diff-formatted output must use the compatibility parser.",
                raw_response=raw_response,
            )
        candidate_span = cls._function_span(candidate_lines, target)
        if candidate_span is None:
            raise PolicyResponseError(
                f"Model response did not contain function {target}.",
                raw_response=raw_response,
            )

        original_start, original_end, destination_indent = original_span
        candidate_start, candidate_end, candidate_indent = candidate_span
        replacement = candidate_lines[candidate_start:candidate_end]

        while replacement and not replacement[-1].strip():
            replacement.pop()

        normalized: list[str] = []
        for line in replacement:
            if not line.strip():
                normalized.append("")
                continue

            if line.startswith(candidate_indent):
                relative = line[len(candidate_indent) :]
            else:
                relative = line.lstrip()
            normalized.append(destination_indent + relative)

        updated = [
            *original[:original_start],
            *normalized,
            *original[original_end:],
        ]
        if updated == original:
            raise PolicyResponseError(
                f"Model returned an unchanged function {target}.",
                raw_response=raw_response,
            )

        diff = list(
            difflib.unified_diff(
                original,
                updated,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
                lineterm="",
            )
        )
        patch_text = f"diff --git a/{path} b/{path}\n" + "\n".join(diff) + "\n"

        changed_lines = sum(
            1
            for line in patch_text.splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )
        if changed_lines > 20:
            raise PolicyResponseError(
                "Corrected function exceeds the 20-line patch boundary.",
                raw_response=raw_response,
            )

        if not cls._has_changed_lines(patch_text):
            raise PolicyResponseError(
                f"Corrected function {target} produced no usable diff.",
                raw_response=raw_response,
            )

        return patch_text

    @classmethod
    def _extract_diff(
        cls,
        raw_response: str,
        path: str,
        read_output: str,
        target: str,
    ) -> str:
        """Convert a corrected function or line into a unified diff."""
        try:
            return cls._synthesise_function_diff(
                raw_response,
                path,
                read_output,
                target,
            )
        except PolicyResponseError as function_error:
            text = cls._unfenced_response(raw_response)
            try:
                return cls._synthesise_single_line_diff(
                    text,
                    path,
                    read_output,
                )
            except PolicyResponseError as line_error:
                raise PolicyResponseError(
                    "Model response contained neither a usable corrected "
                    f"function nor a replacement line: {line_error}",
                    raw_response=raw_response,
                ) from function_error

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

    @staticmethod
    def _previous_patch_texts(state: AgentState) -> list[str]:
        """Return recent model patches that did not complete the task."""
        patches: list[str] = []
        for action in state.actions:
            if action.tool is not ToolName.APPLY_PATCH:
                continue
            patch_text = action.arguments.get("patch_text")
            if isinstance(patch_text, str) and patch_text.strip():
                patches.append(patch_text.strip())
        return patches[-2:]

    @classmethod
    def _ensure_novel_patch(
        cls,
        state: AgentState,
        patch_text: str,
        raw_response: str,
    ) -> None:
        """Reject exact repeats of patches already tried in this run."""
        if patch_text.strip() in cls._previous_patch_texts(state):
            raise PolicyResponseError(
                "The generated patch exactly repeats a previous failed patch. "
                "Revise the repair using the latest failing-test evidence.",
                raw_response=raw_response,
            )

    def _generate_patch_decision(self, state: AgentState) -> AgentDecision:
        path, content = self._last_read_file(state)
        test_output = self._latest_failed_test_output(state)
        focused_content = self._focused_source_context(state, content)
        target = self._target_function_from_failure(state) or "the failing function"
        previous_patches = self._previous_patch_texts(state)
        previous_patch_context = ""
        if previous_patches:
            previous_patch_context = (
                "\nPREVIOUS FAILED PATCHES — DO NOT REPEAT:\n"
                + previous_patches[-1][:1000]
            )

        system_prompt = (
            "You are PatchPilot. Return only the complete corrected Python "
            "function. Satisfy every requirement in the repair goal and every "
            "remaining failing assertion."
        )
        user_prompt = (
            f"REPAIR GOAL: {state.task.goal}\n"
            f"FILE: {path}\n"
            f"TARGET FUNCTION: {target}\n"
            f"SOURCE:\n{focused_content}\n"
            f"FAILING TEST EVIDENCE:\n{test_output[:1400]}\n"
            "Return the complete corrected definition beginning with def "
            "or async def. Preserve the signature. Implement every behavior "
            "named in the repair goal, even when only one test currently "
            "fails. Return source only, without markdown, tests, diff "
            "markers, imports, or explanation."
            f"{previous_patch_context}"
        )

        raw_diff, first_record = _generate_with_trace(
            state=state,
            model=self.model,
            policy_name=type(self).__name__,
            purpose="patch_generation",
            attempt=1,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=None,
        )
        try:
            patch_text = self._extract_diff(
                raw_diff,
                path,
                content,
                target,
            )
            self._ensure_novel_patch(
                state,
                patch_text,
                raw_diff,
            )
            _mark_model_call_parse(
                state,
                first_record,
                succeeded=True,
            )
        except PolicyResponseError as first_error:
            _mark_model_call_parse(
                state,
                first_record,
                succeeded=False,
                error=first_error,
            )
            state.decision_parse_failures += 1
            retry_system_prompt = (
                "You are PatchPilot acting as a repair critic. Return a revised "
                "complete Python function that differs from the failed patch "
                "and resolves the remaining evidence."
            )
            retry_prompt = (
                user_prompt
                + "\nCORRECTION REQUIRED:"
                + str(first_error)
                + "\nThe prior function was incomplete. Preserve "
                "improvements that passed tests, then change at least one "
                "different original source line to address the remaining "
                "assertion. Return only the revised complete function. "
                "Previous answer:" + raw_diff[:500]
            )
            retry_diff, retry_record = _generate_with_trace(
                state=state,
                model=self.model,
                policy_name=type(self).__name__,
                purpose="patch_generation",
                attempt=2,
                system_prompt=retry_system_prompt,
                user_prompt=retry_prompt,
                response_schema=None,
            )
            try:
                patch_text = self._extract_diff(
                    retry_diff,
                    path,
                    content,
                    target,
                )
                self._ensure_novel_patch(
                    state,
                    patch_text,
                    retry_diff,
                )
                _mark_model_call_parse(
                    state,
                    retry_record,
                    succeeded=True,
                )
            except PolicyResponseError as retry_error:
                _mark_model_call_parse(
                    state,
                    retry_record,
                    succeeded=False,
                    error=retry_error,
                )
                state.decision_parse_failures += 1
                raise PolicyResponseError(
                    f"{retry_error} First invalid response: {raw_diff[:300]}",
                    raw_response=retry_diff,
                ) from first_error

        return self._make_decision(
            summary="Generate and apply a model-proposed source patch.",
            plan=("Apply the smallest source patch satisfying all known evidence."),
            tool=ToolName.APPLY_PATCH,
            arguments={"patch_text": patch_text},
            rationale="Apply the model-proposed unified diff.",
        )

    def decide(self, state: AgentState) -> AgentDecision:
        """Generate the next staged repair decision."""
        if state.rollback_required:
            raise PolicyResponseError(
                "Runtime transactional rollback must complete before the "
                "scaffolded policy can continue."
            )

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

            if state.current_attempt_id is not None:
                raise PolicyResponseError(
                    "Failed verification requires runtime transactional rollback "
                    "before the scaffolded policy continues."
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
            relative_path = last_action.arguments.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path:
                if state.last_rolled_back_attempt_files:
                    relative_path = state.last_rolled_back_attempt_files[0]
                else:
                    raise PolicyResponseError(
                        "Rollback completed without a source file to inspect."
                    )

            return self._make_decision(
                summary="Read restored source before retrying repair.",
                plan="Inspect the clean source before another patch attempt.",
                tool=ToolName.READ_FILE,
                arguments={"relative_path": relative_path},
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

            if state.current_attempt_id is not None:
                raise PolicyResponseError(
                    "Failed syntax validation requires runtime transactional "
                    "rollback before the scaffolded policy continues."
                )

        raise PolicyResponseError(
            f"No staged policy transition for {last_action.tool.value}/"
            f"{last_observation.status.value}."
        )
