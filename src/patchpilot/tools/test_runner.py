"""Bounded pytest execution for PatchPilot repair verification."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from patchpilot.schemas import (
    ObservationStatus,
    RepairTask,
    ToolName,
    ToolObservation,
)
from patchpilot.tools.repository import RepositorySandbox

_MAX_COMMAND_ARGUMENTS = 30
_DEFAULT_OUTPUT_LIMIT = 200_000

_SAFE_PYTEST_FLAGS = {
    "-q",
    "-x",
    "-ra",
    "--disable-warnings",
    "--tb=short",
    "--tb=line",
}


class TestExecutionError(ValueError):
    """Raised when a test command violates execution policy."""


class TestRunner:
    """Execute controlled pytest commands without a shell."""

    def __init__(
        self,
        sandbox: RepositorySandbox,
        task: RepairTask,
        timeout_seconds: int = 60,
        max_output_bytes: int = _DEFAULT_OUTPUT_LIMIT,
    ) -> None:
        if not 1 <= timeout_seconds <= 600:
            raise TestExecutionError(
                "timeout_seconds must be between 1 and 600."
            )

        if not 1_000 <= max_output_bytes <= 1_000_000:
            raise TestExecutionError(
                "max_output_bytes must be between 1000 and 1000000."
            )

        self.repository_root = sandbox.repository_root
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.base_arguments = self._validate_test_command(
            task.test_command
        )

    def _validate_repository_target(self, raw_target: str) -> str:
        """Validate a pytest file, directory, or node identifier."""
        if not raw_target or raw_target.startswith("-"):
            raise TestExecutionError(
                "Test targets must be repository-relative paths."
            )

        path_text = raw_target.split("::", maxsplit=1)[0]
        requested = Path(path_text)

        if requested.is_absolute() or ".." in requested.parts:
            raise TestExecutionError(
                "Test targets cannot be absolute or contain '..'."
            )

        try:
            resolved = (
                self.repository_root / requested
            ).resolve(strict=True)
        except FileNotFoundError as exc:
            raise TestExecutionError(
                f"Test target does not exist: {path_text}"
            ) from exc

        if not resolved.is_relative_to(self.repository_root):
            raise TestExecutionError(
                "Test target escapes the repository sandbox."
            )

        return raw_target

    def _validate_test_command(
        self,
        command: list[str],
    ) -> tuple[str, ...]:
        """Allow only bounded ``python -m pytest`` commands."""
        if len(command) < 3 or len(command) > _MAX_COMMAND_ARGUMENTS:
            raise TestExecutionError(
                "The test command has an invalid number of arguments."
            )

        if (
            command[0] not in {"python", "python3"}
            or command[1:3] != ["-m", "pytest"]
        ):
            raise TestExecutionError(
                "Only 'python -m pytest' commands are permitted."
            )

        validated: list[str] = []

        for argument in command[3:]:
            if "\x00" in argument or "\n" in argument:
                raise TestExecutionError(
                    "Test arguments contain forbidden characters."
                )

            if argument in _SAFE_PYTEST_FLAGS:
                validated.append(argument)
                continue

            if argument.startswith("--maxfail="):
                value = argument.partition("=")[2]

                if value.isdigit() and 1 <= int(value) <= 20:
                    validated.append(argument)
                    continue

                raise TestExecutionError(
                    "--maxfail must be an integer from 1 to 20."
                )

            if argument.startswith("-"):
                raise TestExecutionError(
                    f"Unsupported pytest option: {argument}"
                )

            validated.append(
                self._validate_repository_target(argument)
            )

        return tuple(validated)

    def _build_command(
        self,
        target: str | None,
    ) -> list[str]:
        """Build the final shell-free pytest command."""
        arguments = list(self.base_arguments)

        if target is not None:
            arguments.append(
                self._validate_repository_target(target)
            )

        return [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            *arguments,
        ]

    def _environment(self) -> dict[str, str]:
        """Create a minimal environment without inherited secrets."""
        return {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(self.repository_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

    def _bounded_output(
        self,
        stdout: bytes | None,
        stderr: bytes | None,
    ) -> str:
        """Decode and truncate captured process output."""
        combined = (stdout or b"") + (stderr or b"")
        truncated = len(combined) > self.max_output_bytes
        visible = combined[: self.max_output_bytes]
        text = visible.decode("utf-8", errors="replace")

        if truncated:
            text += "\n[PatchPilot truncated test output.]"

        return text.strip()

    def run_tests(
        self,
        target: str | None = None,
    ) -> ToolObservation:
        """Run the complete suite or one validated target."""
        started_at = perf_counter()

        try:
            command = self._build_command(target)

            result = subprocess.run(
                command,
                cwd=self.repository_root,
                env=self._environment(),
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )

            status = (
                ObservationStatus.OK
                if result.returncode == 0
                else ObservationStatus.ERROR
            )
            outcome = (
                "passed"
                if result.returncode == 0
                else "failed"
            )

            return ToolObservation(
                tool=ToolName.RUN_TESTS,
                status=status,
                summary=(
                    f"Tests {outcome} with exit code "
                    f"{result.returncode}."
                ),
                output=self._bounded_output(
                    result.stdout,
                    result.stderr,
                ),
                duration_seconds=perf_counter() - started_at,
            )
        except TestExecutionError as exc:
            return ToolObservation(
                tool=ToolName.RUN_TESTS,
                status=ObservationStatus.REJECTED,
                summary=str(exc),
                duration_seconds=perf_counter() - started_at,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolObservation(
                tool=ToolName.RUN_TESTS,
                status=ObservationStatus.TIMEOUT,
                summary=(
                    "Test execution exceeded the configured "
                    f"{self.timeout_seconds}-second timeout."
                ),
                output=self._bounded_output(
                    exc.stdout,
                    exc.stderr,
                ),
                duration_seconds=perf_counter() - started_at,
            )
