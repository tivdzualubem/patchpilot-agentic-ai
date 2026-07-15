"""Bounded Python syntax validation for changed source files."""

from __future__ import annotations

import ast
from pathlib import Path
from time import perf_counter

from patchpilot.schemas import (
    ObservationStatus,
    ToolName,
    ToolObservation,
)
from patchpilot.tools.repository import RepositorySandbox


class SyntaxChecker:
    """Parse changed Python files without executing repository code."""

    def __init__(self, sandbox: RepositorySandbox) -> None:
        self.repository_root = sandbox.repository_root

    @staticmethod
    def _observation(
        status: ObservationStatus,
        summary: str,
        started_at: float,
        output: str = "",
    ) -> ToolObservation:
        return ToolObservation(
            tool=ToolName.CHECK_SYNTAX,
            status=status,
            summary=summary,
            output=output,
            duration_seconds=perf_counter() - started_at,
        )

    def _resolve_changed_file(self, relative_path: str) -> Path:
        requested = Path(relative_path)

        if requested.is_absolute() or ".." in requested.parts:
            raise ValueError(
                "Changed-file paths must be relative and cannot contain '..'."
            )

        resolved = (self.repository_root / requested).resolve(strict=True)

        if not resolved.is_relative_to(self.repository_root):
            raise ValueError("Changed-file path escapes the repository sandbox.")

        if not resolved.is_file():
            raise ValueError("Syntax checking requires regular files.")

        return resolved

    def check_files(
        self,
        relative_paths: tuple[str, ...],
    ) -> ToolObservation:
        """Parse all changed Python files and report the first syntax error."""
        started_at = perf_counter()
        python_paths = tuple(
            path for path in relative_paths if Path(path).suffix == ".py"
        )

        if not python_paths:
            return self._observation(
                ObservationStatus.REJECTED,
                "No changed Python files are available for syntax checking.",
                started_at,
            )

        checked: list[str] = []

        try:
            for relative_path in python_paths:
                path = self._resolve_changed_file(relative_path)
                source = path.read_text(encoding="utf-8")

                try:
                    ast.parse(source, filename=relative_path)
                except SyntaxError as exc:
                    location = f"{relative_path}:{exc.lineno or 0}:{exc.offset or 0}"
                    message = exc.msg or "invalid syntax"
                    return self._observation(
                        ObservationStatus.ERROR,
                        "Python syntax check failed.",
                        started_at,
                        f"{location}: {message}",
                    )

                checked.append(relative_path)
        except (OSError, UnicodeError, ValueError) as exc:
            return self._observation(
                ObservationStatus.REJECTED,
                str(exc),
                started_at,
            )

        return self._observation(
            ObservationStatus.OK,
            f"Syntax check passed for {len(checked)} Python file(s).",
            started_at,
            "\n".join(checked),
        )
