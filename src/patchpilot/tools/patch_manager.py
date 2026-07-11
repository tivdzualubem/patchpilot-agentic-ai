"""Controlled source-code mutation and rollback for PatchPilot."""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from time import perf_counter

from patchpilot.schemas import (
    ObservationStatus,
    RepairTask,
    ToolName,
    ToolObservation,
)
from patchpilot.tools.repository import RepositorySandbox

_MAX_PATCH_BYTES = 50_000
_MAX_PATCH_FILES = 2
_MAX_CHANGED_LINES = 20
_GIT_TIMEOUT_SECONDS = 15

_BLOCKED_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
}

_UNSUPPORTED_PATCH_MARKERS = {
    "new file mode",
    "deleted file mode",
    "rename from",
    "rename to",
    "copy from",
    "copy to",
    "Binary files ",
    "GIT binary patch",
}


class PatchPolicyError(ValueError):
    """Raised when a proposed patch violates mutation policy."""


@dataclass(frozen=True)
class _PatchAttempt:
    """Exact pre-application snapshots for one successful patch attempt."""

    attempt_id: int
    snapshots: dict[str, bytes]

    @property
    def files(self) -> tuple[str, ...]:
        """Return the files changed by this attempt."""
        return tuple(sorted(self.snapshots))


class PatchManager:
    """Apply bounded patches and retain exact rollback snapshots."""

    def __init__(
        self,
        sandbox: RepositorySandbox,
        task: RepairTask,
        max_patch_bytes: int = _MAX_PATCH_BYTES,
        max_patch_files: int = _MAX_PATCH_FILES,
        max_changed_lines: int = _MAX_CHANGED_LINES,
    ) -> None:
        if shutil.which("git") is None:
            raise PatchPolicyError(
                "Git is required for validated unified-diff application."
            )

        if not 1_000 <= max_patch_bytes <= 1_000_000:
            raise PatchPolicyError("max_patch_bytes must be between 1000 and 1000000.")

        if not 1 <= max_patch_files <= 25:
            raise PatchPolicyError("max_patch_files must be between 1 and 25.")

        if not 1 <= max_changed_lines <= 1_000:
            raise PatchPolicyError("max_changed_lines must be between 1 and 1000.")

        self.repository_root = sandbox.repository_root
        self.allowed_paths = tuple(PurePosixPath(path) for path in task.allowed_paths)
        self.forbidden_paths = tuple(
            PurePosixPath(path) for path in task.forbidden_paths
        )
        self.max_patch_bytes = max_patch_bytes
        self.max_patch_files = max_patch_files
        self.max_changed_lines = max_changed_lines

        self._original_files: dict[str, bytes] = {}
        self._changed_files: set[str] = set()
        self._attempts: list[_PatchAttempt] = []
        self._next_attempt_id = 1

    @property
    def changed_files(self) -> tuple[str, ...]:
        """Return repository-relative files changed in this run."""
        return tuple(sorted(self._changed_files))

    @property
    def current_attempt_id(self) -> int | None:
        """Return the newest active patch-attempt identifier."""
        if not self._attempts:
            return None
        return self._attempts[-1].attempt_id

    @property
    def current_attempt_files(self) -> tuple[str, ...]:
        """Return files changed by the newest active patch attempt."""
        if not self._attempts:
            return ()
        return self._attempts[-1].files

    @staticmethod
    def _within(
        path: PurePosixPath,
        root: PurePosixPath,
    ) -> bool:
        """Return whether path equals or is below root."""
        return path == root or path.is_relative_to(root)

    def _validate_path(self, raw_path: str) -> str:
        """Validate one existing editable repository path."""
        if (
            not raw_path
            or raw_path.startswith("/")
            or "\x00" in raw_path
            or any(character.isspace() for character in raw_path)
        ):
            raise PatchPolicyError(
                "Patch paths must be simple repository-relative paths."
            )

        path = PurePosixPath(raw_path)

        if ".." in path.parts:
            raise PatchPolicyError("Patch paths cannot contain parent traversal.")

        if any(part in _BLOCKED_PARTS for part in path.parts):
            raise PatchPolicyError("Patch path targets a blocked repository area.")

        if any(part == ".env" or part.startswith(".env.") for part in path.parts):
            raise PatchPolicyError("Environment and secret files cannot be modified.")

        if not any(self._within(path, allowed) for allowed in self.allowed_paths):
            raise PatchPolicyError(f"Patch path is outside allowed paths: {raw_path}")

        if any(self._within(path, forbidden) for forbidden in self.forbidden_paths):
            raise PatchPolicyError(f"Patch path is forbidden: {raw_path}")

        candidate = self.repository_root / Path(*path.parts)

        if candidate.is_symlink():
            raise PatchPolicyError("Symbolic-link patch targets are not permitted.")

        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise PatchPolicyError("Patches may modify existing files only.") from exc

        if not resolved.is_relative_to(self.repository_root):
            raise PatchPolicyError("Patch target escapes the repository sandbox.")

        if not resolved.is_file():
            raise PatchPolicyError("Patch target must be a regular file.")

        raw = resolved.read_bytes()

        if b"\x00" in raw:
            raise PatchPolicyError("Binary files cannot be modified.")

        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PatchPolicyError("Only UTF-8 source files can be modified.") from exc

        return path.as_posix()

    @staticmethod
    def _count_changed_lines(patch_text: str) -> int:
        """Count added and removed content lines in a unified diff."""
        return sum(
            1
            for line in patch_text.splitlines()
            if (line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
        )

    @staticmethod
    def _remove_git_prefix(raw_path: str) -> str:
        """Remove the required a/ or b/ unified-diff prefix."""
        if not raw_path.startswith(("a/", "b/")):
            raise PatchPolicyError("Unified-diff paths must use a/ and b/ prefixes.")

        return raw_path[2:]

    def _extract_patch_paths(
        self,
        patch_text: str,
    ) -> tuple[str, ...]:
        """Parse and validate all paths declared by a unified diff."""
        encoded = patch_text.encode("utf-8")

        if not patch_text.strip():
            raise PatchPolicyError("Patch content cannot be empty.")

        if len(encoded) > self.max_patch_bytes:
            raise PatchPolicyError("Patch exceeds the configured size limit.")

        if "\x00" in patch_text:
            raise PatchPolicyError("Patch content contains a null byte.")

        for marker in _UNSUPPORTED_PATCH_MARKERS:
            if marker in patch_text:
                raise PatchPolicyError(
                    "File creation, deletion, rename, copy, and binary "
                    "patches are not permitted."
                )

        changed_lines = self._count_changed_lines(patch_text)
        if changed_lines > self.max_changed_lines:
            raise PatchPolicyError(
                "Patch changes more lines than the configured limit: "
                f"{changed_lines} > {self.max_changed_lines}."
            )

        declared: list[str] = []
        header_paths: list[str] = []

        for line in patch_text.splitlines():
            if line.startswith("diff --git "):
                fields = line.split()

                if len(fields) != 4:
                    raise PatchPolicyError("Malformed diff --git declaration.")

                old_path = self._remove_git_prefix(fields[2])
                new_path = self._remove_git_prefix(fields[3])

                if old_path != new_path:
                    raise PatchPolicyError("Renaming files is not permitted.")

                declared.append(self._validate_path(old_path))

            elif line.startswith(("--- ", "+++ ")):
                fields = line.split()

                if len(fields) < 2 or fields[1] == "/dev/null":
                    raise PatchPolicyError(
                        "File creation and deletion are not permitted."
                    )

                header_path = self._remove_git_prefix(fields[1])
                header_paths.append(self._validate_path(header_path))

        unique_paths = tuple(dict.fromkeys(declared))

        if not unique_paths:
            raise PatchPolicyError(
                "Patch must contain at least one diff --git declaration."
            )

        if len(unique_paths) > self.max_patch_files:
            raise PatchPolicyError(
                "Patch modifies more files than the configured limit."
            )

        if not header_paths:
            raise PatchPolicyError("Patch is missing unified-diff file headers.")

        undeclared = set(header_paths) - set(unique_paths)

        if undeclared:
            raise PatchPolicyError(
                "Patch contains file headers not declared by diff --git."
            )

        return unique_paths

    @staticmethod
    def _observation(
        tool: ToolName,
        status: ObservationStatus,
        summary: str,
        started_at: float,
        output: str = "",
    ) -> ToolObservation:
        """Create one timed patch-tool observation."""
        return ToolObservation(
            tool=tool,
            status=status,
            summary=summary,
            output=output,
            duration_seconds=perf_counter() - started_at,
        )

    def _recompute_changed_files(self) -> None:
        """Synchronize run-level change tracking with repository contents."""
        unchanged: list[str] = []
        for path, original in self._original_files.items():
            current = (self.repository_root / path).read_bytes()
            if current == original:
                unchanged.append(path)
            else:
                self._changed_files.add(path)

        for path in unchanged:
            self._changed_files.discard(path)
            self._original_files.pop(path, None)

    def _restore_snapshot(self, snapshots: dict[str, bytes]) -> None:
        """Restore a group of files atomically from exact byte snapshots."""
        current = {
            path: (self.repository_root / path).read_bytes() for path in snapshots
        }
        written: list[str] = []

        try:
            for path, content in snapshots.items():
                (self.repository_root / path).write_bytes(content)
                written.append(path)
        except OSError:
            for path in written:
                (self.repository_root / path).write_bytes(current[path])
            raise

    def _prune_attempt_path(self, path: str) -> None:
        """Remove a manually restored path from stored attempt snapshots."""
        retained: list[_PatchAttempt] = []
        for attempt in self._attempts:
            snapshots = dict(attempt.snapshots)
            snapshots.pop(path, None)
            if snapshots:
                retained.append(
                    _PatchAttempt(
                        attempt_id=attempt.attempt_id,
                        snapshots=snapshots,
                    )
                )
        self._attempts = retained

    def _run_git_apply(
        self,
        patch_text: str,
        check_only: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run git apply without a shell or inherited secrets."""
        command = ["git", "apply"]

        if check_only:
            command.append("--check")

        command.extend(["--whitespace=nowarn", "-"])

        environment = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "GIT_CEILING_DIRECTORIES": str(self.repository_root.parent),
        }

        return subprocess.run(
            command,
            cwd=self.repository_root,
            env=environment,
            input=patch_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )

    def apply_patch(self, patch_text: str) -> ToolObservation:
        """Validate and atomically apply an existing-file patch."""
        started_at = perf_counter()

        try:
            paths = self._extract_patch_paths(patch_text)

            check = self._run_git_apply(
                patch_text,
                check_only=True,
            )

            if check.returncode != 0:
                message = (
                    check.stderr.strip()
                    or check.stdout.strip()
                    or "Patch does not apply cleanly."
                )
                return self._observation(
                    ToolName.APPLY_PATCH,
                    ObservationStatus.ERROR,
                    "Patch validation failed.",
                    started_at,
                    message,
                )

            snapshots = {
                path: (self.repository_root / path).read_bytes() for path in paths
            }

            result = self._run_git_apply(
                patch_text,
                check_only=False,
            )

            if result.returncode != 0:
                for path, content in snapshots.items():
                    (self.repository_root / path).write_bytes(content)

                message = (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or "Patch application failed."
                )
                return self._observation(
                    ToolName.APPLY_PATCH,
                    ObservationStatus.ERROR,
                    "Patch application failed and files were restored.",
                    started_at,
                    message,
                )

            for path, content in snapshots.items():
                self._original_files.setdefault(path, content)
                self._changed_files.add(path)

            attempt = _PatchAttempt(
                attempt_id=self._next_attempt_id,
                snapshots=snapshots,
            )
            self._attempts.append(attempt)
            self._next_attempt_id += 1

            return self._observation(
                ToolName.APPLY_PATCH,
                ObservationStatus.OK,
                (
                    f"Applied patch attempt {attempt.attempt_id} "
                    f"to {len(paths)} file(s)."
                ),
                started_at,
                "\n".join(paths),
            )
        except PatchPolicyError as exc:
            return self._observation(
                ToolName.APPLY_PATCH,
                ObservationStatus.REJECTED,
                str(exc),
                started_at,
            )
        except subprocess.TimeoutExpired:
            return self._observation(
                ToolName.APPLY_PATCH,
                ObservationStatus.TIMEOUT,
                "Patch validation exceeded the execution timeout.",
                started_at,
            )

    def view_diff(self) -> ToolObservation:
        """Display the complete in-run diff against original snapshots."""
        started_at = perf_counter()
        sections: list[str] = []

        for path in sorted(self._changed_files):
            original = self._original_files[path].decode("utf-8")
            current = (self.repository_root / path).read_text(encoding="utf-8")

            sections.extend(
                difflib.unified_diff(
                    original.splitlines(),
                    current.splitlines(),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                    lineterm="",
                )
            )

        output = "\n".join(sections)

        return self._observation(
            ToolName.VIEW_DIFF,
            ObservationStatus.OK,
            (f"Generated diff for {len(self._changed_files)} changed file(s)."),
            started_at,
            output,
        )

    def restore_file(self, relative_path: str) -> ToolObservation:
        """Restore one modified file to its pre-run content."""
        started_at = perf_counter()

        try:
            path = self._validate_path(relative_path)

            if path not in self._original_files:
                return self._observation(
                    ToolName.RESTORE_FILE,
                    ObservationStatus.ERROR,
                    "No rollback snapshot exists for this file.",
                    started_at,
                )

            self._restore_snapshot({path: self._original_files[path]})
            self._prune_attempt_path(path)
            self._recompute_changed_files()

            return self._observation(
                ToolName.RESTORE_FILE,
                ObservationStatus.OK,
                f"Restored {path} to its pre-run content.",
                started_at,
                path,
            )
        except PatchPolicyError as exc:
            return self._observation(
                ToolName.RESTORE_FILE,
                ObservationStatus.REJECTED,
                str(exc),
                started_at,
            )
        except OSError as exc:
            return self._observation(
                ToolName.RESTORE_FILE,
                ObservationStatus.ERROR,
                "File restoration failed without committing a partial rollback.",
                started_at,
                str(exc),
            )

    def restore_attempt(self) -> ToolObservation:
        """Atomically rollback every file changed by the latest patch attempt."""
        started_at = perf_counter()

        if not self._attempts:
            return self._observation(
                ToolName.RESTORE_FILE,
                ObservationStatus.ERROR,
                "No active patch attempt is available for rollback.",
                started_at,
            )

        attempt = self._attempts[-1]

        try:
            self._restore_snapshot(attempt.snapshots)
        except OSError as exc:
            return self._observation(
                ToolName.RESTORE_FILE,
                ObservationStatus.ERROR,
                (
                    "Patch-attempt rollback failed without committing "
                    "a partial rollback."
                ),
                started_at,
                str(exc),
            )

        self._attempts.pop()
        self._recompute_changed_files()

        return self._observation(
            ToolName.RESTORE_FILE,
            ObservationStatus.OK,
            (
                f"Rolled back patch attempt {attempt.attempt_id} "
                f"across {len(attempt.snapshots)} file(s)."
            ),
            started_at,
            "\n".join(attempt.files),
        )

    def restore_all(self) -> ToolObservation:
        """Restore every file modified during this agent run."""
        started_at = perf_counter()
        restored = sorted(self._original_files)

        try:
            self._restore_snapshot(dict(self._original_files))
        except OSError as exc:
            return self._observation(
                ToolName.RESTORE_FILE,
                ObservationStatus.ERROR,
                "Full rollback failed without committing a partial rollback.",
                started_at,
                str(exc),
            )

        self._original_files.clear()
        self._changed_files.clear()
        self._attempts.clear()

        return self._observation(
            ToolName.RESTORE_FILE,
            ObservationStatus.OK,
            f"Restored {len(restored)} file(s).",
            started_at,
            "\n".join(restored),
        )
