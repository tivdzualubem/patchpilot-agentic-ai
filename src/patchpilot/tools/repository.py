"""Sandboxed read-only repository tools for PatchPilot."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

from patchpilot.schemas import (
    ObservationStatus,
    RepairTask,
    ToolName,
    ToolObservation,
)

_BLOCKED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

_BLOCKED_SUFFIXES = {
    ".key",
    ".pem",
    ".p12",
    ".pfx",
}

_MAX_READ_BYTES = 256_000
_MAX_READ_LINES = 400
_MAX_LISTED_FILES = 500
_MAX_SEARCH_HITS = 100


class RepositoryAccessError(ValueError):
    """Raised when a repository operation violates sandbox policy."""


class RepositorySandbox:
    """Provide bounded read-only access to one repair repository."""

    def __init__(
        self,
        workspace_root: Path,
        task: RepairTask,
    ) -> None:
        self.workspace_root = workspace_root.expanduser().resolve(
            strict=True
        )

        repository_root = (
            self.workspace_root / task.repository_root
        ).resolve(strict=True)

        if not repository_root.is_dir():
            raise RepositoryAccessError(
                "The configured repository root is not a directory."
            )

        if not repository_root.is_relative_to(self.workspace_root):
            raise RepositoryAccessError(
                "The repository must remain inside the workspace."
            )

        self.repository_root = repository_root

    @staticmethod
    def _is_blocked(relative_path: Path) -> bool:
        """Return whether a relative path contains sensitive content."""
        for part in relative_path.parts:
            if part in _BLOCKED_DIRECTORIES:
                return True
            if part == ".env" or part.startswith(".env."):
                return True

        return relative_path.suffix.lower() in _BLOCKED_SUFFIXES

    def _resolve_existing(self, raw_path: str) -> Path:
        """Resolve an existing path while preventing sandbox escape."""
        requested = Path(raw_path)

        if requested.is_absolute() or ".." in requested.parts:
            raise RepositoryAccessError(
                "Paths must be relative and cannot contain '..'."
            )

        if self._is_blocked(requested):
            raise RepositoryAccessError(
                "Access to this path is blocked by repository policy."
            )

        candidate = self.repository_root / requested

        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RepositoryAccessError(
                f"Path does not exist: {raw_path}"
            ) from exc

        if not resolved.is_relative_to(self.repository_root):
            raise RepositoryAccessError(
                "Resolved path escapes the repository sandbox."
            )

        resolved_relative = resolved.relative_to(self.repository_root)

        if self._is_blocked(resolved_relative):
            raise RepositoryAccessError(
                "Access to this path is blocked by repository policy."
            )

        return resolved

    def _iter_safe_files(self, root: Path) -> Iterable[Path]:
        """Yield regular files that remain inside the sandbox."""
        candidates = [root] if root.is_file() else root.rglob("*")

        for candidate in candidates:
            lexical_relative = candidate.relative_to(
                self.repository_root
            )

            if self._is_blocked(lexical_relative):
                continue

            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue

            if not resolved.is_relative_to(self.repository_root):
                continue

            if resolved.is_file():
                yield candidate

    @staticmethod
    def _read_text(path: Path) -> str:
        """Read a bounded UTF-8 text file."""
        size = path.stat().st_size

        if size > _MAX_READ_BYTES:
            raise RepositoryAccessError(
                f"File exceeds the {_MAX_READ_BYTES}-byte read limit."
            )

        raw = path.read_bytes()

        if b"\x00" in raw:
            raise RepositoryAccessError(
                "Binary files cannot be read by the agent."
            )

        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryAccessError(
                "Only UTF-8 text files can be read."
            ) from exc

    @staticmethod
    def _rejected(
        tool: ToolName,
        message: str,
        started_at: float,
    ) -> ToolObservation:
        """Create a policy-rejection observation."""
        return ToolObservation(
            tool=tool,
            status=ObservationStatus.REJECTED,
            summary=message,
            duration_seconds=perf_counter() - started_at,
        )

    def list_files(
        self,
        relative_path: str = ".",
    ) -> ToolObservation:
        """List bounded repository files below a relative path."""
        started_at = perf_counter()

        try:
            root = self._resolve_existing(relative_path)
            files = sorted(
                path.relative_to(self.repository_root).as_posix()
                for path in self._iter_safe_files(root)
            )

            truncated = len(files) > _MAX_LISTED_FILES
            visible_files = files[:_MAX_LISTED_FILES]
            suffix = " Results were truncated." if truncated else ""

            return ToolObservation(
                tool=ToolName.LIST_FILES,
                status=ObservationStatus.OK,
                summary=(
                    f"Listed {len(visible_files)} repository file(s)."
                    f"{suffix}"
                ),
                output="\n".join(visible_files),
                duration_seconds=perf_counter() - started_at,
            )
        except RepositoryAccessError as exc:
            return self._rejected(
                ToolName.LIST_FILES,
                str(exc),
                started_at,
            )

    def read_file(
        self,
        relative_path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> ToolObservation:
        """Read a bounded, numbered range from a UTF-8 file."""
        started_at = perf_counter()

        try:
            if start_line < 1:
                raise RepositoryAccessError(
                    "start_line must be at least 1."
                )

            requested_end = (
                start_line + 199 if end_line is None else end_line
            )

            if requested_end < start_line:
                raise RepositoryAccessError(
                    "end_line cannot be less than start_line."
                )

            if requested_end - start_line + 1 > _MAX_READ_LINES:
                raise RepositoryAccessError(
                    f"A maximum of {_MAX_READ_LINES} lines may be read."
                )

            path = self._resolve_existing(relative_path)

            if not path.is_file():
                raise RepositoryAccessError(
                    "read_file requires a regular file."
                )

            lines = self._read_text(path).splitlines()
            selected = lines[start_line - 1 : requested_end]
            numbered = [
                f"{number}: {line}"
                for number, line in enumerate(
                    selected,
                    start=start_line,
                )
            ]

            return ToolObservation(
                tool=ToolName.READ_FILE,
                status=ObservationStatus.OK,
                summary=(
                    f"Read {len(selected)} line(s) from "
                    f"{relative_path}."
                ),
                output="\n".join(numbered),
                duration_seconds=perf_counter() - started_at,
            )
        except (OSError, RepositoryAccessError) as exc:
            return self._rejected(
                ToolName.READ_FILE,
                str(exc),
                started_at,
            )

    def search_code(
        self,
        query: str,
        relative_path: str = ".",
        max_hits: int = 50,
    ) -> ToolObservation:
        """Search text files using a bounded case-insensitive literal query."""
        started_at = perf_counter()

        try:
            normalized_query = query.strip()

            if not normalized_query:
                raise RepositoryAccessError(
                    "The search query cannot be empty."
                )

            if len(normalized_query) > 200:
                raise RepositoryAccessError(
                    "The search query cannot exceed 200 characters."
                )

            if not 1 <= max_hits <= _MAX_SEARCH_HITS:
                raise RepositoryAccessError(
                    f"max_hits must be between 1 and "
                    f"{_MAX_SEARCH_HITS}."
                )

            root = self._resolve_existing(relative_path)
            lowered_query = normalized_query.casefold()
            hits: list[str] = []

            for path in self._iter_safe_files(root):
                try:
                    lines = self._read_text(path).splitlines()
                except (OSError, RepositoryAccessError):
                    continue

                relative = path.relative_to(
                    self.repository_root
                ).as_posix()

                for line_number, line in enumerate(lines, start=1):
                    if lowered_query in line.casefold():
                        preview = line.strip()[:300]
                        hits.append(
                            f"{relative}:{line_number}: {preview}"
                        )

                        if len(hits) >= max_hits:
                            break

                if len(hits) >= max_hits:
                    break

            return ToolObservation(
                tool=ToolName.SEARCH_CODE,
                status=ObservationStatus.OK,
                summary=(
                    f"Found {len(hits)} match(es) for "
                    f"{normalized_query!r}."
                ),
                output="\n".join(hits),
                duration_seconds=perf_counter() - started_at,
            )
        except RepositoryAccessError as exc:
            return self._rejected(
                ToolName.SEARCH_CODE,
                str(exc),
                started_at,
            )
