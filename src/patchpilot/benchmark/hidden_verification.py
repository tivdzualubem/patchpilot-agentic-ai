"""Post-run hidden verification outside the agent-visible workspace."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from enum import StrEnum
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from patchpilot.benchmark.manifest import BenchmarkManifest


class HiddenVerificationError(ValueError):
    """Raised when hidden verification cannot be prepared safely."""


class HiddenVerificationStatus(StrEnum):
    """Outcome of a post-run hidden test suite."""

    NOT_CONFIGURED = "not_configured"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


class HiddenVerificationResult(BaseModel):
    """Non-sensitive hidden-suite evidence stored with one benchmark run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: HiddenVerificationStatus
    passed: bool | None = None
    test_count: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0)
    return_code: int | None = None
    output_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    error_type: str | None = Field(default=None, max_length=200)


class HiddenTestRunner:
    """Run hidden tests only after the agent loop has terminated."""

    _COUNT_PATTERN = re.compile(
        r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)"
    )

    def __init__(
        self,
        project_root: Path,
        output_root: Path,
        timeout_seconds: int = 60,
    ) -> None:
        if not 1 <= timeout_seconds <= 600:
            raise HiddenVerificationError("timeout_seconds must be between 1 and 600.")

        self.project_root = project_root.expanduser().resolve(strict=True)
        self.output_root = output_root.expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _contains_symlink(root: Path) -> bool:
        return root.is_symlink() or any(path.is_symlink() for path in root.rglob("*"))

    def _resolve_hidden_root(self, manifest: BenchmarkManifest) -> Path | None:
        if manifest.hidden_test_root is None:
            return None

        hidden_candidate = self.project_root / manifest.hidden_test_root
        if hidden_candidate.is_symlink():
            raise HiddenVerificationError("Hidden test roots cannot be symbolic links.")

        hidden_root = hidden_candidate.resolve(strict=True)
        repository_root = (self.project_root / manifest.repository_root).resolve(
            strict=True
        )

        if not hidden_root.is_relative_to(self.project_root):
            raise HiddenVerificationError("Hidden test root escapes the project root.")
        if not hidden_root.is_dir():
            raise HiddenVerificationError("Hidden test root must be a directory.")
        if hidden_root == repository_root or hidden_root.is_relative_to(
            repository_root
        ):
            raise HiddenVerificationError(
                "Hidden tests must remain outside the agent-visible repository."
            )
        if self._contains_symlink(hidden_root):
            raise HiddenVerificationError(
                "Hidden test directories cannot contain symbolic links."
            )

        return hidden_root

    @staticmethod
    def _test_count(output: bytes) -> int:
        text = output.decode("utf-8", errors="replace")
        return sum(
            int(match.group(1))
            for match in HiddenTestRunner._COUNT_PATTERN.finditer(text)
        )

    @staticmethod
    def _environment(repository_root: Path) -> dict[str, str]:
        python_paths = [repository_root]
        source_root = repository_root / "src"
        if source_root.is_dir():
            python_paths.append(source_root)

        return {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join(str(path) for path in python_paths),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

    def run(
        self,
        *,
        manifest: BenchmarkManifest,
        repaired_repository: Path,
        run_id: str,
    ) -> HiddenVerificationResult:
        """Judge the final repository without exposing hidden tests to the agent."""
        hidden_root = self._resolve_hidden_root(manifest)
        if hidden_root is None:
            return HiddenVerificationResult(
                status=HiddenVerificationStatus.NOT_CONFIGURED,
            )

        repaired = repaired_repository.expanduser().resolve(strict=True)
        if not repaired.is_dir():
            raise HiddenVerificationError(
                "The repaired repository must be a directory."
            )
        if self._contains_symlink(repaired):
            raise HiddenVerificationError(
                "The repaired repository cannot contain symbolic links."
            )

        started_at = perf_counter()
        judge_root = Path(
            tempfile.mkdtemp(
                prefix=f"{run_id[:40]}-",
                dir=self.output_root,
            )
        )

        try:
            judge_repository = judge_root / "repository"
            judge_tests = judge_root / "hidden_tests"
            shutil.copytree(repaired, judge_repository)
            shutil.copytree(hidden_root, judge_tests)

            command = [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                str(judge_tests),
            ]
            result = subprocess.run(
                command,
                cwd=judge_root,
                env=self._environment(judge_repository),
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
            output = (result.stdout or b"") + (result.stderr or b"")
            passed = result.returncode == 0
            return HiddenVerificationResult(
                status=(
                    HiddenVerificationStatus.PASSED
                    if passed
                    else HiddenVerificationStatus.FAILED
                ),
                passed=passed,
                test_count=self._test_count(output),
                duration_seconds=perf_counter() - started_at,
                return_code=result.returncode,
                output_sha256=hashlib.sha256(output).hexdigest(),
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or b"") + (exc.stderr or b"")
            return HiddenVerificationResult(
                status=HiddenVerificationStatus.TIMEOUT,
                passed=False,
                test_count=self._test_count(output),
                duration_seconds=perf_counter() - started_at,
                output_sha256=hashlib.sha256(output).hexdigest(),
                error_type=type(exc).__name__,
            )
        except OSError as exc:
            return HiddenVerificationResult(
                status=HiddenVerificationStatus.ERROR,
                passed=False,
                duration_seconds=perf_counter() - started_at,
                error_type=type(exc).__name__,
            )
        finally:
            shutil.rmtree(judge_root, ignore_errors=False)
