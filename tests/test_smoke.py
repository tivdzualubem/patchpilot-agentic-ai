from typer.testing import CliRunner

from patchpilot import __version__
from patchpilot.cli import app


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_cli_reports_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
