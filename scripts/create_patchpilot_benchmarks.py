from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()
BENCHMARKS = ROOT / "benchmarks"
TESTS = ROOT / "tests"

TASKS = [
    {
        "task_id": "calculator-002",
        "title": "Incorrect subtraction operator",
        "goal": "Repair the subtract function so that all regression tests pass.",
        "defect_category": "wrong_binary_operator",
        "difficulty": "easy",
        "module": "calculator.py",
        "source": '''\
            """Simple arithmetic operations."""


            def subtract(left: int, right: int) -> int:
                """Return the difference between two integers."""
                return left + right
        ''',
        "test": """\
            from src.calculator import subtract


            def test_subtract_positive_integers() -> None:
                assert subtract(9, 4) == 5


            def test_subtract_negative_integers() -> None:
                assert subtract(-4, -3) == -1


            def test_subtract_zero() -> None:
                assert subtract(9, 0) == 9
        """,
        "expected_initial_failures": 2,
    },
    {
        "task_id": "calculator-003",
        "title": "Incorrect multiplication operator",
        "goal": "Repair the multiply function so that all regression tests pass.",
        "defect_category": "wrong_binary_operator",
        "difficulty": "easy",
        "module": "calculator.py",
        "source": '''\
            """Simple arithmetic operations."""


            def multiply(left: int, right: int) -> int:
                """Return the product of two integers."""
                return left + right
        ''',
        "test": """\
            from src.calculator import multiply


            def test_multiply_positive_integers() -> None:
                assert multiply(2, 3) == 6


            def test_multiply_negative_integer() -> None:
                assert multiply(-4, 3) == -12


            def test_multiply_by_zero() -> None:
                assert multiply(9, 0) == 0
        """,
        "expected_initial_failures": 3,
    },
    {
        "task_id": "calculator-004",
        "title": "Negation returns unchanged value",
        "goal": "Repair the negate function so that all regression tests pass.",
        "defect_category": "wrong_unary_operator",
        "difficulty": "easy",
        "module": "calculator.py",
        "source": '''\
            """Simple arithmetic operations."""


            def negate(value: int) -> int:
                """Return the arithmetic negation of an integer."""
                return value
        ''',
        "test": """\
            from src.calculator import negate


            def test_negate_positive_integer() -> None:
                assert negate(5) == -5


            def test_negate_negative_integer() -> None:
                assert negate(-4) == 4


            def test_negate_zero() -> None:
                assert negate(0) == 0
        """,
        "expected_initial_failures": 2,
    },
    {
        "task_id": "strings-001",
        "title": "Reverse text returns original string",
        "goal": "Repair the reverse_text function so that all regression tests pass.",
        "defect_category": "incorrect_string_operation",
        "difficulty": "easy",
        "module": "strings.py",
        "source": '''\
            """String helper functions."""


            def reverse_text(text: str) -> str:
                """Return the input text in reverse order."""
                return text
        ''',
        "test": """\
            from src.strings import reverse_text


            def test_reverse_regular_word() -> None:
                assert reverse_text("patch") == "hctap"


            def test_reverse_palindrome() -> None:
                assert reverse_text("level") == "level"


            def test_reverse_with_punctuation() -> None:
                assert reverse_text("AI!") == "!IA"
        """,
        "expected_initial_failures": 2,
    },
    {
        "task_id": "strings-002",
        "title": "Lowercase conversion uses uppercase",
        "goal": "Repair the lowercase function so that all regression tests pass.",
        "defect_category": "incorrect_string_operation",
        "difficulty": "easy",
        "module": "strings.py",
        "source": '''\
            """String helper functions."""


            def lowercase(text: str) -> str:
                """Return the input text converted to lowercase."""
                return text.upper()
        ''',
        "test": """\
            from src.strings import lowercase


            def test_lowercase_mixed_case() -> None:
                assert lowercase("PatchPilot") == "patchpilot"


            def test_lowercase_all_caps() -> None:
                assert lowercase("AI") == "ai"


            def test_lowercase_empty_string() -> None:
                assert lowercase("") == ""
        """,
        "expected_initial_failures": 2,
    },
    {
        "task_id": "lists-001",
        "title": "First item returns last item",
        "goal": "Repair the first_item function so that all regression tests pass.",
        "defect_category": "wrong_index",
        "difficulty": "easy",
        "module": "lists.py",
        "source": '''\
            """List helper functions."""


            def first_item(values: list[int]) -> int:
                """Return the first item in a non-empty list."""
                return values[-1]
        ''',
        "test": """\
            from src.lists import first_item


            def test_first_item_multiple_values() -> None:
                assert first_item([1, 2, 3]) == 1


            def test_first_item_two_values() -> None:
                assert first_item([8, 9]) == 8


            def test_first_item_single_value() -> None:
                assert first_item([42]) == 42
        """,
        "expected_initial_failures": 2,
    },
    {
        "task_id": "lists-002",
        "title": "Total returns list length",
        "goal": "Repair the total function so that all regression tests pass.",
        "defect_category": "wrong_builtin",
        "difficulty": "easy",
        "module": "lists.py",
        "source": '''\
            """List helper functions."""


            def total(values: list[int]) -> int:
                """Return the arithmetic sum of the list values."""
                return len(values)
        ''',
        "test": """\
            from src.lists import total


            def test_total_positive_values() -> None:
                assert total([1, 2, 3]) == 6


            def test_total_empty_list() -> None:
                assert total([]) == 0


            def test_total_mixed_values() -> None:
                assert total([-1, 5]) == 4
        """,
        "expected_initial_failures": 2,
    },
    {
        "task_id": "stats-001",
        "title": "Mean divides by incorrect count",
        "goal": "Repair the mean function so that all regression tests pass.",
        "defect_category": "off_by_one_denominator",
        "difficulty": "medium",
        "module": "stats.py",
        "source": '''\
            """Small statistics helper functions."""


            def mean(values: list[float]) -> float:
                """Return the arithmetic mean of a non-empty list."""
                return sum(values) / (len(values) + 1)
        ''',
        "test": """\
            from src.stats import mean


            def test_mean_multiple_values() -> None:
                assert mean([2.0, 4.0, 6.0]) == 4.0


            def test_mean_single_value() -> None:
                assert mean([10.0]) == 10.0


            def test_mean_symmetric_values() -> None:
                assert mean([-2.0, 2.0]) == 0.0
        """,
        "expected_initial_failures": 2,
    },
    {
        "task_id": "geometry-001",
        "title": "Rectangle area uses perimeter-style addition",
        "goal": "Repair the rectangle_area function so that all regression tests pass.",
        "defect_category": "wrong_binary_operator",
        "difficulty": "easy",
        "module": "geometry.py",
        "source": '''\
            """Geometry helper functions."""


            def rectangle_area(width: float, height: float) -> float:
                """Return the area of a rectangle."""
                return width + height
        ''',
        "test": """\
            from src.geometry import rectangle_area


            def test_rectangle_area_positive_dimensions() -> None:
                assert rectangle_area(3.0, 4.0) == 12.0


            def test_rectangle_area_zero_height() -> None:
                assert rectangle_area(5.0, 0.0) == 0.0


            def test_rectangle_area_float_width() -> None:
                assert rectangle_area(2.5, 4.0) == 10.0
        """,
        "expected_initial_failures": 3,
    },
    {
        "task_id": "geometry-002",
        "title": "Rectangle perimeter uses area multiplication",
        "goal": (
            "Repair the rectangle_perimeter function so that all regression tests pass."
        ),
        "defect_category": "wrong_formula",
        "difficulty": "medium",
        "module": "geometry.py",
        "source": '''\
            """Geometry helper functions."""


            def rectangle_perimeter(width: float, height: float) -> float:
                """Return the perimeter of a rectangle."""
                return width * height
        ''',
        "test": """\
            from src.geometry import rectangle_perimeter


            def test_rectangle_perimeter_positive_dimensions() -> None:
                assert rectangle_perimeter(3.0, 4.0) == 14.0


            def test_rectangle_perimeter_zero_height() -> None:
                assert rectangle_perimeter(5.0, 0.0) == 10.0


            def test_rectangle_perimeter_float_width() -> None:
                assert rectangle_perimeter(2.5, 4.0) == 13.0
        """,
        "expected_initial_failures": 3,
    },
    {
        "task_id": "numbers-001",
        "title": "Even-number predicate is inverted",
        "goal": "Repair the is_even function so that all regression tests pass.",
        "defect_category": "wrong_comparison",
        "difficulty": "easy",
        "module": "numbers.py",
        "source": '''\
            """Number predicate helper functions."""


            def is_even(value: int) -> bool:
                """Return True when the integer is even."""
                return value % 2 == 1
        ''',
        "test": """\
            from src.numbers import is_even


            def test_positive_even_number() -> None:
                assert is_even(8) is True


            def test_positive_odd_number() -> None:
                assert is_even(7) is False


            def test_zero_is_even() -> None:
                assert is_even(0) is True
        """,
        "expected_initial_failures": 3,
    },
]

HIDDEN_TESTS = {
    "calculator-001": """\
        from src.calculator import add


        def test_add_large_mixed_sign_values() -> None:
            assert add(100, -40) == 60


        def test_add_two_negative_values() -> None:
            assert add(-25, -17) == -42
    """,
    "calculator-002": """\
        from src.calculator import subtract


        def test_subtract_from_zero() -> None:
            assert subtract(0, 7) == -7


        def test_subtract_larger_negative_value() -> None:
            assert subtract(-10, 6) == -16
    """,
    "calculator-003": """\
        from src.calculator import multiply


        def test_multiply_larger_values() -> None:
            assert multiply(7, 8) == 56


        def test_multiply_two_negative_values() -> None:
            assert multiply(-6, -9) == 54
    """,
    "calculator-004": """\
        from src.calculator import negate


        def test_negate_large_positive_value() -> None:
            assert negate(123) == -123


        def test_negate_large_negative_value() -> None:
            assert negate(-91) == 91
    """,
    "geometry-001": """\
        from src.geometry import rectangle_area


        def test_rectangle_area_fractional_dimensions() -> None:
            assert rectangle_area(1.25, 8.0) == 10.0


        def test_rectangle_area_square() -> None:
            assert rectangle_area(6.0, 6.0) == 36.0
    """,
    "geometry-002": """\
        from src.geometry import rectangle_perimeter


        def test_rectangle_perimeter_fractional_dimensions() -> None:
            assert rectangle_perimeter(1.5, 2.5) == 8.0


        def test_rectangle_perimeter_square() -> None:
            assert rectangle_perimeter(6.0, 6.0) == 24.0
    """,
    "lists-001": """\
        from src.lists import first_item


        def test_first_item_when_first_is_zero() -> None:
            assert first_item([0, 5, 9]) == 0


        def test_first_item_with_negative_values() -> None:
            assert first_item([-8, -3, -1]) == -8
    """,
    "lists-002": """\
        from src.lists import total


        def test_total_larger_mixed_values() -> None:
            assert total([10, -5, 2, 8]) == 15


        def test_total_repeated_values() -> None:
            assert total([4, 4, 4, 4]) == 16
    """,
    "numbers-001": """\
        from src.numbers import is_even


        def test_negative_even_number() -> None:
            assert is_even(-4) is True


        def test_negative_odd_number() -> None:
            assert is_even(-3) is False
    """,
    "stats-001": """\
        from src.stats import mean


        def test_mean_fractional_values() -> None:
            assert mean([1.5, 2.5]) == 2.0


        def test_mean_larger_collection() -> None:
            assert mean([1.0, 3.0, 5.0, 7.0]) == 4.0
    """,
    "strings-001": """\
        from src.strings import reverse_text


        def test_reverse_longer_word() -> None:
            assert reverse_text("agentic") == "citnega"


        def test_reverse_spaces() -> None:
            assert reverse_text("a b c") == "c b a"
    """,
    "strings-002": """\
        from src.strings import lowercase


        def test_lowercase_with_digits_and_punctuation() -> None:
            assert lowercase("AI-2026!") == "ai-2026!"


        def test_lowercase_preserves_spaces() -> None:
            assert lowercase("  Mixed Case  ") == "  mixed case  "
    """,
}

CATALOG_TEST = """\
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from patchpilot.benchmark import load_manifest

MANIFESTS = sorted(Path("benchmarks").glob("*/task.json"))


def _failure_count(output: str) -> int:
    match = re.search(r"(\\d+) failed", output)
    if match is None:
        return 0
    return int(match.group(1))


@pytest.mark.parametrize("manifest_path", MANIFESTS, ids=lambda p: p.parent.name)
def test_benchmark_manifest_and_initial_failure_count(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    repository = Path(manifest.repository_root)

    assert repository.is_dir()
    assert (repository / "src").is_dir()
    assert (repository / "tests").is_dir()
    assert manifest.allowed_paths == ["src"]
    assert manifest.forbidden_paths == ["tests"]

    result = subprocess.run(
        manifest.test_command,
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert _failure_count(output) == manifest.expected_initial_failures
"""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def write_task(task: dict[str, object]) -> None:
    task_id = str(task["task_id"])
    root = BENCHMARKS / task_id / "repository"
    write_text(root / "src" / "__init__.py", '"""Benchmark package."""')
    write_text(root / "src" / str(task["module"]), str(task["source"]))
    module_stem = Path(str(task["module"])).stem
    write_text(root / "tests" / f"test_{module_stem}.py", str(task["test"]))
    hidden_test = HIDDEN_TESTS[task_id]
    write_text(
        BENCHMARKS / task_id / "hidden_tests" / f"test_hidden_{module_stem}.py",
        hidden_test,
    )

    manifest = {
        "task_id": task_id,
        "title": task["title"],
        "goal": task["goal"],
        "repository_root": f"benchmarks/{task_id}/repository",
        "defect_category": task["defect_category"],
        "difficulty": task["difficulty"],
        "allowed_paths": ["src"],
        "forbidden_paths": ["tests"],
        "test_command": ["python", "-m", "pytest", "-q"],
        "expected_initial_failures": task["expected_initial_failures"],
        "hidden_test_root": f"benchmarks/{task_id}/hidden_tests",
        "expected_hidden_tests": 2,
    }
    write_text(
        BENCHMARKS / task_id / "task.json",
        json.dumps(manifest, indent=2),
    )


def main() -> None:
    for task in TASKS:
        write_task(task)
    write_text(TESTS / "test_benchmark_catalog.py", CATALOG_TEST)
    print(f"Wrote {len(TASKS)} new benchmark tasks plus catalog test.")


if __name__ == "__main__":
    main()
