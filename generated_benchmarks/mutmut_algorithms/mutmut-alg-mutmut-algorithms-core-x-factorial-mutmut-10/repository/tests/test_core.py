import pytest
from mutmut_algorithms.core import (
    add,
    binary_search,
    clamp,
    count_vowels,
    factorial,
    fibonacci,
    flatten_once,
    gcd,
    has_balanced_parentheses,
    is_even,
    is_palindrome,
    lcm,
    median,
    merge_sorted,
    multiply,
    normalize_whitespace,
    safe_divide,
    subtract,
    to_base,
    unique_preserve_order,
)


def test_basic_arithmetic() -> None:
    assert add(2, 3) == 5
    assert subtract(10, 4) == 6
    assert multiply(-3, 7) == -21
    assert safe_divide(9, 2) == 4.5
    with pytest.raises(ZeroDivisionError):
        safe_divide(1, 0)


def test_clamp_and_even() -> None:
    assert clamp(5, 1, 10) == 5
    assert clamp(-3, 0, 10) == 0
    assert clamp(99, 0, 10) == 10
    assert is_even(8) is True
    assert is_even(9) is False
    with pytest.raises(ValueError):
        clamp(3, 10, 1)


def test_factorial_and_fibonacci() -> None:
    assert factorial(0) == 1
    assert factorial(5) == 120
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(7) == 13
    with pytest.raises(ValueError):
        factorial(-1)


def test_number_theory() -> None:
    assert gcd(54, 24) == 6
    assert gcd(-21, 14) == 7
    assert lcm(6, 8) == 24
    assert lcm(0, 8) == 0


def test_text_helpers() -> None:
    assert count_vowels("Agentic AI") == 5
    assert normalize_whitespace("  patch\npilot\t works  ") == "patch pilot works"
    assert is_palindrome("A man, a plan, a canal: Panama") is True
    assert is_palindrome("debugging") is False


def test_list_helpers() -> None:
    assert unique_preserve_order([3, 1, 3, 2, 1, 4]) == [3, 1, 2, 4]
    assert flatten_once([[1, 2], [], [3], [4, 5]]) == [1, 2, 3, 4, 5]


def test_median_and_binary_search() -> None:
    assert median([3, 1, 2]) == 2.0
    assert median([10, 2, 8, 4]) == 6.0
    assert binary_search([1, 4, 6, 9], 6) == 2
    assert binary_search([1, 4, 6, 9], 7) == -1
    with pytest.raises(ValueError):
        median([])


def test_merge_parentheses_and_base() -> None:
    assert merge_sorted([1, 4, 9], [2, 3, 10]) == [1, 2, 3, 4, 9, 10]
    assert has_balanced_parentheses("(a + b) * (c)") is True
    assert has_balanced_parentheses(")(") is False
    assert has_balanced_parentheses("(()") is False
    assert to_base(31, 16) == "1F"
    assert to_base(10, 2) == "1010"
    with pytest.raises(ValueError):
        to_base(-1, 10)
