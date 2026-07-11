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


def test_hidden_arithmetic_and_division() -> None:
    assert add(-20, 7) == -13
    assert subtract(0, 9) == -9
    assert multiply(-8, -6) == 48
    assert safe_divide(-7, 2) == -3.5
    with pytest.raises(ZeroDivisionError):
        safe_divide(5, 0)


def test_hidden_clamp_and_parity() -> None:
    assert clamp(4, 4, 4) == 4
    assert clamp(-20, -5, 10) == -5
    assert clamp(15, -5, 10) == 10
    assert is_even(-12) is True
    assert is_even(-11) is False
    with pytest.raises(ValueError):
        clamp(0, 2, 1)


def test_hidden_sequences() -> None:
    assert factorial(1) == 1
    assert factorial(6) == 720
    assert fibonacci(2) == 1
    assert fibonacci(9) == 34
    with pytest.raises(ValueError):
        fibonacci(-2)


def test_hidden_number_theory() -> None:
    assert gcd(0, 18) == 18
    assert gcd(-45, -30) == 15
    assert lcm(9, 12) == 36
    assert lcm(-4, 6) == 12


def test_hidden_text_helpers() -> None:
    assert count_vowels("QUEUE rhythm") == 4
    assert normalize_whitespace("\n alpha\t beta  gamma \n") == "alpha beta gamma"
    assert is_palindrome("No lemon, no melon") is True
    assert is_palindrome("mutation") is False


def test_hidden_collection_helpers() -> None:
    assert unique_preserve_order([0, 0, -1, 2, -1, 3]) == [0, -1, 2, 3]
    assert flatten_once([[], [9], [1, 2], []]) == [9, 1, 2]


def test_hidden_search_and_statistics() -> None:
    assert median([9, 1, 5, 3, 7]) == 5.0
    assert median([-4, 0, 8, 12]) == 4.0
    assert binary_search([], 3) == -1
    assert binary_search([2, 5, 8, 11, 14], 2) == 0
    assert binary_search([2, 5, 8, 11, 14], 14) == 4


def test_hidden_merge_parentheses_and_base() -> None:
    assert merge_sorted([], [1, 2]) == [1, 2]
    assert merge_sorted([1, 1, 4], [1, 3]) == [1, 1, 1, 3, 4]
    assert has_balanced_parentheses("plain text") is True
    assert has_balanced_parentheses("((x))(y)") is True
    assert has_balanced_parentheses("x)") is False
    assert to_base(255, 16) == "FF"
    assert to_base(8, 2) == "1000"
    assert to_base(35, 16) == "23"
    with pytest.raises(ValueError):
        to_base(4, 1)
    with pytest.raises(ValueError):
        to_base(4, 17)
