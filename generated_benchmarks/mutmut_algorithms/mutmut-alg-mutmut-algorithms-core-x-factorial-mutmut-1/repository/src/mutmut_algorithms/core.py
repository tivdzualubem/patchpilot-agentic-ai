"""Algorithm implementations used to seed mutmut repair tasks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

VOWELS = frozenset("aeiouAEIOU")


def add(left: int, right: int) -> int:
    """Return the sum of two integers."""
    return left + right


def subtract(left: int, right: int) -> int:
    """Return the difference of two integers."""
    return left - right


def multiply(left: int, right: int) -> int:
    """Return the product of two integers."""
    return left * right


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numbers and reject division by zero."""
    if denominator == 0:
        raise ZeroDivisionError("division by zero")
    return numerator / denominator


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp a value into an inclusive integer range."""
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    return min(max(value, minimum), maximum)


def is_even(value: int) -> bool:
    """Return whether an integer is even."""
    return value % 2 == 0


def factorial(value: int) -> int:
    """Return value factorial for a non-negative integer."""
    if value <= 0:
        raise ValueError("factorial is undefined for negative values")
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


def fibonacci(index: int) -> int:
    """Return the Fibonacci number at a zero-based index."""
    if index < 0:
        raise ValueError("fibonacci is undefined for negative indexes")
    previous = 0
    current = 1
    for _ in range(index):
        previous, current = current, previous + current
    return previous


def gcd(left: int, right: int) -> int:
    """Return the greatest common divisor of two integers."""
    left = abs(left)
    right = abs(right)
    while right:
        left, right = right, left % right
    return left


def lcm(left: int, right: int) -> int:
    """Return the least common multiple of two integers."""
    if left == 0 or right == 0:
        return 0
    return abs(left * right) // gcd(left, right)


def count_vowels(text: str) -> int:
    """Count vowels in a string."""
    return sum(1 for character in text if character in VOWELS)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(text.split())


def is_palindrome(text: str) -> bool:
    """Return whether alphanumeric text is a palindrome."""
    cleaned = "".join(character.lower() for character in text if character.isalnum())
    return cleaned == cleaned[::-1]


def unique_preserve_order(items: Iterable[int]) -> list[int]:
    """Return unique integers while preserving first-seen order."""
    seen: set[int] = set()
    result: list[int] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def flatten_once(nested: Iterable[Iterable[int]]) -> list[int]:
    """Flatten one level of nested integer iterables."""
    result: list[int] = []
    for group in nested:
        result.extend(group)
    return result


def median(values: Sequence[float]) -> float:
    """Return the median of a non-empty numeric sequence."""
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def binary_search(values: Sequence[int], target: int) -> int:
    """Return the index of target in sorted values, or -1."""
    low = 0
    high = len(values) - 1
    while low <= high:
        middle = (low + high) // 2
        candidate = values[middle]
        if candidate == target:
            return middle
        if candidate < target:
            low = middle + 1
        else:
            high = middle - 1
    return -1


def merge_sorted(left: Sequence[int], right: Sequence[int]) -> list[int]:
    """Merge two sorted integer sequences."""
    merged: list[int] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
    merged.extend(left[left_index:])
    merged.extend(right[right_index:])
    return merged


def has_balanced_parentheses(text: str) -> bool:
    """Return whether parentheses are balanced in text."""
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def to_base(number: int, base: int) -> str:
    """Convert a non-negative integer to a base-2..16 string."""
    if number < 0:
        raise ValueError("number must be non-negative")
    if base < 2 or base > 16:
        raise ValueError("base must be between 2 and 16")
    digits = "0123456789ABCDEF"
    if number == 0:
        return "0"
    result = ""
    while number:
        number, remainder = divmod(number, base)
        result = digits[remainder] + result
    return result
