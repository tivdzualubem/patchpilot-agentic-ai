"""Text and small-data transformations for mutation repair tasks."""

from __future__ import annotations

import re
from collections.abc import Iterable

_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


def normalize_spaces(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(text.split())


def slugify(text: str) -> str:
    """Create a lowercase ASCII-style hyphenated slug."""
    lowered = text.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", lowered)
    return normalized.strip("-")


def truncate_middle(text: str, limit: int) -> str:
    """Truncate text in the middle while preserving both ends."""
    if limit < 3:
        raise ValueError("limit must be at least 3")
    if len(text) <= limit:
        return text
    remaining = limit - 3
    left = (remaining + 1) // 2
    right = remaining // 2
    return text[:left] + "..." + text[len(text) - right :]


def mask_email(address: str) -> str:
    """Mask an email local part while retaining its first character."""
    if address.count("@") != 1:
        raise ValueError("address must contain one @")
    local, domain = address.split("@")
    if not local or not domain:
        raise ValueError("address parts cannot be empty")
    return local[0] + "*" * max(1, len(local) - 1) + "@" + domain


def word_frequencies(text: str) -> dict[str, int]:
    """Count case-insensitive word occurrences."""
    counts: dict[str, int] = {}
    for match in _WORD_PATTERN.findall(text.lower()):
        counts[match] = counts.get(match, 0) + 1
    return counts


def longest_common_prefix(values: Iterable[str]) -> str:
    """Return the longest prefix shared by every value."""
    items = list(values)
    if not items:
        return ""
    prefix = items[0]
    for item in items[1:]:
        while not item.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def parse_csv_line(line: str) -> list[str]:
    """Parse a simple comma-separated line with whitespace trimming."""
    if not line:
        return []
    return [part.strip() for part in line.split(",")]


def format_initials(name: str) -> str:
    """Return uppercase initials for non-empty name parts."""
    parts = [part for part in name.split() if part]
    return ".".join(part[0].upper() for part in parts) + ("." if parts else "")


def count_substring(text: str, needle: str) -> int:
    """Count non-overlapping occurrences of a non-empty substring."""
    if not needle:
        raise ValueError("needle cannot be empty")
    return text.count(needle)


def remove_duplicate_words(text: str) -> str:
    """Remove repeated words case-insensitively while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for word in text.split():
        key = word.casefold()
        if key in seen:
            seen.add(key)
            result.append(word)
    return " ".join(result)


def split_sentences(text: str) -> list[str]:
    """Split text on sentence punctuation and discard empty pieces."""
    return [piece.strip() for piece in re.split(r"[.!?]+", text) if piece.strip()]


def wrap_words(text: str, width: int) -> list[str]:
    """Greedily wrap words without exceeding a positive width."""
    if width <= 0:
        raise ValueError("width must be positive")
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else current + " " + word
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def parse_key_values(text: str) -> dict[str, str]:
    """Parse newline-separated key=value pairs."""
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError("each line must contain =")
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        if not key:
            raise ValueError("keys cannot be empty")
        result[key] = value.strip()
    return result


def format_bytes(size: int) -> str:
    """Format a non-negative byte count using binary units."""
    if size < 0:
        raise ValueError("size cannot be negative")
    units = ("B", "KiB", "MiB", "GiB")
    value = float(size)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024 or candidate == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def normalize_newlines(text: str) -> str:
    """Convert CRLF and CR line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def redact_terms(text: str, terms: Iterable[str]) -> str:
    """Replace case-insensitive literal terms with asterisks."""
    result = text
    for term in terms:
        if not term:
            continue
        pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
        result = pattern.sub("*" * len(term), result)
    return result


def camel_to_snake(name: str) -> str:
    """Convert simple CamelCase identifiers to snake_case."""
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    second = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first)
    normalized = second.replace("-", "_").lower()
    return re.sub(r"_+", "_", normalized)


def strip_prefixes(text: str, prefixes: Iterable[str]) -> str:
    """Remove the first matching prefix in iteration order."""
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text
