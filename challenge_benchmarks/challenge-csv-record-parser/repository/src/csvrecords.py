"""A small RFC-4180-style single-record CSV parser."""

from __future__ import annotations


def parse_record(record: str) -> list[str]:
    """Parse one CSV record with commas and doubled-quote escapes."""
    fields: list[str] = []
    for part in record.split(","):
        value = part.strip()
        if value.startswith('"'):
            value = value[1:]
        if value.endswith('"'):
            value = value[:-1]
        fields.append(value.replace('""', '"'))
    return fields
