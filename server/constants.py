"""Loads medical reference data from data/reference/ and provides lookup functions."""

import json
from pathlib import Path

_data_dir: Path | None = None
_lab_ranges: dict | None = None
_diagnostic_criteria: dict | None = None
_drug_info: dict | None = None


def _find_reference_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "data" / "reference",
        here / "data" / "reference",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"data/reference/ not found. Checked: {candidates}")


def _load():
    global _lab_ranges, _diagnostic_criteria, _drug_info, _data_dir
    if _lab_ranges is not None:
        return
    _data_dir = _find_reference_dir()
    with open(_data_dir / "lab_ranges.json", encoding="utf-8") as f:
        _lab_ranges = json.load(f)
    with open(_data_dir / "diagnostic_criteria.json", encoding="utf-8") as f:
        _diagnostic_criteria = json.load(f)
    with open(_data_dir / "drug_info.json", encoding="utf-8") as f:
        _drug_info = json.load(f)


def _fuzzy_get(data: dict, key: str) -> tuple[str, any] | None:
    k = key.strip().lower()
    if k in data:
        return k, data[k]
    for dk, dv in data.items():
        if k in dk or dk in k:
            return dk, dv
    return None


def lookup_range(test_name: str) -> str | None:
    _load()
    match = _fuzzy_get(_lab_ranges, test_name)
    if match is None:
        return None
    key, entry = match
    result = f"{key.upper()}: Normal range {entry['low']}-{entry['high']} {entry['unit']}".strip()
    if entry.get("context"):
        result += f"\n  {entry['context']}"
    return result


def lookup_criteria(condition: str) -> str | None:
    _load()
    match = _fuzzy_get(_diagnostic_criteria, condition)
    if match is None:
        return None
    return match[1]


def lookup_drug(drug_name: str) -> str | None:
    _load()
    match = _fuzzy_get(_drug_info, drug_name)
    if match is None:
        return None
    return match[1]


def interpret_value(test_name: str, value_str: str) -> str | None:
    _load()
    match = _fuzzy_get(_lab_ranges, test_name)
    if match is None:
        return None

    try:
        value = float(value_str)
    except (ValueError, TypeError):
        return f"Cannot parse '{value_str}' as a numeric value."

    key, entry = match
    low, high, unit = entry["low"], entry["high"], entry["unit"]

    if value < low:
        status = "LOW"
        severity = "critically low" if value < low * 0.7 else "below normal"
    elif value > high:
        status = "HIGH"
        severity = "critically elevated" if value > high * 1.5 else "above normal"
    else:
        status = "NORMAL"
        severity = "within normal range"

    result = f"{key.upper()} {value} {unit}: {status} — {severity} (normal {low}-{high})"
    if entry.get("context") and status != "NORMAL":
        result += f"\n  {entry['context']}"
    return result
