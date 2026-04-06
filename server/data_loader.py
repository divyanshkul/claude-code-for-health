"""Lazy-loading data access for all three clinical datasets."""

import csv
import json
import os
from pathlib import Path


def _find_data_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "data",
        here / "data",
        Path(os.getcwd()) / "data",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"data/ directory not found. Checked: {candidates}")


class DataLoader:
    def __init__(self):
        self._diagnosis_cases: list[dict] | None = None
        self._calculation_cases: list[dict] | None = None
        self._note_cases: list[dict] | None = None
        self._loaded = False

    def load_all(self) -> None:
        if self._loaded:
            return
        data_dir = _find_data_dir()
        self._load_diagnosis(data_dir / "MedCaseReasoning")
        self._load_calculations(data_dir / "MedCalcBench")
        self._load_notes(data_dir / "MEDEC")
        self._loaded = True

    def _load_diagnosis(self, path: Path) -> None:
        jsonl_path = path / "extracted_cases.jsonl"
        cases = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        self._diagnosis_cases = cases

    def _load_calculations(self, path: Path) -> None:
        cases = []
        for filename in ["train_data.csv", "test_data.csv"]:
            filepath = path / filename
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Ground Truth Answer") and row["Ground Truth Answer"] != "None":
                        cases.append(row)
        self._calculation_cases = cases

    def _load_notes(self, path: Path) -> None:
        cases = []
        filenames = [
            "MEDEC-Full-TrainingSet-with-ErrorType.csv",
            "MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv",
            "MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv",
        ]
        for filename in filenames:
            filepath = path / filename
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    flag = row.get("Error Flag", "0") or "0"
                    try:
                        row["Error Flag"] = int(float(flag))
                    except (ValueError, TypeError):
                        row["Error Flag"] = 0
                    cases.append(row)
        self._note_cases = cases

    def get_diagnosis_cases(self) -> list[dict]:
        self.load_all()
        return self._diagnosis_cases or []

    def get_calculation_cases(self) -> list[dict]:
        self.load_all()
        return self._calculation_cases or []

    def get_note_cases(self) -> list[dict]:
        self.load_all()
        return self._note_cases or []
