"""Difficulty tier definitions and case selection logic."""

from random import Random

SIMPLE_CALCULATORS = {
    "bmi", "body mass index",
    "anion gap",
    "mean arterial pressure", "map",
    "ideal body weight", "ibw",
    "body surface area", "bsa",
    "corrected sodium",
    "corrected calcium",
    "free water deficit",
}

COMPLEX_CALCULATORS = {
    "apache ii", "apache",
    "wells", "wells criteria",
    "cha2ds2-vasc", "cha2ds2",
    "curb-65", "curb",
    "gcs", "glasgow coma scale",
    "meld", "meld score",
    "child-pugh", "child pugh",
    "sofa", "sofa score",
    "ranson", "ranson criteria",
}

SUBTLE_ERROR_TYPES = {"pharmacotherapy", "causalorganism", "causal organism"}


def select_case(task_type: str, difficulty: str, cases: list[dict], rng: Random) -> dict:
    filtered = _filter_by_difficulty(task_type, difficulty, cases)
    if not filtered:
        filtered = cases
    return rng.choice(filtered)


TASK_TYPES = ["diagnosis", "calculation", "note_review"]


def get_default_task_type(difficulty: str, rng: Random | None = None) -> str:
    if rng is None:
        rng = Random()
    return rng.choice(TASK_TYPES)


def _filter_by_difficulty(task_type: str, difficulty: str, cases: list[dict]) -> list[dict]:
    if task_type == "diagnosis":
        return _filter_diagnosis(difficulty, cases)
    elif task_type == "calculation":
        return _filter_calculation(difficulty, cases)
    elif task_type == "note_review":
        return _filter_notes(difficulty, cases)
    return cases


def _filter_diagnosis(difficulty: str, cases: list[dict]) -> list[dict]:
    def score(c: dict) -> int:
        try:
            return int(c.get("score", 0))
        except (ValueError, TypeError):
            return 0

    if difficulty == "easy":
        return [c for c in cases if 12 <= score(c) <= 17]
    elif difficulty == "medium":
        return [c for c in cases if 17 < score(c) <= 22]
    elif difficulty == "hard":
        return [c for c in cases if score(c) > 22]
    return cases


def _matches_set(name: str, keyword_set: set[str]) -> bool:
    return any(kw in name for kw in keyword_set)


def _filter_calculation(difficulty: str, cases: list[dict]) -> list[dict]:
    def calc_name(c: dict) -> str:
        return (c.get("Calculator Name") or "").lower()

    if difficulty == "easy":
        return [c for c in cases if _matches_set(calc_name(c), SIMPLE_CALCULATORS)]
    elif difficulty == "hard":
        return [c for c in cases if _matches_set(calc_name(c), COMPLEX_CALCULATORS)]
    elif difficulty == "medium":
        return [
            c for c in cases
            if not _matches_set(calc_name(c), SIMPLE_CALCULATORS)
            and not _matches_set(calc_name(c), COMPLEX_CALCULATORS)
        ]
    return cases


def _filter_notes(difficulty: str, cases: list[dict]) -> list[dict]:
    def error_flag(c: dict) -> int:
        try:
            return int(float(c.get("Error Flag", 0)))
        except (ValueError, TypeError):
            return 0

    def error_type(c: dict) -> str:
        return (c.get("Error Type") or "").lower().strip()

    if difficulty == "easy":
        return [c for c in cases if error_flag(c) == 0]
    elif difficulty == "medium":
        return [
            c for c in cases
            if error_flag(c) == 1
            and error_type(c) not in SUBTLE_ERROR_TYPES
        ]
    elif difficulty == "hard":
        return [
            c for c in cases
            if error_flag(c) == 1
            and error_type(c) in SUBTLE_ERROR_TYPES
        ]
    return cases
