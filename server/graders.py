"""Dense reward functions for diagnosis, calculation, and note review tasks.

Reward budgets per task type:
  diagnosis:    0.30 intermediate + 0.70 terminal = 1.0
  calculation:  0.15 intermediate + 0.85 terminal = 1.0
  note_review:  0.10 intermediate + 0.90 terminal = 1.0
"""

from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Diagnosis grading
# ---------------------------------------------------------------------------

def diagnosis_step_reward(
    command: str,
    args: list[str],
    accessed_sections: set[str],
    relevant_sections: set[str],
) -> float:
    """Step reward for chart exploration commands. Budget: 0.30 total."""
    n = len(relevant_sections)
    if n == 0:
        return 0.0

    per_section = 0.30 / n
    section_key = _chart_command_to_section_key(command, args)
    if section_key is None:
        return 0.0
    if section_key in accessed_sections:
        return 0.0
    if section_key not in relevant_sections:
        return 0.0

    accessed_sections.add(section_key)
    return round(per_section, 4)


def _chart_command_to_section_key(command: str, args: list[str]) -> str | None:
    if command == "chart.history":
        return "history"
    if command == "chart.vitals":
        return "vitals"
    if command == "chart.labs" and args:
        return f"labs.{args[0].lower()}"
    if command == "chart.imaging" and args:
        return f"imaging.{args[0].lower()}"
    if command == "chart.exam" and args:
        return f"exam.{args[0].lower()}"
    return None


def diagnosis_terminal_reward(
    confirmed: str,
    ground_truth_diagnosis: str,
    accessed_sections: set[str],
    relevant_sections: set[str],
    ddx_list: list[str],
    steps_taken: int,
) -> float:
    """Terminal reward on ddx.confirm. Budget: 0.70 total."""
    n = max(len(relevant_sections), 1)

    # Diagnostic accuracy (0.40)
    ratio = fuzz.token_sort_ratio(confirmed.lower(), ground_truth_diagnosis.lower())
    if ratio >= 80:
        accuracy_score = 1.0
    elif ratio >= 60:
        accuracy_score = 0.5
    else:
        accuracy_score = 0.0
    accuracy = 0.40 * accuracy_score

    # Workup completeness (0.10)
    accessed_relevant = len(accessed_sections & relevant_sections)
    completeness = 0.10 * (accessed_relevant / n)

    # Efficiency (0.10) — baseline is N+2 steps
    excess = max(0, steps_taken - n - 2)
    efficiency = 0.10 * max(0.0, 1.0 - excess / 20.0)

    # Reasoning quality (0.10) — DDX breadth + whether answer was in DDX
    ddx_breadth = min(len(ddx_list), 3) / 3.0 * 0.5
    confirmed_in_ddx = 0.5 if any(
        fuzz.token_sort_ratio(confirmed.lower(), d.lower()) >= 70
        for d in ddx_list
    ) else 0.0
    reasoning = 0.10 * (ddx_breadth + confirmed_in_ddx)

    return round(accuracy + completeness + efficiency + reasoning, 4)


# ---------------------------------------------------------------------------
# Calculation grading
# ---------------------------------------------------------------------------

def calculation_step_reward(command: str, case_read: bool, calculator_declared: bool) -> float:
    """Step reward for case reading and calculator declaration. Budget: 0.15."""
    if command == "case.read" and not case_read:
        return 0.10
    if command == "calculate" and not calculator_declared:
        return 0.05
    return 0.0


def calculation_terminal_reward(
    submitted_value: float,
    ground_truth: float,
    lower_limit: float,
    upper_limit: float,
    calculator_used: str,
    expected_calculator: str,
    steps_taken: int,
) -> float:
    """Terminal reward on submit. Budget: 0.85."""
    # Numeric accuracy (0.50)
    if lower_limit <= submitted_value <= upper_limit:
        numeric_score = 1.0
    else:
        band = upper_limit - lower_limit
        extended_lower = lower_limit - band
        extended_upper = upper_limit + band
        if extended_lower <= submitted_value <= extended_upper:
            numeric_score = 0.5
        else:
            numeric_score = 0.0
    numeric = 0.50 * numeric_score

    # Correct calculator (0.25)
    calc_ratio = fuzz.token_sort_ratio(calculator_used.lower(), expected_calculator.lower())
    calc_match = 0.25 * (1.0 if calc_ratio >= 75 else 0.0)

    # Efficiency (0.10) — perfect if ≤3 steps, linear decay to 0 at 10
    if steps_taken <= 3:
        eff_score = 1.0
    elif steps_taken >= 10:
        eff_score = 0.0
    else:
        eff_score = 1.0 - (steps_taken - 3) / 7.0
    efficiency = 0.10 * eff_score

    return round(numeric + calc_match + efficiency, 4)


# ---------------------------------------------------------------------------
# Note review grading
# ---------------------------------------------------------------------------

def note_step_reward(command: str, note_read: bool) -> float:
    """Step reward for reading the note. Budget: 0.10."""
    if command == "note.read" and not note_read:
        return 0.10
    return 0.0


def note_terminal_reward(
    corrections: dict[str, str],
    has_error: bool,
    error_sentence_id: str | None,
    corrected_sentence: str | None,
) -> float:
    """Terminal reward on note.approve. Budget: 0.90."""
    if not has_error:
        # No error in note — agent should approve without corrections
        if len(corrections) == 0:
            return 0.90
        # False positive penalty
        return round(0.90 * max(0.0, 1.0 - len(corrections) * 0.3), 4)

    # Note has an error — evaluate detection + correction
    found_correct_sentence = False
    correction_quality = 0.0

    if error_sentence_id is not None:
        target_id = str(error_sentence_id).strip()
        if target_id in corrections:
            found_correct_sentence = True
            if corrected_sentence:
                ratio = fuzz.ratio(
                    corrections[target_id].strip().lower(),
                    corrected_sentence.strip().lower(),
                )
                correction_quality = ratio / 100.0

    # Error detection (0.40)
    detection = 0.40 * (1.0 if found_correct_sentence else 0.0)

    # Correction accuracy (0.40)
    correction = 0.40 * correction_quality

    # False positive penalty (0.10)
    total_corrections = len(corrections)
    true_positives = 1 if found_correct_sentence else 0
    false_positives = total_corrections - true_positives
    fp_penalty = 1.0 - (false_positives / max(total_corrections, 1))
    no_fp = 0.10 * max(0.0, fp_penalty)

    return round(detection + correction + no_fp, 4)


# ---------------------------------------------------------------------------
# Utility: compute relevant sections from extracted case data
# ---------------------------------------------------------------------------

def compute_relevant_sections(extracted: dict) -> set[str]:
    """Build the set of non-empty data sections for a diagnosis case."""
    sections = set()

    if _has_data(extracted.get("vitals")):
        sections.add("vitals")
    if _has_data(extracted.get("history")):
        sections.add("history")

    for panel_name, panel_data in (extracted.get("labs") or {}).items():
        if _has_data(panel_data):
            sections.add(f"labs.{panel_name.lower()}")

    for modality, findings in (extracted.get("imaging") or {}).items():
        if _has_data(findings):
            sections.add(f"imaging.{modality.lower()}")

    for system, findings in (extracted.get("physical_exam") or {}).items():
        if _has_data(findings):
            sections.add(f"exam.{system.lower()}")

    return sections


def _has_data(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_data(v) for v in value.values())
    if isinstance(value, list):
        return len(value) > 0
    return True
