"""Core environment: reset/step/state for all three clinical task types."""

from random import Random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import MedAction, MedObservation, MedState
except ImportError:
    from models import MedAction, MedObservation, MedState

from . import command_parser, constants, graders, task_configs
from .data_loader import DataLoader

PROTOCOL_PENALTY = -0.05
SPECIALIZED_LAB_PANELS = {"abg", "coags", "coagulation", "cultures", "cytology"}

REFERENCE_TOOLS = [
    "reference.ranges <test>", "reference.criteria <condition>",
    "reference.drug_info <drug>", "interpret <test> <value>",
]
REFERENCE_TOOL_NAMES = {"reference.ranges", "reference.criteria", "reference.drug_info", "interpret"}

TASK_TOOLS = {
    "diagnosis": [
        "chart.history", "chart.vitals", "chart.labs [panel]",
        "chart.imaging [type]", "chart.exam [system]",
        "chart.medications", "chart.allergies",
        "ddx.list", "ddx.add <diagnosis>", "ddx.remove <diagnosis>",
        "ddx.confirm <diagnosis>", "help",
    ] + REFERENCE_TOOLS,
    "calculation": [
        "case.read", "calculate <calculator_name>",
        "submit <numeric_value>", "help",
    ] + REFERENCE_TOOLS,
    "note_review": [
        "note.read", "note.correct <sentence_id> <corrected_text>",
        "note.approve", "help",
    ] + REFERENCE_TOOLS,
}

VALID_TOOL_NAMES = {
    "diagnosis": {
        "chart.history", "chart.vitals", "chart.labs", "chart.imaging",
        "chart.exam", "chart.medications", "chart.allergies",
        "ddx.list", "ddx.add", "ddx.remove", "ddx.confirm", "help",
    } | REFERENCE_TOOL_NAMES,
    "calculation": {"case.read", "calculate", "submit", "help"} | REFERENCE_TOOL_NAMES,
    "note_review": {"note.read", "note.correct", "note.approve", "help"} | REFERENCE_TOOL_NAMES,
}


class ClaudeCodeForHealthEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._data_loader = DataLoader()
        self._rng = Random()
        self._state = MedState(episode_id=str(uuid4()), step_count=0)
        self._max_steps = 50
        self._reset_episode_vars()

    def _reset_episode_vars(self):
        self._task_type = ""
        self._difficulty = "easy"
        self._task_data: dict = {}
        self._ground_truth: dict = {}
        self._agent_actions: list[str] = []
        self._ddx_list: list[str] = []
        self._confirmed_diagnosis = ""
        self._calculator_used = ""
        self._submitted_value: float | None = None
        self._corrections: dict[str, str] = {}
        self._accessed_sections: set[str] = set()
        self._relevant_sections: set[str] = set()
        self._case_read = False
        self._note_read = False
        self._calculator_declared = False
        self._is_done = False
        self._cumulative_reward = 0.0
        self._seen_commands: set[str] = set()

    # ------------------------------------------------------------------
    # reset / step / state
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> MedObservation:
        self._data_loader.load_all()
        if seed is not None:
            self._rng = Random(seed)

        opts = options or {}
        self._difficulty = opts.get("task", "easy")
        self._task_type = opts.get("task_type") or task_configs.get_default_task_type(self._difficulty, self._rng)

        cases_map = {
            "diagnosis": self._data_loader.get_diagnosis_cases,
            "calculation": self._data_loader.get_calculation_cases,
            "note_review": self._data_loader.get_note_cases,
        }
        cases = cases_map.get(self._task_type, self._data_loader.get_diagnosis_cases)()
        case = task_configs.select_case(self._task_type, self._difficulty, cases, self._rng)

        self._state = MedState(
            episode_id=str(uuid4()),
            step_count=0,
            task_type=self._task_type,
            difficulty=self._difficulty,
        )
        self._reset_episode_vars()
        self._task_type = self._state.task_type
        self._difficulty = self._state.difficulty
        self._task_data = case
        self._setup_ground_truth(case)

        if self._task_type == "diagnosis":
            self._relevant_sections = graders.compute_relevant_sections(case.get("extracted", {}))

        return MedObservation(
            output=self._build_initial_observation(case),
            available_commands=TASK_TOOLS.get(self._task_type, ["help"]),
            task_type=self._task_type,
            step_number=0,
            max_steps=self._max_steps,
            done=False,
            reward=0.0,
        )

    def step(self, action: MedAction) -> MedObservation:
        if self._is_done:
            return self._obs("Episode is over. Call reset() to start a new one.", reward=0.0, done=True)

        self._state.step_count += 1
        self._state.commands_issued += 1
        raw = action.command
        self._agent_actions.append(raw)

        cmd, args = command_parser.parse(raw)

        if not cmd:
            return self._obs("Empty command. Type 'help' for available tools.", reward=0.0)

        valid = VALID_TOOL_NAMES.get(self._task_type, {"help"})
        if cmd not in valid:
            return self._obs(
                f"Unknown tool: '{cmd}'. Type 'help' for available tools.",
                error=f"Unknown command: {cmd}",
                reward=0.0,
            )

        full_cmd = raw.strip().lower()
        is_duplicate = full_cmd in self._seen_commands and cmd not in ("help", "ddx.list")
        self._seen_commands.add(full_cmd)

        output, reward, done = self._dispatch(cmd, args)

        if is_duplicate and not done:
            output += f"\n[NOTE] Duplicate tool call. Efficiency penalty: {PROTOCOL_PENALTY}"
            reward += PROTOCOL_PENALTY

        self._cumulative_reward += reward
        self._state.total_score = round(self._cumulative_reward, 4)

        if done:
            self._is_done = True
            self._state.is_submitted = True

        if not done and self._state.step_count >= self._max_steps:
            terminal_reward = self._force_terminal()
            reward += terminal_reward
            self._cumulative_reward += terminal_reward
            self._state.total_score = round(self._cumulative_reward, 4)
            done = True
            self._is_done = True
            output += "\n\nMax steps reached. Episode ended."

        return self._obs(output, reward=round(reward, 4), done=done)

    @property
    def state(self) -> MedState:
        return self._state

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, cmd: str, args: list[str]) -> tuple[str, float, bool]:
        if cmd == "help":
            return self._handle_help(), 0.0, False

        ref_result = self._dispatch_reference(cmd, args)
        if ref_result is not None:
            return ref_result

        dispatch_map = {
            "diagnosis": self._dispatch_diagnosis,
            "calculation": self._dispatch_calculation,
            "note_review": self._dispatch_note,
        }
        handler = dispatch_map.get(self._task_type)
        if handler:
            return handler(cmd, args)
        return "Internal error: unknown task type.", 0.0, False

    def _dispatch_reference(self, cmd: str, args: list[str]) -> tuple[str, float, bool] | None:
        lookup_map = {
            "reference.ranges": ("test_name", constants.lookup_range),
            "reference.criteria": ("condition", constants.lookup_criteria),
            "reference.drug_info": ("drug_name", constants.lookup_drug),
        }
        if cmd in lookup_map:
            param_name, lookup_fn = lookup_map[cmd]
            if not args:
                return f"Usage: {cmd} <{param_name}>", 0.0, False
            result = lookup_fn(args[0])
            if result is None:
                return f"No results found for '{args[0]}'.", 0.0, False
            return result, 0.0, False

        if cmd == "interpret":
            if not args:
                return "Usage: interpret <test_name> <value>", 0.0, False
            parts = args[0].rsplit(None, 1) if len(args) == 1 else args
            if len(parts) < 2:
                return "Usage: interpret <test_name> <value>", 0.0, False
            result = constants.interpret_value(parts[0], parts[-1])
            if result is None:
                return f"Unknown test '{parts[0]}'. Try: sodium, potassium, troponin, wbc, etc.", 0.0, False
            return result, 0.0, False

        return None

    # ------------------------------------------------------------------
    # Diagnosis tools
    # ------------------------------------------------------------------

    def _diag_step_reward(self, cmd: str, args: list[str]) -> float:
        return graders.diagnosis_step_reward(cmd, args, self._accessed_sections, self._relevant_sections)

    def _handle_chart_keyed(self, data: dict, key_arg: str | None, cmd: str,
                            label: str, list_label: str) -> tuple[str, float, bool]:
        if not key_arg:
            keys = list(data.keys()) if data else []
            if keys:
                return f"Available {list_label}: {', '.join(keys)}", 0.0, False
            return f"No {list_label} available.", 0.0, False

        matched = self._fuzzy_key_match(key_arg, data)
        if matched is None:
            return f"{label} '{key_arg}' not available. Use '{cmd}' to list.", 0.0, False

        value = data[matched]
        output = self._format_dict(value, title=matched) if isinstance(value, dict) else f"{matched}: {value}"
        reward = self._diag_step_reward(cmd, [matched.lower()])
        return output, reward, False

    def _dispatch_diagnosis(self, cmd: str, args: list[str]) -> tuple[str, float, bool]:
        extracted = self._task_data.get("extracted", {})
        penalty, warning = self._check_prerequisites(cmd, args)

        if cmd == "chart.history":
            output = self._format_history(extracted.get("history", {}))
            return (output + warning), self._diag_step_reward(cmd, args) + penalty, False

        if cmd == "chart.vitals":
            output = self._format_vitals(extracted.get("vitals", {}))
            return (output + warning), self._diag_step_reward(cmd, args) + penalty, False

        if cmd == "chart.labs":
            output, reward, done = self._handle_chart_keyed(
                extracted.get("labs", {}), args[0] if args else None,
                "chart.labs", "Lab panel", "lab panels")
            return (output + warning), reward + penalty, done

        if cmd == "chart.imaging":
            output, reward, done = self._handle_chart_keyed(
                extracted.get("imaging", {}), args[0] if args else None,
                "chart.imaging", "Imaging", "imaging")
            return (output + warning), reward + penalty, done

        if cmd == "chart.exam":
            output, reward, done = self._handle_chart_keyed(
                extracted.get("physical_exam", {}), args[0] if args else None,
                "chart.exam", "Exam", "exam systems")
            return output, reward, done

        if cmd == "chart.medications":
            meds = extracted.get("history", {}).get("medications", [])
            return ("Medications: " + ", ".join(meds)) if meds else "No medications listed.", 0.0, False

        if cmd == "chart.allergies":
            allergies = extracted.get("history", {}).get("allergies", [])
            return ("Allergies: " + ", ".join(allergies)) if allergies else "No known allergies.", 0.0, False

        if cmd == "ddx.list":
            if self._ddx_list:
                items = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(self._ddx_list))
                return f"Current differential:\n{items}", 0.0, False
            return "Differential is empty.", 0.0, False

        if cmd == "ddx.add":
            if not args:
                return "Usage: ddx.add <diagnosis>", 0.0, False
            dx = args[0].strip()
            self._ddx_list.append(dx)
            return f"Added '{dx}'. Differential has {len(self._ddx_list)} entry(ies).", 0.0, False

        if cmd == "ddx.remove":
            if not args:
                return "Usage: ddx.remove <diagnosis>", 0.0, False
            dx = args[0].strip().lower()
            before = len(self._ddx_list)
            self._ddx_list = [d for d in self._ddx_list if d.lower() != dx]
            if len(self._ddx_list) < before:
                return f"Removed. Differential has {len(self._ddx_list)} entry(ies).", 0.0, False
            return f"'{args[0]}' not found in differential.", 0.0, False

        if cmd == "ddx.confirm":
            if not args:
                return "Usage: ddx.confirm <diagnosis>", 0.0, False
            self._confirmed_diagnosis = args[0].strip()
            terminal = graders.diagnosis_terminal_reward(
                confirmed=self._confirmed_diagnosis,
                ground_truth_diagnosis=self._ground_truth.get("diagnosis", ""),
                accessed_sections=self._accessed_sections,
                relevant_sections=self._relevant_sections,
                ddx_list=self._ddx_list,
                steps_taken=self._state.step_count,
            ) + penalty
            return f"Diagnosis submitted: '{self._confirmed_diagnosis}'. Score: {terminal:.2f}" + warning, terminal, True

        return f"Unknown diagnosis tool: {cmd}", 0.0, False

    # ------------------------------------------------------------------
    # Calculation tools
    # ------------------------------------------------------------------

    def _dispatch_calculation(self, cmd: str, args: list[str]) -> tuple[str, float, bool]:
        if cmd == "case.read":
            note = self._task_data.get("Patient Note", "No patient note available.")
            question = self._task_data.get("Question", "")
            output = note + (f"\n\nQuestion: {question}" if question else "")
            reward = graders.calculation_step_reward(cmd, self._case_read, self._calculator_declared)
            self._case_read = True
            return output, reward, False

        if cmd == "calculate":
            if not args:
                return "Usage: calculate <calculator_name>", 0.0, False
            self._calculator_used = args[0].strip()
            reward = graders.calculation_step_reward("calculate", self._case_read, self._calculator_declared)
            self._calculator_declared = True
            return f"Calculator noted: {self._calculator_used}. Use 'submit <value>' with your answer.", reward, False

        if cmd == "submit":
            if not args:
                return "Usage: submit <numeric_value>", 0.0, False
            try:
                self._submitted_value = float(args[0].strip())
            except ValueError:
                return f"Cannot parse '{args[0]}' as a number.", 0.0, False

            gt = self._ground_truth
            try:
                gt_answer = float(gt.get("answer", 0))
                lower = float(gt.get("lower_limit", gt_answer))
                upper = float(gt.get("upper_limit", gt_answer))
            except (ValueError, TypeError):
                gt_answer, lower, upper = 0.0, 0.0, 0.0

            terminal = graders.calculation_terminal_reward(
                submitted_value=self._submitted_value,
                ground_truth=gt_answer,
                lower_limit=lower,
                upper_limit=upper,
                calculator_used=self._calculator_used,
                expected_calculator=gt.get("calculator_name", ""),
                steps_taken=self._state.step_count,
            )
            return f"Submitted: {self._submitted_value}. Score: {terminal:.2f}", terminal, True

        return f"Unknown calculation tool: {cmd}", 0.0, False

    # ------------------------------------------------------------------
    # Note review tools
    # ------------------------------------------------------------------

    def _dispatch_note(self, cmd: str, args: list[str]) -> tuple[str, float, bool]:
        if cmd == "note.read":
            sentences_raw = self._task_data.get("Sentences", "")
            output = self._format_note_sentences(sentences_raw) if sentences_raw else self._task_data.get("Text", "No note available.")
            reward = graders.note_step_reward(cmd, self._note_read)
            self._note_read = True
            return output, reward, False

        if cmd == "note.correct":
            if len(args) < 2:
                return "Usage: note.correct <sentence_id> <corrected_text>", 0.0, False
            self._corrections[args[0].strip()] = args[1].strip()
            return f"Correction recorded for sentence {args[0].strip()}.", 0.0, False

        if cmd == "note.approve":
            gt = self._ground_truth
            terminal = graders.note_terminal_reward(
                corrections=self._corrections,
                has_error=bool(gt.get("has_error", False)),
                error_sentence_id=gt.get("error_sentence_id"),
                corrected_sentence=gt.get("corrected_sentence"),
            )
            status = "Corrections submitted." if self._corrections else "Note approved as correct."
            return f"{status} Score: {terminal:.2f}", terminal, True

        return f"Unknown note review tool: {cmd}", 0.0, False

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_ground_truth(self, case: dict):
        if self._task_type == "diagnosis":
            extracted = case.get("extracted", {})
            gt = extracted.get("ground_truth", {})
            self._ground_truth = {
                "diagnosis": gt.get("diagnosis", case.get("final_diagnosis", "")),
                "organ_system": gt.get("organ_system", ""),
                "key_findings": gt.get("key_findings", []),
            }
        elif self._task_type == "calculation":
            self._ground_truth = {
                "answer": case.get("Ground Truth Answer", "0"),
                "lower_limit": case.get("Lower Limit", case.get("Ground Truth Answer", "0")),
                "upper_limit": case.get("Upper Limit", case.get("Ground Truth Answer", "0")),
                "calculator_name": case.get("Calculator Name", ""),
                "explanation": case.get("Ground Truth Explanation", ""),
            }
        elif self._task_type == "note_review":
            try:
                has_error = int(float(case.get("Error Flag", 0))) == 1
            except (ValueError, TypeError):
                has_error = False
            self._ground_truth = {
                "has_error": has_error,
                "error_sentence_id": str(case.get("Error Sentence ID", "")).strip() if has_error else None,
                "error_sentence": case.get("Error Sentence", "") if has_error else None,
                "corrected_sentence": case.get("Corrected Sentence", "") if has_error else None,
            }

    def _build_initial_observation(self, case: dict) -> str:
        if self._task_type == "diagnosis":
            extracted = case.get("extracted", {})
            demo = extracted.get("demographics", {})
            cc = extracted.get("chief_complaint", case.get("case_prompt", "")[:150])
            return f"Patient: {demo.get('age', '?')}{demo.get('sex', '?')}, {cc}\nType 'help' for available tools."
        elif self._task_type == "calculation":
            return (
                f"Medical Calculation Task — {case.get('Calculator Name', '')}\n"
                f"{case.get('Question', '')}\n"
                f"Type 'case.read' to view the full patient note."
            )
        elif self._task_type == "note_review":
            return "Clinical Note Review Task\nReview the note for medical errors. Correct any you find, then approve.\nType 'note.read' to view the clinical note."
        return "Unknown task type."

    def _handle_help(self) -> str:
        tools = TASK_TOOLS.get(self._task_type, ["help"])
        lines = [f"Available tools ({self._task_type}):"]
        for t in tools:
            lines.append(f"  {t}")
        return "\n".join(lines)

    def _force_terminal(self) -> float:
        if self._task_type == "diagnosis":
            return graders.diagnosis_terminal_reward(
                confirmed=self._confirmed_diagnosis or "",
                ground_truth_diagnosis=self._ground_truth.get("diagnosis", ""),
                accessed_sections=self._accessed_sections,
                relevant_sections=self._relevant_sections,
                ddx_list=self._ddx_list,
                steps_taken=self._state.step_count,
            )
        elif self._task_type == "note_review":
            return graders.note_terminal_reward(
                corrections=self._corrections,
                has_error=bool(self._ground_truth.get("has_error", False)),
                error_sentence_id=self._ground_truth.get("error_sentence_id"),
                corrected_sentence=self._ground_truth.get("corrected_sentence"),
            )
        return 0.0

    def _check_prerequisites(self, cmd: str, args: list[str]) -> tuple[float, str]:
        if cmd == "chart.imaging" and args:
            if "vitals" not in self._accessed_sections:
                return PROTOCOL_PENALTY, f"\n[WARNING] Ordering imaging without baseline vitals: {PROTOCOL_PENALTY} protocol penalty"

        if cmd == "chart.labs" and args:
            if args[0].lower() in SPECIALIZED_LAB_PANELS:
                has_basic = any(s.startswith("labs.") and s.split(".")[-1] in ("cbc", "bmp") for s in self._accessed_sections)
                if not has_basic:
                    return PROTOCOL_PENALTY, f"\n[WARNING] Ordering specialized labs without basic panels (CBC/BMP): {PROTOCOL_PENALTY} protocol penalty"

        if cmd == "ddx.confirm" and len(self._ddx_list) < 2:
            return PROTOCOL_PENALTY, f"\n[WARNING] Confirming diagnosis with <2 differentials: {PROTOCOL_PENALTY} protocol penalty"

        return 0.0, ""

    # ------------------------------------------------------------------
    # Observation + status
    # ------------------------------------------------------------------

    def _obs(self, output: str, reward: float = 0.0, done: bool = False, error: str = "") -> MedObservation:
        if not done and self._task_type:
            output = output + "\n\n" + self._status_footer()
        return MedObservation(
            output=output,
            error=error,
            available_commands=TASK_TOOLS.get(self._task_type, ["help"]),
            task_type=self._task_type,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            done=done,
            reward=reward,
        )

    def _status_footer(self) -> str:
        step_info = f"Step: {self._state.step_count}/{self._max_steps}"
        if self._task_type == "diagnosis":
            ddx = ", ".join(self._ddx_list) if self._ddx_list else "empty"
            accessed = ", ".join(sorted(self._accessed_sections)) if self._accessed_sections else "none"
            return f"[STATUS] DDX: [{ddx}] | Accessed: {accessed} | {step_info}"
        if self._task_type == "calculation":
            return f"[STATUS] Case read: {'yes' if self._case_read else 'no'} | Calculator: {self._calculator_used or 'none'} | {step_info}"
        if self._task_type == "note_review":
            corr = str(dict(self._corrections)) if self._corrections else "none"
            return f"[STATUS] Note read: {'yes' if self._note_read else 'no'} | Corrections: {corr} | {step_info}"
        return f"[STATUS] {step_info}"

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_history(self, history: dict) -> str:
        if not history or not any(history.values()):
            return "No history data available."
        field_map = {"pmh": "PMH", "medications": "Medications", "allergies": "Allergies", "social": "Social", "family": "Family"}
        lines = []
        for key, label in field_map.items():
            val = history.get(key)
            if val:
                lines.append(f"{label}: {', '.join(val) if isinstance(val, list) else val}")
        return "\n".join(lines) if lines else "No history data available."

    def _format_vitals(self, vitals: dict) -> str:
        if not vitals or not any(v for v in vitals.values() if v):
            return "No vital signs recorded."
        label_map = {"bp": "BP", "hr": "HR", "temp": "Temp", "rr": "RR", "spo2": "SpO2"}
        parts = [f"{label}: {vitals[key]}" for key, label in label_map.items() if vitals.get(key)]
        return " | ".join(parts) if parts else "No vital signs recorded."

    def _format_dict(self, data, title: str = "") -> str:
        if isinstance(data, dict):
            lines = ([f"{title}:"] if title else []) + [f"  {k}: {v}" for k, v in data.items()]
            return "\n".join(lines)
        return f"{title}: {data}" if title else str(data)

    def _format_note_sentences(self, sentences_raw: str) -> str:
        formatted = []
        for line in sentences_raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if parts[0].isdigit():
                formatted.append(f"[{parts[0]}] {parts[1] if len(parts) > 1 else ''}")
            else:
                formatted.append(line)
        return "\n".join(formatted)

    @staticmethod
    def _fuzzy_key_match(query: str, data: dict) -> str | None:
        q = query.lower().strip()
        for key in data:
            if key.lower() == q:
                return key
        for key in data:
            if q in key.lower() or key.lower() in q:
                return key
        return None
