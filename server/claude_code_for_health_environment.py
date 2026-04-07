"""Core environment: reset/step/state for all three clinical task types."""

from random import Random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import MedAction, MedObservation, MedState
except ImportError:
    from models import MedAction, MedObservation, MedState

from . import command_parser, graders, task_configs
from .data_loader import DataLoader

DIAGNOSIS_COMMANDS = [
    "chart.history", "chart.vitals", "chart.labs [panel]",
    "chart.imaging [type]", "chart.exam [system]",
    "chart.medications", "chart.allergies",
    "ddx.list", "ddx.add <diagnosis>", "ddx.remove <diagnosis>",
    "ddx.confirm <diagnosis>", "help",
]
CALCULATION_COMMANDS = [
    "case.read", "calculate <calculator_name>",
    "submit <numeric_value>", "help",
]
NOTE_COMMANDS = [
    "note.read", "note.correct <sentence_id> <corrected_text>",
    "note.approve", "help",
]
VALID_DIAGNOSIS_CMDS = {
    "chart.history", "chart.vitals", "chart.labs", "chart.imaging",
    "chart.exam", "chart.medications", "chart.allergies",
    "ddx.list", "ddx.add", "ddx.remove", "ddx.confirm", "help",
}
VALID_CALCULATION_CMDS = {"case.read", "calculate", "submit", "help"}
VALID_NOTE_CMDS = {"note.read", "note.correct", "note.approve", "help"}

PROTOCOL_PENALTY = -0.05
SPECIALIZED_LAB_PANELS = {"abg", "coags", "coagulation", "cultures", "cytology"}


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
    # reset
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> MedObservation:
        self._data_loader.load_all()
        if seed is not None:
            self._rng = Random(seed)

        opts = options or {}
        self._difficulty = opts.get("task", "easy")
        self._task_type = opts.get("task_type") or task_configs.get_default_task_type(self._difficulty)

        cases = self._get_cases_for_type()
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
            extracted = case.get("extracted", {})
            self._relevant_sections = graders.compute_relevant_sections(extracted)

        initial_output = self._build_initial_observation(case)
        cmds = self._available_commands()

        return MedObservation(
            output=initial_output,
            available_commands=cmds,
            task_type=self._task_type,
            step_number=0,
            max_steps=self._max_steps,
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: MedAction) -> MedObservation:
        if self._is_done:
            return self._obs("Episode is over. Call reset() to start a new one.", reward=0.0, done=True)

        self._state.step_count += 1
        self._state.commands_issued += 1
        raw = action.command
        self._agent_actions.append(raw)

        cmd, args = command_parser.parse(raw)

        if not cmd:
            return self._obs("Empty command. Type 'help' for available commands.", reward=0.0)

        valid_cmds = self._valid_commands_set()
        if cmd not in valid_cmds:
            return self._obs(
                f"Unknown command: '{cmd}'. Type 'help' for available commands.",
                error=f"Unknown command: {cmd}",
                reward=0.0,
            )

        full_cmd = raw.strip().lower()
        is_duplicate = full_cmd in self._seen_commands and cmd not in ("help", "ddx.list")
        self._seen_commands.add(full_cmd)

        output, reward, done = self._dispatch(cmd, args)

        if is_duplicate and not done:
            output += f"\n[NOTE] Duplicate command — already executed. Efficiency penalty: {PROTOCOL_PENALTY}"
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

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    @property
    def state(self) -> MedState:
        return self._state

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, cmd: str, args: list[str]) -> tuple[str, float, bool]:
        if cmd == "help":
            return self._handle_help(), 0.0, False

        if self._task_type == "diagnosis":
            return self._dispatch_diagnosis(cmd, args)
        elif self._task_type == "calculation":
            return self._dispatch_calculation(cmd, args)
        elif self._task_type == "note_review":
            return self._dispatch_note(cmd, args)

        return "Internal error: unknown task type.", 0.0, False

    # ------------------------------------------------------------------
    # Diagnosis handlers
    # ------------------------------------------------------------------

    def _dispatch_diagnosis(self, cmd: str, args: list[str]) -> tuple[str, float, bool]:
        extracted = self._task_data.get("extracted", {})
        penalty, warning = self._check_prerequisites(cmd, args)

        if cmd == "chart.history":
            output = self._format_history(extracted.get("history", {}))
            reward = graders.diagnosis_step_reward(cmd, args, self._accessed_sections, self._relevant_sections) + penalty
            return (output + warning), reward, False

        if cmd == "chart.vitals":
            output = self._format_vitals(extracted.get("vitals", {}))
            reward = graders.diagnosis_step_reward(cmd, args, self._accessed_sections, self._relevant_sections) + penalty
            return (output + warning), reward, False

        if cmd == "chart.labs":
            labs = extracted.get("labs", {})
            if not args:
                panels = list(labs.keys()) if labs else []
                if panels:
                    return f"Available lab panels: {', '.join(panels)}", 0.0, False
                return "No lab results available.", 0.0, False
            panel_name = args[0]
            matched_panel = self._fuzzy_key_match(panel_name, labs)
            if matched_panel is None:
                return f"Lab panel '{panel_name}' not available. Use 'chart.labs' to list panels.", 0.0, False
            output = self._format_dict(labs[matched_panel], title=matched_panel)
            reward = graders.diagnosis_step_reward(cmd, [matched_panel.lower()], self._accessed_sections, self._relevant_sections) + penalty
            return (output + warning), reward, False

        if cmd == "chart.imaging":
            imaging = extracted.get("imaging", {})
            if not args:
                modalities = list(imaging.keys()) if imaging else []
                if modalities:
                    return f"Available imaging: {', '.join(modalities)}", 0.0, False
                return "No imaging results available.", 0.0, False
            modality = args[0]
            matched = self._fuzzy_key_match(modality, imaging)
            if matched is None:
                return f"Imaging '{modality}' not available. Use 'chart.imaging' to list.", 0.0, False
            output = f"{matched}: {imaging[matched]}"
            reward = graders.diagnosis_step_reward(cmd, [matched.lower()], self._accessed_sections, self._relevant_sections) + penalty
            return (output + warning), reward, False

        if cmd == "chart.exam":
            exam = extracted.get("physical_exam", {})
            if not args:
                systems = list(exam.keys()) if exam else []
                if systems:
                    return f"Available exam systems: {', '.join(systems)}", 0.0, False
                return "No physical exam data available.", 0.0, False
            system = args[0]
            matched = self._fuzzy_key_match(system, exam)
            if matched is None:
                return f"Exam '{system}' not available. Use 'chart.exam' to list.", 0.0, False
            output = f"{matched}: {exam[matched]}"
            reward = graders.diagnosis_step_reward(cmd, [matched.lower()], self._accessed_sections, self._relevant_sections)
            return output, reward, False

        if cmd == "chart.medications":
            meds = extracted.get("history", {}).get("medications", [])
            if meds:
                return "Medications: " + ", ".join(meds), 0.0, False
            return "No medications listed.", 0.0, False

        if cmd == "chart.allergies":
            allergies = extracted.get("history", {}).get("allergies", [])
            if allergies:
                return "Allergies: " + ", ".join(allergies), 0.0, False
            return "No known allergies.", 0.0, False

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
            gt_diagnosis = self._ground_truth.get("diagnosis", "")
            terminal = graders.diagnosis_terminal_reward(
                confirmed=self._confirmed_diagnosis,
                ground_truth_diagnosis=gt_diagnosis,
                accessed_sections=self._accessed_sections,
                relevant_sections=self._relevant_sections,
                ddx_list=self._ddx_list,
                steps_taken=self._state.step_count,
            )
            terminal += penalty
            return f"Diagnosis submitted: '{self._confirmed_diagnosis}'. Score: {terminal:.2f}" + warning, terminal, True

        return f"Unknown diagnosis command: {cmd}", 0.0, False

    # ------------------------------------------------------------------
    # Calculation handlers
    # ------------------------------------------------------------------

    def _dispatch_calculation(self, cmd: str, args: list[str]) -> tuple[str, float, bool]:
        if cmd == "case.read":
            note = self._task_data.get("Patient Note", "No patient note available.")
            question = self._task_data.get("Question", "")
            output = note
            if question:
                output += f"\n\nQuestion: {question}"
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

            gt_answer = float(self._ground_truth.get("answer", 0))
            lower = float(self._ground_truth.get("lower_limit", gt_answer))
            upper = float(self._ground_truth.get("upper_limit", gt_answer))
            expected_calc = self._ground_truth.get("calculator_name", "")

            terminal = graders.calculation_terminal_reward(
                submitted_value=self._submitted_value,
                ground_truth=gt_answer,
                lower_limit=lower,
                upper_limit=upper,
                calculator_used=self._calculator_used,
                expected_calculator=expected_calc,
                steps_taken=self._state.step_count,
            )
            return f"Submitted: {self._submitted_value}. Score: {terminal:.2f}", terminal, True

        return f"Unknown calculation command: {cmd}", 0.0, False

    # ------------------------------------------------------------------
    # Note review handlers
    # ------------------------------------------------------------------

    def _dispatch_note(self, cmd: str, args: list[str]) -> tuple[str, float, bool]:
        if cmd == "note.read":
            sentences_raw = self._task_data.get("Sentences", "")
            if sentences_raw:
                output = self._format_note_sentences(sentences_raw)
            else:
                output = self._task_data.get("Text", "No note available.")
            reward = graders.note_step_reward(cmd, self._note_read)
            self._note_read = True
            return output, reward, False

        if cmd == "note.correct":
            if len(args) < 2:
                return "Usage: note.correct <sentence_id> <corrected_text>", 0.0, False
            sentence_id = args[0].strip()
            corrected_text = args[1].strip()
            self._corrections[sentence_id] = corrected_text
            return f"Correction recorded for sentence {sentence_id}.", 0.0, False

        if cmd == "note.approve":
            has_error = bool(self._ground_truth.get("has_error", False))
            error_sid = self._ground_truth.get("error_sentence_id")
            corrected_sentence = self._ground_truth.get("corrected_sentence")

            terminal = graders.note_terminal_reward(
                corrections=self._corrections,
                has_error=has_error,
                error_sentence_id=error_sid,
                corrected_sentence=corrected_sentence,
            )
            status = "Corrections submitted." if self._corrections else "Note approved as correct."
            return f"{status} Score: {terminal:.2f}", terminal, True

        return f"Unknown note review command: {cmd}", 0.0, False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_cases_for_type(self) -> list[dict]:
        if self._task_type == "diagnosis":
            return self._data_loader.get_diagnosis_cases()
        elif self._task_type == "calculation":
            return self._data_loader.get_calculation_cases()
        elif self._task_type == "note_review":
            return self._data_loader.get_note_cases()
        return self._data_loader.get_diagnosis_cases()

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
            error_flag = case.get("Error Flag", 0)
            try:
                has_error = int(float(error_flag)) == 1
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
            age = demo.get("age", "?")
            sex = demo.get("sex", "?")
            cc = extracted.get("chief_complaint", case.get("case_prompt", "")[:150])
            return (
                f"Patient: {age}{sex}, {cc}\n"
                f"Type 'help' for available commands."
            )
        elif self._task_type == "calculation":
            question = case.get("Question", "")
            calc_name = case.get("Calculator Name", "")
            return (
                f"Medical Calculation Task — {calc_name}\n"
                f"{question}\n"
                f"Type 'case.read' to view the full patient note."
            )
        elif self._task_type == "note_review":
            return (
                "Clinical Note Review Task\n"
                "Review the note for medical errors. Correct any you find, then approve.\n"
                "Type 'note.read' to view the clinical note."
            )
        return "Unknown task type."

    def _handle_help(self) -> str:
        cmds = self._available_commands()
        lines = [f"Available commands ({self._task_type}):"]
        for c in cmds:
            lines.append(f"  {c}")
        return "\n".join(lines)

    def _available_commands(self) -> list[str]:
        if self._task_type == "diagnosis":
            return DIAGNOSIS_COMMANDS
        elif self._task_type == "calculation":
            return CALCULATION_COMMANDS
        elif self._task_type == "note_review":
            return NOTE_COMMANDS
        return ["help"]

    def _valid_commands_set(self) -> set[str]:
        if self._task_type == "diagnosis":
            return VALID_DIAGNOSIS_CMDS
        elif self._task_type == "calculation":
            return VALID_CALCULATION_CMDS
        elif self._task_type == "note_review":
            return VALID_NOTE_CMDS
        return {"help"}

    def _force_terminal(self) -> float:
        if self._task_type == "diagnosis":
            gt = self._ground_truth.get("diagnosis", "")
            return graders.diagnosis_terminal_reward(
                confirmed=self._confirmed_diagnosis or "",
                ground_truth_diagnosis=gt,
                accessed_sections=self._accessed_sections,
                relevant_sections=self._relevant_sections,
                ddx_list=self._ddx_list,
                steps_taken=self._state.step_count,
            )
        elif self._task_type == "calculation":
            if self._submitted_value is not None:
                return 0.0
            return 0.0
        elif self._task_type == "note_review":
            return graders.note_terminal_reward(
                corrections=self._corrections,
                has_error=bool(self._ground_truth.get("has_error", False)),
                error_sentence_id=self._ground_truth.get("error_sentence_id"),
                corrected_sentence=self._ground_truth.get("corrected_sentence"),
            )
        return 0.0

    def _obs(self, output: str, reward: float = 0.0, done: bool = False, error: str = "") -> MedObservation:
        if not done and self._task_type:
            output = output + "\n\n" + self._status_footer()
        return MedObservation(
            output=output,
            error=error,
            available_commands=self._available_commands(),
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
            read = "yes" if self._case_read else "no"
            calc = self._calculator_used or "none"
            return f"[STATUS] Case read: {read} | Calculator: {calc} | {step_info}"

        if self._task_type == "note_review":
            read = "yes" if self._note_read else "no"
            corr = str(dict(self._corrections)) if self._corrections else "none"
            return f"[STATUS] Note read: {read} | Corrections: {corr} | {step_info}"

        return f"[STATUS] {step_info}"

    def _check_prerequisites(self, cmd: str, args: list[str]) -> tuple[float, str]:
        """Returns (penalty, warning_text). Both empty if no violation."""
        if cmd == "chart.imaging" and args:
            if "vitals" not in self._accessed_sections:
                return PROTOCOL_PENALTY, f"\n[WARNING] Ordering imaging without baseline vitals: {PROTOCOL_PENALTY} protocol penalty"

        if cmd == "chart.labs" and args:
            panel = args[0].lower()
            if panel in SPECIALIZED_LAB_PANELS:
                has_basic = any(s.startswith("labs.") and s.split(".")[-1] in ("cbc", "bmp") for s in self._accessed_sections)
                if not has_basic:
                    return PROTOCOL_PENALTY, f"\n[WARNING] Ordering specialized labs without basic panels (CBC/BMP): {PROTOCOL_PENALTY} protocol penalty"

        if cmd == "ddx.confirm":
            if len(self._ddx_list) < 2:
                return PROTOCOL_PENALTY, f"\n[WARNING] Confirming diagnosis with <2 differentials: {PROTOCOL_PENALTY} protocol penalty"

        return 0.0, ""

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_history(self, history: dict) -> str:
        if not history or not any(history.values()):
            return "No history data available."
        lines = []
        if history.get("pmh"):
            lines.append(f"PMH: {', '.join(history['pmh'])}")
        if history.get("medications"):
            lines.append(f"Medications: {', '.join(history['medications'])}")
        if history.get("allergies"):
            lines.append(f"Allergies: {', '.join(history['allergies'])}")
        if history.get("social"):
            lines.append(f"Social: {history['social']}")
        if history.get("family"):
            lines.append(f"Family: {history['family']}")
        return "\n".join(lines) if lines else "No history data available."

    def _format_vitals(self, vitals: dict) -> str:
        if not vitals or not any(v for v in vitals.values() if v):
            return "No vital signs recorded."
        parts = []
        label_map = {"bp": "BP", "hr": "HR", "temp": "Temp", "rr": "RR", "spo2": "SpO2"}
        for key, label in label_map.items():
            val = vitals.get(key)
            if val:
                parts.append(f"{label}: {val}")
        return " | ".join(parts) if parts else "No vital signs recorded."

    def _format_dict(self, data, title: str = "") -> str:
        if isinstance(data, dict):
            lines = [f"{title}:"] if title else []
            for k, v in data.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)
        if isinstance(data, str):
            return f"{title}: {data}" if title else data
        return str(data)

    def _format_note_sentences(self, sentences_raw: str) -> str:
        lines = sentences_raw.strip().split("\n")
        formatted = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if parts[0].isdigit():
                sid = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                formatted.append(f"[{sid}] {text}")
            else:
                formatted.append(line)
        return "\n".join(formatted)

    @staticmethod
    def _fuzzy_key_match(query: str, data: dict) -> str | None:
        query_lower = query.lower().strip()
        for key in data:
            if key.lower() == query_lower:
                return key
        for key in data:
            if query_lower in key.lower() or key.lower() in query_lower:
                return key
        return None
