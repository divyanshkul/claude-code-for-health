"""Client for the Claude Code for Health environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import MedAction, MedObservation, MedState


class ClaudeCodeForHealthEnv(
    EnvClient[MedAction, MedObservation, MedState]
):
    def _step_payload(self, action: MedAction) -> Dict:
        return {"command": action.command}

    def _parse_result(self, payload: Dict) -> StepResult[MedObservation]:
        obs_data = payload.get("observation", {})
        observation = MedObservation(
            output=obs_data.get("output", ""),
            error=obs_data.get("error", ""),
            available_commands=obs_data.get("available_commands", []),
            task_type=obs_data.get("task_type", ""),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 50),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> MedState:
        return MedState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_type=payload.get("task_type", ""),
            difficulty=payload.get("difficulty", "easy"),
            total_score=payload.get("total_score", 0.0),
            commands_issued=payload.get("commands_issued", 0),
            is_submitted=payload.get("is_submitted", False),
        )
