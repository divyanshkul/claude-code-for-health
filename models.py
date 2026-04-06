"""
Data models for the Claude Code for Health Environment.

Three Pydantic models defining the action/observation/state contract:
- MedAction: single CLI command string (terminal metaphor)
- MedObservation: command output + episode metadata
- MedState: episode tracking for state() endpoint
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class MedAction(Action):
    """Agent sends a single CLI command string per step."""

    command: str = Field(..., description="CLI command string, e.g. 'chart.labs CBC'")


class MedObservation(Observation):
    """Environment returns command output and episode context."""

    output: str = Field(default="", description="Command output text")
    error: str = Field(default="", description="Error message if command invalid")
    available_commands: list[str] = Field(default_factory=list)
    task_type: str = Field(default="", description="diagnosis | calculation | note_review")
    step_number: int = Field(default=0)
    max_steps: int = Field(default=50)


class MedState(State):
    """Episode state exposed via the state() endpoint."""

    task_type: str = Field(default="")
    difficulty: str = Field(default="easy")
    total_score: float = Field(default=0.0)
    commands_issued: int = Field(default=0)
    is_submitted: bool = Field(default=False)
