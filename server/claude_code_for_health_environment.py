# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Claude Code For Health Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ClaudeCodeForHealthAction, ClaudeCodeForHealthObservation
except ImportError:
    from models import ClaudeCodeForHealthAction, ClaudeCodeForHealthObservation


class ClaudeCodeForHealthEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = ClaudeCodeForHealthEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Claude Code For Health environment ready!"
        >>>
        >>> obs = env.step(ClaudeCodeForHealthAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the claude_code_for_health environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> ClaudeCodeForHealthObservation:
        """
        Reset the environment.

        Returns:
            ClaudeCodeForHealthObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return ClaudeCodeForHealthObservation(
            echoed_message="Claude Code For Health environment ready!",
            message_length=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: ClaudeCodeForHealthAction) -> ClaudeCodeForHealthObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: ClaudeCodeForHealthAction containing the message to echo

        Returns:
            ClaudeCodeForHealthObservation with the echoed message and its length
        """
        self._state.step_count += 1

        message = action.message
        length = len(message)

        # Simple reward: longer messages get higher rewards
        reward = length * 0.1

        return ClaudeCodeForHealthObservation(
            echoed_message=message,
            message_length=length,
            done=False,
            reward=reward,
            metadata={"original_message": message, "step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
