"""Uniform random legal move agent. This is the first baseline opponent."""

from __future__ import annotations

import random

import numpy as np

from connect4.agents.base import Agent, AgentDecision
from connect4.env.constants import COLS


class RandomAgent(Agent):
    name = "random"

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def select_action(self, state: np.ndarray, valid_actions: list[int]) -> AgentDecision:
        if not valid_actions:
            raise ValueError("No valid actions.")

        # Randomly choose a valid action.
        action = self._rng.choice(valid_actions)
        
        probs = [0.0] * COLS
        p = 1.0 / len(valid_actions)
        for col in valid_actions:
            probs[col] = p
        return AgentDecision(
            action=action,
            probabilities=probs,
            value=None,
            metadata={"policy": "uniform_legal"},
        )
