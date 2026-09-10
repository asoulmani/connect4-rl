"""Not implemented yet — policy/value + MCTS is the next agent."""

from connect4.agents.base import Agent, AgentDecision
import numpy as np


class AlphaZeroAgent(Agent):
    name = "alphazero"

    def select_action(self, state: np.ndarray, valid_actions: list[int]) -> AgentDecision:
        raise NotImplementedError("AlphaZeroAgent is not implemented yet")
