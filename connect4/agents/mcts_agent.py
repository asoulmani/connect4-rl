"""MCTS agent: PUCT tree, uniform prior, random rollouts. No network."""

from __future__ import annotations

import random
import time

import numpy as np

from connect4.agents.alphazero.mcts import (
    DEFAULT_C_PUCT,
    DEFAULT_SIMULATIONS,
    count_nodes,
    env_from_state,
    most_visited_action,
    run_mcts,
    visit_policy,
)
from connect4.agents.base import Agent, AgentDecision
from connect4.env.constants import COLS


class MCTSAgent(Agent):
    name = "mcts"

    def __init__(
        self,
        n_simulations: int = DEFAULT_SIMULATIONS,
        c_puct: float = DEFAULT_C_PUCT,
        rng: random.Random | None = None,
    ) -> None:
        if n_simulations < 1:
            raise ValueError("n_simulations must be >= 1.")
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self._rng = rng or random.Random()

    def select_action(self, state: np.ndarray, valid_actions: list[int]) -> AgentDecision:
        if not valid_actions:
            raise ValueError("No valid actions.")

        t0 = time.perf_counter()
        env = env_from_state(state)
        root = run_mcts(
            env,
            self.n_simulations,
            c_puct=self.c_puct,
            rng=self._rng,
            root_actions=valid_actions,
        )
        action = most_visited_action(root, valid_actions)
        child = root.children[action]
        # Child q is the opponent's outcome. The root player wants the negation of that.
        value = -child.q()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        visits = {str(col): int(root.children[col].visits) for col in valid_actions if col in root.children}
        q_values = {str(col): float(-root.children[col].q()) for col in valid_actions if col in root.children}
        return AgentDecision(
            action=action,
            probabilities=visit_policy(root, valid_actions, COLS),
            value=float(value),
            metadata={
                "policy": "mcts_puct",
                "prior": "uniform",
                "leaf": "random_rollout",
                "n_simulations": self.n_simulations,
                "c_puct": self.c_puct,
                "seat": int(env.current_player),
                "visits": visits,
                "q": q_values,
                "tree_nodes": count_nodes(root),
                "elapsed_ms": float(elapsed_ms),
            },
        )
