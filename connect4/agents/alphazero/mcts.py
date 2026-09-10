"""PUCT Monte Carlo tree search. No network yet.

Uniform prior + random rollouts is still PUCT, not UCT: the exploration term is
c_puct * P(a) * sqrt(N) / (1 + n), even when P is flat. This is the
network-free precursor to AlphaZero search.

A later heuristic or net needs the board, not only the legal-column list. That
interface change waits until the prior is no longer uniform.

PUCT ties and the final visit-count choice break toward center (column 3), same
as minimax move ordering. That is a small piece of Connect Four knowledge in an
otherwise search-only baseline.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from connect4.agents.heuristic_agent import player_to_move
from connect4.env.board import Board
from connect4.env.environment import ConnectFourEnv

PriorFn = Callable[[list[int]], dict[int, float]]

DEFAULT_C_PUCT = 1.5
DEFAULT_SIMULATIONS = 5000


def uniform_prior(valid: list[int]) -> dict[int, float]:
    if not valid:
        return {}
    p = 1.0 / len(valid)
    return {action: p for action in valid}


def env_from_state(state: np.ndarray) -> ConnectFourEnv:
    """Rebuild an env from an absolute board. Same as minimax."""
    env = ConnectFourEnv()
    env.board = Board(state)
    env.current_player = player_to_move(state)
    env.terminated = False
    env.winner = None
    env.winning_cells = None
    winner, cells = env.board.check_winner()
    if winner is not None:
        env.terminated = True
        env.winner = winner
        env.winning_cells = cells
    elif env.board.is_full():
        env.terminated = True
    return env


def terminal_value(env: ConnectFourEnv, player: int) -> float:
    """+1 if player won, -1 if they lost, 0 on a draw. Also same as minimax."""
    if env.winner == player:
        return 1.0
    elif env.winner == -player:
        return -1.0
    else:
        return 0.0


def rollout(env: ConnectFourEnv, rng: random.Random) -> float:
    """Random legal play to the end. Value is from 'env.current_player' at the start."""
    player = env.current_player
    while not env.terminated:
        valid = env.get_valid_actions()
        env.step(rng.choice(valid))  # 'step' ensures the player to move is updated.
    return terminal_value(env, player)


@dataclass
class MCTSNode:
    to_play: int
    prior: float
    parent: MCTSNode | None = None
    action: int | None = None
    children: dict[int, MCTSNode] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    def q(self) -> float:
        """Mean outcome from the player-to-move at this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def puct(self, parent_visits: int, c_puct: float) -> float:
        """Parent's score for the action that led here.

        q() is the child's to-play (the opponent), so the parent wants -q.
        """
        return -self.q() + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)


def _expand(node: MCTSNode, env: ConnectFourEnv, prior_fn: PriorFn) -> None:
    priors = prior_fn(env.get_valid_actions())
    opponent = -node.to_play
    for action, prior in priors.items():
        node.children[action] = MCTSNode(
            to_play=opponent,
            prior=prior,
            parent=node,
            action=action,
        )


def _select_action(node: MCTSNode, c_puct: float) -> int:
    # Center-first on exact PUCT ties (and smaller index after that).
    return max(
        node.children,
        key=lambda action: (
            node.children[action].puct(node.visits, c_puct),
            -abs(action - 3),
            -action,
        ),
    )


def _backup(node: MCTSNode, value: float) -> None:
    while node is not None:
        node.visits += 1
        node.value_sum += value
        value = -value
        node = node.parent


def count_nodes(node: MCTSNode) -> int:
    return 1 + sum(count_nodes(child) for child in node.children.values())


def run_mcts(
    env: ConnectFourEnv,
    n_simulations: int,
    *,
    c_puct: float = DEFAULT_C_PUCT,
    rng: random.Random | None = None,
    prior_fn: PriorFn | None = None,
    root_actions: list[int] | None = None,
) -> MCTSNode:
    """Grow a tree from 'env' and return the root. Does not mutate 'env'."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1.")
    if env.terminated:
        raise ValueError("Cannot search a terminal position.")

    rng = rng or random.Random()
    prior_fn = prior_fn or uniform_prior
    allowed = None if root_actions is None else set(root_actions)

    def root_prior(valid: list[int]) -> dict[int, float]:
        legal = valid if allowed is None else [action for action in valid if action in allowed]
        return prior_fn(legal)

    root = MCTSNode(to_play=env.current_player, prior=1.0)
    _expand(root, env, root_prior)
    if not root.children:
        raise ValueError("No valid actions.")

    for _ in range(n_simulations):
        node = root
        sim = env.clone()
        while node.children and not sim.terminated:
            action = _select_action(node, c_puct)
            sim.step(action)
            node = node.children[action]
        if sim.terminated:
            value = terminal_value(sim, node.to_play)
        else:
            if not node.children:
                _expand(node, sim, prior_fn)
            value = rollout(sim, rng)
        _backup(node, value)

    return root


def visit_policy(root: MCTSNode, valid_actions: list[int], n_cols: int) -> list[float]:
    visits = [0.0] * n_cols
    for action in valid_actions:
        child = root.children.get(action)
        visits[action] = float(child.visits) if child is not None else 0.0
    total = sum(visits)
    if total <= 0:
        p = 1.0 / len(valid_actions)
        probs = [0.0] * n_cols
        for action in valid_actions:
            probs[action] = p
        return probs
    return [v / total for v in visits]


def most_visited_action(root: MCTSNode, valid_actions: list[int]) -> int:
    return max(
        valid_actions,
        key=lambda action: (
            root.children[action].visits if action in root.children else 0,
            -abs(action - 3),
            -action,
        ),
    )
