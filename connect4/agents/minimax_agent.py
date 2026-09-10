"""Depth limited minimax with alpha-beta pruning.

Searches the game tree from the absolute board passed to 'select_action'.
Leaves use a static window/center evaluation (same features as the heuristic
baseline). Terminal outcomes dominate any leaf score.
"""

from __future__ import annotations

import math
import time

import numpy as np

from connect4.agents.base import Agent, AgentDecision
from connect4.agents.heuristic_agent import (
    CENTER_SCORE,
    SCORE_THREE,
    SCORE_TWO,
    count_windows,
    player_to_move,
)
from connect4.env.constants import COLS
from connect4.env.board import Board
from connect4.env.environment import ConnectFourEnv

WIN_SCORE = 1_000_000


def evaluate(grid: np.ndarray, me: int) -> float:
    """Static score from me's perspective. Higher is better for 'me'.

    Zero-sum: 'evaluate(s, me) == -evaluate(s, -me)'.
    """
    opp = -me
    my_threes, my_twos = count_windows(grid, me)
    opp_threes, opp_twos = count_windows(grid, opp)
    center = 0.0
    for col in range(COLS):
        my_pieces = int(np.sum(grid[:, col] == me))
        opp_pieces = int(np.sum(grid[:, col] == opp))
        center += CENTER_SCORE[col] * (my_pieces - opp_pieces)
    return float(
        SCORE_THREE * (my_threes - opp_threes)
        + SCORE_TWO * (my_twos - opp_twos)
        + center
    )


def _ordered_moves(valid: list[int]) -> list[int]:
    """Center-first ordering improves alpha-beta cutoffs."""
    return sorted(valid, key=lambda c: (abs(c - 3), c))


def _env_from_state(state: np.ndarray) -> ConnectFourEnv:
    """Create a ConnectFourEnv from a numpy array state to reuse existing environment methods."""
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


def reference_minimax(
    env: ConnectFourEnv,
    depth: int,
    maximizing: bool,
    root_player: int,
    ply: int,
) -> float:
    """Naive minimax (no pruning). Must match '_alphabeta' for correctness tests."""
    if env.terminated:
        return _terminal_score(env, root_player, ply)
    if depth == 0:
        return evaluate(env.get_state(), root_player)

    moves = _ordered_moves(env.get_valid_actions())

    # Maximizing.
    if maximizing:
        return max(
            reference_minimax(
                _child_after(env, col), depth - 1, False, root_player, ply + 1
            )
            for col in moves
        )
    
    # Minimizing.
    return min(
        reference_minimax(
            _child_after(env, col), depth - 1, True, root_player, ply + 1
        )
        for col in moves
    )


def _child_after(env: ConnectFourEnv, col: int) -> ConnectFourEnv:
    child = env.clone()
    child.step(col)
    return child


def _alphabeta(
    env: ConnectFourEnv,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    root_player: int,
    ply: int,
    stats: dict[str, int],
) -> float:
    stats["nodes"] += 1

    if env.terminated:
        return _terminal_score(env, root_player, ply)
    if depth == 0:
        return evaluate(env.get_state(), root_player)

    moves = _ordered_moves(env.get_valid_actions())

    # Maximizing.
    if maximizing:
        best = -math.inf
        for col in moves:
            child = env.clone()
            child.step(col)
            score = _alphabeta(
                child, depth - 1, alpha, beta, False, root_player, ply + 1, stats
            )
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    # Minimizing.
    best = math.inf
    for col in moves:
        child = env.clone()
        child.step(col)
        score = _alphabeta(
            child, depth - 1, alpha, beta, True, root_player, ply + 1, stats
        )
        if score < best:
            best = score
        if best < beta:
            beta = best
        if alpha >= beta:
            break
    return best


def _terminal_score(env: ConnectFourEnv, root_player: int, ply: int) -> float:
    if env.winner == root_player:
        return WIN_SCORE - ply
    if env.winner == -root_player:
        return -WIN_SCORE + ply
    return 0.0


def root_move_scores(
    state: np.ndarray,
    valid_actions: list[int],
    depth: int,
    *,
    use_alphabeta: bool = True,
    stats: dict[str, int] | None = None,
) -> dict[int, float]:
    """Exact depth-limited minimax value per root column (analysis / tests)."""
    root = _env_from_state(state)
    me = root.current_player
    scores: dict[int, float] = {}
    for col in _ordered_moves(valid_actions):
        child = _child_after(root, col)
        if use_alphabeta:
            counter = stats if stats is not None else {"nodes": 0}
            scores[col] = _alphabeta(
                child, depth - 1, -math.inf, math.inf, False, me, ply=1, stats=counter
            )
        else:
            scores[col] = reference_minimax(child, depth - 1, False, me, ply=1)
    return scores


def best_move_from_scores(scores: dict[int, float], valid_actions: list[int]) -> int:
    best = max(scores.values())
    tied = [c for c in valid_actions if scores[c] == best]
    return min(tied, key=lambda c: (abs(c - 3), c))


class MinimaxAgent(Agent):
    name = "minimax"

    def __init__(self, depth: int = 5) -> None:
        if depth < 1:
            raise ValueError("Depth must be >= 1.")
        self.depth = depth

    def select_action(self, state: np.ndarray, valid_actions: list[int]) -> AgentDecision:
        if not valid_actions:
            raise ValueError("No valid actions.")

        t0 = time.perf_counter()
        stats: dict[str, int] = {"nodes": 1}
        scores = root_move_scores(
            state, valid_actions, self.depth, use_alphabeta=True, stats=stats
        )
        best_action = best_move_from_scores(scores, valid_actions)
        best_score = scores[best_action]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return AgentDecision(
            action=best_action,
            probabilities=_scores_to_probs(scores, valid_actions),
            value=_score_to_value(best_score),
            metadata={
                "policy": "minimax",
                "depth": self.depth,
                "seat": _env_from_state(state).current_player,
                "scores": {str(c): float(s) for c, s in scores.items()},
                "best_score": float(best_score),
                "nodes": int(stats["nodes"]),
                "elapsed_ms": float(elapsed_ms),
            },
        )


def _score_to_value(score: float) -> float:
    if score >= WIN_SCORE / 2:
        return 1.0
    if score <= -WIN_SCORE / 2:
        return -1.0
    return float(np.tanh(score / 200.0))


def _scores_to_probs(scores: dict[int, float], valid_actions: list[int]) -> list[float]:
    xs = np.full(COLS, -1e9, dtype=np.float64)
    for col, score in scores.items():
        # Cap extremes so softmax stays numeric.
        xs[col] = float(np.clip(score, -WIN_SCORE, WIN_SCORE) / 50.0)
    shifted = xs - np.max(xs)
    exp = np.exp(shifted)
    for col in range(COLS):
        if col not in valid_actions:
            exp[col] = 0.0
    total = float(exp.sum())
    if total <= 0:
        probs = [0.0] * COLS
        probs[valid_actions[0]] = 1.0
        return probs
    return (exp / total).tolist()
