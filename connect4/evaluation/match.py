"""Head-to-head match utilities for agent benchmarks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from connect4.agents.base import Agent
from connect4.env.constants import PLAYER_ONE
from connect4.env.environment import ConnectFourEnv


@dataclass
class MoveStat:
    nodes: int
    elapsed_ms: float


@dataclass
class GameResult:
    winner: Optional[int]  # PLAYER_ONE, PLAYER_TWO, or None (draw)
    minimax_seat: int
    move_stats: list[MoveStat] = field(default_factory=list)
    opening: list[int] = field(default_factory=list)


def generate_opening(rng: random.Random, opening_plies: int) -> list[int]:
    """Random legal opening shared across benchmark depths/seats."""
    env = ConnectFourEnv()
    env.reset()
    opening: list[int] = []
    for _ in range(opening_plies):
        if env.terminated:
            break
        col = rng.choice(env.get_valid_actions())
        env.step(col)
        opening.append(col)
    return opening


def generate_paired_openings(
    n_pairs: int, *, seed: int = 0, opening_plies: int = 4
) -> list[list[int]]:
    """'n_pairs' openings; each is played with minimax as P1 and as P2."""
    return [
        generate_opening(random.Random(seed + i), opening_plies) for i in range(n_pairs)
    ]


def play_game(
    minimax: Agent,
    opponent: Agent,
    *,
    minimax_seat: int = PLAYER_ONE,
    rng: random.Random | None = None,
    opening_plies: int = 4,
    opening: list[int] | None = None,
) -> GameResult:
    """Play one game after a fixed or random opening.

    Pass 'opening' for controlled benchmarks (same positions at every depth).
    Collects search cost / latency for the focal agent's moves
    (minimax ``nodes`` or MCTS ``tree_nodes``).
    """
    env = ConnectFourEnv()
    env.reset()
    if opening is not None:
        played = list(opening)
    else:
        rng = rng or random.Random()
        played = generate_opening(rng, opening_plies)
    for col in played:
        if env.terminated:
            break
        env.step(col)

    stats: list[MoveStat] = []
    while not env.terminated:
        state = env.get_state()
        valid = env.get_valid_actions()
        if env.current_player == minimax_seat:
            decision = minimax.select_action(state, valid)
            meta = decision.metadata or {}
            if "nodes" in meta:
                n_nodes = int(meta["nodes"])
            elif "tree_nodes" in meta:
                n_nodes = int(meta["tree_nodes"])
            else:
                n_nodes = 0
            stats.append(
                MoveStat(
                    nodes=n_nodes,
                    elapsed_ms=float(meta.get("elapsed_ms", 0.0)),
                )
            )
            action = decision.action
        else:
            action = opponent.select_action(state, valid).action
        env.step(action)

    return GameResult(
        winner=env.winner,
        minimax_seat=minimax_seat,
        move_stats=stats,
        opening=played,
    )


def outcome_from_minimax_view(result: GameResult) -> str:
    if result.winner is None:
        return "draw"
    if result.winner == result.minimax_seat:
        return "win"
    return "loss"
