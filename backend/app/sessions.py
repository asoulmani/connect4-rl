"""Stateless game helpers. No process-level game registry.

DQN weights may be cached per warm process; ConnectFourEnv is never cached.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from connect4.agents.base import Agent, AgentDecision
from connect4.agents.heuristic_agent import HeuristicAgent
from connect4.agents.mcts_agent import MCTSAgent
from connect4.agents.minimax_agent import MinimaxAgent
from connect4.agents.random_agent import RandomAgent
from connect4.env.constants import COLS, PLAYER_ONE, PLAYER_TWO, ROWS
from connect4.env.environment import ConnectFourEnv

MAX_MOVES = ROWS * COLS  # 42

# One DQN net per process so the demo does not reload weights every game.
_DQN_CACHED: Agent | None = None


class UnknownAgentError(ValueError):
    pass


class InvalidHistoryError(ValueError):
    """Client-supplied move history cannot be replayed."""


@dataclass
class PlayedGame:
    """Ephemeral result of one request; never stored across requests."""

    env: ConnectFourEnv
    moves: list[int]
    human_player: int
    agent_id: str
    agent_depth: Optional[int] = None
    last_action: Optional[int] = None
    last_ai: Optional[AgentDecision] = None


def make_agent(agent_id: str, depth: int | None = None) -> Agent:
    if agent_id == "random":
        return RandomAgent()
    if agent_id == "heuristic":
        return HeuristicAgent()
    if agent_id == "minimax":
        return MinimaxAgent(depth=depth if depth is not None else 5)
    if agent_id == "mcts":
        return MCTSAgent()
    if agent_id == "dqn":
        global _DQN_CACHED
        if _DQN_CACHED is None:
            from connect4.agents.dqn.agent import DEFAULT_CHECKPOINT, DQNAgent

            if DEFAULT_CHECKPOINT.is_file():
                _DQN_CACHED = DQNAgent.from_checkpoint(DEFAULT_CHECKPOINT)
            else:
                _DQN_CACHED = DQNAgent()
        return _DQN_CACHED
    raise UnknownAgentError(agent_id)


def rebuild_env(moves: list[int]) -> ConnectFourEnv:
    """Replay move history into a fresh env; validate rather than trust."""
    if len(moves) > MAX_MOVES:
        raise InvalidHistoryError(f"Move history longer than {MAX_MOVES}")

    env = ConnectFourEnv()
    env.reset()

    for column in moves:
        if not isinstance(column, int) or isinstance(column, bool):
            raise InvalidHistoryError(f"Invalid column {column!r}")
        if column < 0 or column >= COLS:
            raise InvalidHistoryError(f"Column out of range: {column}")
        if env.terminated:
            raise InvalidHistoryError("Move history continues after a finished game")
        if column not in env.get_valid_actions():
            raise InvalidHistoryError(f"Illegal move in history: column {column}")
        env.step(column)

    return env


def create_game(
    agent_id: str, first_player: str, depth: int | None = None
) -> PlayedGame:
    env = ConnectFourEnv()
    env.reset()
    if first_player == "human":
        human = PLAYER_ONE
    elif first_player == "ai":
        human = PLAYER_TWO
    else:
        human = PLAYER_ONE if random.random() < 0.5 else PLAYER_TWO

    agent = make_agent(agent_id, depth=depth)
    game = PlayedGame(
        env=env,
        moves=[],
        human_player=human,
        agent_id=agent_id,
        agent_depth=getattr(agent, "depth", None),
    )
    if env.current_player != human:
        _apply_ai(game, agent)
    return game


def apply_human(
    moves: list[int],
    human_player: int,
    agent_id: str,
    agent_depth: int | None,
    column: int,
) -> PlayedGame:
    env = rebuild_env(moves)
    if env.terminated:
        raise ValueError("Game already finished")
    if env.current_player != human_player:
        raise ValueError("Not your turn")
    if column < 0 or column >= COLS or column not in env.get_valid_actions():
        raise ValueError(f"Illegal action {column}, valid={env.get_valid_actions()}.")
    env.step(column)
    return PlayedGame(
        env=env,
        moves=[*moves, column],
        human_player=human_player,
        agent_id=agent_id,
        agent_depth=agent_depth,
        last_action=column,
        last_ai=None,
    )


def apply_ai(
    moves: list[int],
    human_player: int,
    agent_id: str,
    agent_depth: int | None,
) -> PlayedGame:
    env = rebuild_env(moves)
    agent = make_agent(agent_id, depth=agent_depth)
    game = PlayedGame(
        env=env,
        moves=list(moves),
        human_player=human_player,
        agent_id=agent_id,
        agent_depth=getattr(agent, "depth", None),
    )
    _apply_ai(game, agent)
    return game


def _apply_ai(game: PlayedGame, agent: Agent) -> None:
    env = game.env
    if env.terminated:
        raise ValueError("Game already finished")
    if env.current_player == game.human_player:
        raise ValueError("Not the AI's turn")
    decision = agent.select_action(env.get_state(), env.get_valid_actions())
    env.step(decision.action)
    game.moves.append(decision.action)
    game.last_action = decision.action
    game.last_ai = decision
