"""In-memory games. Fine for local demo; not for multi-process production."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Optional

from connect4.agents.base import Agent, AgentDecision
from connect4.agents.heuristic_agent import HeuristicAgent
from connect4.agents.mcts_agent import MCTSAgent
from connect4.agents.minimax_agent import MinimaxAgent
from connect4.agents.random_agent import RandomAgent
from connect4.env.constants import PLAYER_ONE, PLAYER_TWO
from connect4.env.environment import ConnectFourEnv

# One DQN net per process so the demo does not reload weights every game.
_DQN_CACHED: Agent | None = None


class UnknownAgentError(ValueError):
    pass


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


@dataclass
class GameSession:
    id: str
    env: ConnectFourEnv
    agent: Agent
    agent_id: str
    human_player: int
    agent_depth: Optional[int] = None
    last_action: Optional[int] = None
    last_ai: Optional[AgentDecision] = None


_GAMES: dict[str, GameSession] = {}


def create_game(
    agent_id: str, first_player: str, depth: int | None = None
) -> GameSession:
    env = ConnectFourEnv()
    env.reset()
    if first_player == "human":
        human = PLAYER_ONE
    elif first_player == "ai":
        human = PLAYER_TWO
    else:
        human = PLAYER_ONE if random.random() < 0.5 else PLAYER_TWO
    agent = make_agent(agent_id, depth=depth)
    session = GameSession(
        id=str(uuid.uuid4())[:8],
        env=env,
        agent=agent,
        agent_id=agent_id,
        human_player=human,
        agent_depth=getattr(agent, "depth", None),
    )
    if env.current_player != human:
        apply_ai(session)
    _GAMES[session.id] = session
    return session


def get_game(game_id: str) -> GameSession:
    try:
        return _GAMES[game_id]
    except KeyError as exc:
        raise KeyError(game_id) from exc


def apply_human(session: GameSession, column: int) -> None:
    env = session.env
    if env.terminated:
        raise ValueError("Game already finished")
    if env.current_player != session.human_player:
        raise ValueError("Not your turn")
    env.step(column)
    session.last_action = column


def apply_ai(session: GameSession) -> None:
    env = session.env
    if env.terminated:
        return
    if env.current_player == session.human_player:
        raise ValueError("Not the AI's turn")
    decision = session.agent.select_action(env.get_state(), env.get_valid_actions())
    env.step(decision.action)
    session.last_action = decision.action
    session.last_ai = decision
