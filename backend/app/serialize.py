from __future__ import annotations

from typing import Optional

from connect4.agents.base import AgentDecision
from connect4.env.environment import ConnectFourEnv

from backend.app.schemas.game import AgentDecisionSchema, GameState


def to_state(
    env: ConnectFourEnv,
    moves: list[int],
    human_player: int,
    agent_id: str,
    agent_depth: Optional[int] = None,
    last_action: Optional[int] = None,
    last_ai: Optional[AgentDecision] = None,
) -> GameState:
    cells = None
    if env.winning_cells is not None:
        cells = [[r, c] for r, c in env.winning_cells]
    last_ai_schema = None
    if last_ai is not None:
        last_ai_schema = AgentDecisionSchema(
            action=last_ai.action,
            probabilities=last_ai.probabilities,
            value=last_ai.value,
            metadata=last_ai.metadata,
        )
    return GameState(
        moves=list(moves),
        board=env.get_state().tolist(),
        current_player=int(env.current_player),
        human_player=human_player,
        done=env.terminated,
        winner=env.winner,
        winning_cells=cells,
        valid_actions=env.get_valid_actions(),
        agent_id=agent_id,
        agent_depth=agent_depth,
        last_action=last_action,
        last_ai=last_ai_schema,
        draw=env.terminated and env.winner is None,
    )
