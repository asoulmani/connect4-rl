from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class NewGameRequest(BaseModel):
    agent_id: str = "random"
    first_player: Literal["human", "ai", "random"] = "human"
    # Minimax only; ignored for other agents.
    depth: Optional[int] = Field(default=None, ge=1, le=8)


class GameContext(BaseModel):
    """Client-owned game identity: move history + seat/agent metadata."""

    moves: list[int] = Field(default_factory=list, max_length=42)
    human_player: Literal[1, -1]
    agent_id: str
    agent_depth: Optional[int] = Field(default=None, ge=1, le=8)


class MoveRequest(GameContext):
    column: int = Field(ge=0, le=6)


class AiMoveRequest(GameContext):
    pass


class AgentDecisionSchema(BaseModel):
    action: int
    probabilities: Optional[list[float]] = None
    value: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GameState(BaseModel):
    moves: list[int]
    board: list[list[int]]
    current_player: int
    human_player: int
    done: bool
    winner: Optional[int] = None
    winning_cells: Optional[list[list[int]]] = None
    valid_actions: list[int]
    agent_id: str
    agent_depth: Optional[int] = None
    last_action: Optional[int] = None
    last_ai: Optional[AgentDecisionSchema] = None
    draw: bool = False
