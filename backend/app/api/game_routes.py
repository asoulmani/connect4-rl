from fastapi import APIRouter, HTTPException

from backend.app.schemas.game import GameState, MoveRequest, NewGameRequest
from backend.app.sessions import (
    UnknownAgentError,
    apply_ai,
    apply_human,
    create_game,
    get_game,
)
from backend.app.serialize import to_state

router = APIRouter(prefix="/game", tags=["game"])


@router.post("/new", response_model=GameState)
def new_game(body: NewGameRequest) -> GameState:
    try:
        session = create_game(body.agent_id, body.first_player, depth=body.depth)
    except UnknownAgentError:
        raise HTTPException(status_code=400, detail=f"Agent '{body.agent_id}' is not available yet")
    return to_state(session)


@router.get("/{game_id}", response_model=GameState)
def read_game(game_id: str) -> GameState:
    try:
        session = get_game(game_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Game not found")
    return to_state(session)


@router.post("/{game_id}/move", response_model=GameState)
def human_move(game_id: str, body: MoveRequest) -> GameState:
    try:
        session = get_game(game_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Game not found")
    try:
        apply_human(session, body.column)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return to_state(session)


@router.post("/{game_id}/ai-move", response_model=GameState)
def ai_move(game_id: str) -> GameState:
    try:
        session = get_game(game_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Game not found")
    try:
        apply_ai(session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return to_state(session)
