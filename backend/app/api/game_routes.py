from fastapi import APIRouter, HTTPException

from backend.app.schemas.game import AiMoveRequest, GameState, MoveRequest, NewGameRequest
from backend.app.serialize import to_state
from backend.app.sessions import (
    InvalidHistoryError,
    UnknownAgentError,
    apply_ai,
    apply_human,
    create_game,
)

router = APIRouter(prefix="/game", tags=["game"])


def _state_from_played(game) -> GameState:
    return to_state(
        env=game.env,
        moves=game.moves,
        human_player=game.human_player,
        agent_id=game.agent_id,
        agent_depth=game.agent_depth,
        last_action=game.last_action,
        last_ai=game.last_ai,
    )


@router.post("/new", response_model=GameState)
def new_game(body: NewGameRequest) -> GameState:
    try:
        game = create_game(body.agent_id, body.first_player, depth=body.depth)
    except UnknownAgentError:
        raise HTTPException(
            status_code=400,
            detail=f"Agent '{body.agent_id}' is not available yet",
        )
    return _state_from_played(game)


@router.post("/move", response_model=GameState)
def human_move(body: MoveRequest) -> GameState:
    try:
        game = apply_human(
            moves=body.moves,
            human_player=body.human_player,
            agent_id=body.agent_id,
            agent_depth=body.agent_depth,
            column=body.column,
        )
    except InvalidHistoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnknownAgentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _state_from_played(game)


@router.post("/ai-move", response_model=GameState)
def ai_move(body: AiMoveRequest) -> GameState:
    try:
        game = apply_ai(
            moves=body.moves,
            human_player=body.human_player,
            agent_id=body.agent_id,
            agent_depth=body.agent_depth,
        )
    except InvalidHistoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnknownAgentError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Agent '{body.agent_id}' is not available yet",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _state_from_played(game)
