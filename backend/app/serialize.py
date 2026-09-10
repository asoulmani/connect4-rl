from backend.app.schemas.game import AgentDecisionSchema, GameState
from backend.app.sessions import GameSession


def to_state(session: GameSession) -> GameState:
    env = session.env
    cells = None
    if env.winning_cells is not None:
        cells = [[r, c] for r, c in env.winning_cells]
    last_ai = None
    if session.last_ai is not None:
        last_ai = AgentDecisionSchema(
            action=session.last_ai.action,
            probabilities=session.last_ai.probabilities,
            value=session.last_ai.value,
            metadata=session.last_ai.metadata,
        )
    return GameState(
        id=session.id,
        board=env.get_state().tolist(),
        current_player=int(env.current_player),
        human_player=session.human_player,
        done=env.terminated,
        winner=env.winner,
        winning_cells=cells,
        valid_actions=env.get_valid_actions(),
        agent_id=session.agent_id,
        agent_depth=session.agent_depth,
        last_action=session.last_action,
        last_ai=last_ai,
        draw=env.terminated and env.winner is None,
    )
