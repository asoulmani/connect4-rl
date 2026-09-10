from fastapi import APIRouter

from backend.app.schemas.agent import AgentInfo

router = APIRouter(prefix="/agents", tags=["agents"])

AGENTS = [
    AgentInfo(id="random", name="Random", subtitle="Beginner · uniform legal moves", available=True),
    AgentInfo(id="heuristic", name="Heuristic", subtitle="Intermediate · win / block / center", available=True),
    AgentInfo(id="minimax", name="Minimax", subtitle="Hard · alpha-beta search", available=True),
    AgentInfo(id="mcts", name="MCTS", subtitle="Search · PUCT, random rollouts", available=True),
    AgentInfo(id="dqn", name="RL Agent", subtitle="DQN · replay, target net, ε-greedy", available=True),
    AgentInfo(id="alphazero", name="AlphaZero", subtitle="Expert · coming in Milestone 6", available=False),
]


@router.get("", response_model=list[AgentInfo])
def list_agents() -> list[AgentInfo]:
    return AGENTS
