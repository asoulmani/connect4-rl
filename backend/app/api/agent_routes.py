from fastapi import APIRouter

from backend.app.schemas.agent import AgentInfo

router = APIRouter(prefix="/agents", tags=["agents"])

AGENTS = [
    AgentInfo(id="random", name="Random", subtitle="Uniform Legal Moves", available=True),
    AgentInfo(id="heuristic", name="Heuristic", subtitle="Win / Block / Center", available=True),
    AgentInfo(id="minimax", name="Minimax", subtitle="Alpha-Beta Search", available=True),
    AgentInfo(id="mcts", name="MCTS", subtitle="Search: PUCT, Random Rollouts", available=True),
    AgentInfo(id="dqn", name="RL Agent", subtitle="DQN: Replay, Target Net, ε-greedy", available=True),
    AgentInfo(id="alphazero", name="AlphaZero", subtitle="Coming Soon", available=False),
]


@router.get("", response_model=list[AgentInfo])
def list_agents() -> list[AgentInfo]:
    return AGENTS
