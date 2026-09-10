"""Text CLI: human vs agent."""

from __future__ import annotations

import argparse
import random
import sys

from connect4.agents.base import Agent
from connect4.agents.heuristic_agent import HeuristicAgent
from connect4.agents.mcts_agent import MCTSAgent
from connect4.agents.minimax_agent import MinimaxAgent
from connect4.agents.random_agent import RandomAgent
from connect4.env.constants import PLAYER_ONE
from connect4.env.environment import ConnectFourEnv


def _prompt_column(valid: list[int]) -> int:
    while True:
        raw = input(f"Your column {valid}: ").strip()
        try:
            col = int(raw)
        except ValueError:
            print("Enter an integer column index.")
            continue
        if col not in valid:
            print("Illegal column.")
            continue
        return col


def _make_agent(name: str, seed: int | None) -> Agent:
    if name == "heuristic":
        return HeuristicAgent()
    if name == "minimax":
        return MinimaxAgent()
    if name == "mcts":
        return MCTSAgent(rng=random.Random(seed) if seed is not None else None)
    if name == "dqn":
        from connect4.agents.dqn.agent import DEFAULT_CHECKPOINT, DQNAgent

        if DEFAULT_CHECKPOINT.is_file():
            return DQNAgent.from_checkpoint(DEFAULT_CHECKPOINT)
        return DQNAgent()
    return RandomAgent(rng=random.Random(seed) if seed is not None else None)


def play(human_first: bool = True, seed: int | None = None, agent_name: str = "random") -> None:
    env = ConnectFourEnv()
    agent = _make_agent(agent_name, seed)
    human_id = PLAYER_ONE if human_first else -PLAYER_ONE

    print(f"Connect Four — you vs {agent.name}. You are X (1) if you move first, else O (-1).")
    print(env.render())
    print()

    while not env.terminated:
        valid = env.get_valid_actions()
        if env.current_player == human_id:
            action = _prompt_column(valid)
        else:
            decision = agent.select_action(env.get_state(), valid)
            action = decision.action
            reason = (decision.metadata or {}).get("reason", "")
            extra = f" ({reason})" if reason else ""
            print(f"{agent.name} plays column {action}{extra}")
        env.step(action)
        print(env.render())
        print()

    if env.winner == human_id:
        print("You win.")
    elif env.winner is None:
        print("Draw.")
    else:
        print(f"{agent.name} wins.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play Connect Four vs a baseline agent")
    parser.add_argument("--ai-first", action="store_true", help="Agent moves first")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--agent",
        choices=["random", "heuristic", "minimax", "mcts", "dqn"],
        default="random",
    )
    )
    args = parser.parse_args(argv)
    try:
        play(human_first=not args.ai_first, seed=args.seed, agent_name=args.agent)
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
