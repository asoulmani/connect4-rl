"""RandomAgent stays in the legal-move set."""

import random

import numpy as np
import pytest

from connect4.agents.random_agent import RandomAgent
from connect4.env.environment import ConnectFourEnv


def test_random_agent_only_plays_legal_moves():
    env = ConnectFourEnv()
    env.reset()
    agent = RandomAgent(rng=random.Random(0))
    while not env.terminated:
        valid = env.get_valid_actions()
        decision = agent.select_action(env.get_state(), valid)
        assert decision.action in valid
        assert decision.probabilities is not None
        assert pytest.approx(sum(decision.probabilities)) == 1.0
        env.step(decision.action)


def test_random_agent_empty_actions_raises():
    agent = RandomAgent()
    with pytest.raises(ValueError):
        agent.select_action(np.zeros((6, 7), dtype=np.int8), [])


def test_two_random_agents_always_terminate():
    env = ConnectFourEnv()
    env.reset()
    a = RandomAgent(rng=random.Random(1))
    b = RandomAgent(rng=random.Random(2))
    steps = 0
    while not env.terminated:
        agent = a if env.current_player == 1 else b
        move = agent.select_action(env.get_state(), env.get_valid_actions())
        env.step(move.action)
        steps += 1
        assert steps <= 42
    assert env.terminated
    assert env.winner in (1, -1, None)
