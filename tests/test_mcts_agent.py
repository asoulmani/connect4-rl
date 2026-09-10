"""MCTSAgent: uniform-prior PUCT with random rollouts."""

import random

import numpy as np
import pytest

from connect4.agents.mcts_agent import MCTSAgent
from connect4.env.constants import PLAYER_ONE, PLAYER_TWO
from connect4.env.environment import ConnectFourEnv


def _play(env: ConnectFourEnv, columns: list[int]) -> None:
    for col in columns:
        env.step(col)


def _agent(n_simulations: int = 128, seed: int = 0) -> MCTSAgent:
    return MCTSAgent(n_simulations=n_simulations, rng=random.Random(seed))


def test_takes_immediate_win():
    env = ConnectFourEnv()
    _play(env, [0, 6, 1, 6, 2, 6])
    decision = _agent(64).select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 3
    visits = decision.metadata["visits"]
    assert visits["3"] == max(visits.values())
    env.step(decision.action)
    assert env.winner == PLAYER_ONE
    assert decision.value is not None
    assert decision.value > 0.5


def test_takes_immediate_win_not_center():
    """Vertical win in column 0 — not the center-first PUCT tie-break."""
    env = ConnectFourEnv()
    _play(env, [0, 3, 0, 4, 0, 5])
    decision = _agent(64).select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 0
    visits = decision.metadata["visits"]
    assert visits["0"] == max(visits.values())
    env.step(decision.action)
    assert env.winner == PLAYER_ONE


def test_blocks_opponent_win():
    env = ConnectFourEnv()
    _play(env, [6, 0, 6, 1, 4, 2])
    assert env.current_player == PLAYER_ONE
    decision = _agent(256).select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 3


def test_avoids_hanging_move():
    """Filling under an opponent three gifts an instant win. Tree terminals should punish it."""
    grid = np.zeros((6, 7), dtype=np.int8)
    grid[5, 0] = PLAYER_ONE
    grid[5, 1] = PLAYER_TWO
    grid[5, 2] = PLAYER_ONE
    grid[4, 0:3] = PLAYER_TWO
    env = ConnectFourEnv()
    env.board.grid = grid.copy()
    env.current_player = PLAYER_ONE
    decision = _agent(256).select_action(env.get_state(), env.get_valid_actions())
    assert decision.action != 3
    visits = decision.metadata["visits"]
    assert visits["3"] < max(visits.values())


def test_only_legal_moves_and_does_not_mutate_state():
    env = ConnectFourEnv()
    env.step(3)
    state = env.get_state()
    snapshot = state.copy()
    valid = env.get_valid_actions()
    decision = _agent(32).select_action(state, valid)
    assert decision.action in valid
    np.testing.assert_array_equal(state, snapshot)
    assert env.current_player == PLAYER_TWO
    assert pytest.approx(sum(decision.probabilities or [])) == 1.0
    assert decision.metadata["policy"] == "mcts_puct"
    assert decision.metadata["prior"] == "uniform"
    assert "tree_nodes" in decision.metadata
    assert "nodes" not in decision.metadata


def test_visit_counts_sum_to_simulations():
    env = ConnectFourEnv()
    n_sim = 80
    decision = _agent(n_sim).select_action(env.get_state(), env.get_valid_actions())
    assert sum(decision.metadata["visits"].values()) == n_sim


def test_fresh_tree_each_call():
    env = ConnectFourEnv()
    agent = _agent(40, seed=1)
    first = agent.select_action(env.get_state(), env.get_valid_actions())
    env.step(first.action)
    second = agent.select_action(env.get_state(), env.get_valid_actions())
    assert second.metadata["n_simulations"] == 40
    assert sum(second.metadata["visits"].values()) == 40


def test_empty_actions_raises():
    with pytest.raises(ValueError):
        _agent(8).select_action(np.zeros((6, 7), dtype=np.int8), [])


def test_simulations_must_be_positive():
    with pytest.raises(ValueError):
        MCTSAgent(n_simulations=0)
