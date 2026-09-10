"""MinimaxAgent: depth-limited alpha-beta over the Connect Four env."""

import random

import numpy as np
import pytest

from connect4.agents.heuristic_agent import HeuristicAgent
from connect4.agents.minimax_agent import (
    MinimaxAgent,
    _child_after,
    _terminal_score,
    best_move_from_scores,
    evaluate,
    root_move_scores,
)
from connect4.agents.random_agent import RandomAgent
from connect4.env.constants import PLAYER_ONE, PLAYER_TWO
from connect4.env.environment import ConnectFourEnv


def _play(env: ConnectFourEnv, columns: list[int]) -> None:
    for col in columns:
        env.step(col)


def test_empty_board_prefers_center():
    env = ConnectFourEnv()
    env.reset()
    decision = MinimaxAgent(depth=4).select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 3
    assert decision.metadata["policy"] == "minimax"
    assert decision.value is not None


def test_takes_immediate_win():
    env = ConnectFourEnv()
    env.reset()
    _play(env, [0, 6, 1, 6, 2, 6])
    decision = MinimaxAgent(depth=3).select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 3
    env.step(decision.action)
    assert env.winner == PLAYER_ONE
    assert decision.value == 1.0


def test_blocks_opponent_win():
    env = ConnectFourEnv()
    env.reset()
    _play(env, [6, 0, 6, 1, 4, 2])
    assert env.current_player == PLAYER_ONE
    decision = MinimaxAgent(depth=3).select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 3


def test_win_beats_block():
    env = ConnectFourEnv()
    env.reset()
    _play(env, [0, 3, 0, 4, 0, 5])
    decision = MinimaxAgent(depth=3).select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 0
    env.step(decision.action)
    assert env.winner == PLAYER_ONE


def test_only_legal_moves_and_does_not_mutate_state():
    env = ConnectFourEnv()
    env.reset()
    env.step(3)
    state = env.get_state()
    snapshot = state.copy()
    valid = env.get_valid_actions()
    decision = MinimaxAgent(depth=3).select_action(state, valid)
    assert decision.action in valid
    np.testing.assert_array_equal(state, snapshot)
    assert pytest.approx(sum(decision.probabilities or [])) == 1.0
    assert env.current_player == PLAYER_TWO


def test_depth_must_be_positive():
    with pytest.raises(ValueError):
        MinimaxAgent(depth=0)


def test_evaluate_symmetric_empty():
    empty = np.zeros((6, 7), dtype=np.int8)
    assert evaluate(empty, PLAYER_ONE) == 0.0


def test_evaluate_is_zero_sum():
    grid = np.zeros((6, 7), dtype=np.int8)
    grid[5, 3] = PLAYER_ONE
    grid[5, 2] = PLAYER_TWO
    grid[4, 3] = PLAYER_ONE
    assert evaluate(grid, PLAYER_ONE) == pytest.approx(-evaluate(grid, PLAYER_TWO))


def test_root_scores_cover_all_legal_actions():
    env = ConnectFourEnv()
    env.reset()
    valid = env.get_valid_actions()
    decision = MinimaxAgent(depth=3).select_action(env.get_state(), valid)
    assert set(decision.metadata["scores"]) == {str(c) for c in valid}
    assert decision.metadata["nodes"] >= len(valid)


def test_alphabeta_matches_reference_on_random_positions():
    rng = random.Random(42)
    for trial in range(80):
        env = ConnectFourEnv()
        env.reset()
        for _ in range(rng.randint(0, 10)):
            if env.terminated:
                break
            env.step(rng.choice(env.get_valid_actions()))
        if env.terminated:
            continue
        state = env.get_state()
        valid = env.get_valid_actions()
        for depth in (1, 2, 3):
            ab = root_move_scores(state, valid, depth, use_alphabeta=True)
            ref = root_move_scores(state, valid, depth, use_alphabeta=False)
            for col in valid:
                assert ab[col] == pytest.approx(ref[col]), f"trial={trial} depth={depth} col={col}"


def test_alphabeta_action_matches_reference():
    env = ConnectFourEnv()
    env.reset()
    for col in (3, 3, 4, 4):
        env.step(col)
    state = env.get_state()
    valid = env.get_valid_actions()
    for depth in (2, 3, 4):
        ab = root_move_scores(state, valid, depth, use_alphabeta=True)
        ref = root_move_scores(state, valid, depth, use_alphabeta=False)
        assert best_move_from_scores(ab, valid) == best_move_from_scores(ref, valid)


def test_depth_two_blocks_opponent_win():
    env = ConnectFourEnv()
    env.reset()
    _play(env, [6, 0, 6, 1, 4, 2])
    decision = MinimaxAgent(depth=2).select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 3


def test_terminal_score_prefers_faster_win_and_slower_loss():
    env = ConnectFourEnv()
    env.reset()
    env.step(0)
    env.step(1)
    env.step(0)
    env.step(1)
    env.step(0)
    env.step(1)
    win_now = _child_after(env, 0)
    assert win_now.winner == PLAYER_ONE
    assert _terminal_score(win_now, PLAYER_ONE, ply=1) > _terminal_score(
        win_now, PLAYER_ONE, ply=5
    )
    assert _terminal_score(win_now, PLAYER_TWO, ply=5) > _terminal_score(
        win_now, PLAYER_TWO, ply=1
    )


def test_evaluate_zero_sum_on_random_boards():
    rng = random.Random(0)
    for _ in range(50):
        grid = np.zeros((6, 7), dtype=np.int8)
        for _ in range(rng.randint(0, 20)):
            empties = np.argwhere(grid == 0)
            if len(empties) == 0:
                break
            r, c = empties[rng.randint(0, len(empties) - 1)]
            grid[r, c] = PLAYER_ONE if rng.random() < 0.5 else PLAYER_TWO
        assert evaluate(grid, PLAYER_ONE) == pytest.approx(-evaluate(grid, PLAYER_TWO))


def test_minimax_beats_random_as_first_player():
    wins = 0
    games = 20
    for seed in range(games):
        env = ConnectFourEnv()
        env.reset()
        m = MinimaxAgent(depth=4)
        r = RandomAgent(rng=random.Random(seed))
        while not env.terminated:
            agent = m if env.current_player == PLAYER_ONE else r
            move = agent.select_action(env.get_state(), env.get_valid_actions())
            env.step(move.action)
        if env.winner == PLAYER_ONE:
            wins += 1
    assert wins >= 18


def test_minimax_beats_heuristic_head_to_head():
    """Depth-4 search should beat the shallow heuristic most of the time as P1."""
    wins = 0
    games = 10
    for _ in range(games):
        env = ConnectFourEnv()
        env.reset()
        m = MinimaxAgent(depth=4)
        h = HeuristicAgent()
        while not env.terminated:
            agent = m if env.current_player == PLAYER_ONE else h
            move = agent.select_action(env.get_state(), env.get_valid_actions())
            env.step(move.action)
        if env.winner == PLAYER_ONE:
            wins += 1
    assert wins >= 7
