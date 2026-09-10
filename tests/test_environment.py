"""Gym-style environment: turns, rewards, clone, canonical state, termination."""

import numpy as np
import pytest

from connect4.env.constants import EMPTY, PLAYER_ONE, PLAYER_TWO, ROWS
from connect4.env.environment import ConnectFourEnv


def test_reset_and_first_player():
    env = ConnectFourEnv()
    state = env.reset()
    assert state.shape == (6, 7)
    assert np.all(state == EMPTY)
    assert env.current_player == PLAYER_ONE
    assert not env.terminated
    assert env.get_valid_actions() == list(range(7))


def test_player_switching():
    env = ConnectFourEnv()
    env.reset()
    env.step(0)
    assert env.current_player == PLAYER_TWO
    env.step(1)
    assert env.current_player == PLAYER_ONE


def test_reward_on_win_and_zero_otherwise():
    env = ConnectFourEnv()
    env.reset()
    # Horizontal P1 win on bottom: 0,1,2,3 with P2 playing 6.
    _, reward, done, info = env.step(0)
    assert reward == 0.0 and not done
    env.step(6)
    env.step(1)
    env.step(6)
    env.step(2)
    env.step(6)
    state, reward, done, info = env.step(3)
    assert done
    assert reward == 1.0
    assert info["winner"] == PLAYER_ONE
    assert info["winning_cells"] is not None
    assert len(info["winning_cells"]) == 4
    assert env.current_player == PLAYER_ONE  # winner does not switch away


def test_illegal_column_and_full_column():
    env = ConnectFourEnv()
    env.reset()
    with pytest.raises(ValueError, match="Illegal"):
        env.step(99)

    env = ConnectFourEnv()
    env.reset()
    for _ in range(ROWS):
        env.step(0)
    assert env.board.is_column_full(0)
    assert 0 not in env.get_valid_actions()
    with pytest.raises(ValueError, match="Illegal"):
        env.step(0)


def test_game_already_finished():
    env = ConnectFourEnv()
    env.reset()
    for col in (0, 6, 1, 6, 2, 6, 3):
        env.step(col)
    assert env.terminated
    with pytest.raises(ValueError, match="already finished"):
        env.step(4)
    assert env.get_valid_actions() == []
    assert not env.is_valid_action(4)


def test_clone_correctness():
    env = ConnectFourEnv()
    env.reset()
    env.step(3)
    clone = env.clone()
    clone.step(4)
    assert env.board.grid[ROWS - 1, 4] == EMPTY
    assert clone.current_player == PLAYER_ONE
    assert env.current_player == PLAYER_TWO
    np.testing.assert_array_equal(env.get_state()[ROWS - 1, 3], PLAYER_ONE)


def test_canonical_representation():
    env = ConnectFourEnv()
    env.reset()
    env.step(3)  # P1 in col 3
    # Now P2 to move: canonical should flip signs so P2 discs are +1
    canonical = env.get_canonical_state()
    raw = env.get_state()
    np.testing.assert_array_equal(canonical, raw * PLAYER_TWO)
    assert canonical[ROWS - 1, 3] == PLAYER_TWO  # opponent from P2's view is -1... wait
    # raw[5,3] = +1 (P1). current_player = -1. canonical = raw * -1 = -1 at (5,3)
    # meaning opponent piece is -1, current player pieces would be +1. Correct:
    # the existing P1 piece is the opponent for P2, so it should be -1.
    assert canonical[ROWS - 1, 3] == -1
    assert env.current_player == PLAYER_TWO


def test_is_draw_false_midgame():
    env = ConnectFourEnv()
    env.reset()
    env.step(3)
    assert not env.is_draw()
