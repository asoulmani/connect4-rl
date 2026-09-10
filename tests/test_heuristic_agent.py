"""HeuristicAgent: win, block, then scored positional play."""

import numpy as np
import pytest

from connect4.agents.heuristic_agent import HeuristicAgent, player_to_move
from connect4.env.constants import PLAYER_ONE, PLAYER_TWO
from connect4.env.environment import ConnectFourEnv


def _play(env: ConnectFourEnv, columns: list[int]) -> None:
    for col in columns:
        env.step(col)


def test_empty_board_prefers_center():
    env = ConnectFourEnv()
    env.reset()
    decision = HeuristicAgent().select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 3
    assert decision.metadata["reason"] == "score"


def test_takes_immediate_win():
    env = ConnectFourEnv()
    env.reset()
    # P1: 0,1,2 on the bottom row; P2 dumps in 6.
    _play(env, [0, 6, 1, 6, 2, 6])
    decision = HeuristicAgent().select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 3
    assert decision.metadata["reason"] == "win"
    env.step(decision.action)
    assert env.winner == PLAYER_ONE


def test_blocks_opponent_win():
    env = ConnectFourEnv()
    env.reset()
    # P1 has no immediate win; P2 occupies bottom 0,1,2 so P1 must block 3.
    _play(env, [6, 0, 6, 1, 4, 2])
    assert env.current_player == PLAYER_ONE
    decision = HeuristicAgent().select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 3
    assert decision.metadata["reason"] == "block"


def test_win_beats_block():
    env = ConnectFourEnv()
    env.reset()
    # P1 has 0,1,2 bottom; P2 has 4,5,6 bottom. P1 to play: win at 3, also would
    # "block" at 3. Either way 3. Build a case where win col != block col.
    # P1 vertical 3-stack in col 0, P2 horizontal threat on row 5 cols 3,4,5.
    # Moves: 0,3, 0,4, 0,5  → P1 can win by 0, or must block 6.
    # Win should take 0.
    _play(env, [0, 3, 0, 4, 0, 5])
    decision = HeuristicAgent().select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 0
    assert decision.metadata["reason"] == "win"


def test_only_legal_moves_and_does_not_mutate_state():
    env = ConnectFourEnv()
    env.reset()
    env.step(3)
    state = env.get_state()
    snapshot = state.copy()
    valid = env.get_valid_actions()
    decision = HeuristicAgent().select_action(state, valid)
    assert decision.action in valid
    np.testing.assert_array_equal(state, snapshot)
    assert pytest.approx(sum(decision.probabilities or [])) == 1.0


def test_player_to_move_from_piece_count():
    empty = np.zeros((6, 7), dtype=np.int8)
    assert player_to_move(empty) == PLAYER_ONE
    empty[5, 3] = PLAYER_ONE
    assert player_to_move(empty) == PLAYER_TWO


def test_avoids_open_ended_three_fork():
    """After X on 3 then 4 with O stacked on 3, do not stack 4: that allows _XXX_."""
    env = ConnectFourEnv()
    env.reset()
    _play(env, [3, 3, 4])
    decision = HeuristicAgent().select_action(env.get_state(), env.get_valid_actions())
    assert decision.action != 4
    env.step(decision.action)
    env.step(5)
    # Human's 5 should not create two immediate wins (fork already prevented).
    from connect4.agents.heuristic_agent import immediate_win_columns

    threats = immediate_win_columns(env.get_state(), PLAYER_ONE)
    assert len(threats) <= 1


def test_hard_rejects_move_that_gifts_immediate_win():
    """Filling under an opponent 3-on-row-4 must score −∞, not a soft penalty."""
    from connect4.agents.heuristic_agent import HARD_LOSS, score_move

    # Row4: OOO_; row5 under them is mixed so dropping col 3 is not our win,
    # but it makes (4,3) playable for O's immediate win.
    grid = np.zeros((6, 7), dtype=np.int8)
    grid[5, 0] = PLAYER_ONE
    grid[5, 1] = PLAYER_TWO
    grid[5, 2] = PLAYER_ONE
    grid[4, 0:3] = PLAYER_TWO
    assert score_move(grid, 3, PLAYER_ONE, PLAYER_TWO) == HARD_LOSS
    assert score_move(grid, 4, PLAYER_ONE, PLAYER_TWO) > HARD_LOSS

    env = ConnectFourEnv()
    env.reset()
    # 6 discs → P1 to move; col 3 would gift P2 a win on row 4.
    env.board.grid = grid.copy()
    env.current_player = PLAYER_ONE
    decision = HeuristicAgent().select_action(env.get_state(), env.get_valid_actions())
    assert decision.action != 3
    assert decision.metadata["reason"] != "win"


def test_prefers_self_fork():
    """`. X . X .` → drop middle creates `_ X X X _` double threat."""
    env = ConnectFourEnv()
    env.reset()
    _play(env, [1, 6, 3, 6])
    decision = HeuristicAgent().select_action(env.get_state(), env.get_valid_actions())
    assert decision.action == 2
    assert decision.metadata["reason"] == "fork"


def test_human_345_sequence_does_not_win_immediately():
    env = ConnectFourEnv()
    env.reset()
    agent = HeuristicAgent()
    for col in (3, 4, 5):
        env.step(col)
        assert env.winner != PLAYER_ONE
        move = agent.select_action(env.get_state(), env.get_valid_actions())
        env.step(move.action)
        assert env.winner != PLAYER_ONE


def test_heuristic_beats_random_as_first_player():
    from connect4.agents.random_agent import RandomAgent
    import random

    wins = 0
    games = 30
    for seed in range(games):
        env = ConnectFourEnv()
        env.reset()
        h = HeuristicAgent()
        r = RandomAgent(rng=random.Random(seed))
        while not env.terminated:
            agent = h if env.current_player == PLAYER_ONE else r
            move = agent.select_action(env.get_state(), env.get_valid_actions())
            env.step(move.action)
        if env.winner == PLAYER_ONE:
            wins += 1
    assert wins >= 20
