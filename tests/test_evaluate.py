"""Benchmark harness: minimax instrumentation + depth sweep helpers."""

import random

from connect4.agents.heuristic_agent import HeuristicAgent
from connect4.agents.minimax_agent import MinimaxAgent
from connect4.env.constants import PLAYER_ONE, PLAYER_TWO
from connect4.env.environment import ConnectFourEnv
from connect4.evaluation.match import generate_paired_openings, play_game
from connect4.evaluation.metrics import summarize_depth
from connect4.training.evaluate import format_table, make_opponent, run_depth_sweep


def test_minimax_reports_nodes_and_latency():
    env = ConnectFourEnv()
    env.reset()
    d = MinimaxAgent(depth=2).select_action(env.get_state(), env.get_valid_actions())
    assert d.metadata["nodes"] >= 1
    assert d.metadata["elapsed_ms"] >= 0.0


def test_deeper_search_visits_more_nodes_on_empty_board():
    env = ConnectFourEnv()
    env.reset()
    state = env.get_state()
    valid = env.get_valid_actions()
    n2 = MinimaxAgent(depth=2).select_action(state, valid).metadata["nodes"]
    n3 = MinimaxAgent(depth=3).select_action(state, valid).metadata["nodes"]
    assert n3 > n2


def test_play_game_with_fixed_opening():
    opening = [3, 3, 4, 4]
    r1 = play_game(
        MinimaxAgent(depth=2),
        HeuristicAgent(),
        minimax_seat=PLAYER_ONE,
        opening=opening,
    )
    r2 = play_game(
        MinimaxAgent(depth=2),
        HeuristicAgent(),
        minimax_seat=PLAYER_ONE,
        opening=opening,
    )
    assert r1.opening == opening
    assert r2.opening == opening
    assert r1.winner == r2.winner


def test_run_depth_sweep_uses_same_openings_at_every_depth():
    summaries = run_depth_sweep([1, 3], games_per_depth=4, seed=7, opening_plies=3)
    openings = generate_paired_openings(2, seed=7, opening_plies=3)
    for depth in (1, 3):
        m = MinimaxAgent(depth=depth)
        h = HeuristicAgent()
        for opening in openings:
            g1 = play_game(m, h, minimax_seat=PLAYER_ONE, opening=opening)
            g2 = play_game(m, h, minimax_seat=PLAYER_TWO, opening=opening)
            assert g1.opening == opening
            assert g2.opening == opening
    assert len(summaries) == 2
    assert summaries[0].games == 4


def test_run_depth_sweep_smoke():
    summaries = run_depth_sweep([1, 2], games_per_depth=4, seed=0, opening_plies=2)
    assert len(summaries) == 2
    assert summaries[0].games == 4
    assert summaries[0].games_as_p1 == 2
    assert summaries[0].games_as_p2 == 2
    assert "Win%" in format_table(summaries)


def test_paired_openings_vary_by_index():
    openings = generate_paired_openings(8, seed=0, opening_plies=4)
    assert len(openings) == 8
    assert len({tuple(o) for o in openings}) > 1


def test_make_opponent_kinds():
    assert make_opponent("heuristic").name == "heuristic"
    d1 = make_opponent("d1")
    assert d1.name == "minimax"
    assert d1.depth == 1


def test_run_depth_sweep_d1_smoke():
    summaries = run_depth_sweep(
        [1, 2], games_per_depth=4, seed=0, opening_plies=2, opponent="d1"
    )
    assert len(summaries) == 2
    assert summaries[0].games == 4
    assert summaries[0].games_as_p1 == 2
    assert summaries[0].games_as_p2 == 2


def test_d1_vs_d1_wins_equal_losses():
    """Identical deterministic players + seat swap ⇒ each win is someone else's loss."""
    summaries = run_depth_sweep(
        [1], games_per_depth=8, seed=1, opening_plies=3, opponent="d1"
    )
    s = summaries[0]
    assert s.wins == s.losses
    assert s.wins + s.draws + s.losses == s.games


def test_empty_board_play_has_no_opening():
    r = play_game(
        MinimaxAgent(depth=1),
        HeuristicAgent(),
        minimax_seat=PLAYER_ONE,
        opening=[],
    )
    assert r.opening == []
    assert r.move_stats
    assert all(m.elapsed_ms >= 0.0 for m in r.move_stats)


def test_run_empty_board_sweep_two_seats():
    from connect4.training.evaluate import format_empty_table, run_empty_board_sweep

    rows = run_empty_board_sweep([1], opponent="heuristic")
    assert len(rows) == 1
    assert rows[0]["depth"] == 1
    assert rows[0]["as_p1"] in ("win", "draw", "loss")
    assert rows[0]["as_p2"] in ("win", "draw", "loss")
    assert rows[0]["labelled_moves"] >= 2
    assert "As P1" in format_empty_table(rows)
