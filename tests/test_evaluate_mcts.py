"""MCTS simulation-budget harness."""

import random

from connect4.agents.heuristic_agent import HeuristicAgent
from connect4.agents.mcts_agent import MCTSAgent
from connect4.env.constants import PLAYER_ONE, PLAYER_TWO
from connect4.evaluation.match import generate_paired_openings, play_game
from connect4.training.evaluate_mcts import (
    format_table,
    make_opponent,
    run_sim_sweep,
)


def test_make_opponent_kinds():
    assert make_opponent("heuristic").name == "heuristic"
    m = make_opponent("minimax", minimax_depth=2)
    assert m.name == "minimax"
    assert m.depth == 2


def test_run_sim_sweep_smoke():
    summaries = run_sim_sweep(
        [8, 16], games_per_budget=4, seed=0, opening_plies=2, opponent="heuristic"
    )
    assert len(summaries) == 2
    assert summaries[0].games == 4
    assert summaries[0].games_as_p1 == 2
    assert summaries[0].games_as_p2 == 2
    assert summaries[0].depth == 8
    assert summaries[1].depth == 16
    assert summaries[0].mean_ms >= 0.0
    assert summaries[0].mean_nodes > 0
    assert "Sims" in format_table(summaries)


def test_run_sim_sweep_uses_same_openings_at_every_budget():
    openings = generate_paired_openings(2, seed=11, opening_plies=3)
    summaries = run_sim_sweep(
        [8, 12], games_per_budget=4, seed=11, opening_plies=3, opponent="heuristic"
    )
    h = HeuristicAgent()
    for n_sim in (8, 12):
        m = MCTSAgent(n_simulations=n_sim, rng=random.Random(0))
        for opening in openings:
            g1 = play_game(m, h, minimax_seat=PLAYER_ONE, opening=opening)
            g2 = play_game(m, h, minimax_seat=PLAYER_TWO, opening=opening)
            assert g1.opening == opening
            assert g2.opening == opening
    assert len(summaries) == 2


def test_play_game_records_mcts_tree_nodes():
    opening = [3, 3, 4, 4]
    result = play_game(
        MCTSAgent(n_simulations=16, rng=random.Random(0)),
        HeuristicAgent(),
        minimax_seat=PLAYER_ONE,
        opening=opening,
    )
    assert result.opening == opening
    assert result.move_stats
    assert all(m.nodes > 0 for m in result.move_stats)
    assert all(m.elapsed_ms >= 0.0 for m in result.move_stats)
