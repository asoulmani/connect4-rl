"""Aggregate win/draw/loss and search-cost metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from connect4.evaluation.match import GameResult, MoveStat, outcome_from_minimax_view


@dataclass
class DepthSummary:
    depth: int
    games: int
    wins: int
    draws: int
    losses: int
    wins_as_p1: int
    games_as_p1: int
    wins_as_p2: int
    games_as_p2: int
    mean_nodes: float
    mean_ms: float
    median_ms: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def win_rate_p1(self) -> float:
        return self.wins_as_p1 / self.games_as_p1 if self.games_as_p1 else 0.0

    @property
    def win_rate_p2(self) -> float:
        return self.wins_as_p2 / self.games_as_p2 if self.games_as_p2 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["win_rate"] = self.win_rate
        d["draw_rate"] = self.draw_rate
        d["win_rate_p1"] = self.win_rate_p1
        d["win_rate_p2"] = self.win_rate_p2
        return d


def summarize_depth(depth: int, games: list[GameResult]) -> DepthSummary:
    wins = draws = losses = 0
    wins_p1 = games_p1 = wins_p2 = games_p2 = 0
    all_moves: list[MoveStat] = []

    for g in games:
        outcome = outcome_from_minimax_view(g)
        if outcome == "win":
            wins += 1
        elif outcome == "draw":
            draws += 1
        else:
            losses += 1

        if g.minimax_seat == 1:
            games_p1 += 1
            if outcome == "win":
                wins_p1 += 1
        else:
            games_p2 += 1
            if outcome == "win":
                wins_p2 += 1

        all_moves.extend(g.move_stats)

    nodes = [m.nodes for m in all_moves]
    times = [m.elapsed_ms for m in all_moves]
    times_sorted = sorted(times)
    mid = len(times_sorted) // 2
    if not times_sorted:
        median_ms = 0.0
    elif len(times_sorted) % 2:
        median_ms = times_sorted[mid]
    else:
        median_ms = 0.5 * (times_sorted[mid - 1] + times_sorted[mid])

    return DepthSummary(
        depth=depth,
        games=len(games),
        wins=wins,
        draws=draws,
        losses=losses,
        wins_as_p1=wins_p1,
        games_as_p1=games_p1,
        wins_as_p2=wins_p2,
        games_as_p2=games_p2,
        mean_nodes=float(sum(nodes) / len(nodes)) if nodes else 0.0,
        mean_ms=float(sum(times) / len(times)) if times else 0.0,
        median_ms=float(median_ms),
    )
