"""Minimax depth sweeps.

Two opponents, two questions:

- ``heuristic`` — can search beat a 1-ply tactician? The heuristic still
  does win/block/fork; the minimax leaf does not.
- ``d1`` — same ``evaluate()``, same search code, only depth changes.
  Opponent is ``MinimaxAgent(depth=1)``. Depth 1 vs itself is the control.

Example:
    python -m connect4.training.evaluate --opponent d1 --games 100 --depths 1,2,3,4,5
    python -m connect4.training.evaluate --empty-board --opponent heuristic
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from connect4.agents.base import Agent
from connect4.agents.heuristic_agent import HeuristicAgent
from connect4.agents.minimax_agent import MinimaxAgent
from connect4.env.constants import PLAYER_ONE, PLAYER_TWO
from connect4.evaluation.match import GameResult, generate_paired_openings, play_game, outcome_from_minimax_view
from connect4.evaluation.metrics import DepthSummary, summarize_depth

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"

OPPONENTS = ("heuristic", "d1")
DEFAULT_OUT = {
    "heuristic": RESULTS_DIR / "minimax_vs_heuristic",
    "d1": RESULTS_DIR / "minimax_vs_d1",
}


def make_opponent(kind: str) -> Agent:
    if kind == "heuristic":
        return HeuristicAgent()
    if kind == "d1":
        return MinimaxAgent(depth=1)
    raise ValueError(f"Unknown opponent {kind!r}; expected one of {OPPONENTS}")


def run_depth_sweep(
    depths: list[int],
    games_per_depth: int,
    *,
    seed: int = 0,
    opening_plies: int = 4,
    opponent: str = "heuristic",
) -> list[DepthSummary]:
    """Paired openings: every depth sees the same positions and seats.

    For opening ``i``: the deeper agent plays as P1, then the same opening
    as P2.
    """
    if games_per_depth < 2:
        raise ValueError("games_per_depth must be >= 2 (need both seats)")
    if games_per_depth % 2:
        raise ValueError("games_per_depth must be even so seats are balanced")

    n_pairs = games_per_depth // 2
    openings = generate_paired_openings(n_pairs, seed=seed, opening_plies=opening_plies)
    opp = make_opponent(opponent)
    summaries: list[DepthSummary] = []

    for depth in depths:
        minimax = MinimaxAgent(depth=depth)
        games = []
        for i, opening in enumerate(openings):
            games.append(
                play_game(
                    minimax, opp, minimax_seat=PLAYER_ONE, opening=opening
                )
            )
            games.append(
                play_game(
                    minimax, opp, minimax_seat=PLAYER_TWO, opening=opening
                )
            )
            done = (i + 1) * 2
            if done == games_per_depth or done % max(2, games_per_depth // 4) == 0:
                print(
                    f"  depth {depth} vs {opponent}: {done}/{games_per_depth} games",
                    flush=True,
                )
        summaries.append(summarize_depth(depth, games))
    return summaries


def _seat_outcome(result: GameResult) -> str:
    return outcome_from_minimax_view(result)


def _move_means(result: GameResult) -> tuple[float, float, float]:
    stats = result.move_stats
    if not stats:
        return 0.0, 0.0, 0.0
    nodes = [m.nodes for m in stats]
    times = [m.elapsed_ms for m in stats]
    times_sorted = sorted(times)
    mid = len(times_sorted) // 2
    if len(times_sorted) % 2:
        median_ms = times_sorted[mid]
    else:
        median_ms = 0.5 * (times_sorted[mid - 1] + times_sorted[mid])
    return (
        float(sum(nodes) / len(nodes)),
        float(sum(times) / len(times)),
        float(median_ms),
    )


def run_empty_board_sweep(
    depths: list[int],
    *,
    opponent: str = "heuristic",
) -> list[dict]:
    """One empty-board game per seat, per depth. Deterministic; n = 2."""
    opp = make_opponent(opponent)
    rows: list[dict] = []
    for depth in depths:
        agent = MinimaxAgent(depth=depth)
        as_p1 = play_game(agent, opp, minimax_seat=PLAYER_ONE, opening=[])
        as_p2 = play_game(agent, opp, minimax_seat=PLAYER_TWO, opening=[])
        nodes_p1, ms_p1, med_p1 = _move_means(as_p1)
        nodes_p2, ms_p2, med_p2 = _move_means(as_p2)
        all_stats = as_p1.move_stats + as_p2.move_stats
        all_nodes = [m.nodes for m in all_stats]
        all_ms = [m.elapsed_ms for m in all_stats]
        row = {
            "depth": depth,
            "as_p1": _seat_outcome(as_p1),
            "as_p2": _seat_outcome(as_p2),
            "p1_winner": None if as_p1.winner is None else int(as_p1.winner),
            "p2_winner": None if as_p2.winner is None else int(as_p2.winner),
            "mean_nodes": float(sum(all_nodes) / len(all_nodes)) if all_nodes else 0.0,
            "mean_ms": float(sum(all_ms) / len(all_ms)) if all_ms else 0.0,
            "median_ms": float(sorted(all_ms)[len(all_ms) // 2]) if all_ms else 0.0,
            "as_p1_mean_ms": ms_p1,
            "as_p2_mean_ms": ms_p2,
            "as_p1_mean_nodes": nodes_p1,
            "as_p2_mean_nodes": nodes_p2,
            "as_p1_median_ms": med_p1,
            "as_p2_median_ms": med_p2,
            "labelled_moves": len(all_stats),
        }
        rows.append(row)
        print(
            f"  empty depth {depth} vs {opponent}: "
            f"P1={row['as_p1']} P2={row['as_p2']} "
            f"{row['mean_ms']:.1f} ms/move",
            flush=True,
        )
    return rows


def format_empty_table(rows: list[dict]) -> str:
    lines = [
        "| Depth | As P1 | As P2 | Nodes/move | Mean ms | Median ms |",
        "|------:|------:|------:|-----------:|--------:|----------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['depth']} | {r['as_p1']} | {r['as_p2']} | "
            f"{r['mean_nodes']:,.0f} | {r['mean_ms']:.1f} | {r['median_ms']:.1f} |"
        )
    return "\n".join(lines)


def write_empty_markdown(path: Path, rows: list[dict], meta: dict) -> None:
    opponent = meta.get("opponent", "heuristic")
    vs = "Minimax(d=1)" if opponent == "d1" else "heuristic"
    body = "\n".join(
        [
            f"# Minimax vs {vs} — empty board (sanity check)",
            "",
            "Deterministic agents, empty board, both seats. Two games per depth, "
            "not a sample. Game-theoretic Connect Four: first player wins with "
            "perfect play (center). This table is a labeled sanity line, not a "
            "win-rate estimate.",
            "",
            f"- Opponent: **{vs}**",
            f"- Depths: `{meta['depths']}`",
            f"- Generated: `{meta['generated_at']}`",
            "",
            format_empty_table(rows),
            "",
            "Latency is mean ms per labelled minimax move, averaged over both seats.",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def format_table(summaries: list[DepthSummary]) -> str:
    lines = [
        "| Depth | Win% | Draw% | P1 win% | P2 win% | Nodes/move | Latency |",
        "|------:|-----:|------:|--------:|--------:|-----------:|--------:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.depth} | {100 * s.win_rate:.1f}% | {100 * s.draw_rate:.1f}% | "
            f"{100 * s.win_rate_p1:.1f}% | {100 * s.win_rate_p2:.1f}% | "
            f"{s.mean_nodes:,.0f} | {s.mean_ms:.1f} ms |"
        )
    return "\n".join(lines)


def _copy_and_takeaway(opponent: str) -> tuple[str, str, str]:
    if opponent == "d1":
        title = "Minimax vs Minimax(d=1) — same evaluator, depth changes"
        blurb = (
            "Opponent is `MinimaxAgent(depth=1)`. Same `evaluate()`, same search, "
            "only the depth integer changes. Depth 1 vs itself is the control: "
            "identical players, so wins and losses should split (draws aside)."
        )
        takeaway = (
            "If win rate rises with depth here, that gain is lookahead — the "
            "leaf did not change. Depth 1 vs d=1 should sit near 50% (or be "
            "all draws). If most of the lift is already at depth 2, later "
            "plies are mostly cost. Compare to `--opponent heuristic`, which "
            "asks whether search beats a tactician."
        )
        return title, blurb, takeaway

    title = "Minimax vs Heuristic — search vs a 1-ply tactician"
    blurb = (
        "Minimax against the full heuristic (win / block / fork), not against "
        "its own leaf. The window/center features are shared; the tactics are "
        "not. Depth 1 is a weak 1-ply scorer. Depth 2 is the first ply that "
        "can see a block."
    )
    takeaway = (
        "Depth 1 should lose: it cannot see blocks. The jump at depth 2 is "
        "mostly those missing tactics. Later depths are extra lookahead "
        "against someone who already knows the cheap stuff. For a clean "
        "depth ablation see `--opponent d1`."
    )
    return title, blurb, takeaway


def write_markdown(path: Path, summaries: list[DepthSummary], meta: dict) -> None:
    opponent = meta.get("opponent", "heuristic")
    title, blurb, takeaway = _copy_and_takeaway(opponent)
    body = "\n".join(
        [
            f"# {title}",
            "",
            blurb,
            "",
            f"- Games per depth: **{meta['games_per_depth']}** "
            f"({meta['paired_openings']} paired openings × 2 seats)",
            f"- Random opening plies: **{meta.get('opening_plies', 4)}** "
            "(same opening at every depth)",
            f"- Depths: `{meta['depths']}`",
            f"- Generated: `{meta['generated_at']}`",
            "",
            format_table(summaries),
            "",
            "## Takeaway",
            "",
            takeaway,
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def write_plot(path: Path, summaries: list[DepthSummary], *, opponent: str) -> bool:
    """Dark-lab style dual panel. Returns False if matplotlib is missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    depths = [s.depth for s in summaries]
    win = [100 * s.win_rate for s in summaries]
    draw = [100 * s.draw_rate for s in summaries]
    nodes = [s.mean_nodes for s in summaries]
    ms = [s.mean_ms for s in summaries]
    vs = "Minimax(d=1)" if opponent == "d1" else "heuristic"
    subtitle = (
        "Same evaluate() · paired openings · only depth changes"
        if opponent == "d1"
        else "Vs a tactician · paired openings · leaf is not the heuristic"
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    fig.patch.set_facecolor("#0a0e18")
    for ax in (ax0, ax1):
        ax.set_facecolor("#0a0e18")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        ax.xaxis.label.set_color("#cbd5e1")
        ax.yaxis.label.set_color("#cbd5e1")
        ax.title.set_color("#e2e8f0")

    ax0.plot(depths, win, "o-", color="#5ce1e6", lw=2.2, ms=8, label="Win rate")
    ax0.plot(depths, draw, "s--", color="#94a3b8", lw=1.4, ms=6, label="Draw rate")
    ax0.set_xlabel("Minimax depth")
    ax0.set_ylabel("Rate (%)")
    ax0.set_title(f"Outcome vs {vs}")
    ax0.set_xticks(depths)
    ax0.set_ylim(0, 105)
    ax0.legend(facecolor="#12182a", edgecolor="#1e293b", labelcolor="#e2e8f0")
    ax0.grid(True, color="#1e293b", alpha=0.8)

    ax1.plot(depths, nodes, "o-", color="#ff4d8d", lw=2.2, ms=8, label="Nodes / move")
    ax1.set_xlabel("Minimax depth")
    ax1.set_ylabel("Nodes / move", color="#ff4d8d")
    ax1.tick_params(axis="y", colors="#ff4d8d")
    ax1.set_xticks(depths)
    ax1.set_title("Search cost")
    ax1.grid(True, color="#1e293b", alpha=0.8)

    ax1b = ax1.twinx()
    ax1b.plot(depths, ms, "^-", color="#ffd166", lw=1.8, ms=7, label="Latency")
    ax1b.set_ylabel("Latency (ms)", color="#ffd166")
    ax1b.tick_params(axis="y", colors="#ffd166")
    ax1b.spines["top"].set_color("#1e293b")
    ax1b.spines["bottom"].set_color("#1e293b")
    ax1b.spines["left"].set_color("#1e293b")
    ax1b.spines["right"].set_color("#1e293b")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        facecolor="#12182a",
        edgecolor="#1e293b",
        labelcolor="#e2e8f0",
        loc="upper left",
    )

    fig.suptitle(subtitle, color="#e2e8f0", fontsize=12, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _notes(opponent: str) -> str:
    if opponent == "d1":
        return (
            "Paired openings. Opponent is MinimaxAgent(depth=1): same evaluate(), "
            "same search, only depth changes. Depth 1 vs d=1 is the identical-player control."
        )
    return (
        "Paired openings. Opponent is the full heuristic (win/block/fork). "
        "Minimax leaves use window/center features only — not the heuristic's tactics."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Minimax depth sweep vs heuristic or Minimax(d=1)"
    )
    parser.add_argument(
        "--opponent",
        choices=OPPONENTS,
        default="heuristic",
        help="heuristic = tactician; d1 = same leaf, depth frozen at 1",
    )
    parser.add_argument(
        "--depths",
        default="1,2,3,4,5",
        help="Comma-separated depths (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=40,
        help="Games per depth, even, split across seats (default: 40)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=4,
        help="Random legal moves before agents take over (default: 4)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path prefix (writes .json, .md, .png). Default depends on --opponent.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--empty-board",
        action="store_true",
        help="Sanity check: empty board, both seats (2 games/depth). Ignores --games and --opening-plies.",
    )
    args = parser.parse_args(argv)

    depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]
    if not depths:
        print("No depths provided", file=sys.stderr)
        return 2

    if args.empty_board:
        if args.depths == parser.get_default("depths"):
            depths = list(range(1, 8))
        vs = "heuristic" if args.opponent == "heuristic" else "minimax(d=1)"
        print(f"Empty-board sanity opponent={vs} depths={depths} (2 games/depth)")
        rows = run_empty_board_sweep(depths, opponent=args.opponent)
        meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "opponent": args.opponent,
            "depths": depths,
            "games_per_depth": 2,
            "opening_plies": 0,
            "note": (
                "Empty board, both seats. Deterministic; two games per depth, "
                "not a win-rate sample."
            ),
        }
        payload = {"meta": meta, "results": rows}
        default_name = (
            "minimax_vs_heuristic_empty"
            if args.opponent == "heuristic"
            else "minimax_vs_d1_empty"
        )
        out = args.out or (RESULTS_DIR / default_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        json_path = out.with_suffix(".json")
        md_path = out.with_suffix(".md")
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_empty_markdown(md_path, rows, meta)
        print()
        print(format_empty_table(rows))
        print()
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")
        return 0

    n_pairs = args.games // 2
    print(
        f"Sweep opponent={args.opponent} depths={depths} games/depth={args.games} "
        f"({n_pairs} paired openings, opening_plies={args.opening_plies})"
    )
    summaries = run_depth_sweep(
        depths,
        args.games,
        seed=args.seed,
        opening_plies=args.opening_plies,
        opponent=args.opponent,
    )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "opponent": args.opponent,
        "depths": depths,
        "games_per_depth": args.games,
        "paired_openings": n_pairs,
        "seed": args.seed,
        "opening_plies": args.opening_plies,
        "note": _notes(args.opponent),
    }
    payload = {"meta": meta, "results": [s.to_dict() for s in summaries]}

    out = args.out or DEFAULT_OUT[args.opponent]
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    md_path = out.with_suffix(".md")
    png_path = out.with_suffix(".png")

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, summaries, meta)

    print()
    print(format_table(summaries))
    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    if not args.no_plot:
        if write_plot(png_path, summaries, opponent=args.opponent):
            print(f"Wrote {png_path}")
        else:
            print(
                "matplotlib not installed — skip plot. "
                "Install with: pip install 'connect4-rl[eval]'",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
