"""MCTS simulation-budget sweeps.

PUCT + uniform prior + random rollout evaluation. Same paired-opening
harness as the minimax depth sweeps. The axis is simulations per move,
not depth.

Two opponents:

- ``heuristic`` — can extra rollouts beat a 1-ply tactician?
- ``minimax`` — MCTS vs ``MinimaxAgent`` at ``--minimax-depth`` (pick a
  depth whose latency is in the same ballpark as the probe).

Example::

    python -m connect4.training.evaluate_mcts --probe
    python -m connect4.training.evaluate_mcts --opponent heuristic --games 100 --sims 50,200,800
    python -m connect4.training.evaluate_mcts --opponent minimax --minimax-depth 3 --games 100 --sims 50,200,800
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from connect4.agents.base import Agent
from connect4.agents.heuristic_agent import HeuristicAgent
from connect4.agents.mcts_agent import MCTSAgent
from connect4.agents.minimax_agent import MinimaxAgent
from connect4.env.constants import PLAYER_ONE, PLAYER_TWO
from connect4.env.environment import ConnectFourEnv
from connect4.evaluation.match import generate_paired_openings, play_game
from connect4.evaluation.metrics import DepthSummary, summarize_depth

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"

OPPONENTS = ("heuristic", "minimax")
PROBE_SIMS = (50, 200, 800)
PROBE_MINIMAX_DEPTHS = (2, 3, 5)


def make_opponent(kind: str, *, minimax_depth: int = 3) -> Agent:
    if kind == "heuristic":
        return HeuristicAgent()
    if kind == "minimax":
        return MinimaxAgent(depth=minimax_depth)
    raise ValueError(f"Unknown opponent {kind!r}; expected one of {OPPONENTS}")


def run_sim_sweep(
    sims: list[int],
    games_per_budget: int,
    *,
    seed: int = 0,
    opening_plies: int = 4,
    opponent: str = "heuristic",
    minimax_depth: int = 3,
) -> list[DepthSummary]:
    """Paired openings: every simulation budget sees the same positions and seats."""
    if games_per_budget < 2:
        raise ValueError("games_per_budget must be >= 2 (need both seats)")
    if games_per_budget % 2:
        raise ValueError("games_per_budget must be even so seats are balanced")

    n_pairs = games_per_budget // 2
    openings = generate_paired_openings(n_pairs, seed=seed, opening_plies=opening_plies)
    opp = make_opponent(opponent, minimax_depth=minimax_depth)
    summaries: list[DepthSummary] = []

    for n_sim in sims:
        agent = MCTSAgent(n_simulations=n_sim, rng=random.Random(seed + n_sim))
        games = []
        for i, opening in enumerate(openings):
            games.append(
                play_game(agent, opp, minimax_seat=PLAYER_ONE, opening=opening)
            )
            games.append(
                play_game(agent, opp, minimax_seat=PLAYER_TWO, opening=opening)
            )
            done = (i + 1) * 2
            if done == games_per_budget or done % max(2, games_per_budget // 4) == 0:
                label = "heuristic" if opponent == "heuristic" else f"minimax(d={minimax_depth})"
                print(
                    f"  sims {n_sim} vs {label}: {done}/{games_per_budget} games",
                    flush=True,
                )
        summaries.append(summarize_depth(n_sim, games))
    return summaries


def format_table(summaries: list[DepthSummary]) -> str:
    lines = [
        "| Sims | Win% | Draw% | P1 win% | P2 win% | Tree nodes/move | Latency |",
        "|-----:|-----:|------:|--------:|--------:|----------------:|--------:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.depth} | {100 * s.win_rate:.1f}% | {100 * s.draw_rate:.1f}% | "
            f"{100 * s.win_rate_p1:.1f}% | {100 * s.win_rate_p2:.1f}% | "
            f"{s.mean_nodes:,.0f} | {s.mean_ms:.1f} ms |"
        )
    return "\n".join(lines)


def _copy_and_takeaway(opponent: str, minimax_depth: int) -> tuple[str, str, str]:
    if opponent == "minimax":
        title = (
            f"MCTS vs Minimax(d={minimax_depth}) — "
            "PUCT + uniform prior + random rollouts"
        )
        blurb = (
            f"Opponent is `MinimaxAgent(depth={minimax_depth})`. "
            "MCTS is PUCT with a uniform prior and random rollout evaluation. "
            "The axis is simulations per move, not depth. Pick a minimax depth "
            "whose latency is comparable (see `--probe`)."
        )
        takeaway = (
            "If win rate rises with simulations here, extra test-time compute "
            "is buying strength against a fixed-depth searcher. Latency is the "
            "fair comparison, not tree_nodes vs alphabeta nodes."
        )
        return title, blurb, takeaway

    title = "MCTS vs Heuristic — PUCT + uniform prior + random rollouts"
    blurb = (
        "MCTS against the full heuristic (win / block / fork). Random rollouts "
        "are noisy; terminals in the tree still encode instant wins and hangs. "
        "The question is whether a simulation budget beats a 1-ply tactician."
    )
    takeaway = (
        "Low budgets can lose to the heuristic: a 1-ply tactician is cheap and "
        "sharp. If the curve climbs, extra simulations are finding those tactics "
        "the hard way. Compare to `--opponent minimax` at similar latency."
    )
    return title, blurb, takeaway


def write_markdown(
    path: Path, summaries: list[DepthSummary], meta: dict
) -> None:
    opponent = meta.get("opponent", "heuristic")
    depth = int(meta.get("minimax_depth") or 3)
    title, blurb, takeaway = _copy_and_takeaway(opponent, depth)
    body = "\n".join(
        [
            f"# {title}",
            "",
            blurb,
            "",
            f"- Games per budget: **{meta['games_per_budget']}** "
            f"({meta['paired_openings']} paired openings × 2 seats)",
            f"- Random opening plies: **{meta.get('opening_plies', 4)}** "
            "(same opening at every simulation count)",
            f"- Simulations: `{meta['sims']}`",
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


def write_plot(
    path: Path,
    summaries: list[DepthSummary],
    *,
    opponent: str,
    minimax_depth: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    sims = [s.depth for s in summaries]
    win = [100 * s.win_rate for s in summaries]
    draw = [100 * s.draw_rate for s in summaries]
    nodes = [s.mean_nodes for s in summaries]
    ms = [s.mean_ms for s in summaries]
    vs = "heuristic" if opponent == "heuristic" else f"Minimax(d={minimax_depth})"
    subtitle = "PUCT + uniform prior + random rollouts · paired openings"

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

    ax0.plot(sims, win, "o-", color="#5ce1e6", lw=2.2, ms=8, label="Win rate")
    ax0.plot(sims, draw, "s--", color="#94a3b8", lw=1.4, ms=6, label="Draw rate")
    ax0.set_xlabel("Simulations / move")
    ax0.set_ylabel("Rate (%)")
    ax0.set_title(f"Outcome vs {vs}")
    ax0.set_xticks(sims)
    ax0.set_ylim(0, 105)
    ax0.legend(facecolor="#12182a", edgecolor="#1e293b", labelcolor="#e2e8f0")
    ax0.grid(True, color="#1e293b", alpha=0.8)

    ax1.plot(sims, nodes, "o-", color="#ff4d8d", lw=2.2, ms=8, label="Tree nodes / move")
    ax1.set_xlabel("Simulations / move")
    ax1.set_ylabel("Tree nodes / move", color="#ff4d8d")
    ax1.tick_params(axis="y", colors="#ff4d8d")
    ax1.set_xticks(sims)
    ax1.set_title("Search cost")
    ax1.grid(True, color="#1e293b", alpha=0.8)

    ax1b = ax1.twinx()
    ax1b.plot(sims, ms, "^-", color="#ffd166", lw=1.8, ms=7, label="Latency")
    ax1b.set_ylabel("Latency (ms)", color="#ffd166")
    ax1b.tick_params(axis="y", colors="#ffd166")
    for spine in ax1b.spines.values():
        spine.set_color("#1e293b")

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


def _timed_move(agent: Agent, env: ConnectFourEnv, repeats: int) -> tuple[float, dict]:
    state = env.get_state()
    valid = env.get_valid_actions()
    last_meta: dict = {}
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        decision = agent.select_action(state, valid)
        times.append((time.perf_counter() - t0) * 1000.0)
        last_meta = decision.metadata or {}
    return sum(times) / len(times), last_meta


def _midgame(seed: int = 0, plies: int = 12) -> ConnectFourEnv:
    rng = random.Random(seed)
    env = ConnectFourEnv()
    for _ in range(plies):
        if env.terminated:
            break
        env.step(rng.choice(env.get_valid_actions()))
    return env


def run_probe(*, repeats: int = 3, seed: int = 0) -> list[dict]:
    """Time one move on empty and midgame boards. No match play."""
    empty = ConnectFourEnv()
    mid = _midgame(seed=seed)
    rows: list[dict] = []

    print("Latency probe (mean of "
          f"{repeats} calls). Empty board vs a 12-ply random position.")
    print()
    header = (
        f"{'agent':<18} {'empty ms':>10} {'mid ms':>10} {'tree_nodes':>12}"
    )
    print(header)
    print("-" * len(header))

    for n_sim in PROBE_SIMS:
        agent = MCTSAgent(n_simulations=n_sim, rng=random.Random(seed))
        empty_ms, meta_e = _timed_move(agent, empty, repeats)
        mid_ms, meta_m = _timed_move(agent, mid, repeats)
        tree = int(meta_m.get("tree_nodes") or meta_e.get("tree_nodes") or 0)
        label = f"MCTS s={n_sim}"
        print(f"{label:<18} {empty_ms:10.1f} {mid_ms:10.1f} {tree:12d}")
        rows.append(
            {
                "agent": "mcts",
                "n_simulations": n_sim,
                "empty_ms": empty_ms,
                "mid_ms": mid_ms,
                "tree_nodes": tree,
            }
        )

    for depth in PROBE_MINIMAX_DEPTHS:
        agent = MinimaxAgent(depth=depth)
        empty_ms, meta_e = _timed_move(agent, empty, repeats)
        mid_ms, meta_m = _timed_move(agent, mid, repeats)
        nodes = int(meta_m.get("nodes") or meta_e.get("nodes") or 0)
        label = f"Minimax d={depth}"
        print(f"{label:<18} {empty_ms:10.1f} {mid_ms:10.1f} {nodes:12d}")
        rows.append(
            {
                "agent": "minimax",
                "depth": depth,
                "empty_ms": empty_ms,
                "mid_ms": mid_ms,
                "nodes": nodes,
            }
        )

    print()
    print(
        "Pick three --sims values from the MCTS rows, then --minimax-depth "
        "whose midgame latency is in the same band."
    )
    return rows


def _default_out(opponent: str, minimax_depth: int) -> Path:
    if opponent == "minimax":
        return RESULTS_DIR / f"mcts_vs_minimax_d{minimax_depth}"
    return RESULTS_DIR / "mcts_vs_heuristic"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MCTS (PUCT + uniform prior + random rollouts) "
            "vs heuristic or minimax"
        )
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Time 50/200/800 sims and minimax d=2/3/5; do not play matches",
    )
    parser.add_argument(
        "--opponent",
        choices=OPPONENTS,
        default="heuristic",
        help="heuristic = tactician; minimax = MinimaxAgent(--minimax-depth)",
    )
    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=3,
        help="Minimax depth when --opponent minimax (default: 3)",
    )
    parser.add_argument(
        "--sims",
        default="50,200,800",
        help="Comma-separated simulation budgets (default: 50,200,800)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=40,
        help="Games per budget, even, split across seats (default: 40)",
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
        help="Output path prefix (writes .json, .md, .png)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    if args.probe:
        run_probe(seed=args.seed)
        return 0

    sims = [int(x.strip()) for x in args.sims.split(",") if x.strip()]
    if not sims:
        print("No simulation budgets provided", file=sys.stderr)
        return 2
    if args.minimax_depth < 1:
        print("--minimax-depth must be >= 1", file=sys.stderr)
        return 2

    n_pairs = args.games // 2
    vs = (
        "heuristic"
        if args.opponent == "heuristic"
        else f"minimax(d={args.minimax_depth})"
    )
    print(
        f"Sweep opponent={vs} sims={sims} games/budget={args.games} "
        f"({n_pairs} paired openings, opening_plies={args.opening_plies})"
    )
    summaries = run_sim_sweep(
        sims,
        args.games,
        seed=args.seed,
        opening_plies=args.opening_plies,
        opponent=args.opponent,
        minimax_depth=args.minimax_depth,
    )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent": "mcts",
        "search": "PUCT + uniform prior + random rollouts",
        "opponent": args.opponent,
        "minimax_depth": args.minimax_depth if args.opponent == "minimax" else None,
        "sims": sims,
        "games_per_budget": args.games,
        "paired_openings": n_pairs,
        "seed": args.seed,
        "opening_plies": args.opening_plies,
        "note": (
            "Paired openings. Focal agent is MCTSAgent (uniform PUCT, random "
            "rollouts). Cost column is mean tree_nodes/move, not alphabeta nodes."
        ),
    }
    payload = {"meta": meta, "results": [s.to_dict() for s in summaries]}

    out = args.out or _default_out(args.opponent, args.minimax_depth)
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
        if write_plot(
            png_path,
            summaries,
            opponent=args.opponent,
            minimax_depth=args.minimax_depth,
        ):
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
