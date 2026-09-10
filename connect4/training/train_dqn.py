"""Train compact DQN by self-play. Does not go through HTTP.

Examples::

    python -m connect4.training.train_dqn --smoke
    python -m connect4.training.train_dqn --episodes 30000
    python -m connect4.training.train_dqn --eval-only --checkpoint models/dqn/dqn.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from connect4.agents.base import Agent
from connect4.agents.dqn.agent import DEFAULT_CHECKPOINT, DQNAgent
from connect4.agents.dqn.trainer import DQNConfig, DQNTrainer
from connect4.agents.heuristic_agent import HeuristicAgent
from connect4.agents.minimax_agent import MinimaxAgent
from connect4.agents.random_agent import RandomAgent
from connect4.env.constants import PLAYER_ONE, PLAYER_TWO
from connect4.env.environment import ConnectFourEnv
from connect4.evaluation.match import generate_paired_openings, play_game
from connect4.evaluation.metrics import DepthSummary, summarize_depth
from connect4.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"

SMOKE_CONFIG = DQNConfig(
    hidden=64,
    buffer_size=512,
    batch_size=32,
    warmup_steps=32,
    target_update_every=50,
    epsilon_decay_steps=200,
    seed=0,
)


def play_self_play_episode(trainer: DQNTrainer) -> dict[str, float]:
    env = ConnectFourEnv()
    env.reset()
    losses: list[float] = []
    moves = 0
    while not env.terminated:
        action, planes, _valid = trainer.act_epsilon_greedy(env)
        _state, reward, done, _info = env.step(action)
        trainer.store_transition(planes, action, reward, env, done)
        loss = trainer.maybe_update()
        if loss is not None:
            losses.append(loss)
        moves += 1
    trainer.episodes += 1
    return {
        "moves": float(moves),
        "mean_loss": float(sum(losses) / len(losses)) if losses else float("nan"),
        "winner": float(env.winner) if env.winner is not None else 0.0,
        "draw": 1.0 if env.winner is None else 0.0,
    }


def make_eval_opponent(kind: str, *, seed: int = 0) -> Agent:
    if kind == "random":
        return RandomAgent(rng=random.Random(seed))
    if kind == "heuristic":
        return HeuristicAgent()
    if kind == "minimax-d1":
        return MinimaxAgent(depth=1)
    if kind == "minimax-d2":
        return MinimaxAgent(depth=2)
    raise ValueError(f"Unknown opponent {kind!r}")


def run_dqn_eval(
    agent: Agent,
    opponent: Agent,
    games: int,
    *,
    seed: int = 0,
    opening_plies: int = 4,
) -> DepthSummary:
    """Paired frozen openings, both seats — same philosophy as the search sweeps."""
    if games < 2 or games % 2:
        raise ValueError("games must be even and >= 2")
    openings = generate_paired_openings(games // 2, seed=seed, opening_plies=opening_plies)
    results = []
    for opening in openings:
        results.append(play_game(agent, opponent, minimax_seat=PLAYER_ONE, opening=opening))
        results.append(play_game(agent, opponent, minimax_seat=PLAYER_TWO, opening=opening))
    return summarize_depth(0, results)


def format_eval_table(rows: list[tuple[str, DepthSummary]]) -> str:
    lines = [
        "| Opponent | Games | Win% | Draw% | P1 win% | P2 win% |",
        "|----------|------:|-----:|------:|--------:|--------:|",
    ]
    for name, s in rows:
        lines.append(
            f"| {name} | {s.games} | {100 * s.win_rate:.1f}% | {100 * s.draw_rate:.1f}% | "
            f"{100 * s.win_rate_p1:.1f}% | {100 * s.win_rate_p2:.1f}% |"
        )
    return "\n".join(lines)


def write_eval_markdown(path: Path, rows: list[tuple[str, DepthSummary]], meta: dict) -> None:
    body = "\n".join(
        [
            "# DQN vs baselines",
            "",
            "Compact from-scratch DQN (replay, target net, ε-greedy, illegal-action "
            "masking, negamax Bellman target). Frozen 4-ply openings, both seats.",
            "",
            f"- Games per opponent: **{meta['games']}** "
            f"({meta['games'] // 2} paired openings × 2 seats)",
            f"- Opening plies: **{meta['opening_plies']}**",
            f"- Training episodes: **{meta.get('episodes', 'unknown')}**",
            f"- Checkpoint: `{meta['checkpoint']}`",
            f"- Generated: `{meta['generated_at']}`",
            "",
            format_eval_table(rows),
            "",
            "This is a time-boxed strength check, not a claim of SOTA Connect Four play.",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    games: int,
    seed: int,
    opening_plies: int,
    opponents: list[str],
    out: Path | None,
) -> list[tuple[str, DepthSummary]]:
    agent = DQNAgent.from_checkpoint(checkpoint)
    rows: list[tuple[str, DepthSummary]] = []
    for kind in opponents:
        print(f"  eval vs {kind} ({games} games)...", flush=True)
        summary = run_dqn_eval(
            agent,
            make_eval_opponent(kind, seed=seed),
            games,
            seed=seed,
            opening_plies=opening_plies,
        )
        rows.append((kind, summary))
        print(
            f"    win%={100 * summary.win_rate:.1f} draw%={100 * summary.draw_rate:.1f}",
            flush=True,
        )

    ckpt_path = Path(checkpoint)
    try:
        rel = ckpt_path.resolve().relative_to(ROOT)
        ckpt_str = str(rel)
    except ValueError:
        ckpt_str = str(ckpt_path)
    extra = getattr(agent, "_ckpt_meta", {}) or {}
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": ckpt_str,
        "episodes": extra.get("episodes"),
        "steps": extra.get("steps"),
        "games": games,
        "opening_plies": opening_plies,
        "seed": seed,
        "opponents": opponents,
        "note": (
            "Paired openings, both seats. DQN is greedy (ε=0). "
            "Win rates are from the DQN seat."
        ),
    }
    payload = {
        "meta": meta,
        "results": [{"opponent": name, **s.to_dict()} for name, s in rows],
    }
    dest = out or (RESULTS_DIR / "dqn_vs_baselines")
    dest.parent.mkdir(parents=True, exist_ok=True)
    json_path = dest.with_suffix(".json")
    md_path = dest.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_eval_markdown(md_path, rows, meta)
    print()
    print(format_eval_table(rows))
    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return rows


def train(
    *,
    episodes: int,
    checkpoint: Path,
    config: DQNConfig,
    log_every: int,
) -> DQNTrainer:
    set_seed(config.seed)
    trainer = DQNTrainer(config)
    finite_losses = 0
    for ep in range(1, episodes + 1):
        stats = play_self_play_episode(trainer)
        loss = stats["mean_loss"]
        if loss == loss:  # not NaN
            finite_losses += 1
            if not (loss == loss and abs(loss) < 1e6):
                raise RuntimeError(f"Non-finite or exploding loss at episode {ep}: {loss}")
        if ep == 1 or ep % log_every == 0 or ep == episodes:
            print(
                f"episode {ep}/{episodes}  steps={trainer.steps}  "
                f"eps={trainer.epsilon():.3f}  buffer={len(trainer.buffer)}  "
                f"loss={loss:.4f}" if loss == loss else
                f"episode {ep}/{episodes}  steps={trainer.steps}  "
                f"eps={trainer.epsilon():.3f}  buffer={len(trainer.buffer)}  loss=n/a",
                flush=True,
            )
    trainer.save(checkpoint)
    print(f"Wrote {checkpoint}")
    return trainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train compact from-scratch Connect Four DQN")
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Weight path (default: {DEFAULT_CHECKPOINT}; smoke uses models/dqn/dqn_smoke.pt)",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run: few episodes, small nets, no eval. Checks finite loss.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate an existing checkpoint.",
    )
    parser.add_argument("--eval-games", type=int, default=40, help="Even, both seats")
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument(
        "--opponents",
        default="random,heuristic,minimax-d1",
        help="Comma-separated: random,heuristic,minimax-d1,minimax-d2",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    if args.checkpoint is None:
        args.checkpoint = (
            (ROOT / "models" / "dqn" / "dqn_smoke.pt") if args.smoke else DEFAULT_CHECKPOINT
        )

    if args.smoke:
        cfg = replace(SMOKE_CONFIG, seed=args.seed)
        train(
            episodes=min(args.episodes, 8),
            checkpoint=args.checkpoint,
            config=cfg,
            log_every=1,
        )
        return 0

    if args.eval_only:
        if not args.checkpoint.is_file():
            print(f"No checkpoint at {args.checkpoint}", file=sys.stderr)
            return 2
        evaluate_checkpoint(
            args.checkpoint,
            games=args.eval_games,
            seed=args.seed,
            opening_plies=args.opening_plies,
            opponents=opponents,
            out=args.out,
        )
        return 0

    cfg = DQNConfig(seed=args.seed)
    train(
        episodes=args.episodes,
        checkpoint=args.checkpoint,
        config=cfg,
        log_every=args.log_every,
    )
    if not args.skip_eval:
        evaluate_checkpoint(
            args.checkpoint,
            games=args.eval_games,
            seed=args.seed,
            opening_plies=args.opening_plies,
            opponents=opponents,
            out=args.out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
