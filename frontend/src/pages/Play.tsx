import { useEffect, useMemo, useState } from "react";
import { newGame, playAi, playMove } from "../api";
import { Board } from "../components/Board";
import type { AgentInfo, GameState } from "../types";

const MINIMAX_DEPTHS = [3, 4, 5, 6, 7] as const;
const DEFAULT_DEPTH = 5;

export function PlayScreen({
  initial,
  agent,
  firstPlayer,
  onExit,
  onError,
}: {
  initial: GameState;
  agent: AgentInfo;
  firstPlayer: "human" | "ai" | "random";
  onExit: () => void;
  onError: (msg: string | null) => void;
}) {
  const [game, setGame] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [hoverCol, setHoverCol] = useState<number | null>(null);
  const [depth, setDepth] = useState(initial.agent_depth ?? DEFAULT_DEPTH);

  const humanTurn = !game.done && game.current_player === game.human_player;
  const isMinimax = agent.id === "minimax";
  const aiLabel = isMinimax ? `${agent.name} · d${depth}` : agent.name;

  useEffect(() => {
    if (game.done || game.current_player === game.human_player || busy) return;
    const t = window.setTimeout(async () => {
      setBusy(true);
      try {
        setGame(await playAi(game));
      } catch (err) {
        onError(err instanceof Error ? err.message : "AI move failed");
      } finally {
        setBusy(false);
      }
    }, 420);
    return () => window.clearTimeout(t);
  }, [game, busy, onError]);

  const status = useMemo(() => {
    if (game.draw) return "Draw — board is full";
    if (game.winner === game.human_player) return "You win";
    if (game.winner != null) return `${aiLabel} wins`;
    if (humanTurn) return "Your turn";
    return `${aiLabel} is thinking…`;
  }, [game, aiLabel, humanTurn]);

  async function drop(column: number) {
    if (!humanTurn || busy) return;
    if (!game.valid_actions.includes(column)) return;
    setBusy(true);
    onError(null);
    try {
      setGame(await playMove(game, column));
    } catch (err) {
      onError(err instanceof Error ? err.message : "Move failed");
    } finally {
      setBusy(false);
    }
  }

  async function restart(nextDepth = depth) {
    onError(null);
    setBusy(true);
    try {
      const next = await newGame(
        agent.id,
        firstPlayer,
        isMinimax ? nextDepth : undefined,
      );
      setDepth(next.agent_depth ?? nextDepth);
      setGame(next);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Could not start game");
    } finally {
      setBusy(false);
    }
  }

  async function changeDepth(next: number) {
    if (next === depth || busy) return;
    setDepth(next);
    await restart(next);
  }

  const youColor = game.human_player === 1 ? "rose" : "gold";
  const aiColor = game.human_player === 1 ? "gold" : "rose";

  const analysisHint = isMinimax
    ? `minimax · depth ${depth}`
    : agent.id === "heuristic"
      ? "heuristic scores"
      : "uniform over legal columns";

  return (
    <section className="flex w-full min-w-0 flex-col items-center overflow-x-hidden">
      <div className="mb-6 flex w-full max-w-3xl items-center justify-between gap-2 sm:mb-8 sm:gap-4">
        <PlayerCard label="You" color={youColor} active={humanTurn} />
        <p className="shrink-0 font-display text-sm font-bold tracking-[0.3em] text-slate-500">VS</p>
        <PlayerCard label={aiLabel} color={aiColor} active={!humanTurn && !game.done} />
      </div>

      <div className="flex w-full min-w-0 justify-center">
        <Board
          board={game.board}
          valid={game.valid_actions}
          winning={game.winning_cells}
          hoverCol={humanTurn ? hoverCol : null}
          lastAction={game.last_action}
          interactive={humanTurn}
          onHover={setHoverCol}
          onDrop={drop}
        />
      </div>

      <p className="mt-8 font-display text-2xl font-bold tracking-tight">{status}</p>

      {isMinimax && (
        <div className="mt-6 w-full max-w-sm">
          <div className="mb-2.5 flex items-baseline justify-between px-1">
            <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">
              Search depth
            </span>
            <span className="font-mono text-[10px] text-slate-600">
              {depth <= 3 ? "quick" : depth >= 7 ? "deep" : "balanced"}
              <span className="text-slate-700"> · new game</span>
            </span>
          </div>
          <div
            className="relative flex items-center justify-between gap-1 rounded-full border border-white/10 bg-ink-900/70 px-3 py-2"
            role="group"
            aria-label="Minimax search depth"
          >
            <div
              className="pointer-events-none absolute left-5 right-5 top-1/2 h-px -translate-y-1/2 bg-gradient-to-r from-transparent via-white/15 to-transparent"
              aria-hidden
            />
            {MINIMAX_DEPTHS.map((d) => {
              const selected = depth === d;
              return (
                <button
                  key={d}
                  type="button"
                  disabled={busy}
                  onClick={() => void changeDepth(d)}
                  aria-pressed={selected}
                  className={`relative z-10 flex h-8 w-8 items-center justify-center rounded-full font-mono text-sm transition disabled:opacity-50 ${
                    selected
                      ? "bg-neon-cyan text-ink-950 shadow-[0_0_18px_rgba(92,225,230,0.35)]"
                      : "border border-white/10 bg-ink-950/80 text-slate-400 hover:border-white/25 hover:text-slate-200"
                  }`}
                >
                  {d}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {game.last_ai?.probabilities && (
        <div className="mt-8 w-full max-w-xl rounded-2xl border border-white/10 bg-ink-900/70 p-5 shadow-glow">
          <div className="mb-3 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500">
            <span>AI analysis</span>
            <span>{analysisHint}</span>
          </div>
          <div className="grid grid-cols-7 gap-2">
            {game.last_ai.probabilities.map((p, i) => (
              <div key={i} className="text-center">
                <div className="mx-auto flex h-16 w-3 items-end overflow-hidden rounded-full bg-white/10">
                  <div
                    className="w-full rounded-full bg-neon-cyan"
                    style={{ height: `${Math.max(p * 100, 4)}%` }}
                  />
                </div>
                <p className="mt-2 font-mono text-[10px] text-slate-400">C{i}</p>
                <p className="font-mono text-xs text-slate-200">{Math.round(p * 100)}%</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-8 flex gap-3">
        <button
          type="button"
          onClick={onExit}
          className="rounded-full border border-white/15 px-5 py-2 text-sm text-slate-300 hover:border-white/40"
        >
          Menu
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => void restart()}
          className="rounded-full bg-white px-5 py-2 text-sm font-semibold text-ink-950 hover:bg-slate-200 disabled:opacity-50"
        >
          New game
        </button>
      </div>
    </section>
  );
}

function PlayerCard({
  label,
  color,
  active,
}: {
  label: string;
  color: "rose" | "gold";
  active: boolean;
}) {
  const disc =
    color === "rose"
      ? "bg-neon-rose shadow-[0_0_24px_rgba(255,77,109,0.55)]"
      : "bg-neon-gold shadow-[0_0_24px_rgba(255,209,102,0.5)]";
  return (
    <div
      className={`flex min-w-0 flex-1 items-center gap-2 rounded-2xl border px-3 py-2.5 sm:gap-3 sm:px-4 sm:py-3 ${
        active ? "border-white/25 bg-white/5" : "border-white/10 bg-ink-900/50"
      }`}
    >
      <span className={`h-7 w-7 shrink-0 rounded-full sm:h-8 sm:w-8 ${disc}`} />
      <div className="min-w-0">
        <p className="font-mono text-[10px] uppercase tracking-widest text-slate-500">
          {active ? "To move" : "Waiting"}
        </p>
        <p className="truncate font-display text-base font-bold sm:text-lg">{label}</p>
      </div>
    </div>
  );
}
