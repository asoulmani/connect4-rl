import { useEffect, useState } from "react";
import { listAgents, newGame } from "./api";
import { PlayScreen } from "./pages/Play";
import type { AgentInfo, GameState } from "./types";

type Screen = { kind: "menu" } | { kind: "play"; game: GameState; agent: AgentInfo };

const ICONS: Record<string, string> = {
  random: "🎲",
  heuristic: "🧠",
  minimax: "🌳",
  mcts: "🎯",
  dqn: "🤖",
  alphazero: "✨",
};

export default function App() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [screen, setScreen] = useState<Screen>({ kind: "menu" });
  const [error, setError] = useState<string | null>(null);
  const [first, setFirst] = useState<"human" | "ai" | "random">("human");

  useEffect(() => {
    listAgents()
      .then(setAgents)
      .catch((err: Error) => setError(err.message));
  }, []);

  async function start(agent: AgentInfo) {
    if (!agent.available) return;
    setError(null);
    try {
      const game = await newGame(agent.id, first);
      setScreen({ kind: "play", game, agent });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start game");
    }
  }

  return (
    <div className="relative min-h-screen">
      <div className="grid-overlay pointer-events-none absolute inset-0" />
      <header className="relative z-10 flex items-center justify-between px-6 py-5 md:px-10">
        <div>
          <p className="font-mono text-[11px] tracking-[0.28em] text-neon-cyan/80">CONNECT FOUR RL</p>
          <h1 className="font-display text-xl font-extrabold tracking-tight md:text-2xl">Lab Playground</h1>
        </div>
        <p className="hidden font-mono text-xs text-slate-500 md:block">Random · Heuristic · Minimax · MCTS · DQN</p>
      </header>

      <main className="relative z-10 mx-auto max-w-5xl px-4 pb-16">
        {error && (
          <div className="mb-6 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            {error}
            {error.toLowerCase().includes("fetch") && (
              <span className="mt-1 block text-rose-200/70">
                Start the API: <code className="font-mono">uvicorn backend.app.main:app --reload</code>
              </span>
            )}
          </div>
        )}

        {screen.kind === "menu" ? (
          <Menu agents={agents} first={first} onFirst={setFirst} onPick={start} />
        ) : (
          <PlayScreen
            initial={screen.game}
            agent={screen.agent}
            firstPlayer={first}
            onExit={() => setScreen({ kind: "menu" })}
            onError={setError}
          />
        )}
      </main>
    </div>
  );
}

function Menu({
  agents,
  first,
  onFirst,
  onPick,
}: {
  agents: AgentInfo[];
  first: "human" | "ai" | "random";
  onFirst: (v: "human" | "ai" | "random") => void;
  onPick: (agent: AgentInfo) => void;
}) {
  return (
    <section className="mt-4">
      <p className="font-display text-4xl font-extrabold leading-none tracking-tight md:text-6xl">
        Choose your
        <span className="bg-gradient-to-r from-neon-cyan to-neon-magenta bg-clip-text text-transparent"> opponent</span>
      </p>
      <p className="mt-4 max-w-xl text-slate-400">
        A dark lab for Connect Four agents. Random through DQN are playable. AlphaZero is next.
      </p>

      <div className="mt-8 flex flex-wrap gap-2">
        {(
          [
            ["human", "You start"],
            ["ai", "AI starts"],
            ["random", "Coin flip"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => onFirst(id)}
            className={`rounded-full px-4 py-1.5 font-mono text-xs tracking-wide transition ${
              first === id
                ? "bg-white text-ink-950"
                : "border border-white/10 bg-white/5 text-slate-300 hover:border-white/25"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {agents.map((agent) => (
          <button
            key={agent.id}
            type="button"
            disabled={!agent.available}
            onClick={() => onPick(agent)}
            className={`group rounded-2xl border p-5 text-left transition ${
              agent.available
                ? "border-cyan-400/25 bg-ink-900/80 shadow-glow hover:-translate-y-0.5 hover:border-cyan-300/50"
                : "cursor-not-allowed border-white/5 bg-ink-900/40 opacity-45"
            }`}
          >
            <div className="flex items-start justify-between">
              <span className="text-2xl">{ICONS[agent.id] ?? "◆"}</span>
              <span className="font-mono text-[10px] uppercase tracking-widest text-slate-500">
                {agent.available ? "Ready" : "Locked"}
              </span>
            </div>
            <h2 className="mt-4 font-display text-2xl font-bold">{agent.name}</h2>
            <p className="mt-1 text-sm text-slate-400">{agent.subtitle}</p>
          </button>
        ))}
      </div>
    </section>
  );
}
