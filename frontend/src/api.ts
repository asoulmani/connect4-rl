import type { AgentInfo, GameState } from "./types";

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function listAgents(): Promise<AgentInfo[]> {
  return fetch("/agents").then((r) => parse<AgentInfo[]>(r));
}

export function newGame(
  agentId: string,
  firstPlayer: "human" | "ai" | "random",
  depth?: number,
): Promise<GameState> {
  const body: Record<string, unknown> = {
    agent_id: agentId,
    first_player: firstPlayer,
  };
  if (depth != null) body.depth = depth;
  return fetch("/game/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => parse<GameState>(r));
}

export function playMove(gameId: string, column: number): Promise<GameState> {
  return fetch(`/game/${gameId}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ column }),
  }).then((r) => parse<GameState>(r));
}

export function playAi(gameId: string): Promise<GameState> {
  return fetch(`/game/${gameId}/ai-move`, { method: "POST" }).then((r) => parse<GameState>(r));
}
