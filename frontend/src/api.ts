import type { AgentInfo, GameState } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "";

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

function gameContext(game: GameState) {
  return {
    moves: game.moves,
    human_player: game.human_player,
    agent_id: game.agent_id,
    agent_depth: game.agent_depth,
  };
}

export function listAgents(): Promise<AgentInfo[]> {
  return fetch(`${API_URL}/agents`).then((r) =>
    parse<AgentInfo[]>(r),
  );
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

  if (depth != null) {
    body.depth = depth;
  }

  return fetch(`${API_URL}/game/new`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  }).then((r) => parse<GameState>(r));
}

export function playMove(
  game: GameState,
  column: number,
): Promise<GameState> {
  return fetch(`${API_URL}/game/move`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ...gameContext(game),
      column,
    }),
  }).then((r) => parse<GameState>(r));
}

export function playAi(game: GameState): Promise<GameState> {
  return fetch(`${API_URL}/game/ai-move`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(gameContext(game)),
  }).then((r) => parse<GameState>(r));
}
