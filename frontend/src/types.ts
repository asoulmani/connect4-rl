export type AgentInfo = {
  id: string;
  name: string;
  subtitle: string;
  available: boolean;
};

export type AgentDecision = {
  action: number;
  probabilities: number[] | null;
  value: number | null;
  metadata: Record<string, unknown>;
};

export type GameState = {
  id: string;
  board: number[][];
  current_player: number;
  human_player: number;
  done: boolean;
  winner: number | null;
  winning_cells: number[][] | null;
  valid_actions: number[];
  agent_id: string;
  agent_depth: number | null;
  last_action: number | null;
  last_ai: AgentDecision | null;
  draw: boolean;
};
