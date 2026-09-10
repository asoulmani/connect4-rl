"""Rule-based Connect Four policy with no search, no learning.

Priority:
1. Win immediately if a legal drop makes 4 in a row.
2. Block if the opponent would win by dropping in a column now.
3. Otherwise score legal drops: 3-threats, 2-windows, center, self-fork
   bonus, and a penalty for allowing an opponent fork. Moves that hand the
   opponent an immediate win are hard rejected.
"""

from __future__ import annotations

import numpy as np

from connect4.agents.base import Agent, AgentDecision
from connect4.env.board import Board
from connect4.env.constants import (
    COLS, 
    CONNECT, 
    EMPTY, 
    PLAYER_ONE, 
    ROWS, 
    WIN_DIRECTIONS,
)

# Distance to center bonus. Column 3 (given score 4) is the strongest opening.
CENTER_SCORE = (1, 2, 3, 4, 3, 2, 1)
SCORE_THREE = 100
SCORE_TWO = 10
# Double threat after our drop, usually forcing a win next turn.
BONUS_SELF_FORK = 100_000
# Soft penalty only (opponent fork is not always forced if we can still respond).
PENALTY_FORK = 800
# Dominates any positional / fork score.
HARD_LOSS = -1_000_000.0


def player_to_move(state: np.ndarray) -> int:
    """P1 starts; even number of discs ⇒ P1 to move. Absolute +1/-1 board."""
    n = int(np.count_nonzero(state))
    return PLAYER_ONE if n % 2 == 0 else -PLAYER_ONE


def landing_row(grid: np.ndarray, column: int) -> int | None:
    for row in range(ROWS - 1, -1, -1):
        if int(grid[row, column]) == EMPTY:
            return row
    return None


def after_drop(grid: np.ndarray, column: int, player: int) -> np.ndarray:
    """Return the hypothetical board after the player drops in the given column."""
    nxt = np.array(grid, copy=True)
    row = landing_row(nxt, column)
    if row is None:
        raise ValueError(f"Column {column} is full")
    nxt[row, column] = player
    return nxt


def is_win(grid: np.ndarray, player: int) -> bool:
    winner, _ = Board(grid).check_winner()
    return winner == player


def count_windows(grid: np.ndarray, player: int) -> tuple[int, int]:
    """Count unblocked 4-windows with exactly 3 or exactly 2 of 'player'.

    A window is blocked if it contains the opponent. This is the usual
    'threat / potential' feature, not a search.
    """
    threes = 0
    twos = 0
    opponent = -player

    for dr, dc in WIN_DIRECTIONS:
        for row in range(ROWS):
            for col in range(COLS):
                cells = []
                ok = True
                for k in range(CONNECT):
                    nr = row + dr * k
                    nc = col + dc * k
                    if not (0 <= nr < ROWS and 0 <= nc < COLS):
                        ok = False
                        break
                    cells.append(int(grid[nr, nc]))
                if not ok or opponent in cells:
                    continue
                n = sum(1 for v in cells if v == player)
                if n == 3:
                    threes += 1
                elif n == 2:
                    twos += 1
    return threes, twos


def legal_columns(grid: np.ndarray) -> list[int]:
    return [c for c in range(COLS) if int(grid[0, c]) == EMPTY]


def immediate_win_columns(grid: np.ndarray, player: int) -> list[int]:
    """Columns where 'player' would win if it were their turn to drop."""
    return [c for c in legal_columns(grid) if is_win(after_drop(grid, c, player), player)]


def creates_self_fork(grid: np.ndarray, column: int, me: int) -> bool:
    """True if dropping 'column' leaves two+ immediate wins for 'me'."""
    nxt = after_drop(grid, column, me)
    if immediate_win_columns(nxt, -me):
        return False
    return len(immediate_win_columns(nxt, me)) >= 2


def opponent_can_create_fork(grid: np.ndarray, opponent: int) -> bool:
    """True if opponent has a non-winning drop that leaves >= 2 immediate wins.

    Classic Connect Four fork: open-ended three-in-a-row (e.g. _ X X X _).
    One block cannot stop both ends.
    """
    for col in legal_columns(grid):
        nxt = after_drop(grid, col, opponent)
        if is_win(nxt, opponent):
            continue
        if len(immediate_win_columns(nxt, opponent)) >= 2:
            return True
    return False


def score_move(grid: np.ndarray, column: int, me: int, opp: int) -> float:
    nxt = after_drop(grid, column, me)
    # Tactical truth dominates soft scoring features: never gift an immediate loss.
    if immediate_win_columns(nxt, opp):
        return HARD_LOSS
    threes, twos = count_windows(nxt, me)
    score = float(CENTER_SCORE[column] + SCORE_THREE * threes + SCORE_TWO * twos)
    if len(immediate_win_columns(nxt, me)) >= 2:
        score += BONUS_SELF_FORK
    if opponent_can_create_fork(nxt, opp):
        score -= PENALTY_FORK
    return score


def prefer_center(columns: list[int]) -> int:
    return min(columns, key=lambda c: (abs(c - 3), c))


class HeuristicAgent(Agent):
    name = "heuristic"

    def select_action(self, state: np.ndarray, valid_actions: list[int]) -> AgentDecision:
        if not valid_actions:
            raise ValueError("No valid actions")

        # Determine if the agent is -1 or 1 before making a move.
        me = player_to_move(state)
        opp = -me

        # Win immediately if possible.
        wins = [c for c in valid_actions if is_win(after_drop(state, c, me), me)]
        if wins:
            action = prefer_center(wins)
            return self._decision(action, valid_actions, reason="win", me=me)

        # Block if the opponent would win by dropping in a column now.
        blocks = [c for c in immediate_win_columns(state, opp) if c in valid_actions]
        if blocks:
            action = prefer_center(blocks)
            reason = "fork_unavoidable" if len(blocks) >= 2 else "block"
            return self._decision(action, valid_actions, reason=reason, me=me)

        # Otherwise score legal drops according to the heuristic.
        scores = {c: score_move(state, c, me, opp) for c in valid_actions}
        best = max(scores.values())
        tied = [c for c, s in scores.items() if s == best]
        action = prefer_center(tied)
        reason = "fork" if creates_self_fork(state, action, me) else "score"
        return self._decision(action, valid_actions, reason=reason, me=me, scores=scores)

    def _decision(
        self,
        action: int,
        valid_actions: list[int],
        *,
        reason: str,
        me: int,
        scores: dict[int, float] | None = None,
    ) -> AgentDecision:
        probs = _to_probs(action, valid_actions, scores)
        return AgentDecision(
            action=action,
            probabilities=probs,
            value=None,
            metadata={
                "policy": "heuristic",
                "reason": reason,
                "seat": me,
                "scores": {str(c): float(s) for c, s in (scores or {}).items()},
            },
        )


def _to_probs(
    action: int,
    valid_actions: list[int],
    scores: dict[int, float] | None,
) -> list[float]:
    # If the move is to immediately win or block.
    if scores is None:
        probs = [0.0] * COLS
        probs[action] = 1.0
        return probs
    
    xs = np.full(COLS, -1e9, dtype=np.float64)
    for col, score in scores.items():
        xs[col] = score
    shifted = xs - np.max(xs)
    exp = np.exp(np.clip(shifted / 20.0, -50.0, 50.0))
    for col in range(COLS):
        if col not in valid_actions:
            exp[col] = 0.0
    total = float(exp.sum())
    if total <= 0:
        probs = [0.0] * COLS
        probs[action] = 1.0
        return probs
    return (exp / total).tolist()
