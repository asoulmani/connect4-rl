"""Gym-style Connect Four environment. No knowledge of agents, APIs, or neural nets."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from connect4.env.board import Board, Cell
from connect4.env.constants import (
    COLS, 
    ROWS,
    PLAYER_ONE, 
    PLAYER_TWO,
)


class ConnectFourEnv:
    ROWS = ROWS
    COLS = COLS

    def __init__(self) -> None:
        self.board = Board()
        self.current_player: int = PLAYER_ONE
        self.terminated: bool = False
        self.winner: Optional[int] = None
        self.winning_cells: Optional[list[Cell]] = None

    def reset(self) -> np.ndarray:
        self.board = Board()
        self.current_player = PLAYER_ONE
        self.terminated = False
        self.winner = None
        self.winning_cells = None
        return self.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Drop a disc for the current player.

        Returns (state, reward, terminated, info). Reward is from the mover's
        perspective: +1 win, 0 otherwise (including draws).
        """
        if self.terminated:
            raise ValueError("Game already finished; call reset().")
        if not self.is_valid_action(action):
            raise ValueError(f"Illegal action {action}, valid={self.get_valid_actions()}.")

        mover = self.current_player
        self.board.drop(action, mover)

        winner, cells = self.board.check_winner()
        draw = winner is None and self.board.is_full()
        self.terminated = winner is not None or draw
        self.winner = winner
        self.winning_cells = cells

        reward = 1.0 if winner == mover else 0.0
        if not self.terminated:
            self.current_player = PLAYER_TWO if mover == PLAYER_ONE else PLAYER_ONE

        info = {
            "winner": self.winner,
            "winning_cells": self.winning_cells,
            "action": action,
            "draw": draw,
        }
        return self.get_state(), reward, self.terminated, info

    def get_valid_actions(self) -> list[int]:
        if self.terminated:
            return []
        return self.board.valid_columns()

    def is_valid_action(self, action: int) -> bool:
        if self.terminated:
            return False
        if not 0 <= action < self.COLS:
            return False
        return not self.board.is_column_full(action)

    def check_winner(self) -> tuple[Optional[int], Optional[list[Cell]]]:
        return self.board.check_winner()

    def is_draw(self) -> bool:
        return self.board.is_draw()

    def clone(self) -> ConnectFourEnv:
        env = ConnectFourEnv()
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.terminated = self.terminated
        env.winner = self.winner
        env.winning_cells = None if self.winning_cells is None else list(self.winning_cells)
        return env

    def get_state(self) -> np.ndarray:
        return self.board.grid.copy()

    def get_canonical_state(self) -> np.ndarray:
        """Board from the current player's view: +1 self, -1 opponent."""
        return (self.board.grid.astype(np.int8) * self.current_player).astype(np.int8)

    def render(self) -> str:
        header = f"Player to move: {self.current_player}  (X=1, O=-1)"
        if self.terminated:
            if self.winner is not None:
                header = f"Winner: {self.winner}"
            else:
                header = "Draw"
        return f"{header}\n{self.board.render()}"
