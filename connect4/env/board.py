"""Pure board mechanics: gravity, legal columns, win/draw detection."""

from __future__ import annotations

from typing import Optional

import numpy as np

from connect4.env.constants import (
    COLS,
    ROWS,
    CONNECT,
    EMPTY,
    WIN_DIRECTIONS,
)

BoardArray = np.ndarray
Cell = tuple[int, int]


class Board:
    """6x7 Connect Four grid. Knows rules, not players' identities beyond piece values."""

    def __init__(self, grid: Optional[BoardArray] = None) -> None:
        if grid is None:
            self.grid = np.zeros((ROWS, COLS), dtype=np.int8)
        else:
            if grid.shape != (ROWS, COLS):
                raise ValueError(f"Board must be shape {(ROWS, COLS)}, got {grid.shape}")
            self.grid = np.array(grid, dtype=np.int8, copy=True)

    def copy(self) -> Board:
        return Board(self.grid)

    def valid_columns(self) -> list[int]:
        """Columns whose top cell is empty (a piece can still drop)."""
        return [c for c in range(COLS) if self.grid[0, c] == EMPTY]

    def is_column_full(self, column: int) -> bool:
        self._require_column(column)
        return bool(self.grid[0, column] != EMPTY)

    def drop(self, column: int, player: int) -> Cell:
        """Place 'player' in 'column' with gravity. Returns (row, col) of the disc."""
        if self.is_column_full(column):
            raise ValueError(f"Column {column} is full")
        for row in range(ROWS - 1, -1, -1):
            if self.grid[row, column] == EMPTY:
                self.grid[row, column] = player
                return row, column
        raise RuntimeError("Unreachable: column reported not full but no empty cell.")

    def check_winner(self) -> tuple[Optional[int], Optional[list[Cell]]]:
        """Return (winner, four cells) or (None, None). First line found wins the scan."""
        for row in range(ROWS):
            for col in range(COLS):
                player = int(self.grid[row, col])
                if player == EMPTY:
                    continue
                for dr, dc in WIN_DIRECTIONS:
                    cells = self._line_from(row, col, dr, dc, player)
                    if cells is not None:
                        return player, cells
        return None, None

    def is_full(self) -> bool:
        return bool(np.all(self.grid != EMPTY))

    def is_draw(self) -> bool:
        winner, _ = self.check_winner()
        return winner is None and self.is_full()

    def render(self) -> str:
        """Return a string representation of the board for terminal display."""
        symbols = {EMPTY: ".", 1: "X", -1: "O"}
        lines = ["  " + " ".join(str(c) for c in range(COLS))]
        for r in range(ROWS):
            cells = " ".join(symbols[int(self.grid[r, c])] for c in range(COLS))
            lines.append(f"{r} {cells}")
        return "\n".join(lines)

    def _line_from(
        self, row: int, col: int, dr: int, dc: int, player: int
    ) -> Optional[list[Cell]]:
        """Return a list of cells in the direction of the player, or None if the line is not a win."""
        cells: list[Cell] = []
        for k in range(CONNECT):
            nr = row + dr * k
            nc = col + dc * k
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                return None
            if int(self.grid[nr, nc]) != player:
                return None
            cells.append((nr, nc))
        return cells

    @staticmethod
    def _require_column(column: int) -> None:
        if not 0 <= column < COLS:
            raise ValueError(f"Column must be in 0..{COLS - 1}, got {column}.")
