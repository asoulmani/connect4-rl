"""Board gravity, legal columns, and clone independence."""

import numpy as np
import pytest

from connect4.env.board import Board
from connect4.env.constants import COLS, EMPTY, PLAYER_ONE, PLAYER_TWO, ROWS


def test_board_starts_empty():
    board = Board()
    assert board.grid.shape == (ROWS, COLS)
    assert board.grid.dtype == np.int8
    assert np.all(board.grid == EMPTY)
    assert board.valid_columns() == list(range(COLS))


def test_piece_gravity_fills_from_bottom():
    board = Board()
    r0, c0 = board.drop(3, PLAYER_ONE)
    assert (r0, c0) == (ROWS - 1, 3)
    r1, c1 = board.drop(3, PLAYER_TWO)
    assert (r1, c1) == (ROWS - 2, 3)
    assert board.grid[ROWS - 1, 3] == PLAYER_ONE
    assert board.grid[ROWS - 2, 3] == PLAYER_TWO


def test_full_column_detection():
    board = Board()
    for i in range(ROWS):
        player = PLAYER_ONE if i % 2 == 0 else PLAYER_TWO
        board.drop(0, player)
    assert board.is_column_full(0)
    assert 0 not in board.valid_columns()
    with pytest.raises(ValueError, match="full"):
        board.drop(0, PLAYER_ONE)


def test_illegal_column_index():
    board = Board()
    with pytest.raises(ValueError, match="Column must be"):
        board.drop(-1, PLAYER_ONE)
    with pytest.raises(ValueError, match="Column must be"):
        board.drop(COLS, PLAYER_ONE)


def test_copy_is_independent():
    board = Board()
    board.drop(1, PLAYER_ONE)
    clone = board.copy()
    clone.drop(2, PLAYER_TWO)
    assert board.grid[ROWS - 1, 2] == EMPTY
    assert clone.grid[ROWS - 1, 1] == PLAYER_ONE
