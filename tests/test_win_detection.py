"""Horizontal, vertical, and diagonal four-in-a-row, plus draws."""

from connect4.env.board import Board
from connect4.env.constants import PLAYER_ONE, PLAYER_TWO, ROWS


def _drop_many(board: Board, columns: list[int], start=PLAYER_ONE) -> None:
    player = start
    for col in columns:
        board.drop(col, player)
        player = -player


def test_horizontal_victory():
    board = Board()
    # P1: 0,1,2,3 on the bottom row; P2 fills column 6 to stay out of the way.
    moves = [0, 6, 1, 6, 2, 6, 3]
    _drop_many(board, moves)
    winner, cells = board.check_winner()
    assert winner == PLAYER_ONE
    assert cells == [(ROWS - 1, 0), (ROWS - 1, 1), (ROWS - 1, 2), (ROWS - 1, 3)]


def test_vertical_victory():
    board = Board()
    # P1 stacks column 2 four times; P2 plays column 3.
    moves = [2, 3, 2, 3, 2, 3, 2]
    _drop_many(board, moves)
    winner, cells = board.check_winner()
    assert winner == PLAYER_ONE
    assert set(cells) == {(2, 2), (3, 2), (4, 2), (5, 2)}


def test_positive_diagonal_victory():
    """Diagonal up-right from bottom-left: (5,0),(4,1),(3,2),(2,3)."""
    board = Board()
    # Heights before P1's winning drop:
    # col0: P1
    # col1: P2, P1
    # col2: P2, P1, P1  — need extra P2 filler
    # col3: P2, P1, P2, P1
    #
    # Sequence (P1 starts):
    # 0, 1, 1, 2, 2, 3, 2, 3, 4, 3, 3
    # After these, P1 has (5,0),(4,1),(3,2) and plays 3 for (2,3).
    # Let's simulate carefully with explicit drops.
    board.drop(0, PLAYER_ONE)  # (5,0)
    board.drop(1, PLAYER_TWO)
    board.drop(1, PLAYER_ONE)  # (4,1)
    board.drop(2, PLAYER_TWO)
    board.drop(2, PLAYER_ONE)
    board.drop(2, PLAYER_TWO)  # wait this messes heights

    board = Board()
    # Build stairs with explicit players:
    # Bottom row mixed so P1 can climb.
    # (5,0)=P1
    board.drop(0, PLAYER_ONE)
    # col1: P2 then P1 → (5,1)=P2 (4,1)=P1
    board.drop(1, PLAYER_TWO)
    board.drop(1, PLAYER_ONE)
    # col2: P2, P2, P1 → (5,2)=P2 (4,2)=P2 (3,2)=P1
    board.drop(2, PLAYER_TWO)
    board.drop(2, PLAYER_TWO)
    board.drop(2, PLAYER_ONE)
    # col3: P1, P2, P1, P1 → we need (2,3)=P1 with three underneath
    board.drop(3, PLAYER_ONE)
    board.drop(3, PLAYER_TWO)
    board.drop(3, PLAYER_ONE)
    board.drop(3, PLAYER_ONE)  # (2,3)

    winner, cells = board.check_winner()
    assert winner == PLAYER_ONE
    assert set(cells) == {(5, 0), (4, 1), (3, 2), (2, 3)}


def test_negative_diagonal_victory():
    """Diagonal down-right: (2,0),(3,1),(4,2),(5,3)."""
    board = Board()
    # (5,3)=P1
    board.drop(3, PLAYER_ONE)
    # col2: P2 then P1 → (5,2)=P2 (4,2)=P1
    board.drop(2, PLAYER_TWO)
    board.drop(2, PLAYER_ONE)
    # col1: three discs, top is P1: (5,1),(4,1) not P1, (3,1)=P1
    board.drop(1, PLAYER_TWO)
    board.drop(1, PLAYER_TWO)
    board.drop(1, PLAYER_ONE)
    # col0: four discs, (2,0)=P1
    board.drop(0, PLAYER_ONE)
    board.drop(0, PLAYER_TWO)
    board.drop(0, PLAYER_ONE)
    board.drop(0, PLAYER_ONE)

    winner, cells = board.check_winner()
    assert winner == PLAYER_ONE
    assert set(cells) == {(2, 0), (3, 1), (4, 2), (5, 3)}


def test_no_winner_on_empty_and_partial():
    board = Board()
    assert board.check_winner() == (None, None)
    board.drop(3, PLAYER_ONE)
    board.drop(3, PLAYER_TWO)
    assert board.check_winner() == (None, None)
    assert not board.is_draw()


def test_draw_full_board_without_four():
    """Full board with paired stacks so neither color has four in a row."""
    import numpy as np

    board = Board()
    # Bottom-up stacks: AABBAA vs BBAABB; column 3 uses AABBAA, others BBAABB.
    pattern = [
        [-1, -1, -1, 1, -1, -1, -1],
        [-1, -1, -1, 1, -1, -1, -1],
        [1, 1, 1, -1, 1, 1, 1],
        [1, 1, 1, -1, 1, 1, 1],
        [-1, -1, -1, 1, -1, -1, -1],
        [-1, -1, -1, 1, -1, -1, -1],
    ]
    board.grid = np.array(pattern, dtype=np.int8)
    winner, cells = board.check_winner()
    assert winner is None
    assert cells is None
    assert board.is_full()
    assert board.is_draw()
