"""Connect Four board constants."""

ROWS = 6
COLS = 7
CONNECT = 4

EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = -1

# (row_delta, col_delta): scan 4 in a row in these directions only
# (the opposite direction is covered by starting from the other end).
WIN_DIRECTIONS = (
    (0, 1),   # horizontal
    (1, 0),   # vertical
    (1, 1),   # diagonal down-right
    (1, -1),  # diagonal down-left
)
