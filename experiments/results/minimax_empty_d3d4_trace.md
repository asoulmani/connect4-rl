# Empty-board d3 vs d4: where the games split

Replay of the four deterministic games vs the heuristic (empty board).

Not a bug: both seats diverge on the first unforced minimax move. The scores at that ply are static `evaluate()` values, not `WIN_SCORE` terminals. Odd depth ends on our ply (leaf likes the attack we just made). Even depth ends on the opponent's reply.

## As P1 (d3 loss, d4 win)

Shared prefix: `3` (minimax), `3` (heuristic). Board:

```
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . O . . .
. . . X . . .
```

Ply 2, P1 to move:

| col | d3 | d4 |
|----:|---:|---:|
| 0 | 1 | −22 |
| 1 | 12 | −12 |
| 2 | **92** (played) | −1 |
| 3 | 24 | **0** (played) |
| 4 | 92 | −1 |
| 5 | 12 | −12 |
| 6 | 1 | −22 |

Playing 2 (or 4) puts `X X` on the bottom next to the center stone. That is three unblocked 2-windows; `evaluate()` loves it. Depth 3's horizon stops after P1's follow-up, so +92. Depth 4 sees P2 answer that shape and scores it −1, so it stacks the center instead.

Full winners: d3 → heuristic (P2); d4 → minimax (P1).

## As P2 (d3 win, d4 loss)

Shared prefix: `3` (heuristic). Ply 1, P2 to move:

| col | d3 | d4 |
|----:|---:|---:|
| 0 | −13 | −105 |
| 1 | −3 | **−27** (played) |
| 2 | −1 | −44 |
| 3 | **0** (played) | −92 |
| 4 | −1 | −44 |
| 5 | −3 | −27 |
| 6 | −13 | −105 |

Depth 3 stacks the center. Depth 4 treats stacking as −92 (one extra P1 ply in the leaf) and plays column 1. That line loses; the stacking line is the one that beat the heuristic.

Full winners: d3 → minimax (P2); d4 → heuristic (P1).
