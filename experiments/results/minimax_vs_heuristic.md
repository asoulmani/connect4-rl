# Minimax vs Heuristic — search depth sweep

Minimax against the full heuristic (win / block / fork), not against its own leaf. The window/center features are shared; the tactics are not. Depth 1 is a weak 1-ply scorer. Depth 2 is the first ply that can see a block.

- Games per depth: **200** (100 paired openings × 2 seats)
- Random opening plies: **4** (same opening at every depth)
- Depths: `[1, 2, 3, 4, 5, 6]`
- Generated: `2026-08-22T14:11:13.705366+00:00`

| Depth | Win% | Draw% | P1 win% | P2 win% | Nodes/move | Latency |
|------:|-----:|------:|--------:|--------:|-----------:|--------:|
| 1 | 14.5% | 0.5% | 22.0% | 7.0% | 7 | 1.3 ms |
| 2 | 57.0% | 6.5% | 57.0% | 57.0% | 41 | 6.6 ms |
| 3 | 58.5% | 2.5% | 60.0% | 57.0% | 159 | 22.7 ms |
| 4 | 68.5% | 6.5% | 73.0% | 64.0% | 601 | 88.1 ms |
| 5 | 79.0% | 2.5% | 85.0% | 73.0% | 2,022 | 267.9 ms |
| 6 | 77.0% | 4.5% | 83.0% | 71.0% | 7,456 | 1028.9 ms |

## Takeaway

Depth 1 loses (14.5%) because it cannot see blocks. Depth 2 jumps to 57% once search looks one ply ahead. 2→3 is almost flat. 5→6 spends ~4× the nodes and gets slightly worse — odd/even horizon, or just this matchup.

This experiment answers "can search beat a tactician?" For "what does extra depth buy with the leaf held fixed?" see `minimax_vs_d1`.
