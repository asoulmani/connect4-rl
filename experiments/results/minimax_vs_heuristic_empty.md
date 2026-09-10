# Minimax vs heuristic — empty board (sanity check)

Deterministic agents, empty board, both seats. Two games per depth, not a sample. Game-theoretic Connect Four: first player wins with perfect play (center). This table is a labeled sanity line, not a win-rate estimate.

- Opponent: **heuristic**
- Depths: `[1, 2, 3, 4, 5, 6, 7]`
- Generated: `2026-09-04T13:49:54.528857+00:00`

| Depth | As P1 | As P2 | Nodes/move | Mean ms | Median ms |
|------:|------:|------:|-----------:|--------:|----------:|
| 1 | loss | loss | 8 | 1.3 | 1.3 |
| 2 | loss | draw | 38 | 6.1 | 8.4 |
| 3 | loss | win | 166 | 23.4 | 26.6 |
| 4 | win | loss | 637 | 90.9 | 93.2 |
| 5 | win | win | 2,119 | 273.6 | 219.4 |
| 6 | win | win | 5,579 | 755.7 | 423.3 |
| 7 | win | win | 20,313 | 2517.6 | 1796.0 |

Latency is mean ms per labelled minimax move, averaged over both seats.
