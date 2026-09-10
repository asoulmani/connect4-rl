# Minimax vs Minimax(d=1) — same evaluator, depth changes

Opponent is `MinimaxAgent(depth=1)`. Same `evaluate()`, same search, only the depth integer changes. Depth 1 vs itself is the control: identical players, so wins and losses should split (draws aside).

- Games per depth: **100** (50 paired openings × 2 seats)
- Random opening plies: **4** (same opening at every depth)
- Depths: `[1, 2, 3, 4, 5]`
- Generated: `2026-08-31T16:12:17.057659+00:00`

| Depth | Win% | Draw% | P1 win% | P2 win% | Nodes/move | Latency |
|------:|-----:|------:|--------:|--------:|-----------:|--------:|
| 1 | 50.0% | 0.0% | 58.0% | 42.0% | 8 | 1.3 ms |
| 2 | 87.0% | 3.0% | 94.0% | 80.0% | 47 | 7.3 ms |
| 3 | 88.0% | 1.0% | 94.0% | 82.0% | 187 | 25.6 ms |
| 4 | 90.0% | 2.0% | 92.0% | 88.0% | 748 | 107.8 ms |
| 5 | 91.0% | 1.0% | 98.0% | 84.0% | 2,666 | 355.7 ms |

## Takeaway

Depth 1 vs itself is exactly 50–50, so the harness isn't cooking the books. Almost all of the strength shows up at depth 2 (87%). Depths 3–5 add a few points while nodes go 47 → 2,666.

Against the heuristic the curve kept climbing through depth 5. Extra search still matters when the opponent has tactics the leaf doesn't. When the opponent is just you at depth 1, one extra ply is most of the story.
