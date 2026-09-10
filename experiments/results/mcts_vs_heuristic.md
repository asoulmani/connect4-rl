# MCTS vs Heuristic — PUCT + uniform prior + random rollouts

MCTS against the full heuristic (win / block / fork). Random rollouts are noisy; terminals in the tree still encode instant wins and hangs. The question is whether a simulation budget beats a 1-ply tactician.

- Games per budget: **100** (50 paired openings × 2 seats)
- Random opening plies: **4** (same opening at every simulation count)
- Simulations: `[50, 200, 800]`
- Generated: `2026-09-02T16:51:23.755052+00:00`

| Sims | Win% | Draw% | P1 win% | P2 win% | Tree nodes/move | Latency |
|-----:|-----:|------:|--------:|--------:|----------------:|--------:|
| 50 | 9.0% | 3.0% | 10.0% | 8.0% | 278 | 16.4 ms |
| 200 | 36.0% | 3.0% | 40.0% | 32.0% | 890 | 62.8 ms |
| 800 | 64.0% | 3.0% | 76.0% | 52.0% | 3,126 | 264.8 ms |

## Takeaway

50 simulations lose (9%). A 1-ply tactician is cheap and sharp; random rollouts are not. 200 sims reach 36% — still below Minimax depth 2 vs the same opponent (57% at 7 ms). 800 sims reach 64% at ~265 ms/move, about the same wall time as Minimax depth 5 (79% at 268 ms). Extra simulations help. At matched latency, this leaf is weaker than alpha-beta.
