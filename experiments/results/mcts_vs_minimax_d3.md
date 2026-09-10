# MCTS vs Minimax(d=3) — PUCT + uniform prior + random rollouts

Opponent is `MinimaxAgent(depth=3)`. MCTS is PUCT with a uniform prior and random rollout evaluation. The axis is simulations per move, not depth. Pick a minimax depth whose latency is comparable (see `--probe`).

- Games per budget: **100** (50 paired openings × 2 seats)
- Random opening plies: **4** (same opening at every simulation count)
- Simulations: `[50, 200, 800]`
- Generated: `2026-09-02T17:03:07.956834+00:00`

| Sims | Win% | Draw% | P1 win% | P2 win% | Tree nodes/move | Latency |
|-----:|-----:|------:|--------:|--------:|----------------:|--------:|
| 50 | 9.0% | 0.0% | 14.0% | 4.0% | 298 | 16.4 ms |
| 200 | 26.0% | 5.0% | 30.0% | 22.0% | 949 | 65.0 ms |
| 800 | 66.0% | 3.0% | 70.0% | 62.0% | 3,285 | 276.0 ms |

## Takeaway

Minimax depth 3 is ~22–36 ms/move on the latency probe. 50 sims are cheaper (16 ms) and lose 9–9. 200 sims are ~2× depth-3 time and still lose (26%). 800 sims are ~8–10× depth-3 time and win 66%. Extra test-time compute buys strength, but you only beat this fixed-depth searcher when you spend several times the compute. Latency is the fair comparison, not tree_nodes vs alphabeta nodes.
