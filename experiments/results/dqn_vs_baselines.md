# DQN vs baselines

Compact from-scratch DQN (replay, target net, ε-greedy, illegal-action masking, negamax Bellman target). Frozen 4-ply openings, both seats.

- Games per opponent: **40** (20 paired openings × 2 seats)
- Opening plies: **4**
- Training episodes: **30000** (~563k env steps)
- Checkpoint: `models/dqn/dqn.pt`
- Generated: `2026-09-09T19:35:28.484171+00:00`

| Opponent | Games | Win% | Draw% | P1 win% | P2 win% |
|----------|------:|-----:|------:|--------:|--------:|
| random | 40 | 80.0% | 0.0% | 80.0% | 80.0% |
| heuristic | 40 | 0.0% | 0.0% | 0.0% | 0.0% |
| minimax-d1 | 40 | 15.0% | 0.0% | 25.0% | 5.0% |
| minimax-d2 | 40 | 5.0% | 0.0% | 10.0% | 0.0% |

Same protocol as the 3000-episode run. 10× games did not produce a tactician. n = 40: 80% vs Random is 32/40 vs 36/40 at 3k; treat that dip as noise.

This is a time-boxed strength check, not a claim of SOTA Connect Four play.
