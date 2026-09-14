# Connect Four RL

I'm building Connect Four agents one step at a time: Random, Heuristic, Minimax, MCTS, DQN, then AlphaZero.

The point is to understand what each new idea adds, then measure it under the same evaluation setup.

**Play:** [play.ahmedsoulmani.com](https://play.ahmedsoulmani.com)

**Technical report:** [ahmedsoulmani.com/projects/connect-four](https://www.ahmedsoulmani.com/projects/connect-four)

**Playable now:** Random, Heuristic, Minimax, MCTS, and DQN. AlphaZero is next.

![Opponent menu](docs/figures/ui_menu.png)

![Human vs Minimax depth 5](docs/figures/ui_play.png)

## Play locally

Run the backend and frontend in two terminals from the repo root.

Backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl]"
uvicorn backend.app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

Pick who starts, choose an opponent, and click a column to play. Minimax also has an in-game depth control.

The bars below the board show the agent's analysis of the legal moves. For Minimax they come from its move scores. For MCTS they come from visit counts. They are there to make the agents easier to inspect.

The CLI is still available too:

```bash
python -m connect4.play --agent minimax
python -m connect4.play --agent mcts
python -m connect4.play --agent dqn
```

The trained DQN checkpoint used by the demo is included at:

```text
models/dqn/dqn.pt
```

To train a new one:

```bash
python -m connect4.training.train_dqn --episodes 30000
```

Training and evaluation do not go through the web API. The FastAPI backend only handles the browser demo.

## Agents

Every agent implements the same interface:

```text
select_action(state, valid_actions) -> AgentDecision
```

The CLI, evaluation harness, and web app all use that same interface.


| Agent         | What it does                                                                                                                       |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Random**    | Picks uniformly from the legal columns.                                                                                            |
| **Heuristic** | No search. Checks wins and blocks first, then scores windows, center control, and forks. Moves that lose immediately are rejected. |
| **Minimax**   | Depth-limited minimax with alpha-beta pruning and center-first move ordering. Leaf positions use window and center features.       |
| **MCTS**      | PUCT with a uniform prior and random rollouts. The move with the most visits is played. No neural network yet.                     |
| **DQN**       | Small MLP trained from scratch with replay, a target network, epsilon-greedy exploration, and illegal-action masking.              |


Connect Four is solved and perfect play gives the first player a win. None of these agents is meant to be a solver (yet).

## Experiments

Unless stated otherwise, evaluations use paired random openings, both player seats, and four opening plies.

The full setup, plots, and discussion are in the [technical report](https://www.ahmedsoulmani.com/projects/connect-four).

### Minimax vs Heuristic

The heuristic has explicit win, block, and fork logic. Minimax does not get those rules directly. Its leaf evaluation only uses window and center features.

Depth 1 is therefore a weak one-ply scorer. At depth 2 it can finally see the opponent's reply.

```bash
pip install -e ".[eval]"
python -m connect4.training.evaluate --opponent heuristic --games 200 --depths 1,2,3,4,5,6
```

200 games per depth:


| Depth | Win%      | Draw% | Nodes/move | Latency |
| ----- | --------- | ----- | ---------- | ------- |
| 1     | 14.5%     | 0.5%  | 7          | 1.3 ms  |
| 2     | 57.0%     | 6.5%  | 41         | 6.6 ms  |
| 3     | 58.5%     | 2.5%  | 159        | 23 ms   |
| 4     | 68.5%     | 6.5%  | 601        | 88 ms   |
| 5     | **79.0%** | 2.5%  | 2,022      | 268 ms  |
| 6     | 77.0%     | 4.5%  | 7,456      | 1029 ms |


The big jump is depth 1 to depth 2. Depth 5 is the best tradeoff here. Depth 6 searches almost four times as many nodes and does slightly worse in this matchup.

### Minimax vs Minimax depth 1

This one holds the evaluator fixed and only changes search depth.

Depth 1 against itself is the 50% control.

```bash
python -m connect4.training.evaluate --opponent d1 --games 100 --depths 1,2,3,4,5
```

100 games per depth:


| Depth | Win%  | Draw% | Nodes/move | Latency |
| ----- | ----- | ----- | ---------- | ------- |
| 1     | 50.0% | 0.0%  | 8          | 1.3 ms  |
| 2     | 87.0% | 3.0%  | 47         | 7.3 ms  |
| 3     | 88.0% | 1.0%  | 187        | 26 ms   |
| 4     | 90.0% | 2.0%  | 748        | 108 ms  |
| 5     | 91.0% | 1.0%  | 2,666      | 356 ms  |


Again, most of the gain comes from depth 2. Going deeper still helps against the stronger heuristic, but adds little against a depth-1 copy using the same evaluator.

### MCTS

This version of MCTS deliberately starts simple. PUCT is used for tree selection, priors are uniform, and non-terminal leaves are evaluated with random rollouts.

```bash
python -m connect4.training.evaluate_mcts --opponent heuristic --games 100 --sims 50,200,800
python -m connect4.training.evaluate_mcts --opponent minimax --minimax-depth 3 --games 100 --sims 50,200,800
```

100 games per simulation budget:


| Sims | vs Heuristic | vs Minimax d=3 | Latency |
| ---- | ------------ | -------------- | ------- |
| 50   | 9%           | 9%             | ~16 ms  |
| 200  | 36%          | 26%            | ~64 ms  |
| 800  | 64%          | 66%            | ~270 ms |


More simulations help a lot. At 800 simulations, MCTS beats both opponents, but it still trails depth-5 Minimax against the heuristic at roughly the same wall-clock cost.

That gives a clear next step for MCTS: improve what happens at the leaves instead of only adding more rollouts.

### DQN

The DQN is also intentionally small and from scratch. No Stable-Baselines3.

It uses an MLP, replay buffer, online and target networks, epsilon-greedy exploration, illegal-action masking, and a mover-relative Bellman target.

I trained it for 30,000 self-play episodes and evaluated the greedy policy over 40 games per opponent.

```bash
python -m connect4.training.train_dqn --episodes 30000 --eval-games 40
```


| Opponent    | Win% | At 3k episodes |
| ----------- | ---- | -------------- |
| Random      | 80%  | 90%            |
| Minimax d=1 | 15%  | 10%            |
| Minimax d=2 | 5%   | 0%             |
| Heuristic   | 0%   | 2.5%           |


It learns enough to beat Random consistently, but it never becomes tactically strong.

Going from 3,000 to 30,000 episodes did not really change that. I stopped tuning it there rather than turning this project into a DQN hyperparameter search.

## Repository layout

```text
connect4/env          game rules and environment
connect4/agents       agent implementations
connect4/evaluation   match runner and metrics
connect4/training     training and experiment CLIs
backend/              stateless FastAPI API for the web demo
frontend/             React playground
experiments/results   saved benchmark results
docs/figures/         screenshots used in the README
models/dqn            DQN checkpoint used by the demo
```



## Next

The next step is AlphaZero.

The policy/value network will replace the uniform prior and random rollout evaluation used by the current MCTS agent, then train from games generated by its own search.

I'm leaving the current MCTS and DQN baselines mostly frozen. I want them to stay useful as reference points when the AlphaZero version is added.

## Tools

I used [Cursor](https://cursor.com) as a coding assistant, mainly for tests, the React frontend, and API and deployment work.

The agent logic, experiments, evaluation setup, and [technical report](https://www.ahmedsoulmani.com/projects/connect-four) are the main focus of the project.