"""From-scratch DQN trainer: replay, online net, target net, Huber loss.

Bellman target (why not vanilla r + gamma *max Q(s'))
----------------------------------------------------
Replay stores canonical planes. After a non-terminal move the environment
flips 'current_player', so 's_next' is the opponent's player-to-move
view. 'env.step' reward is mover-centric: +1 on a win, else 0
(the loser never receives -1). Changing that reward would break existing
tests, the sign flip belongs here.

Let Q(s, a) be the value of a column for the player about to move.

- Terminal (we won or drew): y = r  (r in {0, 1}).
- Non-terminal: opponent to move from 's_next', so

  y = r + gamma * (- max_{a' legal} Q_target(s_next, a'))

  with r = 0 in this env. Vectorized:

  y = r + (1 - done) * gamma * (- max_legal Q_target(s_next))

Illegal next columns are filled with :data:'ILLEGAL_Q' before the max.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from connect4.agents.dqn.network import (
    ILLEGAL_Q,
    QNetwork,
    epsilon_greedy_action,
    legal_mask,
    planes_from_canonical,
)
from connect4.agents.dqn.replay_buffer import ReplayBuffer, ReplayBatch
from connect4.env.environment import ConnectFourEnv


@dataclass
class DQNConfig:
    hidden: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30_000
    target_update_every: int = 500
    warmup_steps: int = 1_000
    max_grad_norm: float = 10.0
    seed: int = 0


def compute_bellman_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q: torch.Tensor,
    legal_next: torch.Tensor,
    gamma: float,
    illegal_value: float = ILLEGAL_Q,
) -> torch.Tensor:
    """Negamax DQN targets. ``dones`` is float ``(B,)`` in ``{0, 1}``."""
    masked = next_q.masked_fill(~legal_next, illegal_value)
    max_next = masked.max(dim=1).values
    return rewards + (1.0 - dones) * gamma * (-max_next)


def config_from_dict(data: dict[str, Any]) -> DQNConfig:
    allowed = {f.name for f in fields(DQNConfig)}
    return DQNConfig(**{k: v for k, v in data.items() if k in allowed})


class DQNTrainer:
    def __init__(
        self,
        config: DQNConfig | None = None,
        *,
        device: torch.device | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.device = device or torch.device("cpu")
        self.rng = rng or random.Random(self.config.seed)
        np_rng = np.random.default_rng(self.config.seed)

        self.online = QNetwork(hidden=self.config.hidden).to(self.device)
        self.target = QNetwork(hidden=self.config.hidden).to(self.device)
        self.sync_target()
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_size, rng=np_rng)
        self.steps = 0
        self.episodes = 0
        self.last_loss: float | None = None

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def epsilon(self, step: int | None = None) -> float:
        t = min(1.0, (step if step is not None else self.steps) / max(1, self.config.epsilon_decay_steps))
        return float(
            self.config.epsilon_start
            + t * (self.config.epsilon_end - self.config.epsilon_start)
        )

    def act_epsilon_greedy(self, env: ConnectFourEnv) -> tuple[int, np.ndarray, list[int]]:
        """Choose a legal action from ``env`` with the current ε. Returns (a, planes, valid)."""
        valid = env.get_valid_actions()
        planes = planes_from_canonical(env.get_canonical_state())
        self.online.eval()
        with torch.no_grad():
            q = self.online(torch.from_numpy(planes).unsqueeze(0).to(self.device))[0]
        action = epsilon_greedy_action(q, valid, self.epsilon(), self.rng)
        return action, planes, valid

    def store_transition(
        self,
        planes: np.ndarray,
        action: int,
        reward: float,
        env: ConnectFourEnv,
        done: bool,
    ) -> None:
        next_planes = planes_from_canonical(env.get_canonical_state())
        self.buffer.push(
            planes,
            action,
            reward,
            next_planes,
            done,
            legal_mask(env.get_valid_actions()),
        )

    def train_step(self) -> float:
        cfg = self.config
        batch: ReplayBatch = self.buffer.sample(cfg.batch_size, device=self.device)
        self.online.train()
        q = self.online(batch.states)
        q_sa = q.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target(batch.next_states)
            targets = compute_bellman_targets(
                batch.rewards,
                batch.dones,
                next_q,
                batch.legal_next,
                cfg.gamma,
            )

        loss = F.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        loss_f = float(loss.item())
        self.last_loss = loss_f
        return loss_f

    def maybe_update(self) -> float | None:
        """One SGD step if the buffer is warm. Periodic hard target copy."""
        self.steps += 1
        loss: float | None = None
        cfg = self.config
        if self.steps >= cfg.warmup_steps and len(self.buffer) >= cfg.batch_size:
            loss = self.train_step()
        if self.steps % cfg.target_update_every == 0:
            self.sync_target()
        return loss

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.online.state_dict(),
                "config": asdict(self.config),
                "steps": self.steps,
                "episodes": self.episodes,
            },
            path,
        )

    def load_weights(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["model"])
        self.sync_target()
        self.steps = int(ckpt.get("steps", self.steps))
        self.episodes = int(ckpt.get("episodes", self.episodes))
