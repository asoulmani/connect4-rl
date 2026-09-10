"""Circular replay buffer for compact DQN.

Each row is one mover-centric transition already encoded as network planes:

(s, a, r, s_next, done, legal_next)``

s / s_next are (2, 6, 7) float32. 'legal_next' is a length-7bool mask
of columns that are legal at s_next (all False if 'done').
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from connect4.agents.dqn.network import PLANE_SHAPE
from connect4.env.constants import COLS


@dataclass
class ReplayBatch:
    states: torch.Tensor  # (B, 2, 6, 7) float32
    actions: torch.Tensor  # (B,) int64
    rewards: torch.Tensor  # (B,) float32
    next_states: torch.Tensor  # (B, 2, 6, 7) float32
    dones: torch.Tensor  # (B,) float32 in {0, 1}
    legal_next: torch.Tensor  # (B, 7) bool


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._states = np.zeros((capacity, *PLANE_SHAPE), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, *PLANE_SHAPE), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._legal_next = np.zeros((capacity, COLS), dtype=bool)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        legal_next: np.ndarray,
    ) -> None:
        i = self._idx
        self._states[i] = np.asarray(state, dtype=np.float32)
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._next_states[i] = np.asarray(next_state, dtype=np.float32)
        self._dones[i] = 1.0 if done else 0.0
        mask = np.asarray(legal_next, dtype=bool)
        if mask.shape != (COLS,):
            raise ValueError(f"legal_next must have shape ({COLS},), got {mask.shape}.")
        self._legal_next[i] = mask
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | None = None) -> ReplayBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self._size < batch_size:
            raise ValueError(
                f"Not enough transitions to sample {batch_size} (have {self._size})."
            )
        idx = self._rng.integers(0, self._size, size=batch_size, dtype=np.int64)
        dev = device or torch.device("cpu")
        return ReplayBatch(
            states=torch.from_numpy(self._states[idx]).to(dev),
            actions=torch.from_numpy(self._actions[idx]).to(dev),
            rewards=torch.from_numpy(self._rewards[idx]).to(dev),
            next_states=torch.from_numpy(self._next_states[idx]).to(dev),
            dones=torch.from_numpy(self._dones[idx]).to(dev),
            legal_next=torch.from_numpy(self._legal_next[idx]).to(dev),
        )
