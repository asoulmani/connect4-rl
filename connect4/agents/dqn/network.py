"""Compact Q-network: 2x6x7 canonical planes -> 7 column Q-values.

Input convention
----------------
Planes are built from 'get_canonical_state()': channel 0 is the
player-to-move discs (+1), channel 1 is the opponent (-1). Empty cells
are 0 on both channels.

Illegal actions are not a network output constraint. Callers mask Q-values
with :data:'ILLEGAL_Q' before 'argmax' / Bellman max / softmax.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connect4.env.constants import COLS, PLAYER_ONE, ROWS

# Finite so softmax stays defined. Large enough that masked-max ignores it.
ILLEGAL_Q = -1.0e9

N_PLANES = 2
PLANE_SHAPE = (N_PLANES, ROWS, COLS)


def seat_from_state(state: np.ndarray) -> int:
    """Absolute-board seat: even disc count ⇒ P1. Same rule as the heuristic."""
    n = int(np.count_nonzero(state))
    return PLAYER_ONE if n % 2 == 0 else -PLAYER_ONE


def canonical_from_state(state: np.ndarray) -> np.ndarray:
    """Absolute '+1/-1/0' board → player-to-move canonical '+1/-1/0'."""
    seat = seat_from_state(state)
    return (np.asarray(state, dtype=np.int8) * seat).astype(np.int8)


def planes_from_canonical(canonical: np.ndarray) -> np.ndarray:
    """(6, 7) canonical board → float32 (2, 6, 7) (me, opponent)."""
    board = np.asarray(canonical, dtype=np.int8)
    if board.shape != (ROWS, COLS):
        raise ValueError(f"canonical board must be {(ROWS, COLS)}, got {board.shape}.")
    planes = np.empty(PLANE_SHAPE, dtype=np.float32)
    planes[0] = board == 1
    planes[1] = board == -1
    return planes


def planes_from_state(state: np.ndarray) -> np.ndarray:
    """Absolute board (agent handshake) → network planes."""
    return planes_from_canonical(canonical_from_state(state))


def legal_mask(valid_actions: list[int], n_actions: int = COLS) -> np.ndarray:
    """Bool vector, True on legal columns. Empty list ⇒ all False."""
    mask = np.zeros(n_actions, dtype=bool)
    if valid_actions:
        mask[list(valid_actions)] = True
    return mask


def mask_q(q: torch.Tensor, legal: torch.Tensor, fill: float = ILLEGAL_Q) -> torch.Tensor:
    """'q' and 'legal' broadcast on the last dim (size 7)."""
    return q.masked_fill(~legal, fill)


def greedy_action(q_values: np.ndarray | torch.Tensor, valid_actions: list[int]) -> int:
    """Argmax over legal columns only. Ties keep the first legal index."""
    if not valid_actions:
        raise ValueError("No valid actions.")
    if isinstance(q_values, torch.Tensor):
        q_values = q_values.detach().cpu().numpy()
    q = np.asarray(q_values, dtype=np.float64).reshape(-1)
    best = valid_actions[0]
    best_q = q[best]
    for a in valid_actions[1:]:
        if q[a] > best_q:
            best_q = q[a]
            best = a
    return int(best)


def epsilon_greedy_action(
    q_values: np.ndarray | torch.Tensor,
    valid_actions: list[int],
    epsilon: float,
    rng: random.Random,
) -> int:
    """epsilon-random samples legal columns only. Greedy is masked argmax."""
    if not valid_actions:
        raise ValueError("No valid actions.")
    if rng.random() < epsilon:
        return int(rng.choice(valid_actions))
    return greedy_action(q_values, valid_actions)


def softmax_probs(q_values: np.ndarray | torch.Tensor, valid_actions: list[int]) -> list[float]:
    """Softmax over masked Q. Visualization only, not the training policy."""
    if isinstance(q_values, torch.Tensor):
        q_values = q_values.detach().cpu().numpy()
    q = np.asarray(q_values, dtype=np.float64).reshape(-1)
    logits = np.full(COLS, ILLEGAL_Q, dtype=np.float64)
    for a in valid_actions:
        logits[a] = q[a]
    t = torch.tensor(logits, dtype=torch.float64)
    probs = F.softmax(t, dim=0).numpy()
    return [float(p) for p in probs]


class QNetwork(nn.Module):
    """MLP over flattened 2x6x7 planes. Hidden size is a DQNConfig field."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        n_in = N_PLANES * ROWS * COLS
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, COLS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, 6, 7) → (B, 7) Q-values."""
        return self.net(x)
