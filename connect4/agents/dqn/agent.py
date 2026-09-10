"""Greedy DQN policy. Training epsilon-greedy lives in DQNTrainer, not here.

'select_action' takes the absolute board (same handshake as every other
agent) and converts it to canonical planes internally.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from connect4.agents.base import Agent, AgentDecision
from connect4.agents.dqn.network import (
    QNetwork,
    greedy_action,
    planes_from_state,
    softmax_probs,
)
from connect4.agents.dqn.trainer import DQNConfig, config_from_dict
from connect4.env.constants import COLS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "dqn" / "dqn.pt"


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | None = None,
) -> tuple[QNetwork, DQNConfig, dict]:
    device = device or torch.device("cpu")
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    config = config_from_dict(ckpt.get("config", {}))
    net = QNetwork(hidden=config.hidden)
    net.load_state_dict(ckpt["model"])
    net.to(device)
    net.eval()
    return net, config, ckpt


class DQNAgent(Agent):
    name = "dqn"

    def __init__(
        self,
        net: QNetwork | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: torch.device | None = None,
        config: DQNConfig | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.checkpoint_path = Path(checkpoint) if checkpoint is not None else None
        extra: dict = {}
        if checkpoint is not None:
            self.net, self.config, extra = load_checkpoint(checkpoint, device=self.device)
        else:
            self.config = config or DQNConfig()
            self.net = (net or QNetwork(hidden=self.config.hidden)).to(self.device)
            self.net.eval()
        self._ckpt_meta = {
            "steps": extra.get("steps"),
            "episodes": extra.get("episodes"),
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path | None = None,
        *,
        device: torch.device | None = None,
    ) -> DQNAgent:
        return cls(checkpoint=path or DEFAULT_CHECKPOINT, device=device)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        planes = planes_from_state(state)
        with torch.inference_mode():
            q = self.net(torch.from_numpy(planes).unsqueeze(0).to(self.device))
        return q[0].detach().cpu().numpy()

    def select_action(self, state: np.ndarray, valid_actions: list[int]) -> AgentDecision:
        if not valid_actions:
            raise ValueError("No valid actions.")
        t0 = time.perf_counter()
        q = self.q_values(state)
        action = greedy_action(q, valid_actions)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        q_meta = {str(c): float(q[c]) for c in range(COLS)}
        return AgentDecision(
            action=action,
            probabilities=softmax_probs(q, valid_actions),
            value=float(q[action]),
            metadata={
                "policy": "dqn_greedy",
                "epsilon": 0.0,
                "q": q_meta,
                "hidden": self.config.hidden,
                "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
                "elapsed_ms": float(elapsed_ms),
                **{k: v for k, v in self._ckpt_meta.items() if v is not None},
            },
        )
