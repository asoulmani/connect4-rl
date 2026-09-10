"""Shared agent interface. All opponents (random through AlphaZero) implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class AgentDecision:
    action: int
    probabilities: Optional[list[float]] = None
    value: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def select_action(self, state: np.ndarray, valid_actions: list[int]) -> AgentDecision:
        """Choose a legal column. 'state' is the raw board (1 / -1 / 0)."""
