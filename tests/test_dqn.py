"""Compact from-scratch DQN: masking, replay shapes, Bellman target, smoke step, ckpt."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from connect4.agents.dqn.agent import DQNAgent
from connect4.agents.dqn.network import (
    COLS,
    QNetwork,
    epsilon_greedy_action,
    greedy_action,
    legal_mask,
    planes_from_canonical,
    planes_from_state,
)
from connect4.agents.dqn.replay_buffer import ReplayBuffer
from connect4.agents.dqn.trainer import DQNConfig, DQNTrainer, compute_bellman_targets
from connect4.env.constants import ROWS
from connect4.env.environment import ConnectFourEnv


pytest.importorskip("torch")


class ConstQ(torch.nn.Module):
    """Ignores the board; returns a fixed Q vector. For masking tests."""

    def __init__(self, q: list[float]) -> None:
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q.unsqueeze(0).expand(x.size(0), -1)


def _full_column_state(column: int) -> tuple[np.ndarray, list[int]]:
    env = ConnectFourEnv()
    env.reset()
    # Alternate players down `column` until it is full (6 rows).
    for _ in range(ROWS):
        env.step(column)
        if not env.terminated and column not in env.get_valid_actions():
            break
        if env.terminated:
            break
    # If the game ended (shouldn't with a single column of 6), still return the grid.
    return env.get_state(), env.get_valid_actions()


def test_greedy_never_chooses_illegal_column():
    state, valid = _full_column_state(0)
    assert 0 not in valid
    net = ConstQ([100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    agent = DQNAgent(net=net)
    decision = agent.select_action(state, valid)
    assert decision.action in valid
    assert decision.action != 0
    assert decision.probabilities is not None
    assert decision.probabilities[0] == pytest.approx(0.0, abs=1e-6)
    assert pytest.approx(sum(decision.probabilities), abs=1e-6) == 1.0


def test_epsilon_random_never_chooses_illegal_column():
    q = np.array([99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    valid = [2, 3, 5]
    rng = random.Random(0)
    for _ in range(200):
        a = epsilon_greedy_action(q, valid, epsilon=1.0, rng=rng)
        assert a in valid


def test_greedy_action_empty_raises():
    with pytest.raises(ValueError, match="No valid"):
        greedy_action(np.zeros(COLS), [])


def test_replay_buffer_sample_shapes_and_dtypes():
    buf = ReplayBuffer(capacity=16, rng=np.random.default_rng(0))
    env = ConnectFourEnv()
    env.reset()
    s = planes_from_canonical(env.get_canonical_state())
    env.step(3)
    s2 = planes_from_canonical(env.get_canonical_state())
    buf.push(s, 3, 0.0, s2, False, legal_mask(env.get_valid_actions()))
    for i in range(7):
        buf.push(s, i % 7, 0.0, s2, i == 6, legal_mask([0, 1, 2]))
    batch = buf.sample(4, device=torch.device("cpu"))
    assert batch.states.shape == (4, 2, 6, 7)
    assert batch.next_states.shape == (4, 2, 6, 7)
    assert batch.actions.shape == (4,)
    assert batch.rewards.shape == (4,)
    assert batch.dones.shape == (4,)
    assert batch.legal_next.shape == (4, 7)
    assert batch.states.dtype == torch.float32
    assert batch.next_states.dtype == torch.float32
    assert batch.actions.dtype == torch.int64
    assert batch.rewards.dtype == torch.float32
    assert batch.dones.dtype == torch.float32
    assert batch.legal_next.dtype == torch.bool


def test_bellman_max_masks_illegal_next_actions():
    # Illegal column 0 has a huge Q. If masking is skipped, max is 50.
    next_q = torch.tensor([[50.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    legal = torch.tensor([[False, True, True, True, True, True, True]])
    rewards = torch.tensor([0.0])
    dones = torch.tensor([0.0])
    y = compute_bellman_targets(rewards, dones, next_q, legal, gamma=1.0)
    assert y.item() == pytest.approx(-1.5)
    y_unmasked = compute_bellman_targets(
        rewards,
        dones,
        next_q,
        torch.ones_like(legal),
        gamma=1.0,
    )
    assert y_unmasked.item() == pytest.approx(-50.0)


def test_bellman_terminal_ignores_next_q():
    next_q = torch.tensor([[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]])
    legal = torch.zeros(1, 7, dtype=torch.bool)  # terminated: no legal moves
    y = compute_bellman_targets(
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        next_q,
        legal,
        gamma=0.99,
    )
    assert y.item() == pytest.approx(1.0)


def test_bellman_uses_negamax_not_vanilla_max():
    next_q = torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    legal = torch.ones(1, 7, dtype=torch.bool)
    y = compute_bellman_targets(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        next_q,
        legal,
        gamma=0.5,
    )
    # Vanilla DQN would be +0.5*2 = +1; zero-sum flip is −0.5*2 = −1.
    assert y.item() == pytest.approx(-1.0)


def test_one_training_step_finite_loss():
    cfg = DQNConfig(
        hidden=32,
        batch_size=8,
        buffer_size=64,
        warmup_steps=0,
        seed=0,
    )
    trainer = DQNTrainer(cfg)
    env = ConnectFourEnv()
    env.reset()
    while len(trainer.buffer) < cfg.batch_size:
        if env.terminated:
            env.reset()
        action, planes, _ = trainer.act_epsilon_greedy(env)
        _, reward, done, _ = env.step(action)
        trainer.store_transition(planes, action, reward, env, done)
    loss = trainer.train_step()
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_checkpoint_save_load_reproducible_inference(tmp_path: Path):
    cfg = DQNConfig(hidden=32, seed=1)
    trainer = DQNTrainer(cfg)
    path = tmp_path / "dqn.pt"
    trainer.save(path)

    env = ConnectFourEnv()
    env.reset()
    env.step(3)
    env.step(2)
    state = env.get_state()
    valid = env.get_valid_actions()

    a = DQNAgent.from_checkpoint(path)
    b = DQNAgent.from_checkpoint(path)
    da = a.select_action(state, valid)
    db = b.select_action(state, valid)
    assert da.action == db.action
    assert da.action in valid
    np.testing.assert_allclose(da.probabilities, db.probabilities, rtol=1e-6, atol=1e-6)
    assert da.value == pytest.approx(db.value, rel=1e-6, abs=1e-6)


def test_select_action_empty_raises():
    agent = DQNAgent(net=QNetwork(hidden=16))
    with pytest.raises(ValueError, match="No valid"):
        agent.select_action(np.zeros((6, 7), dtype=np.int8), [])


def test_planes_from_state_matches_env_canonical():
    env = ConnectFourEnv()
    env.reset()
    env.step(3)
    env.step(4)
    planes_agent = planes_from_state(env.get_state())
    planes_env = planes_from_canonical(env.get_canonical_state())
    np.testing.assert_array_equal(planes_agent, planes_env)
    # P1 to move: the P1 disc in col 3 is "me".
    assert planes_agent[0, 5, 3] == 1.0
    assert planes_agent[1, 5, 4] == 1.0


def test_self_play_episode_fills_buffer_and_can_train():
    cfg = DQNConfig(
        hidden=32,
        batch_size=8,
        buffer_size=256,
        warmup_steps=8,
        target_update_every=20,
        epsilon_decay_steps=50,
        seed=0,
    )
    trainer = DQNTrainer(cfg)
    from connect4.training.train_dqn import play_self_play_episode

    for _ in range(4):
        stats = play_self_play_episode(trainer)
        assert stats["moves"] >= 7
        assert stats["moves"] <= 42
    assert len(trainer.buffer) >= 8
    if trainer.last_loss is not None:
        assert np.isfinite(trainer.last_loss)


def test_run_dqn_eval_paired_openings():
    from connect4.agents.random_agent import RandomAgent
    from connect4.training.train_dqn import run_dqn_eval

    agent = DQNAgent(net=QNetwork(hidden=16))
    summary = run_dqn_eval(agent, RandomAgent(rng=random.Random(0)), games=4, seed=0, opening_plies=2)
    assert summary.games == 4
    assert summary.games_as_p1 == 2
    assert summary.games_as_p2 == 2
    assert summary.wins + summary.draws + summary.losses == 4
