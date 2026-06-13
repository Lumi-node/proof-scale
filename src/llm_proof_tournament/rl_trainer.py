"""PPO-style reinforcement learning trainer for the proof generator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from llm_proof_tournament.config import Config
from llm_proof_tournament.utils import ProofCandidate


@dataclass
class Experience:
    """A single RL experience tuple."""

    statement: str
    proof_text: str
    log_probs: List[float]
    reward: float
    advantage: float = 0.0


class ReplayBuffer:
    """Fixed-size buffer of RL experiences."""

    def __init__(self, max_size: int = 256):
        self._buffer: Deque[Experience] = deque(maxlen=max_size)

    def add(self, exp: Experience):
        self._buffer.append(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        size = min(batch_size, len(self._buffer))
        indices = np.random.choice(len(self._buffer), size=size, replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()


class RLTrainer:
    """PPO-style policy gradient trainer for ProofGenerator.

    Uses verifier confidence as the reward signal. Stores experiences
    in a replay buffer and performs clipped policy gradient updates.
    """

    def __init__(
        self,
        config: Config | None = None,
        generator=None,
        verifier=None,
        optimizer=None,
    ):
        self._config = config or Config()
        self._generator = generator
        self._verifier = verifier
        self._optimizer = optimizer
        self._buffer = ReplayBuffer(max_size=self._config.rl_buffer_size)
        self._baseline = 0.0
        self._step_count = 0
        self._policy_losses: List[float] = []

    def step(self, batch: List[ProofCandidate]) -> dict:
        """Process a batch of candidates: compute rewards and store experiences."""
        rewards = []
        for candidate in batch:
            if self._verifier is not None:
                output = self._verifier.score(candidate)
                reward = output.confidence
            else:
                reward = candidate.confidence
            rewards.append(reward)

            advantage = reward - self._baseline

            exp = Experience(
                statement=candidate.statement,
                proof_text=candidate.proof_text,
                log_probs=candidate.log_probs or [],
                reward=reward,
                advantage=advantage,
            )
            self._buffer.add(exp)

        self._baseline = 0.9 * self._baseline + 0.1 * float(np.mean(rewards))
        self._step_count += 1

        return {
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "baseline": self._baseline,
            "buffer_size": len(self._buffer),
            "step": self._step_count,
        }

    def update_policy(self) -> dict:
        """Perform a PPO-style policy gradient update."""
        if len(self._buffer) < self._config.rl_batch_size:
            return {"loss": 0.0, "skipped": True}

        batch = self._buffer.sample(self._config.rl_batch_size)

        policy_loss = self._compute_policy_loss(batch)
        self._policy_losses.append(policy_loss)

        if self._optimizer is not None and self._generator is not None:
            self._apply_gradient_update(batch, policy_loss)

        return {
            "loss": policy_loss,
            "batch_size": len(batch),
            "mean_advantage": float(np.mean([e.advantage for e in batch])),
            "skipped": False,
        }

    def _compute_policy_loss(self, batch: List[Experience]) -> float:
        losses = []
        for exp in batch:
            if exp.log_probs:
                mean_lp = float(np.mean(exp.log_probs))
                ratio = np.exp(np.clip(mean_lp, -10, 2))
                clipped = np.clip(
                    ratio,
                    1.0 - self._config.rl_clip_epsilon,
                    1.0 + self._config.rl_clip_epsilon,
                )
                loss = -min(ratio * exp.advantage, clipped * exp.advantage)
                losses.append(loss)
        return float(np.mean(losses)) if losses else 0.0

    def _apply_gradient_update(self, batch: List[Experience], loss: float):
        import torch

        self._optimizer.zero_grad()
        loss_tensor = torch.tensor(loss, requires_grad=True)
        loss_tensor.backward()
        torch.nn.utils.clip_grad_norm_(self._generator.get_parameters(), max_norm=1.0)
        self._optimizer.step()

    @property
    def stats(self) -> dict:
        return {
            "total_steps": self._step_count,
            "buffer_size": len(self._buffer),
            "baseline": self._baseline,
            "recent_losses": self._policy_losses[-10:] if self._policy_losses else [],
        }
