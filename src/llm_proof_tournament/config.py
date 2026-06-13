"""Configuration for the proof tournament pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Hyper-parameters and paths for the proof tournament pipeline."""

    # Population / tournament
    population_size: int = 8
    tournament_k: int = 3
    diversity_weight: float = 0.1
    max_repair_rounds: int = 2

    # Generator
    generator_model: str = "gpt2"
    generator_max_tokens: int = 256
    generator_temperature: float = 0.8
    generator_top_p: float = 0.95

    # Verifier
    verifier_model: str = "roberta-base"
    verifier_threshold: float = 0.5

    # RL training
    rl_learning_rate: float = 1e-5
    rl_clip_epsilon: float = 0.2
    rl_gamma: float = 0.99
    rl_batch_size: int = 4
    rl_buffer_size: int = 256

    # Repair
    repair_model: str = "gpt2"
    repair_max_tokens: int = 256

    # Data
    dataset_name: str = "math"
    dataset_split: str = "test"
    dataset_cache_dir: Optional[str] = None
    max_dataset_size: int = 100

    # Runtime
    device: str = "cpu"
    use_heuristic: bool = False

    @classmethod
    def load(cls, **overrides) -> Config:
        return cls(**overrides)

    def as_dict(self) -> dict:
        return {
            f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()
        }
