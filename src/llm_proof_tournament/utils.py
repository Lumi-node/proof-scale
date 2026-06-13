"""Data containers and utility functions for the proof tournament pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ProofCandidate:
    """A single proof attempt with associated metadata."""

    statement: str
    proof_text: str
    log_probs: Optional[List[float]] = None
    confidence: float = 0.0
    generation_id: int = 0

    @property
    def mean_log_prob(self) -> float:
        if not self.log_probs:
            return 0.0
        return float(np.mean(self.log_probs))


@dataclass
class VerifierOutput:
    """Result from the proof verifier."""

    confidence: float
    error_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    is_valid: bool = False
    details: str = ""

    def __post_init__(self):
        self.is_valid = self.confidence >= 0.5


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for lightweight use."""
    import re
    return re.findall(r"\w+|[^\w\s]", text.lower())


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(
                min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost)
            )
        prev_row = curr_row
    return prev_row[-1]


def calibrate_confidence(raw_score: float, temperature: float = 1.0) -> float:
    """Apply temperature scaling to calibrate a raw confidence score to [0, 1]."""
    scaled = raw_score / max(temperature, 1e-8)
    return 1.0 / (1.0 + np.exp(-scaled))
