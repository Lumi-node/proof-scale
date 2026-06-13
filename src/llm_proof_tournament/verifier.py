"""Proof verification: scores candidates for mathematical validity."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from llm_proof_tournament.config import Config
from llm_proof_tournament.utils import ProofCandidate, VerifierOutput, tokenize


LOGICAL_CONNECTIVES = {
    "therefore", "since", "because", "hence", "thus", "implies",
    "so", "then", "assume", "let", "suppose", "given", "consider",
    "if", "follows", "conclude", "contradiction", "contradicting",
}

CONCLUSION_MARKERS = {"qed", "proven", "completes", "established", "demonstrated"}

MATHEMATICAL_TERMS = {
    "integer", "even", "odd", "prime", "divisible", "factor", "sum",
    "product", "square", "equals", "divides", "positive", "negative",
    "rational", "irrational", "induction", "base", "hypothesis",
}

CORRUPTION_SIGNALS = [
    r"\b(\d+)\s*=\s*(\d+)\b",
    r"not\s+not\s+not",
    r"obviously|clearly\s+wrong|magic|abracadabra",
    r"therefore.*therefore.*therefore",
]


class ProofVerifier:
    """Scores proof candidates for mathematical validity.

    Operates in two modes:
    - Heuristic mode (default for lightweight/testing): rule-based scoring
    - Model mode: uses a transformer encoder for learned verification
    """

    def __init__(self, config: Config | None = None, model=None, tokenizer_obj=None):
        self._config = config or Config()
        self._model = model
        self._tokenizer = tokenizer_obj
        self._threshold = self._config.verifier_threshold

    def score(self, candidate: ProofCandidate) -> VerifierOutput:
        if self._model is not None and self._tokenizer is not None:
            return self._score_with_model(candidate)
        return self._score_heuristic(candidate)

    def _score_heuristic(self, candidate: ProofCandidate) -> VerifierOutput:
        tokens = tokenize(candidate.proof_text)
        if not tokens:
            return VerifierOutput(confidence=0.0, error_mask=np.array([], dtype=np.float32))

        scores = []

        connective_count = sum(1 for t in tokens if t in LOGICAL_CONNECTIVES)
        connective_ratio = connective_count / len(tokens)
        scores.append(min(connective_ratio * 10, 1.0))

        has_conclusion = any(t in CONCLUSION_MARKERS for t in tokens)
        scores.append(1.0 if has_conclusion else 0.2)

        math_count = sum(1 for t in tokens if t in MATHEMATICAL_TERMS)
        math_ratio = math_count / len(tokens)
        scores.append(min(math_ratio * 8, 1.0))

        length_score = min(len(tokens) / 20.0, 1.0)
        scores.append(length_score)

        corruption_penalty = 0.0
        text_lower = candidate.proof_text.lower()
        for pattern in CORRUPTION_SIGNALS:
            matches = re.findall(pattern, text_lower)
            if matches:
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        a, b = match
                        if a != b and a.isdigit() and b.isdigit():
                            corruption_penalty += 0.5
                    else:
                        corruption_penalty += 0.3
        corruption_penalty = min(corruption_penalty, 1.0)

        mentions_statement = self._check_relevance(candidate)
        scores.append(0.8 if mentions_statement else 0.3)

        raw_confidence = float(np.mean(scores)) - corruption_penalty * 0.5
        confidence = float(np.clip(raw_confidence, 0.0, 1.0))

        error_mask = self._build_error_mask(tokens, text_lower)

        return VerifierOutput(
            confidence=confidence,
            error_mask=error_mask,
            details=f"connectives={connective_count}, math_terms={math_count}, "
                    f"corruption_penalty={corruption_penalty:.2f}",
        )

    def _check_relevance(self, candidate: ProofCandidate) -> bool:
        stmt_tokens = set(tokenize(candidate.statement))
        proof_tokens = set(tokenize(candidate.proof_text))
        overlap = stmt_tokens & proof_tokens - {"the", "a", "is", "that", "of", "and", "for"}
        return len(overlap) >= 2

    def _build_error_mask(self, tokens: list, text_lower: str) -> np.ndarray:
        mask = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            for pattern in CORRUPTION_SIGNALS:
                window = " ".join(tokens[max(0, i - 2):i + 3])
                if re.search(pattern, window):
                    for j in range(max(0, i - 2), min(len(tokens), i + 3)):
                        mask[j] = max(mask[j], 0.8)
        return mask

    def _score_with_model(self, candidate: ProofCandidate) -> VerifierOutput:
        inputs = self._tokenizer(
            candidate.statement,
            candidate.proof_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        import torch
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs[0, 1]) if probs.shape[-1] > 1 else float(probs[0, 0])

        n_tokens = inputs["input_ids"].shape[-1]
        error_mask = np.zeros(n_tokens, dtype=np.float32)

        return VerifierOutput(confidence=confidence, error_mask=error_mask)

    @property
    def threshold(self) -> float:
        return self._threshold
