"""Tournament selection over a population of proof candidates."""

from __future__ import annotations

import random
from typing import List, Optional

from llm_proof_tournament.config import Config
from llm_proof_tournament.utils import ProofCandidate, levenshtein


class TournamentSelector:
    """Selects the best proof from a population using tournament-style ranking.

    Primary ranking: verifier confidence.
    Secondary: diversity penalty (Levenshtein distance) to prevent population collapse.
    """

    def __init__(self, config: Config | None = None):
        self._config = config or Config()

    def select(self, population: List[ProofCandidate]) -> ProofCandidate:
        if not population:
            raise ValueError("Cannot select from empty population")
        if len(population) == 1:
            return population[0]

        scored = self._rank(population)
        return scored[0][0]

    def select_top_k(self, population: List[ProofCandidate], k: int = 1) -> List[ProofCandidate]:
        if not population:
            raise ValueError("Cannot select from empty population")
        scored = self._rank(population)
        return [candidate for candidate, _ in scored[:k]]

    def _rank(self, population: List[ProofCandidate]) -> List[tuple]:
        diversity_scores = self._compute_diversity(population)

        scored = []
        for i, candidate in enumerate(population):
            diversity = diversity_scores[i]
            combined = (
                candidate.confidence
                + self._config.diversity_weight * diversity
            )
            scored.append((candidate, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _compute_diversity(self, population: List[ProofCandidate]) -> List[float]:
        n = len(population)
        if n <= 1:
            return [0.0] * n

        diversity = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = levenshtein(
                        population[i].proof_text, population[j].proof_text
                    )
                    max_len = max(
                        len(population[i].proof_text),
                        len(population[j].proof_text),
                        1,
                    )
                    distances.append(dist / max_len)
            diversity.append(float(sum(distances) / len(distances)) if distances else 0.0)
        return diversity

    def tournament_round(
        self, population: List[ProofCandidate], k: int = 0
    ) -> ProofCandidate:
        """Run a single tournament round: pick k random candidates, return the best."""
        if k <= 0:
            k = self._config.tournament_k
        k = min(k, len(population))
        contenders = random.sample(population, k)
        return max(contenders, key=lambda c: c.confidence)
