"""High-level orchestrator for the proof tournament pipeline."""

from __future__ import annotations

from typing import List, Optional

from llm_proof_tournament.config import Config
from llm_proof_tournament.data import ProofDataset
from llm_proof_tournament.generator import ProofGenerator
from llm_proof_tournament.verifier import ProofVerifier
from llm_proof_tournament.repair import ProofRepair
from llm_proof_tournament.tournament import TournamentSelector
from llm_proof_tournament.rl_trainer import RLTrainer
from llm_proof_tournament.utils import ProofCandidate


class ProofPipeline:
    """Orchestrates the full generate-verify-repair-tournament pipeline.

    For a given mathematical statement:
    1. Generate a population of proof candidates
    2. Score each with the verifier
    3. Repair low-scoring candidates using the error mask
    4. Re-score repaired candidates
    5. Run tournament selection to pick the best proof
    """

    def __init__(
        self,
        config: Config | None = None,
        generator: ProofGenerator | None = None,
        verifier: ProofVerifier | None = None,
        repair: ProofRepair | None = None,
        selector: TournamentSelector | None = None,
        trainer: RLTrainer | None = None,
    ):
        self._config = config or Config()
        self._generator = generator or ProofGenerator(self._config)
        self._verifier = verifier or ProofVerifier(self._config)
        self._repair = repair or ProofRepair(self._config)
        self._selector = selector or TournamentSelector(self._config)
        self._trainer = trainer or RLTrainer(
            self._config, generator=self._generator, verifier=self._verifier
        )

    def run(self, statement: str) -> ProofCandidate:
        population = self._generator.generate(
            statement, num_samples=self._config.population_size
        )

        population = self._score_population(population)

        for _ in range(self._config.max_repair_rounds):
            repaired_any = False
            for i, candidate in enumerate(population):
                if candidate.confidence < self._verifier.threshold:
                    vo = self._verifier.score(candidate)
                    repaired = self._repair.repair(candidate, vo.error_mask)
                    repaired_vo = self._verifier.score(repaired)
                    repaired.confidence = repaired_vo.confidence
                    if repaired.confidence > candidate.confidence:
                        population[i] = repaired
                        repaired_any = True
            if not repaired_any:
                break

        self._trainer.step(population)

        return self._selector.select(population)

    def run_batch(self, statements: List[str]) -> List[ProofCandidate]:
        results = []
        for statement in statements:
            result = self.run(statement)
            results.append(result)

        self._trainer.update_policy()
        return results

    def _score_population(self, population: List[ProofCandidate]) -> List[ProofCandidate]:
        for candidate in population:
            vo = self._verifier.score(candidate)
            candidate.confidence = vo.confidence
        return population
