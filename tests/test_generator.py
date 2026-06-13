"""Tests for the proof generator module."""

import pytest

from llm_proof_tournament.config import Config
from llm_proof_tournament.generator import ProofGenerator
from llm_proof_tournament.utils import ProofCandidate


class TestProofGenerator:
    def test_generate_returns_correct_count(self, config):
        gen = ProofGenerator(config)
        candidates = gen.generate("Prove that 1 + 1 = 2.", num_samples=5)
        assert len(candidates) == 5

    def test_generate_uses_population_size_default(self, config):
        gen = ProofGenerator(config)
        candidates = gen.generate("Prove that 1 + 1 = 2.")
        assert len(candidates) == config.population_size

    def test_candidates_are_proof_candidate_type(self, config):
        gen = ProofGenerator(config)
        candidates = gen.generate("Prove n^2 - n is even.", num_samples=3)
        for c in candidates:
            assert isinstance(c, ProofCandidate)

    def test_candidates_have_nonempty_proof_text(self, config):
        gen = ProofGenerator(config)
        candidates = gen.generate("Prove the sum of two even numbers is even.", num_samples=4)
        for c in candidates:
            assert len(c.proof_text) > 10

    def test_candidates_preserve_statement(self, config):
        stmt = "Prove that n^2 is odd when n is odd."
        gen = ProofGenerator(config)
        candidates = gen.generate(stmt, num_samples=3)
        for c in candidates:
            assert c.statement == stmt

    def test_candidates_have_log_probs(self, config):
        gen = ProofGenerator(config)
        candidates = gen.generate("Prove 2 is prime.", num_samples=2)
        for c in candidates:
            assert c.log_probs is not None
            assert len(c.log_probs) > 0

    def test_candidates_have_unique_generation_ids(self, config):
        gen = ProofGenerator(config)
        candidates = gen.generate("Prove there are infinitely many primes.", num_samples=4)
        ids = [c.generation_id for c in candidates]
        assert len(set(ids)) == len(ids)

    def test_mean_log_prob_is_negative(self, config):
        gen = ProofGenerator(config)
        candidates = gen.generate("Prove sqrt(2) is irrational.", num_samples=3)
        for c in candidates:
            assert c.mean_log_prob < 0

    def test_even_statement_generates_relevant_proof(self, config):
        gen = ProofGenerator(config)
        candidates = gen.generate(
            "Prove that the sum of two even integers is even.", num_samples=3
        )
        has_even_mention = any("even" in c.proof_text.lower() for c in candidates)
        assert has_even_mention

    def test_reward_baseline_property(self, config):
        gen = ProofGenerator(config)
        assert gen.reward_baseline == 0.0
        gen.reward_baseline = 0.5
        assert gen.reward_baseline == 0.5
