"""Tests for the proof repair module."""

import numpy as np
import pytest

from llm_proof_tournament.config import Config
from llm_proof_tournament.repair import ProofRepair
from llm_proof_tournament.verifier import ProofVerifier
from llm_proof_tournament.utils import ProofCandidate


class TestProofRepair:
    def test_repair_returns_proof_candidate(self, config, corrupted_proof):
        r = ProofRepair(config)
        mask = np.zeros(20, dtype=np.float32)
        result = r.repair(corrupted_proof, mask)
        assert isinstance(result, ProofCandidate)

    def test_repair_preserves_statement(self, config, corrupted_proof):
        r = ProofRepair(config)
        mask = np.zeros(20, dtype=np.float32)
        result = r.repair(corrupted_proof, mask)
        assert result.statement == corrupted_proof.statement

    def test_repair_removes_corruption_patterns(self, config):
        bad = ProofCandidate(
            statement="Prove X.",
            proof_text="Since 2 = 3 and obviously wrong, magic happens. abracadabra.",
        )
        r = ProofRepair(config)
        mask = np.zeros(20, dtype=np.float32)
        result = r.repair(bad, mask)
        assert "obviously wrong" not in result.proof_text.lower()
        assert "magic" not in result.proof_text.lower()
        assert "abracadabra" not in result.proof_text.lower()

    def test_repair_adds_qed_if_missing(self, config):
        no_qed = ProofCandidate(
            statement="Prove X.",
            proof_text="Let n be an integer. Then the property holds.",
        )
        r = ProofRepair(config)
        mask = np.zeros(10, dtype=np.float32)
        result = r.repair(no_qed, mask)
        assert result.proof_text.rstrip().endswith(("QED.", "QED"))

    def test_repair_improves_corrupted_score(self, config, corrupted_proof):
        v = ProofVerifier(config)
        r = ProofRepair(config)

        before = v.score(corrupted_proof)
        repaired = r.repair(corrupted_proof, before.error_mask)
        after = v.score(repaired)

        assert after.confidence > before.confidence

    def test_repair_with_error_mask(self, config):
        flawed = ProofCandidate(
            statement="Prove even + even = even.",
            proof_text="Let a = 2k. not not not the sum is even. QED.",
        )
        r = ProofRepair(config)
        mask = np.array([0, 0, 0, 0, 0.9, 0.9, 0.9, 0, 0, 0, 0], dtype=np.float32)
        result = r.repair(flawed, mask)
        assert "not not not" not in result.proof_text

    def test_repair_does_not_destroy_valid_proof(self, config, valid_proof):
        r = ProofRepair(config)
        mask = np.zeros(50, dtype=np.float32)
        result = r.repair(valid_proof, mask)
        assert len(result.proof_text) > 20

    def test_multiple_repair_rounds(self, config):
        bad = ProofCandidate(
            statement="Prove X.",
            proof_text="Therefore therefore therefore 2 = 3. obviously wrong.",
        )
        v = ProofVerifier(config)
        r = ProofRepair(config)

        current = bad
        for _ in range(3):
            vo = v.score(current)
            current = r.repair(current, vo.error_mask)

        final = v.score(current)
        initial = v.score(bad)
        assert final.confidence >= initial.confidence
