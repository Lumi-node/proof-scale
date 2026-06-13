"""Tests for the proof verifier module."""

import numpy as np
import pytest

from llm_proof_tournament.config import Config
from llm_proof_tournament.verifier import ProofVerifier
from llm_proof_tournament.utils import ProofCandidate, VerifierOutput


class TestProofVerifier:
    def test_score_returns_verifier_output(self, config, valid_proof):
        v = ProofVerifier(config)
        result = v.score(valid_proof)
        assert isinstance(result, VerifierOutput)

    def test_valid_proof_scores_higher_than_corrupted(
        self, config, valid_proof, corrupted_proof
    ):
        v = ProofVerifier(config)
        valid_score = v.score(valid_proof)
        corrupt_score = v.score(corrupted_proof)
        assert valid_score.confidence > corrupt_score.confidence

    def test_valid_proof_marked_valid(self, config, valid_proof):
        v = ProofVerifier(config)
        result = v.score(valid_proof)
        assert result.is_valid is True

    def test_corrupted_proof_marked_invalid(self, config, corrupted_proof):
        v = ProofVerifier(config)
        result = v.score(corrupted_proof)
        assert result.is_valid is False

    def test_confidence_in_range(self, config, valid_proof, corrupted_proof):
        v = ProofVerifier(config)
        for candidate in [valid_proof, corrupted_proof]:
            result = v.score(candidate)
            assert 0.0 <= result.confidence <= 1.0

    def test_error_mask_shape_matches_tokens(self, config, corrupted_proof):
        v = ProofVerifier(config)
        result = v.score(corrupted_proof)
        assert isinstance(result.error_mask, np.ndarray)
        assert result.error_mask.ndim == 1

    def test_corrupted_proof_has_nonzero_error_mask(self, config, corrupted_proof):
        v = ProofVerifier(config)
        result = v.score(corrupted_proof)
        assert np.any(result.error_mask > 0)

    def test_valid_proof_has_low_error_mask(self, config, valid_proof):
        v = ProofVerifier(config)
        result = v.score(valid_proof)
        assert np.mean(result.error_mask) < 0.3

    def test_empty_proof_scores_zero(self, config):
        empty = ProofCandidate(statement="Prove X.", proof_text="")
        v = ProofVerifier(config)
        result = v.score(empty)
        assert result.confidence == 0.0

    def test_mediocre_proof_between_valid_and_corrupt(
        self, config, valid_proof, corrupted_proof, mediocre_proof
    ):
        v = ProofVerifier(config)
        valid_s = v.score(valid_proof).confidence
        corrupt_s = v.score(corrupted_proof).confidence
        mediocre_s = v.score(mediocre_proof).confidence
        assert corrupt_s < mediocre_s < valid_s

    def test_threshold_property(self, config):
        v = ProofVerifier(config)
        assert v.threshold == config.verifier_threshold

    def test_details_string_nonempty(self, config, valid_proof):
        v = ProofVerifier(config)
        result = v.score(valid_proof)
        assert len(result.details) > 0

    def test_contradiction_pattern_detected(self, config):
        bad = ProofCandidate(
            statement="Prove X.",
            proof_text="We know that 5 = 7 and therefore the proof is complete. QED.",
        )
        v = ProofVerifier(config)
        result = v.score(bad)
        assert result.confidence < 0.5
