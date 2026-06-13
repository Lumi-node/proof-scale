"""Tests for the tournament selection module."""

import pytest

from llm_proof_tournament.config import Config
from llm_proof_tournament.tournament import TournamentSelector
from llm_proof_tournament.utils import ProofCandidate


def _make_candidate(confidence: float, text: str = "proof", gen_id: int = 0):
    return ProofCandidate(
        statement="Prove X.",
        proof_text=text,
        confidence=confidence,
        generation_id=gen_id,
    )


class TestTournamentSelector:
    def test_select_returns_proof_candidate(self, config):
        sel = TournamentSelector(config)
        pop = [_make_candidate(0.5), _make_candidate(0.8)]
        result = sel.select(pop)
        assert isinstance(result, ProofCandidate)

    def test_select_picks_highest_confidence(self, config):
        sel = TournamentSelector(Config.load(diversity_weight=0.0))
        pop = [
            _make_candidate(0.3, "low", 0),
            _make_candidate(0.9, "high", 1),
            _make_candidate(0.5, "mid", 2),
        ]
        result = sel.select(pop)
        assert result.confidence == 0.9

    def test_select_from_single_candidate(self, config):
        sel = TournamentSelector(config)
        pop = [_make_candidate(0.7)]
        result = sel.select(pop)
        assert result.confidence == 0.7

    def test_select_empty_raises(self, config):
        sel = TournamentSelector(config)
        with pytest.raises(ValueError):
            sel.select([])

    def test_diversity_affects_ranking(self):
        cfg = Config.load(diversity_weight=1.0)
        sel = TournamentSelector(cfg)
        pop = [
            _make_candidate(0.8, "exactly the same proof text here", 0),
            _make_candidate(0.8, "exactly the same proof text here", 1),
            _make_candidate(0.75, "a completely different and unique proof approach", 2),
        ]
        result = sel.select(pop)
        assert result.generation_id == 2

    def test_select_top_k(self, config):
        sel = TournamentSelector(Config.load(diversity_weight=0.0))
        pop = [
            _make_candidate(0.3, "a", 0),
            _make_candidate(0.9, "b", 1),
            _make_candidate(0.6, "c", 2),
            _make_candidate(0.1, "d", 3),
        ]
        top2 = sel.select_top_k(pop, k=2)
        assert len(top2) == 2
        assert top2[0].confidence == 0.9
        assert top2[1].confidence == 0.6

    def test_tournament_round(self, config):
        sel = TournamentSelector(config)
        pop = [_make_candidate(float(i) / 10, f"proof_{i}", i) for i in range(10)]
        winner = sel.tournament_round(pop, k=3)
        assert isinstance(winner, ProofCandidate)

    def test_valid_beats_corrupted_in_tournament(
        self, config, valid_proof, corrupted_proof
    ):
        from llm_proof_tournament.verifier import ProofVerifier

        v = ProofVerifier(config)
        valid_proof.confidence = v.score(valid_proof).confidence
        corrupted_proof.confidence = v.score(corrupted_proof).confidence

        sel = TournamentSelector(Config.load(diversity_weight=0.0))
        result = sel.select([valid_proof, corrupted_proof])
        assert result.statement == valid_proof.statement
        assert result.confidence == valid_proof.confidence

    def test_select_top_k_empty_raises(self, config):
        sel = TournamentSelector(config)
        with pytest.raises(ValueError):
            sel.select_top_k([], k=1)

    def test_deterministic_with_zero_diversity(self):
        cfg = Config.load(diversity_weight=0.0)
        sel = TournamentSelector(cfg)
        pop = [
            _make_candidate(0.4, "proof_a", 0),
            _make_candidate(0.8, "proof_b", 1),
            _make_candidate(0.6, "proof_c", 2),
        ]
        results = [sel.select(pop) for _ in range(10)]
        assert all(r.confidence == 0.8 for r in results)
