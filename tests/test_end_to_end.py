"""End-to-end integration tests for the proof tournament pipeline."""

import pytest

from llm_proof_tournament.config import Config
from llm_proof_tournament.data import ProofDataset
from llm_proof_tournament.generator import ProofGenerator
from llm_proof_tournament.verifier import ProofVerifier
from llm_proof_tournament.repair import ProofRepair
from llm_proof_tournament.tournament import TournamentSelector
from llm_proof_tournament.rl_trainer import RLTrainer, ReplayBuffer
from llm_proof_tournament.main import ProofPipeline
from llm_proof_tournament.utils import ProofCandidate, VerifierOutput, levenshtein, calibrate_confidence


class TestEndToEnd:
    def test_pipeline_returns_proof_candidate(self, config):
        pipeline = ProofPipeline(config)
        result = pipeline.run("Prove that n^2 - n is even for all integers n.")
        assert isinstance(result, ProofCandidate)
        assert len(result.proof_text) > 0

    def test_pipeline_selects_valid_over_corrupted(self, config):
        """Core requirement: the pipeline must rank a valid proof above a corrupted one."""
        v = ProofVerifier(config)
        sel = TournamentSelector(Config.load(diversity_weight=0.0))

        valid = ProofCandidate(
            statement="Prove that n^2 - n is even.",
            proof_text=(
                "Let n be an arbitrary integer. Then n^2 - n = n(n - 1). "
                "Since n and n - 1 are consecutive integers, one of them must be even. "
                "Therefore their product n(n - 1) is even. QED."
            ),
            generation_id=0,
        )
        corrupted = ProofCandidate(
            statement="Prove that n^2 - n is even.",
            proof_text=(
                "Let n be something. Then 2 = 3 obviously wrong. "
                "Therefore therefore therefore magic abracadabra done."
            ),
            generation_id=1,
        )

        valid.confidence = v.score(valid).confidence
        corrupted.confidence = v.score(corrupted).confidence

        assert valid.confidence > corrupted.confidence
        winner = sel.select([valid, corrupted])
        assert winner.generation_id == valid.generation_id

    def test_pipeline_batch(self, config):
        pipeline = ProofPipeline(config)
        statements = [
            "Prove that the sum of two even numbers is even.",
            "Prove that if n is odd, then n^2 is odd.",
        ]
        results = pipeline.run_batch(statements)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ProofCandidate)
            assert len(r.proof_text) > 0

    def test_repair_improves_pipeline_output(self, config):
        v = ProofVerifier(config)
        r = ProofRepair(config)

        corrupted = ProofCandidate(
            statement="Prove X.",
            proof_text="Since 2 = 3 and obviously wrong magic abracadabra.",
            generation_id=0,
        )

        before_score = v.score(corrupted).confidence
        vo = v.score(corrupted)
        repaired = r.repair(corrupted, vo.error_mask)
        after_score = v.score(repaired).confidence

        assert after_score > before_score

    def test_rl_trainer_records_experiences(self, config):
        trainer = RLTrainer(config, verifier=ProofVerifier(config))
        candidates = [
            ProofCandidate(
                statement="Prove X.",
                proof_text=f"Proof attempt {i}. Therefore it holds. QED.",
                log_probs=[-0.5, -0.3],
                generation_id=i,
            )
            for i in range(4)
        ]
        stats = trainer.step(candidates)
        assert stats["buffer_size"] == 4
        assert stats["mean_reward"] > 0

    def test_rl_trainer_update_policy(self, config):
        trainer = RLTrainer(config, verifier=ProofVerifier(config))
        candidates = [
            ProofCandidate(
                statement="Prove X.",
                proof_text=f"Since n is integer, therefore the property holds. QED.",
                log_probs=[-0.5, -0.3, -0.4],
                generation_id=i,
            )
            for i in range(8)
        ]
        trainer.step(candidates)
        result = trainer.update_policy()
        assert not result["skipped"]

    def test_dataset_loads(self, config):
        ds = ProofDataset(config)
        assert len(ds) > 0
        stmt, proof = ds[0]
        assert isinstance(stmt, str)
        assert isinstance(proof, str)
        assert len(stmt) > 5
        assert len(proof) > 10

    def test_full_generate_verify_select_loop(self, config):
        """Full loop: generate candidates + inject corrupted one, verify corrupted never wins."""
        gen = ProofGenerator(config)
        v = ProofVerifier(config)
        sel = TournamentSelector(Config.load(diversity_weight=0.0))

        stmt = "Prove that the product of two odd numbers is odd."
        candidates = gen.generate(stmt, num_samples=4)

        corrupted = ProofCandidate(
            statement=stmt,
            proof_text=(
                "Obviously wrong, 2 = 3 therefore therefore therefore "
                "magic abracadabra the product is odd."
            ),
            generation_id=99,
        )
        candidates.append(corrupted)

        for c in candidates:
            c.confidence = v.score(c).confidence

        winner = sel.select(candidates)
        assert winner.generation_id != 99
        assert winner.confidence > corrupted.confidence

    def test_replay_buffer(self):
        from llm_proof_tournament.rl_trainer import Experience

        buf = ReplayBuffer(max_size=5)
        for i in range(10):
            buf.add(Experience(
                statement="s", proof_text=f"p{i}",
                log_probs=[-0.5], reward=float(i) / 10,
            ))
        assert len(buf) == 5
        sample = buf.sample(3)
        assert len(sample) == 3

    def test_levenshtein_utility(self):
        assert levenshtein("kitten", "sitting") == 3
        assert levenshtein("", "") == 0
        assert levenshtein("abc", "abc") == 0
        assert levenshtein("abc", "") == 3

    def test_calibrate_confidence(self):
        low = calibrate_confidence(-5.0)
        mid = calibrate_confidence(0.0)
        high = calibrate_confidence(5.0)
        assert low < mid < high
        assert abs(mid - 0.5) < 0.01

    def test_verifier_output_is_valid_flag(self):
        valid = VerifierOutput(confidence=0.8)
        invalid = VerifierOutput(confidence=0.3)
        assert valid.is_valid is True
        assert invalid.is_valid is False

    def test_imports_from_package_root(self):
        """Verify that all public classes are importable from the package root."""
        from llm_proof_tournament import (
            Config,
            ProofCandidate,
            VerifierOutput,
            ProofDataset,
            ProofGenerator,
            ProofVerifier,
            ProofRepair,
            TournamentSelector,
            RLTrainer,
            ProofPipeline,
        )
        assert Config is not None
        assert ProofPipeline is not None
