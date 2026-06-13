"""Shared test fixtures for llm_proof_tournament."""

import pytest

from llm_proof_tournament.config import Config
from llm_proof_tournament.utils import ProofCandidate


@pytest.fixture
def config():
    return Config.load(
        population_size=4,
        use_heuristic=True,
        max_repair_rounds=2,
        tournament_k=2,
    )


@pytest.fixture
def valid_proof():
    return ProofCandidate(
        statement="Prove that for every integer n, n^2 - n is even.",
        proof_text=(
            "Let n be an arbitrary integer. Then n^2 - n = n(n - 1). "
            "Since n and n - 1 are consecutive integers, one of them must be even. "
            "Therefore their product n(n - 1) is even. QED."
        ),
        log_probs=[-0.5, -0.3, -0.4, -0.6, -0.2],
        generation_id=0,
    )


@pytest.fixture
def corrupted_proof():
    return ProofCandidate(
        statement="Prove that for every integer n, n^2 - n is even.",
        proof_text=(
            "Let n be an integer. Then 2 = 3 obviously wrong. "
            "Therefore therefore therefore it is magic. abracadabra done."
        ),
        log_probs=[-2.0, -3.0, -2.5, -1.8],
        generation_id=1,
    )


@pytest.fixture
def mediocre_proof():
    return ProofCandidate(
        statement="Prove that for every integer n, n^2 - n is even.",
        proof_text=(
            "Consider n. The value n^2 - n can be factored. "
            "It equals n times n minus 1. One factor is even."
        ),
        log_probs=[-1.0, -1.2, -0.8],
        generation_id=2,
    )
