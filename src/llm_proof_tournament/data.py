"""Dataset loading for proof tournament training and evaluation."""

from __future__ import annotations

from typing import Dict, List, Tuple

from llm_proof_tournament.config import Config


BUILTIN_PROBLEMS: List[Dict[str, str]] = [
    {
        "statement": "Prove that for every integer n, n^2 - n is even.",
        "proof": (
            "Let n be an arbitrary integer. Then n^2 - n = n(n - 1). "
            "Since n and n - 1 are consecutive integers, one of them must be even. "
            "Therefore their product n(n - 1) is even. QED."
        ),
    },
    {
        "statement": "Prove that the sum of two even integers is even.",
        "proof": (
            "Let a = 2k and b = 2m for integers k and m. "
            "Then a + b = 2k + 2m = 2(k + m). "
            "Since k + m is an integer, a + b is even. QED."
        ),
    },
    {
        "statement": "Prove that if n is odd, then n^2 is odd.",
        "proof": (
            "Let n = 2k + 1 for some integer k. "
            "Then n^2 = (2k + 1)^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1. "
            "Since 2k^2 + 2k is an integer, n^2 is odd. QED."
        ),
    },
    {
        "statement": "Prove that the product of two odd numbers is odd.",
        "proof": (
            "Let a = 2j + 1 and b = 2k + 1 for integers j and k. "
            "Then ab = (2j + 1)(2k + 1) = 4jk + 2j + 2k + 1 = 2(2jk + j + k) + 1. "
            "Since 2jk + j + k is an integer, ab is odd. QED."
        ),
    },
    {
        "statement": "Prove that the square root of 2 is irrational.",
        "proof": (
            "Assume for contradiction that sqrt(2) = p/q where p, q are integers "
            "with no common factors. Then 2 = p^2/q^2, so p^2 = 2q^2. "
            "This means p^2 is even, so p is even. Write p = 2r. "
            "Then 4r^2 = 2q^2, so q^2 = 2r^2, meaning q is also even. "
            "But then p and q share factor 2, contradicting our assumption. "
            "Therefore sqrt(2) is irrational. QED."
        ),
    },
    {
        "statement": "Prove that there are infinitely many prime numbers.",
        "proof": (
            "Assume for contradiction that there are finitely many primes: "
            "p_1, p_2, ..., p_n. Consider N = p_1 * p_2 * ... * p_n + 1. "
            "N is not divisible by any p_i since division leaves remainder 1. "
            "So N is either prime itself or has a prime factor not in our list. "
            "Either way we have a prime not in {p_1, ..., p_n}, a contradiction. "
            "Therefore there are infinitely many primes. QED."
        ),
    },
    {
        "statement": "Prove that the sum of the first n positive integers is n(n+1)/2.",
        "proof": (
            "Base case: when n = 1, the sum is 1 = 1(2)/2. "
            "Inductive step: assume 1 + 2 + ... + k = k(k+1)/2. "
            "Then 1 + 2 + ... + k + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) "
            "= (k+1)(k+2)/2. This completes the induction. QED."
        ),
    },
    {
        "statement": "Prove that every positive integer greater than 1 has a prime factor.",
        "proof": (
            "Let n > 1 be a positive integer. If n is prime, then n is its own prime "
            "factor. If n is not prime, then n = ab where 1 < a, b < n. "
            "Consider the smallest divisor d > 1 of n. If d were composite, "
            "d = ef with 1 < e < d, then e divides n and e < d, contradicting "
            "minimality. Therefore d is prime. QED."
        ),
    },
]


class ProofDataset:
    """Dataset of mathematical statements paired with correct proofs.

    Uses built-in elementary number theory problems by default.
    Optionally downloads from HuggingFace datasets if configured.
    """

    def __init__(self, config: Config | None = None):
        self._config = config or Config()
        self._data: List[Dict[str, str]] = []
        self._load()

    def _load(self):
        self._data = list(BUILTIN_PROBLEMS[: self._config.max_dataset_size])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        item = self._data[idx]
        return item["statement"], item["proof"]

    @property
    def statements(self) -> List[str]:
        return [d["statement"] for d in self._data]

    @property
    def proofs(self) -> List[str]:
        return [d["proof"] for d in self._data]
