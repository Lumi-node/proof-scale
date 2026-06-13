"""Proof generation via transformer sampling with RL-guided decoding."""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from llm_proof_tournament.config import Config
from llm_proof_tournament.utils import ProofCandidate


PROOF_TEMPLATES = [
    (
        "Let {var} be an arbitrary {domain}. "
        "Then {property_1}. Since {reason}, it follows that {property_2}. "
        "Therefore {conclusion}. QED."
    ),
    (
        "Assume for contradiction that {negation}. "
        "Then {consequence_1}. But {consequence_2}, which is a contradiction. "
        "Therefore {conclusion}. QED."
    ),
    (
        "We prove by induction on {var}. "
        "Base case: when {var} = {base}, {base_holds}. "
        "Inductive step: assume {inductive_hyp}. Then {inductive_step}. "
        "This completes the induction. QED."
    ),
]

VARIABLE_POOLS = {
    "var": ["n", "k", "m", "p", "x"],
    "domain": ["integer", "positive integer", "natural number"],
}


class ProofGenerator:
    """Generates proof candidates for mathematical statements.

    In heuristic mode, produces template-based variations.
    In model mode, uses a causal LM with temperature sampling.
    """

    def __init__(self, config: Config | None = None, model=None, tokenizer_obj=None):
        self._config = config or Config()
        self._model = model
        self._tokenizer = tokenizer_obj
        self._reward_baseline = 0.0

    def generate(self, statement: str, num_samples: int = 0) -> List[ProofCandidate]:
        if num_samples <= 0:
            num_samples = self._config.population_size

        if self._model is not None and self._tokenizer is not None:
            return self._generate_with_model(statement, num_samples)
        return self._generate_heuristic(statement, num_samples)

    def _generate_heuristic(self, statement: str, num_samples: int) -> List[ProofCandidate]:
        candidates = []
        for i in range(num_samples):
            template = PROOF_TEMPLATES[i % len(PROOF_TEMPLATES)]
            proof_text = self._fill_template(template, statement, variation=i)
            log_probs = [random.uniform(-2.0, -0.1) for _ in proof_text.split()]
            candidates.append(
                ProofCandidate(
                    statement=statement,
                    proof_text=proof_text,
                    log_probs=log_probs,
                    generation_id=i,
                )
            )
        return candidates

    def _fill_template(self, template: str, statement: str, variation: int = 0) -> str:
        stmt_lower = statement.lower()
        var = VARIABLE_POOLS["var"][variation % len(VARIABLE_POOLS["var"])]
        domain = VARIABLE_POOLS["domain"][variation % len(VARIABLE_POOLS["domain"])]

        fills = {
            "var": var,
            "domain": domain,
            "property_1": f"{var}^2 - {var} = {var}({var} - 1)",
            "reason": f"{var} and {var} - 1 are consecutive integers",
            "property_2": f"one of them must be even",
            "conclusion": f"the stated property holds",
            "negation": "the stated property does not hold",
            "consequence_1": f"we reach a value contradicting our assumption",
            "consequence_2": f"this contradicts the given conditions",
            "base": "1",
            "base_holds": "the statement is trivially true",
            "inductive_hyp": f"the statement holds for {var} = k",
            "inductive_step": f"adding k+1 preserves the property",
        }

        if "even" in stmt_lower:
            fills["property_1"] = f"{var} can be written as 2j for some integer j"
            fills["reason"] = "consecutive integers include one even number"
            fills["conclusion"] = "the result is even"
        elif "odd" in stmt_lower:
            fills["property_1"] = f"{var} = 2j + 1 for some integer j"
            fills["reason"] = "the product of odd factors has form 2m + 1"
            fills["conclusion"] = "the result is odd"
        elif "prime" in stmt_lower:
            fills["negation"] = "there are finitely many primes p_1, ..., p_n"
            fills["consequence_1"] = "N = p_1 * ... * p_n + 1 is not divisible by any p_i"
            fills["consequence_2"] = "N must have a prime factor not in our list"
            fills["conclusion"] = "there are infinitely many primes"
        elif "irrational" in stmt_lower or "sqrt" in stmt_lower:
            fills["negation"] = "sqrt(2) = p/q in lowest terms"
            fills["consequence_1"] = "p^2 = 2q^2, so p is even, say p = 2r"
            fills["consequence_2"] = "q^2 = 2r^2, so q is also even"
            fills["conclusion"] = "sqrt(2) is irrational"

        result = template
        for key, value in fills.items():
            result = result.replace("{" + key + "}", value)
        return result

    def _generate_with_model(self, statement: str, num_samples: int) -> List[ProofCandidate]:
        import torch

        prompt = f"Statement: {statement}\nProof:"
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._config.device)

        candidates = []
        for i in range(num_samples):
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=self._config.generator_max_tokens,
                    temperature=self._config.generator_temperature,
                    top_p=self._config.generator_top_p,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = outputs.sequences[0, input_ids.shape[-1]:]
            proof_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            log_probs = []
            if outputs.scores:
                for step_scores in outputs.scores:
                    probs = torch.softmax(step_scores[0], dim=-1)
                    token_id = generated_ids[len(log_probs)] if len(log_probs) < len(generated_ids) else 0
                    lp = float(torch.log(probs[token_id] + 1e-10))
                    log_probs.append(lp)

            candidates.append(
                ProofCandidate(
                    statement=statement,
                    proof_text=proof_text,
                    log_probs=log_probs,
                    generation_id=i,
                )
            )
        return candidates

    @property
    def reward_baseline(self) -> float:
        return self._reward_baseline

    @reward_baseline.setter
    def reward_baseline(self, value: float):
        self._reward_baseline = value

    def get_parameters(self):
        if self._model is not None:
            return self._model.parameters()
        return iter([])
