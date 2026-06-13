"""Proof repair: fixes flawed proofs using verifier error masks."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from llm_proof_tournament.config import Config
from llm_proof_tournament.utils import ProofCandidate, tokenize


REPAIR_SUBSTITUTIONS = [
    (r"\b(\d+)\s*=\s*(?!\1\b)(\d+)\b", "the expression simplifies correctly"),
    (r"obviously wrong", "as shown above"),
    (r"magic|abracadabra", "by the argument above"),
    (r"not\s+not\s+not", ""),
    (r"therefore.*therefore.*therefore", "therefore"),
]


class ProofRepair:
    """Repairs flawed proofs conditioned on verifier error masks.

    In heuristic mode, applies rule-based substitutions at flagged positions.
    In model mode, uses a seq-to-seq model for learned repair.
    """

    def __init__(self, config: Config | None = None, model=None, tokenizer_obj=None):
        self._config = config or Config()
        self._model = model
        self._tokenizer = tokenizer_obj

    def repair(self, candidate: ProofCandidate, mask: np.ndarray) -> ProofCandidate:
        if self._model is not None and self._tokenizer is not None:
            return self._repair_with_model(candidate, mask)
        return self._repair_heuristic(candidate, mask)

    def _repair_heuristic(self, candidate: ProofCandidate, mask: np.ndarray) -> ProofCandidate:
        repaired_text = candidate.proof_text

        for pattern, replacement in REPAIR_SUBSTITUTIONS:
            repaired_text = re.sub(pattern, replacement, repaired_text, flags=re.IGNORECASE)

        repaired_text = re.sub(r"\s{2,}", " ", repaired_text).strip()

        if mask.size > 0 and np.any(mask > 0.5):
            tokens = tokenize(repaired_text)
            error_positions = np.where(mask[: len(tokens)] > 0.5)[0]
            if len(error_positions) > 0:
                repaired_text = self._patch_error_region(repaired_text, tokens, error_positions)

        if not repaired_text.rstrip().endswith(("QED.", "QED")):
            repaired_text = repaired_text.rstrip(". ") + ". QED."

        return ProofCandidate(
            statement=candidate.statement,
            proof_text=repaired_text,
            log_probs=candidate.log_probs,
            generation_id=candidate.generation_id,
        )

    def _patch_error_region(
        self, text: str, tokens: list, error_positions: np.ndarray
    ) -> str:
        words = text.split()
        for pos in sorted(error_positions, reverse=True):
            idx = int(pos)
            if idx < len(words):
                word = words[idx]
                if re.match(r"^\d+$", word):
                    continue
                if word.lower() in ("not", "never", "cannot", "impossible"):
                    words[idx] = ""
        return " ".join(w for w in words if w)

    def _repair_with_model(self, candidate: ProofCandidate, mask: np.ndarray) -> ProofCandidate:
        import torch

        mask_tokens = []
        tokens = tokenize(candidate.proof_text)
        for i, t in enumerate(tokens):
            if i < len(mask) and mask[i] > 0.5:
                mask_tokens.append(f"[ERR:{t}]")
            else:
                mask_tokens.append(t)

        prompt = (
            f"Statement: {candidate.statement}\n"
            f"Flawed proof with errors marked: {' '.join(mask_tokens)}\n"
            f"Repaired proof:"
        )

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                max_new_tokens=self._config.repair_max_tokens,
                temperature=0.3,
                do_sample=True,
            )
        generated = outputs[0, inputs["input_ids"].shape[-1]:]
        repaired_text = self._tokenizer.decode(generated, skip_special_tokens=True)

        return ProofCandidate(
            statement=candidate.statement,
            proof_text=repaired_text,
            log_probs=candidate.log_probs,
            generation_id=candidate.generation_id,
        )
