# ProofScale

**Population‑level test‑time scaling for LLM‑driven mathematical proofs.**

ProofScale turns a single proof attempt into a population: generate many candidates, verify and critique each, repair the promising ones, and select a winner via a diversity‑aware tournament — with an optional RL loop to improve the generator. A heuristic mode runs the full pipeline with no model downloads.

## Quick Start

```python
from llm_proof_tournament import Config, ProofPipeline

# Heuristic mode runs the whole pipeline with no model downloads
config = Config.load(use_heuristic=True)
pipeline = ProofPipeline(config)

result = pipeline.run("Prove that n^2 - n is even for all integers n.")
print(result.proof_text)
print(result.confidence)
```

See [Installation](getting-started/installation.md) and the [Quick Start guide](getting-started/quick-start.md) to go further, or the [API Reference](reference.md).
