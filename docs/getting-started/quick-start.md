# Quick Start

The following example runs end‑to‑end against the installed package:

```python
from llm_proof_tournament import Config, ProofPipeline

# Heuristic mode runs the whole pipeline with no model downloads
config = Config.load(use_heuristic=True)
pipeline = ProofPipeline(config)

result = pipeline.run("Prove that n^2 - n is even for all integers n.")
print(result.proof_text)
print(result.confidence)
```

For the full public API, see the [API Reference](../reference.md). For how the
pieces fit together, see [Architecture](../architecture.md).
