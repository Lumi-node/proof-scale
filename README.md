<p align="center">
  <img src="assets/hero.jpg" alt="ProofScale" width="900">
</p>

<h1 align="center">ProofScale</h1>

<p align="center"><strong>Population‑level test‑time scaling for LLM‑driven mathematical proofs.</strong></p>

<p align="center">
  <a href="https://github.com/Lumi-node/proof-scale"><img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github" alt="GitHub"></a>
  <a href="https://github.com/Lumi-node/proof-scale/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/Lumi-node/proof-scale/actions"><img src="https://img.shields.io/badge/tests-54-success.svg" alt="Tests"></a>
  <a href="https://lumi-node.github.io/proof-scale/"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Docs"></a>
</p>

---

ProofScale turns a single proof attempt into a population: generate many candidates, verify and critique each, repair the promising ones, and select a winner via a diversity‑aware tournament — with an optional RL loop to improve the generator. A heuristic mode runs the full pipeline with no model downloads.

## Installation

```bash
pip install git+https://github.com/Lumi-node/proof-scale.git
```

Requires Python ≥ 3.10. To work on the project locally:

```bash
git clone https://github.com/Lumi-node/proof-scale.git
cd proof-scale
pip install -e ".[dev]"
pytest -q
```

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

## Features

- **generate → verify → critique → repair → select** pipeline
- **Diversity‑aware tournament** selection
- **Optional PPO‑style RL loop** to improve the generator
- **Heuristic mode** for fast, dependency‑free runs

## Modules

| Module | Description |
|--------|-------------|
| `config` | Configuration for the proof tournament pipeline. |
| `data` | Dataset loading for proof tournament training and evaluation. |
| `generator` | Proof generation via transformer sampling with RL-guided decoding. |
| `main` | High-level orchestrator for the proof tournament pipeline. |
| `repair` | Proof repair: fixes flawed proofs using verifier error masks. |
| `rl_trainer` | PPO-style reinforcement learning trainer for the proof generator. |
| `tournament` | Tournament selection over a population of proof candidates. |
| `utils` | Data containers and utility functions for the proof tournament pipeline. |
| `verifier` | Proof verification: scores candidates for mathematical validity. |

## Documentation

📖 Full documentation: [https://lumi-node.github.io/proof-scale/](https://lumi-node.github.io/proof-scale/)
📄 Technical paper: see [`paper/`](paper/) for the LaTeX source and compiled PDF.

> This is a reference implementation produced by an autonomous research pipeline. It is not published to PyPI; install from source as shown above.

## License

[MIT](LICENSE) © Andrew Young / Automate Capture Research
