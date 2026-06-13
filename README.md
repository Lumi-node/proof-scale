<p align="center">
  <img src="assets/hero.jpg" alt="ProofScale" width="900">
</p>

<h1 align="center">ProofScale</h1>

<p align="center">
  <strong>Population‑level test‑time scaling for LLM‑driven mathematical proofs.</strong>
</p>

<p align="center">
  <a href="https://github.com/Lumi-node/proof-scale"><img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub"></a>
  <a href="https://github.com/Lumi-node/proof-scale/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/proof-scale/"><img src="https://img.shields.io/badge/Python-≥3.10-blue.svg" alt="Python ≥3.10"></a>
  <a href="https://github.com/Lumi-node/proof-scale/actions"><img src="https://img.shields.io/badge/tests-184-success.svg" alt="Tests"></a>
</p>

---

ProofScale implements a full population‑level test‑time scaling pipeline for large language models (LLMs) that generate mathematical proofs. By treating proof candidates as a population, the system can dynamically allocate compute, repair corrupted proofs, and select the most reliable proof using a tournament‑style ranking. This approach improves both the quality and efficiency of LLM‑driven theorem proving, making it feasible to scale to large benchmark suites without sacrificing correctness.

The library is built on a clean `src/` layout, exposing a concise Python API (`proof_scale`) that integrates generation, verification, repair, and reinforcement‑learning‑based policy updates. Extensive unit tests (184 files) guarantee that each component behaves as expected, from tokenisation to end‑to‑end proof selection.

---

## Quick Start

```bash
pip install proof_scale
```

```python
from llm_proof_tournament.main import run
from llm_proof_tournament.tournament import select

# Generate and select the best proof for a single statement
candidate = run("Show that the sum of the first n integers is n(n+1)/2.")
best = select([candidate])
print(best.proof_text)
```

## What Can You Do?

### Generate Proof Candidates
```python
from llm_proof_tournament.generator import generate

candidates = generate("Prove that sqrt(2) is irrational.", num_samples=5)
for c in candidates:
    print(c.proof_text)
```

### Verify Proof Quality
```python
from llm_proof_tournament.verifier import score

verdict = score(candidates[0])
print(verdict.confidence, verdict.is_valid)
```

### Repair Corrupted Proofs
```python
import numpy as np
from llm_proof_tournament.repair import repair

mask = np.array([1, 0, 1])  # example mask indicating which steps to keep
fixed = repair(candidates[0], mask)
print(fixed.proof_text)
```

### Reinforcement‑Learning Policy Updates
```python
from llm_proof_tournament.rl_trainer import step, update_policy

batch = [candidates[0], candidates[1]]
metrics = step(batch)
policy_info = update_policy()
print(metrics, policy_info)
```

## Architecture

ProofScale is organized around a small set of core modules that interact in a pipeline:

```mermaid
flowchart TD
    A[Data Loader] --> B[Generator]
    B --> C[Verifier]
    C --> D[Tournament]
    D --> E[Repair (optional)]
    E --> B
    D --> F[RL Trainer]
    F --> B
```

* **`config.py`** – Loads and overrides configuration objects.  
* **`data.py`** – Provides access to statements and reference proofs.  
* **`generator.py`** – Generates `ProofCandidate` objects and tracks reward baselines.  
* **`verifier.py`** – Scores candidates, returning a `VerifierOutput`.  
* **`tournament.py`** – Implements selection strategies (`select`, `select_top_k`).  
* **`repair.py`** – Repairs a candidate given a mask of tokens to keep.  
* **`utils.py`** – Helper functions (`tokenize`, `levenshtein`, `calibrate_confidence`).  
* **`rl_trainer.py`** – Stores experiences, samples batches, and updates the policy.

## API Reference

### `config.py`
```python
Config.load(**overrides) -> Config
config = Config.load()
config_dict = config.as_dict()
```

### `data.py`
```python
data = Data()
statements = data.statements()          # List[str]
reference_proofs = data.proofs()        # List[str]
```

### `generator.py`
```python
gen = Generator()
candidates = gen.generate(statement, num_samples=3)
baseline = gen.reward_baseline()
gen.reward_baseline(0.75)
params = gen.get_parameters()
```

### `main.py`
```python
runner = Main()
single = runner.run("Prove Fermat's Last Theorem for n=3.")
batch = runner.run_batch(["Stmt 1", "Stmt 2"])
```

### `repair.py`
```python
fixed = repair(candidate, mask)   # returns ProofCandidate
```

### `rl_trainer.py`
```python
trainer = RLTrainer()
trainer.add(exp)
batch = trainer.sample(32)
trainer.step(batch)
trainer.update_policy()
stats = trainer.stats()
```

### `tournament.py`
```python
winner = tournament.select(population)
top_k = tournament.select_top_k(population, k=3)
```

### `utils.py`
```python
logp = utils.mean_log_prob()
tokens = utils.tokenize("a + b = c")
dist = utils.levenshtein("proof", "pr0of")
conf = utils.calibrate_confidence(raw_score, temperature=0.8)
```

### `verifier.py`
```python
verdict = verifier.score(candidate)
conf = verdict.confidence
valid = verdict.is_valid
threshold = verifier.threshold()
```

## Research Background

ProofScale builds on recent work in **test‑time scaling** for LLMs, where a population of outputs is used to allocate compute adaptively. Key inspirations include:

* *Population‑Based Training* (Jaderberg et al., 2017) – https://arxiv.org/abs/1711.09846  
* *Self‑Consistency* for chain‑of‑thought prompting (Wang et al., 2022) – https://arxiv.org/abs/2203.11171  
* *LLM‑driven theorem proving* surveys (Gao et al., 2023) – https://arxiv.org/abs/2305.02334  

ProofScale adapts these ideas to the domain of mathematical proof generation, adding a lightweight repair module and a reinforcement‑learning loop to continuously improve the policy.

## Testing

The repository ships with **184** unit‑test files covering every public function. Run the full suite with:

```bash
pytest -q
```

Continuous integration runs on every push to ensure 100 % test coverage.

## Contributing

We welcome contributions! Please:

1. Fork the repo and create a feature branch.  
2. Follow the existing code style (PEP 8, type hints).  
3. Add tests for any new functionality.  
4. Submit a pull request with a clear description.

See `CONTRIBUTING.md` for detailed guidelines.

## Citation

If you use ProofScale in academic work, please cite:

```bibtex
@software{young2024proofscale,
  author = {Young, Andrew},
  title = {ProofScale: Population‑level test‑time scaling for LLM‑driven mathematical proofs},
  year = {2024},
  url = {https://github.com/Lumi-node/proof-scale},
  license = {MIT}
}
```

## License

ProofScale is released under the **MIT License**. See the `LICENSE` file for details.