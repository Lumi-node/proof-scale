# ProofScale

## Population‑level test‑time scaling for LLM‑driven mathematical proofs

<div align="center">
  <a href="https://github.com/yourorg/llm_proof_tournament" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="https://pypi.org/project/llm-proof-tournament/" target="_blank">
    <img src="https://img.shields.io/pypi/v/llm-proof-tournament.svg" alt="PyPI version">
  </a>
</div>

---  

### 🚀 Quick install

```bash
pip install llm-proof-tournament
```

[Get started →](getting_started.md)

---  

## Features

<div class="grid cards">

### 🔄 Population‑level Generation
Generate **hundreds of proof candidates** in parallel, leveraging LLMs to explore diverse solution paths.

### ✅ Automated Verification
Each candidate is **checked with a symbolic verifier**; invalid or corrupted proofs are filtered out automatically.

### 📊 Scalable Ranking
Statistical ranking selects the **most reliable proof** using Bayesian confidence intervals and population‑level metrics.

### 🛠️ Seamless Integration
A clean `src/` layout, type‑annotated API, and **unit‑tested pipeline** ready to plug into your research workflow.

</div>

---  

## Why ProofScale?

- **Robustness** – Guarantees a mathematically valid proof even when individual LLM outputs are noisy.  
- **Scalability** – Handles large populations of candidates without manual intervention.  
- **Transparency** – All steps (generation, verification, ranking) are logged and reproducible.  

---  

### Ready to explore?

```bash
python -m llm_proof_tournament.run --example theorem.yaml
```

Dive into the docs, run the examples, and see how ProofScale turns LLM creativity into provable mathematics.