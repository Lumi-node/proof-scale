# Quick‑Start Guide – **llm_proof_tournament**

Welcome! This guide shows you how to install the **llm_proof_tournament** package and start generating, repairing, verifying, and ranking proof candidates with just a few lines of Python.

---

## 1. Install the package

```bash
# From PyPI (once the package is released)
pip install llm_proof_tournament

# Or, install from a local checkout (recommended for development)
git clone https://github.com/your‑org/llm_proof_tournament.git
cd llm_proof_tournament
pip install -e .
```

The package follows a `src/` layout, so the import path is:

```python
from llm_proof_tournament import (
    config,
    data,
    generator,
    main,
    repair,
    rl_trainer,
    tournament,
    utils,
    verifier,
)
```

---

## 2. Load a configuration

All components read their settings from a single `Config` object.  
You can load the default configuration and optionally override any field:

```python
from llm_proof_tournament.config import load

# Load defaults
cfg = load()

# Override a couple of values (e.g., model name and temperature)
cfg = load(model_name="gpt‑4o-mini", temperature=0.7)

print(cfg.as_dict())
```

---

## 3. Prepare data (statements & reference proofs)

```python
from llm_proof_tournament.data import statements, proofs

# Example statements (you could also load from a file)
my_statements = statements()
# Reference proofs (used by the verifier for scoring)
reference_proofs = proofs()
```

`statements()` returns a `List[str]` of theorem statements, while `proofs()` returns the matching ground‑truth proofs.

---

## 4. Generate proof candidates

```python
from llm_proof_tournament.generator import generate, reward_baseline, get_parameters

# Generate up to 5 candidates for a single statement
candidates = generate("For all n ∈ ℕ, Σ_{i=1}^n i = n(n+1)/2", num_samples=5)

# Inspect the first candidate
print(candidates[0].text)          # the raw proof string
print(candidates[0].log_prob)     # log‑probability assigned by the LLM
```

You can also query or set the baseline reward that the RL trainer uses:

```python
baseline = reward_baseline()          # current baseline
reward_baseline(baseline + 0.1)       # bump it a little
```

Retrieve the underlying model parameters (e.g., temperature) if needed:

```python
params = get_parameters()
print(params)
```

---

## 5. Verify a candidate

The verifier scores a proof against the reference proof(s) and returns a confidence score.

```python
from llm_proof_tournament.verifier import score, threshold

candidate = candidates[0]
verif_out = score(candidate)

print("Raw score:", verif_out.raw_score)
print("Calibrated confidence:", verif_out.confidence)

# Simple pass/fail using the built‑in threshold
if verif_out.confidence >= threshold():
    print("✅ Proof is accepted")
else:
    print("❌ Proof is rejected")
```

`VerifierOutput` (the return type of `score`) contains at least `raw_score` and `confidence` fields.

---

## 6. Repair a bad candidate

If a candidate fails verification, you can ask the model to “repair” it.  
You need to provide a binary mask (`np.ndarray`) indicating which tokens to keep (1) and which to regenerate (0).

```python
import numpy as np
from llm_proof_tournament.repair import repair

# Example: keep the first half of the tokens, regenerate the rest
tokens = utils.tokenize(candidate.text)
mask = np.concatenate([np.ones(len(tokens)//2), np.zeros(len(tokens) - len(tokens)//2)])

repaired = repair(candidate, mask)

print("Repaired proof:", repaired.text)
```

After repair you can re‑run verification on `repaired`.

---

## 7. Run a full tournament (selection)

The tournament module implements the population‑level ranking logic.

```python
from llm_proof_tournament.tournament import select, select_top_k, tournament_round

# Suppose we have a mixed population (some good, some corrupted)
population = candidates + [repaired]  # list of ProofCandidate objects

# Single‑winner selection
winner = select(population)
print("🏆 Winner:", winner.text)

# Top‑3 selection
top3 = select_top_k(population, k=3)
print("🥇🥈🥉", [c.text for c in top3])

# Run a full round (e.g., for RL training)
new_population = tournament_round(population)
```

`select` returns the highest‑scoring candidate (according to the verifier), while `select_top_k` returns the best *k* candidates.

---

## 8. Reinforcement‑Learning trainer (optional)

If you want to fine‑tune the LLM policy with RL, use the trainer:

```python
from llm_proof_tournament.rl_trainer import add, sample, step, update_policy, stats, clear

# Add an experience (proof + reward)
add(Experience(proof=candidate, reward=verif_out.confidence))

# Sample a batch for a training step
batch = sample(batch_size=32)

# Perform a gradient step on the batch
step_info = step(batch)
print("Step loss:", step_info["loss"])

# Update the policy (e.g., PPO clip)
policy_info = update_policy()
print("Policy updated:", policy_info)

# Inspect trainer statistics
print(stats())

# Reset the replay buffer when needed
clear()
```

`Experience` is a simple data container (proof + reward) used by the trainer.

---

## 9. End‑to‑end example (single statement)

```python
from llm_proof_tournament import (
    config, generator, verifier, tournament, repair, utils,
)

# 1️⃣ Load config
cfg = config.load(model_name="gpt-4o-mini", temperature=0.6)

# 2️⃣ Generate candidates
cands = generator.generate("If a function f is continuous on [a,b], then it attains a maximum.", num_samples=4)

# 3️⃣ Verify & keep the best
best = tournament.select(cands)

# 4️⃣ If the best fails, try repairing
if verifier.score(best).confidence < verifier.threshold():
    mask = utils.tokenize(best.text)[:len(best.text)//2]  # keep first half
    mask_arr = np.array([1]*len(mask) + [0]*(len(best.text)-len(mask)))
    best = repair.repair(best, mask_arr)

# 5️⃣