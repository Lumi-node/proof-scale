# ProofScale – API Reference  

*Package name:* `llm_proof_tournament`  
*Source layout:* `src/llm_proof_tournament/`  

The following reference documents **only** the public symbols that are part of the official API. Each entry shows the exact signature, a short description, and a minimal example of how to call the function / method.

---  

## `src/llm_proof_tournament/__init__.py`

The package’s top‑level `__init__` re‑exports the most‑used classes so they can be imported directly:

```python
from llm_proof_tournament import (
    Config,
    ProofCandidate,
    Experience,
    VerifierOutput,
    Generator,
    Repair,
    RLTrainer,
    Tournament,
    Utils,
    Verifier,
    Main,
)
```

*(No additional symbols are defined here; see the modules below for the concrete API.)*  

---  

## `src/llm_proof_tournament/config.py`

### `load(cls, **overrides) -> Config`

*Factory function* that creates a `Config` instance (the concrete class is defined in this module).  
`overrides` can be any configuration field name to replace the default value.

**Example**

```python
from llm_proof_tournament.config import load

cfg = load(model_name="gpt-4", max_tokens=512)
```

### `Config.as_dict(self) -> dict`

Returns the configuration as a plain Python dictionary – useful for logging or for passing the settings to downstream components.

**Example**

```python
cfg_dict = cfg.as_dict()
print(cfg_dict["model_name"])   # → "gpt-4"
```

---  

## `src/llm_proof_tournament/data.py`

### `statements(self) -> List[str]`

Returns the list of problem statements that the dataset contains.

**Example**

```python
from llm_proof_tournament.data import DataLoader

loader = DataLoader()
all_statements = loader.statements()
print(all_statements[:3])
```

### `proofs(self) -> List[str]`

Returns the list of reference (ground‑truth) proofs aligned with the statements returned by `statements()`.

**Example**

```python
reference_proofs = loader.proofs()
print(reference_proofs[0])
```

---  

## `src/llm_proof_tournament/generator.py`

### `generate(self, statement: str, num_samples: int = 0) -> List[ProofCandidate]`

Given a mathematical statement, produce `num_samples` candidate proofs (or a default number defined by the generator). Returns a list of `ProofCandidate` objects.

**Example**

```python
from llm_proof_tournament.generator import Generator

gen = Generator()
candidates = gen.generate("Every even integer > 2 is the sum of two primes.", num_samples=5)
print(len(candidates))   # → 5
```

### `reward_baseline(self) -> float`

Getter for the current baseline reward used by the generator (e.g., the average verification score of recent candidates).

**Example**

```python
baseline = gen.reward_baseline()
print(baseline)   # e.g. 0.73
```

### `reward_baseline(self, value: float)`

Setter that updates the baseline reward. The new value will influence the generator’s sampling temperature.

**Example**

```python
gen.reward_baseline(0.85)   # set a higher baseline
```

### `get_parameters(self)`

Returns the internal hyper‑parameters of the generator (learning rate, temperature, etc.) as a dictionary.

**Example**

```python
params = gen.get_parameters()
print(params["temperature"])
```

---  

## `src/llm_proof_tournament/main.py`

### `run(self, statement: str) -> ProofCandidate`

Runs the full pipeline for a single statement: generation, verification, possible repair, and tournament selection. Returns the best `ProofCandidate`.

**Example**

```python
from llm_proof_tournament.main import Main

pipeline = Main()
best = pipeline.run("The square root of 2 is irrational.")
print(best.text)
```

### `run_batch(self, statements: List[str]) -> List[ProofCandidate]`

Runs the pipeline on a batch of statements, returning the best candidate for each input.

**Example**

```python
batch = ["Pythagoras theorem", "Fundamental theorem of calculus"]
results = pipeline.run_batch(batch)
for pc in results:
    print(pc.text)
```

---  

## `src/llm_proof_tournament/repair.py`

### `repair(self, candidate: ProofCandidate, mask: np.ndarray) -> ProofCandidate`

Attempts to fix a proof candidate by masking out (i.e., removing) parts of the text indicated by `mask` (a boolean NumPy array aligned with token positions) and re‑generating those sections. Returns a new `ProofCandidate`.

**Example**

```python
import numpy as np
from llm_proof_tournament.repair import Repair

repairer = Repair()
bad = candidates[0]                     # assume this one is flawed
mask = np.array([0, 0, 1, 0, 1, 0])    # mask token 2 and 4
fixed = repairer.repair(bad, mask)
print(fixed.text)
```

---  

## `src/llm_proof_tournament/rl_trainer.py`

### `add(self, exp: Experience)`

Adds a single experience tuple (state, action, reward, next_state) to the replay buffer.

**Example**

```python
from llm_proof_tournament.rl_trainer import RLTrainer, Experience

trainer = RLTrainer()
exp = Experience(state="...", action="...", reward=1.0, next_state="...")
trainer.add(exp)
```

### `sample(self, batch_size: int) -> List[Experience]`

Draws a random minibatch of experiences from the buffer.

**Example**

```python
batch = trainer.sample(32)
print(len(batch))   # → 32
```

### `clear(self)`

Empties the replay buffer.

**Example**

```python
trainer.clear()
```

### `step(self, batch: List[ProofCandidate]) -> dict`

Performs a training step using a batch of proof candidates (treated as the “environment” observations). Returns a dictionary of loss metrics.

**Example**

```python
metrics = trainer.step(candidates)
print(metrics["policy_loss"])
```

### `update_policy(self) -> dict`

Updates the policy network after enough gradient steps. Returns a summary of the update (e.g., KL divergence, new learning rate).

**Example**

```python
update_info = trainer.update_policy()
print(update_info["kl"])
```

### `stats(self) -> dict`

Provides current statistics of the trainer (buffer size, total steps, average reward, etc.).

**Example**

```python
print(trainer.stats()["buffer_len"])
```

---  

## `src/llm_proof_tournament/tournament.py`

### `select(self, population: List[ProofCandidate]) -> ProofCandidate`

Runs a single tournament selection (pairwise comparisons) and returns the winner.

**Example**

```python
from llm_proof_tournament.tournament import Tournament

tour = Tournament()
winner = tour.select(candidates)
print(winner.score)
```

### `select_top_k(self, population: List[ProofCandidate], k: int = 1) -> List[ProofCandidate]`

Selects the top‑`k` candidates from the population according to their verification scores.

**Example**

```python
top3 = tour.select_top_k(candidates, k=3)
for pc in top3:
    print(pc.score)
```

### `tournament_round(`*…*`)`

*Signature omitted for brevity* – the function orchestrates a full round of pairwise matches, updates scores, and returns the updated population. (The exact signature is internal; the public entry points are `select` and `select_top_k`.)

---  

## `src/llm_proof_tournament/utils.py`

### `mean_log_prob(self) -> float`

Computes the mean log‑probability of the tokens in a `ProofCandidate` (used for confidence calibration).

**Example**

```python
mean_lp = candidate.mean_log_prob()
print(mean_lp)
```

### `tokenize(text: str) -> List[str]`

Simple whitespace tokenizer that splits a string into tokens.

**Example**

```python
tokens = tokenize("a + b = c")
print(tokens)   # ['a', '+', 'b', '=', 'c']
```

### `levenshtein(a: str, b: str) -> int`

Returns the Levenshtein edit distance between two strings.

**Example**

```python
dist = levenshtein("proof", "pr0of")
print(dist)   # → 1
```

### `calibrate_confidence(raw_score: float, temperature: float = 1.0) -> float`

Applies temperature scaling to a raw verification score to obtain a calibrated confidence value between 0 and 1.

**Example**

```python
conf = calibrate_confidence(0.78, temperature=0.5)
print(conf)   # higher confidence after scaling
```

---  

## `src/llm