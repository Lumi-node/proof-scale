# API Reference

`ProofScale` is imported as `llm_proof_tournament`. The public API:

```python
import llm_proof_tournament
```

### `Config`

Key methods: `as_dict()`

### `ProofCandidate`

Data model.

### `ProofDataset`

Data model.

### `ProofGenerator`

Key methods: `generate()`, `get_parameters()`

### `ProofPipeline`

Key methods: `run()`, `run_batch()`

### `ProofRepair`

Key methods: `repair()`

### `ProofVerifier`

Key methods: `score()`

### `RLTrainer`

Key methods: `step()`, `update_policy()`

### `TournamentSelector`

Key methods: `select()`, `select_top_k()`, `tournament_round()`

### `VerifierOutput`

Data model.

