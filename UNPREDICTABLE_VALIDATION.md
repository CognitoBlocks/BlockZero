# Unpredictable Validation System

## Overview

This document describes the anti-cheat validation system designed to prevent miners from gaming validation metrics through dataset memorization or prediction.

---

## The Problem: Predictable Validation

If validators use predictable validation datasets, miners can:

1. **Memorize validation data** during training
2. **Overfit to specific test sets** instead of generalizing
3. **Predict which examples** will be used for scoring
4. **Game the system** by focusing only on known validation patterns

This defeats the purpose of validation and rewards memorization over actual capability.

---

## The Solution: Multi-Layer Unpredictability

Our system makes validation **cryptographically unpredictable** through multiple layers:

### 1. **Blockchain Randomness** 🔐
```
Validation Seed = H(block_hash || validator_seeds || cycle_number || secret_salt)
```

**Why it works:**
- **Block hash doesn't exist** during training phase
- Miners train during "Train" phase (blocks 0-10)
- Block hash for "Validate" phase (blocks 30-40) doesn't exist yet
- **Cannot be predicted** without mining the future blockchain

**Example Timeline:**
```
Cycle 1:
  Block 0-10:  Training (miners train models)
  Block 10-15: Commit phase
  Block 15-25: Submission (miners submit)
  Block 25-35: Validation (uses block 30 hash - didn't exist during training!)
  Block 35-45: Merge

During blocks 0-10: Block 30's hash = ??? (unknown, literally doesn't exist)
At block 30: Block hash revealed = "0x4f3a..." (now miners know, but too late!)
```

### 2. **Validator Consensus** 🤝

Multiple validators must commit seeds:
- Each validator commits `miner_seed` to chain
- Seeds are hashed together: `H(seed_v1 || seed_v2 || seed_v3 || ...)`
- Requires collusion of **all validators** to predict

**Security:**
- If 5 validators participate, miner needs to compromise all 5
- Even one honest validator makes seed unpredictable

### 3. **Rotating Validation Strategies** 🔄

Seven different validation approaches rotate unpredictably:

| Strategy | Description | Anti-Gaming Measure |
|----------|-------------|---------------------|
| **Random Holdout** | Pure random from 1% holdout set | Baseline unpredictability |
| **Stratified Difficulty** | Sample across easy/medium/hard | Prevents easy-only optimization |
| **Compositional** | Unseen combinations of known parts | Tests true generalization |
| **Perturbed** | Modified versions of data | Prevents exact memorization |
| **Adversarial** | Known failure modes | Tests robustness |
| **Temporal Split** | Recent data only | Tests distribution shift handling |
| **Cross-Domain** | Different dataset mixtures | Prevents dataset-specific overfitting |

**Strategy Selection:**
```python
# Weighted random selection each cycle
strategy = select_strategy(validation_seed)  # Different every cycle!
```

### 4. **Domain-Specific Perturbations** ✨

Each expert group has custom perturbations:

**Math Expert (Group 0):**
```python
# Original: "Alice has 12 apples..."
# Perturbed: "Alice has 13 apples..."  (numbers changed)
```

**Agentic Expert (Group 1):**
```python
# Original: use_tool("calculator", ...)
# Perturbed: use_tool("math_tool", ...)  (tool names randomized)
```

**Planning Expert (Group 2):**
```python
# Original: [Step 1, Step 2, Step 3]
# Perturbed: [Step 1, Step 3, Step 2]  (non-dependent steps shuffled)
```

### 5. **Holdout Set Isolation** 🔒

- **1% of all data** held out permanently
- **Never seen during training**
- Deterministically selected via content hash
- Same examples always in holdout (can't "rotate in")

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Blockchain Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Validator 1  │  │ Validator 2  │  │ Validator 3  │      │
│  │ Commits seed │  │ Commits seed │  │ Commits seed │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └──────────────────┴──────────────────┘              │
│                            │                                  │
│                  Combined Validator Seed                      │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Block Hash at  │
                    │ Validation Time │
                    └────────┬────────┘
                             │
                  ┌──────────▼──────────┐
                  │  ValidationSeed     │
                  │  H(block || seeds   │
                  │    || cycle || salt)│
                  └──────────┬──────────┘
                             │
              ┌──────────────▼──────────────┐
              │  UnpredictableValidator     │
              │  - Select strategy          │
              │  - Load datasets            │
              │  - Apply perturbations      │
              │  - Sample examples          │
              └──────────────┬──────────────┘
                             │
                  ┌──────────▼──────────┐
                  │ Validation Dataset  │
                  │ (500 examples,      │
                  │  different each     │
                  │  cycle!)            │
                  └─────────────────────┘
```

---

## Implementation

### 1. Basic Usage (Validator Side)

```python
from mycelia.shared.validation_integration import get_unpredictable_validation_dataset

# Each validation cycle:
validation_dataset, validation_seed = get_unpredictable_validation_dataset(
    config=config,
    subtensor=subtensor,
    tokenizer=tokenizer,
    samples_per_validation=500,
)

# Use this dataset to evaluate miners
for miner in miners:
    score = evaluate_model(miner.model, validation_dataset)
    rankings[miner.uid] = score
```

### 2. Advanced: Custom Strategy

```python
# Force a specific strategy (for testing)
validation_dataset, seed = get_unpredictable_validation_dataset(
    config=config,
    subtensor=subtensor,
    tokenizer=tokenizer,
    force_strategy="perturbed",  # Always use perturbed strategy
)
```

### 3. Verification: Check Unpredictability

```python
# Generate seeds for 10 consecutive cycles
seeds = []
for cycle in range(10):
    seed = ValidationSeed(
        block_hash=f"0x{cycle:064x}",  # Mock block hashes
        validator_seed="combined_seed",
        cycle_number=cycle,
    )
    seeds.append(seed.to_int())

# Each should be completely different
assert len(set(seeds)) == 10  # All unique
print("Entropy:", np.std(seeds))  # Should be very high
```

---

## Security Analysis

### Attack Vector 1: "Memorize the Holdout Set"

**Attack:** Miner tries to find and memorize the 1% holdout.

**Defense:**
- Holdout selected via cryptographic hash of content
- Miner would need to hash **every possible training example**
- Even if found, doesn't know **which subset** will be sampled
- Different strategies use different compositions

**Difficulty:** ⭐⭐⭐⭐⭐ (Essentially impossible)

### Attack Vector 2: "Predict the Validation Seed"

**Attack:** Miner predicts future block hash.

**Defense:**
- Requires predicting blockchain state (PoW/PoS mining)
- Exponentially harder the further in advance
- Multiple validators must be compromised

**Difficulty:** ⭐⭐⭐⭐⭐ (Cryptographically hard)

### Attack Vector 3: "Collude with Validators"

**Attack:** Bribe all validators to reveal seeds early.

**Defense:**
- Requires **unanimous collusion** (all validators)
- Even one honest validator breaks the scheme
- Secret salt adds additional entropy
- Economic cost > potential reward

**Difficulty:** ⭐⭐⭐⭐ (Economically infeasible with many validators)

### Attack Vector 4: "Overfit to All Strategies"

**Attack:** Train models that work well on all 7 strategies.

**Defense:**
- **This is exactly what we want!**
- A model that generalizes to all strategies is a good model
- The goal is true capability, not gaming a single test set

**Difficulty:** ⭐ (Not actually an attack - this is the desired outcome)

### Attack Vector 5: "Replay Previous Validation Sets"

**Attack:** Collect past validation sets and memorize.

**Defense:**
- Each cycle uses **different seed** → different dataset
- Cycle number in seed ensures no repeats
- Probability of repeat selection = (500 / 100,000)^500 ≈ 0

**Difficulty:** ⭐⭐⭐⭐⭐ (Combinatorially impossible)

---

## Configuration

### Per-Expert Group Datasets

Edit `mycelia/shared/validation_integration.py`:

```python
EXPERT_DATASETS = {
    0: [  # Math group
        "openai/gsm8k",
        "lighteval/MATH",
        "meta-math/MetaMathQA",
    ],
    1: [  # Your custom group
        "your/dataset1",
        "your/dataset2",
    ],
}
```

### Validation Parameters

```python
# In validator config
samples_per_validation = 500  # More = better coverage, slower
holdout_fraction = 0.01       # 1% = good balance
use_secret_salt = True        # Additional entropy (recommended)
```

### Strategy Weights

Edit `UnpredictableValidator.select_strategy()` weights:

```python
weights = np.array([
    3.0,  # RANDOM_HOLDOUT - increase for more random sampling
    2.0,  # STRATIFIED_DIFFICULTY
    1.5,  # COMPOSITIONAL
    2.0,  # PERTURBED
    1.0,  # ADVERSARIAL
    1.5,  # TEMPORAL_SPLIT
    1.0,  # CROSS_DOMAIN
])
```

---

## Monitoring & Debugging

### Log Validation Seeds

```python
logger.info(
    "Validation metadata",
    cycle=validation_seed.cycle_number,
    block_hash=validation_seed.block_hash[:16] + "...",
    seed_int=validation_seed.to_int(),
    strategy=selected_strategy.value,
    dataset_size=len(validation_dataset),
)
```

### Verify Randomness

```python
# Test script to verify seed uniqueness
python -c "
from mycelia.shared.validation_integration import get_validation_seed_from_chain

# Generate 100 seeds
seeds = []
for i in range(100):
    seed = get_validation_seed_from_chain(config, subtensor)
    seeds.append(seed.to_int())

print(f'Unique seeds: {len(set(seeds))}/100')
print(f'Collision rate: {(100 - len(set(seeds)))/100:.2%}')
"
```

### Reproducibility

Save seeds for debugging:

```python
import json

validation_log = {
    "cycle": validation_seed.cycle_number,
    "block_hash": validation_seed.block_hash,
    "validator_seed": validation_seed.validator_seed,
    "seed_int": validation_seed.to_int(),
    "strategy": selected_strategy.value,
}

with open(f"validation_log_cycle_{cycle}.json", "w") as f:
    json.dump(validation_log, f, indent=2)
```

---

## Migration from Old System

### Step 1: Add Imports

```python
from mycelia.shared.validation_integration import (
    get_unpredictable_validation_dataset,
    create_unpredictable_validator,
)
```

### Step 2: Replace Dataloader Creation

**Before:**
```python
eval_dataloader = get_dataloader(
    config,
    rank=rank,
    world_size=world_size,
    tokenizer=tokenizer,
)
```

**After:**
```python
validation_dataset, validation_seed = get_unpredictable_validation_dataset(
    config=config,
    subtensor=subtensor,
    tokenizer=tokenizer,
    samples_per_validation=500,
)

# Convert to dataloader
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

eval_dataloader = DataLoader(
    validation_dataset,
    batch_size=config.task.data.per_device_train_batch_size,
    collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
```

### Step 3: Log Metadata

```python
logger.info(
    "Using unpredictable validation",
    seed_preview=str(validation_seed.to_int())[:16],
    dataset_size=len(validation_dataset),
)
```

---

## Testing

### Unit Tests

```python
def test_seed_uniqueness():
    """Verify seeds are unique across cycles."""
    seeds = []
    for cycle in range(100):
        seed = ValidationSeed(
            block_hash=f"block_{cycle}",
            validator_seed="test",
            cycle_number=cycle,
        )
        seeds.append(seed.to_int())

    assert len(set(seeds)) == 100, "Seeds must be unique"

def test_strategy_rotation():
    """Verify strategies rotate."""
    validator = UnpredictableValidator(...)
    strategies = []

    for cycle in range(50):
        seed = ValidationSeed(...)
        strategy = validator.select_strategy(seed)
        strategies.append(strategy)

    # Should see multiple different strategies
    assert len(set(strategies)) >= 3, "Not enough variety"

def test_perturbation_preserves_structure():
    """Math perturbations should keep problem structure."""
    example = {"text": "Alice has 12 apples and buys 5 more."}
    perturbed = validator._perturb_math_example(example, rng)

    # Numbers changed but structure preserved
    assert "Alice has" in perturbed["text"]
    assert "apples" in perturbed["text"]
    assert "12" not in perturbed["text"]  # Number changed
```

---

## Performance

### Computational Cost

| Operation | Time | Memory |
|-----------|------|--------|
| Generate seed | ~1ms | Negligible |
| Load holdout (first time) | ~10-60s | ~1GB |
| Load holdout (cached) | ~1ms | 0 |
| Sample 500 examples | ~100ms | ~10MB |
| Apply perturbations | ~500ms | ~20MB |
| **Total per cycle** | **~1-2s** | **~1GB** |

### Caching Strategy

```python
# Holdout datasets are cached after first load
_holdout_cache: dict[str, Dataset] = {}

# First cycle: 60s load time
# Subsequent cycles: <1s (cached)
```

---

## Future Enhancements

### 1. Multi-Round Validation

```python
# Run validation multiple times per cycle with different seeds
for round in range(3):
    sub_seed = validation_seed.derive_subseed(f"round_{round}")
    round_dataset = validator.get_validation_dataset(sub_seed)
    scores[round] = evaluate(miners, round_dataset)

# Aggregate scores
final_score = np.mean(scores)
```

### 2. Adaptive Difficulty

```python
# Increase difficulty for consistently high-scoring miners
if miner.avg_score > 0.95:
    validator.samples_per_validation = 1000  # More examples
    force_strategy = "adversarial"  # Harder strategy
```

### 3. Cross-Validator Verification

```python
# Validators compare their validation results
my_rankings = evaluate_all_miners(my_dataset)
other_rankings = fetch_from_other_validators()

# Verify agreement
correlation = np.corrcoef(my_rankings, other_rankings)
assert correlation > 0.8, "Validators disagree on rankings"
```

---

## FAQ

**Q: Can miners still memorize data?**
A: They can memorize the training data (expected), but not the validation data (it's different each time).

**Q: What if a miner gets lucky?**
A: With 500+ examples and multiple strategies, luck has minimal impact. Averaged over many cycles, skill dominates.

**Q: How do I debug validation failures?**
A: Save the `validation_seed` and `strategy` used. You can reproduce the exact dataset later.

**Q: Does this slow down validation?**
A: First load ~60s, subsequent cycles <1s (cached). Negligible impact.

**Q: Can I test with deterministic seeds?**
A: Yes! Set `secret_salt=None` and use fixed `block_hash` for reproducible testing.

---

## Summary

The unpredictable validation system provides:

✅ **Cryptographic unpredictability** (blockchain + validator consensus)
✅ **Strategy diversity** (7 different validation approaches)
✅ **Domain-specific robustness** (custom perturbations per expert)
✅ **Anti-memorization** (rotating holdout samples)
✅ **Attack resistance** (economically + computationally hard to game)

This ensures **miners must build genuinely capable models**, not memorize test sets.

---

**Last Updated:** 2025-12-30
**Version:** 1.0
**Status:** Ready for production
