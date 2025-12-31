# Expert Group Design

## Overview

Three specialized expert groups, each with curated datasets and anti-cheat validation benchmarks.

---

## Expert 1: Math (`exp_math`)

### Purpose
Mathematical reasoning, theorem proving, numerical computation

### Datasets (merged)
| Dataset | Size | Focus |
|---------|------|-------|
| `openai/gsm8k` | 8.5K | Grade school math |
| `hendrycks/MATH` | 12.5K | Competition math |
| `deepmind/aqua_rat` | 100K | Algebraic word problems |
| `allenai/lila` | 140K | Math + language |
| `EleutherAI/math_qa` | 37K | GRE/GMAT style |
| `ChilleD/SVAMP` | 1K | Simple variations |
| `meta-math/MetaMathQA` | 395K | Augmented math |
| `nvidia/OpenMathInstruct-2` | 14M | Synthetic math |

**Total: ~15M+ examples**

### Benchmark Sampling
- Random 1% held out for validation
- Stratified by difficulty level
- New random sample each epoch

---

## Expert 2: Agentic (`exp_agentic`)

### Purpose
Tool use, function calling, ReAct reasoning, API interaction

### Datasets (merged)
| Dataset | Size | Focus |
|---------|------|-------|
| `glaiveai/glaive-function-calling-v2` | 113K | Function calling |
| `Salesforce/xlam-function-calling-60k` | 60K | API calls |
| `berkeley-nest/Gorilla-API` | 16K | Real APIs |
| `ToolBench/ToolBench` | 500K+ | Tool chains |
| `microsoft/AgentInstruct` | 25K | Agent traces |
| `THUDM/AgentTuning` | 100K | Agent behavior |
| `NousResearch/hermes-function-calling-v1` | 12K | Structured output |

**Total: ~800K+ examples**

### Benchmark Sampling
- Random API/tool combinations
- Unseen tool compositions
- Multi-step chains

---

## Expert 3: Planning (`exp_planning`)

### Purpose
Multi-step planning with **uncertainty decay modeling**

### Core Concept: Uncertainty Decay
```
P(success at step n) = P_base × decay^n × confidence(state_n)

Where:
- P_base = initial success probability
- decay = 0.9-0.95 per step (learned)
- confidence(state_n) = model's uncertainty at step n
```

### Datasets (merged)
| Dataset | Size | Focus |
|---------|------|-------|
| `bigcode/self-oss-instruct-sc2` | 75K | Code planning |
| `google/natural-instructions` | 1.6M | Task decomposition |
| `allenai/natural-instructions-v2` | 1.6M | Multi-step instructions |
| `TIGER-Lab/ARC-AGI-100` | 100 | Abstract reasoning |
| `Muennighoff/muennighoff-flan` | 1M | Reasoning chains |
| `kaist-ai/CoT-Collection` | 1.88M | Chain-of-thought |
| `allenai/WildChat` | 1M | Real planning dialogues |

**Total: ~7M+ examples**

### Uncertainty Decay Features
1. **Step confidence tokens**: `[CONF:0.95]`, `[CONF:0.8]`
2. **Branching markers**: `[ALT]` for alternative paths
3. **Failure anticipation**: `[RISK:high]` annotations
4. **Rollback points**: `[CHECKPOINT]` for recovery

### Benchmark Sampling
- Long-horizon plans (10+ steps)
- Plans with intentional failure points
- Recovery scenario testing

---

## Anti-Cheat Measures

### 1. Dynamic Benchmark Sampling
```python
def get_validation_sample(dataset, epoch, seed=42):
    """Never same sample twice"""
    rng = np.random.default_rng(seed + epoch)
    indices = rng.choice(len(dataset), size=min(1000, len(dataset)//100))
    return dataset.select(indices)
```

### 2. Synthetic Perturbations
- Number swapping in math problems
- Tool name randomization
- Step order shuffling

### 3. Holdout Test Set
- 5% completely hidden from training
- Only used for final validation scoring

### 4. Compositional Generalization
- Test on unseen combinations of known primitives
- "Can it plan with tools it hasn't seen together?"

