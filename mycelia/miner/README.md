# Miner

Miner is a two-process setup for training and syncing a model on a decentralized network.

## Overview

A miner runs two cooperating components:

**Local Training**: trains your model and writes checkpoints.

**Model I/O**: handles chain communication: checks chain status, pushes your latest checkpoint for validator evaluation, and pulls new model updates from the validator.

Both read the same .

## Installation

Use a virtual environment and install project dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration Setup

### Step 1: Create Expert Group Configs (One-time setup)

Before creating a miner config, each expert group needs a `config.yaml` file.

**Option A: Automated Setup (Recommended)**

```bash
cd expert_groups
./setup_configs.sh
```

This creates all configs with correct defaults automatically.

**Option B: Manual Setup**

```bash
cd expert_groups

# Copy template for each expert group
cp config.template.yaml exp_math/config.yaml
cp config.template.yaml exp_agentic/config.yaml
cp config.template.yaml exp_planning/config.yaml
cp config.template.yaml exp_dummy/config.yaml
```

Then edit each file to set the correct values:

| Expert Group | ID | Dataset Class | Edit Required |
|--------------|----|--------------|--------------|
| exp_math | 0 | `MergedMathDataset` | Update IDs only |
| exp_agentic | 1 | `MergedAgenticDataset` | Update IDs only |
| exp_planning | 2 | `MergedPlanningDataset` | Update IDs only |
| exp_dummy | 99 | None (uses C4) | Update IDs + dataset_name |

**Example** for `exp_agentic/config.yaml`:
```yaml
data:
  dataset_name: "merged_agentic"
  dataset_class: "expert_groups.exp_agentic.dataset:MergedAgenticDataset"  # Note: colon separator
  # ... other settings from template

expert_group_id: 1
expert_group_name: "exp_agentic"
```

### Step 2: Generate Miner Config

Create a miner configuration file by running:

```bash
python mycelia/shared/config.py \
  --get_template miner \
  --coldkey_name <your_coldkey> \
  --hotkey_name <your_hotkey> \
  --run_name <run_identifier>
```

**Example**:
```bash
python mycelia/shared/config.py \
  --get_template miner \
  --coldkey_name testnet \
  --hotkey_name default \
  --run_name 1
```

This creates:
```
checkpoints/miner/<coldkey>/<hotkey>/<run_name>/config.yaml
```

### Step 3: Customize Miner Config (Optional)

Edit the generated config file to adjust:
- **Expert group assignment**: `moe.my_expert_group_id` (0=math, 1=agentic, 2=planning)
- **Batch size**: `task.data.per_device_train_batch_size`
- **Learning rate**: `opt.lr`
- **Training steps**: `sched.total_steps`
- **Checkpoint frequency**: `ckpt.checkpoint_interval`
- **Wandb logging**: `log.log_wandb`

### Step 4: Use the Config

Point to your config when running miner commands:

```bash
python mycelia/miner/train.py \
  --path checkpoints/miner/<coldkey>/<hotkey>/<run_name>/
```

> **Note**: If no path is provided, the system uses default config values from `mycelia/shared/config.py`

## Usage

### 1) Run Local Training

Trains the model locally and saves checkpoints.

```bash
python mycelia/miner/train.py \
  --path checkpoints/miner/<coldkey>/<hotkey>/<run_name>/
```

**Example**:
```bash
python mycelia/miner/train.py \
  --path checkpoints/miner/testnet/default/1/
```

### 2) Run Model I/O

Maintains chain communication, pushes checkpoints to the validator, and pulls updates.

```bash
python mycelia/miner/model_io.py \
  --path checkpoints/miner/<coldkey>/<hotkey>/<run_name>/
```

**Example**:
```bash
python mycelia/miner/model_io.py \
  --path checkpoints/miner/testnet/default/1/
```

## Running Both Together

Use two terminals (or tmux/screen):

### Terminal A: Training
```bash
python mycelia/miner/train.py \
  --path checkpoints/miner/testnet/default/1/
```

### Terminal B: Model I/O
```bash
python mycelia/miner/model_io.py \
  --path checkpoints/miner/testnet/default/1/
```

## Tips

1. Keep both processes pointed at the same config directory
2. Use a separate directory per hotkey to avoid mixing artifacts
3. Monitor `metrics/<run_name>.csv` for training progress
4. Check logs in `checkpoints/miner/<coldkey>/<hotkey>/<run_name>/` for debugging

## Key Config Parameters

- `moe.my_expert_group_id`: Which expert group you're training (0=math, 1=agentic, 2=planning)
- `moe.aux_load_balance`: Enable MoE load balancing (should be `true`)
- `moe.router_aux_loss_coef`: Load balancing loss weight (default: 1.0)
- `model.precision`: Mixed precision mode (default: fp16-mixed)
- `task.data.per_device_train_batch_size`: Batch size per device
- `sched.total_steps`: Total training steps
- `ckpt.checkpoint_interval`: How often to save checkpoints


# Troubleshooting


# Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you’d like to change. Please add/update tests as appropriate.