# Miner

Miner is a two-process setup for training and syncing a model on a decentralized network.

## Overview

A miner runs two cooperating components:

**Local Training**: trains your model and writes checkpoints.

**Model I/O**: handles chain communication: checks chain status, pushes your latest checkpoint for validator evaluation, and pulls new model updates from the validator.

Both read the same config.json.

## Installation

Use a virtual environment and install project dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
1) Run Local Training

Trains the model locally and saves checkpoints.
```bash
python mycelia/miner/train.py \
  --path /home/isabella/crucible/subnet-MoE/checkpoints/miner/<your hotkey>/<run name>/config.json
```
2) Run Model I/O

Maintains chain communication, pushes checkpoints to the validator, and pulls updates.

```bash
python mycelia/miner/model_io.py \
  --path /home/isabella/crucible/subnet-MoE/checkpoints/miner/<your hotkey>/<run name>/config.json
```

## Running both together

Use two terminals (or tmux/screen):

### Terminal A: training
python mycelia/miner/train.py --path /home/isabella/.../foundation/config.json

### Terminal B: model I/O
python mycelia/miner/model_io.py --path /home/isabella/.../foundation/config.json

## Tips

1) Keep both processes pointed at the same config.json.

2) Use a separate directory per hotkey (e.g., hk1/, hk2/) to avoid mixing artifacts.


# Troubleshooting


# Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you’d like to change. Please add/update tests as appropriate.