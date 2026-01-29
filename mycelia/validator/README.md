# Validator

Validator is a two-process setup for continuously evaluating miners and serving updated models on a decentralized network.

## Overview

A validator runs two cooperating components:

1. **Constant Evaluation**: discovers miners ready for evaluation, evaluates their submissions, aggregates results, and publishes scores to the chain (and shares with peer validators).
2. **Model Serving**: serves updated models to two audiences:

   * **Peers/clients** who may need the **full model**
   * **Miners** who may need a **partial model** (e.g., a subset of experts plus shared weights)

Both processes read the same validator config file.

---

## Prerequisites

* Python **3.10+**
* Project dependencies installed in a virtual environment:

```
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

```
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install -e .
  ```

* Register to the subnet (creates the wallet/hotkey on chain):

  ```bash
  btcli subnets register \
    --netuid <NETUID> \
    --wallet-name <MY_COLDKEY_NAME> \
    --hotkey <MY_HOTKEY_NAME> \
    --subtensor.network test
  ```

---

## Paths & Layout

All miner/validator settings are controlled via a YAML config file.

Create a template config file:

```bash
python mycelia/shared/config.py \
  --get_template <validator/miner> \
  --coldkey_name <your coldkey name> \
  --hotkey_name <your hotkey name> \
  --run_name <your naming to specify this run>
```

Edit the generated YAML config to change any specifics you need.

When running the validator, use `--path` to point at the directory that contains the validator config:

```
~/subnet-MoE/checkpoints/validator/<your hotkey>/<run name>/
```

> If `--path` is not provided, the validator falls back to the default config from `mycelia/config.py`.

---

## 1) Constant Evaluation

Continuously gathers miners to evaluate, runs standardized evaluation, aggregates results, and publishes to the chain (and shares results with other validators).

```bash
python mycelia/validator/run.py \
  --path <path to the config directory you built in Paths & Layout>
```

**What `mycelia/validator/run.py` does (step-by-step):**

* Connects to the subtensor network and wallet defined in the config.
* Discovers miners ready for evaluation by polling on-chain state and the local queue.
* Fetches miner submissions and resolves model metadata (version/hash, hotkey, and UID).
* Downloads or loads submitted checkpoints (with resume/retry to handle partial downloads).
* Evaluates submissions on the configured dataset/dataloader and records per-miner metrics.
* Aggregates scores per UID/hotkey and resets history on hotkey changes.
* Publishes scores to the chain and optionally shares summaries with peers.

---

## 2) Model Serving (Updated Model)

Serves the updated model to two groups:

* **(a) Other validators/clients** — download the **full model**
* **(b) Miners** — download the **partial model** required for training

```bash
python3 mycelia/shared/server.py \
  --path <path to the config directory you built in Paths & Layout>
```

**What `mycelia/shared/server.py` does (step-by-step):**

* Loads the validator config and resolves the model artifacts on disk.
* Starts the HTTP/RPC server described in the config and binds to the configured host/port.
* Exposes endpoints for full vs partial model artifacts (validators/clients vs miners).
* Streams model files and/or metadata to peers, with caching where configured.
* Applies auth rules if enabled in the config.
* Logs requests and maintains indices under `serve/`.

---

## Tips

* Keep both processes pointed at the **same** config directory.
* Separate directories per validator/hotkey keep artifacts clean (`hk3/`, `hk4/`, …).
* Consider enabling **log rotation** for long-running services.
* If serving public endpoints, place the server behind a reverse proxy (nginx/traefik) and enable TLS.

---

## Troubleshooting

* **Evaluation stalls / no miners fetched**: check wallet registration, netuid, and chain connectivity.
* **Download errors / timeouts**: verify peer reachability and retry settings in the config.
* **Dataloader worker exited unexpectedly**: reduce worker count or confirm dataset paths.
* **High GPU memory usage**: lower batch size or reduce evaluation concurrency.
* **Serve endpoints not reachable**: confirm host/port and any reverse proxy settings.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your proposal. Remember to update tests and docs as appropriate.
