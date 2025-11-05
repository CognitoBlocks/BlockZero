"""
Distributed evaluation utilities for sharded model checkpoints (RPC-based).

This module provides:
  * Lightweight RPC helpers (`send_shard`, `mark_done`) for workers to stream
    shard state_dicts to a dedicated evaluator node.
  * An `Evaluator` actor that collects shards per step, assembles the full model,
    runs evaluation, and logs metrics.
  * Helpers to assemble the model and to run the evaluation loop.

Assumptions
-----------
* Workers send CPU tensors (cheaper to serialize) as shard state_dicts.
* There exists a project-level logger named `logger`.
* Project utilities provide:
    - get_base_model(config)       -> nn.Module
    - get_base_tokenizer(config)   -> HF tokenizer
    - get_dataloader(config, ...)  -> eval dataloader
    - MetricLogger(config).log(dict)
"""

from __future__ import annotations

import gc
import threading
import time
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm

import torch
from torch import nn
import logging

import hashlib
from typing import Dict, List, Any

from torch.distributed import rpc
from torch.futures import Future
from mycelia.shared.chain import get_status, MinerStatus
from mycelia.shared.config import BaseConfig
from mycelia.shared.app_logging import structlog
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.model import get_base_model
from mycelia.shared.modeling.modeling_mycelia import get_base_tokenizer
from mycelia.shared.datasets import get_dataloader
from mycelia.shared.expert_manager import ExpertManager

logger = structlog.getLogger(__name__)

tqdm(disable=True, total=0)

@torch.no_grad
def evaluate_model(
    step: int,
    model: nn.Module,
    eval_dataloader,
    device: torch.device,
    max_eval_batches: Optional[int] = 50,
    rank: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run a lightweight eval pass and return scalar metrics.

    Parameters
    ----------
    step : int
        Training step for logging context.
    model : nn.Module
        Fully-assembled model placed on the correct device.
    eval_dataloader :
        Iterable of evaluation batches (dicts of Tensors).
    device : torch.device
        Device to run evaluation on.
    max_eval_batches : Optional[int]
        Optional cap on the number of batches to evaluate.

    Returns
    -------
    Dict[str, float]
        e.g., {"val_loss": 2.345}
    """
    logger.info("evaluate model", step = step)
    model.eval()
    loss_sum: float = 0.0
    aux_loss_sum: float = 0.0
    with torch.inference_mode():
        for batch_step, batch in enumerate(iterable=eval_dataloader):
            device_batch = {}
            for key in batch.keys():
                device_batch[key] = batch[key].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**device_batch)

            loss_sum += float(outputs.loss.detach().item())
            aux_loss_sum += float(outputs.aux_loss.detach().item()) if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None else 0

            del outputs, device_batch

            if max_eval_batches is not None and batch_step >= max_eval_batches:
                break

    return {"val_loss": (loss_sum - aux_loss_sum) / batch_step, "val_aux_loss": aux_loss_sum / batch_step}

def get_validator_miner_assignment(config: BaseConfig, subtensor: bittensor.Subtensor):
    status = get_status(config, subtensor)

    validator_seeds: Dict[str, int] = {
        hotkey: entry["status"].miner_seed
        for hotkey, entry in status.items()
        if entry.get("status")
        and getattr(entry["status"], "expert_group", None) == config.moe.my_expert_group_id
        and getattr(entry["status"], "miner_seed", None) is not None
    }

    miners: List[str] = [
        hk for hk, e in status.items()
        if isinstance(e.get("status"), MinerStatus)
        and getattr(e["status"], "expert_group", None) == config.moe.my_expert_group_id
    ]

    return assign_miners_to_validators(validator_seeds, miners) # type: ignore

def h256_int(*parts: Any) -> int:
    """Deterministic 256-bit hash -> int."""
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode("utf-8"))
        m.update(b"\x00")  # separator
    return int.from_bytes(m.digest(), "big")

def assign_miners_to_validators(
    validators: Dict[str, Any],  # {validator_id: seed}
    miners: List[str],
) -> Dict[str, List[str]]:
    n_v = len(validators)
    n_m = len(miners)
    if n_v == 0:
        raise ValueError("No validators provided")

    # --- 0) Combined seed (hash of all validator seeds)
    combined_seed_str = "".join(str(validators[v]) for v in sorted(validators.keys()))
    combined_seed = hashlib.sha256(combined_seed_str.encode()).hexdigest()

    # --- 1) Balanced capacities
    base = n_m // n_v
    rem = n_m % n_v
    v_ids = list(validators.keys())

    ranked_for_bonus = sorted(
        v_ids,
        key=lambda vid: h256_int("cap_bonus", validators[vid], combined_seed),
        reverse=True,
    )
    capacities = {vid: base for vid in v_ids}
    for vid in ranked_for_bonus[:rem]:
        capacities[vid] += 1

    # --- 2) Deterministic miner order seeded by combined validator seed
    miners_sorted = sorted(miners, key=lambda mid: h256_int("miner_order", mid, combined_seed))

    # --- 3) Preference per miner (based on validator seed + combined seed)
    def validator_prefs(mid: str) -> List[str]:
        return sorted(
            v_ids,
            key=lambda vid: h256_int("preference", mid, validators[vid], combined_seed),
            reverse=True,
        )

    # --- 4) Assign miners evenly, respecting capacities
    assignment: Dict[str, List[str]] = {vid: [] for vid in v_ids}
    for mid in miners_sorted:
        prefs = validator_prefs(mid)
        for vid in prefs:
            if capacities[vid] > 0:
                assignment[vid].append(mid)
                capacities[vid] -= 1
                break
        else:
            # Should never happen if capacities sum == len(miners)
            assignment[prefs[-1]].append(mid)

    return assignment