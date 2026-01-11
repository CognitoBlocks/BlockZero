from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import nn

from mycelia.shared.app_logging import structlog
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.evaluate import evaluate_model
from mycelia.shared.helper import get_model_hash

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MinerEvalJob:
    uid: int
    hotkey: str
    model_path: str
    step: int


# -------------------------- Pipeline Config -----------------------------------
MAX_CONCURRENT_DOWNLOADS = 4
EVAL_WORKERS = 1
DOWNLOAD_TIMEOUT_SEC = 60
EVAL_MAX_BATCHES = 50
# ------------------------------------------------------------------------------

# def load_model_from_path(path: str, base_model, device: torch.device) -> nn.Module:
#     sd = torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
#     model = copy.deepcopy(base_model)
#     model.load_state_dict(sd, strict=False)
#     return model.to(device)


def load_model_from_path(path: str, base_model: nn.Module, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["model_state_dict"]  # checkpoint state_dict

    model = copy.deepcopy(base_model)

    # Keys in each state_dict (before loading)
    base_sd = base_model.state_dict()
    base_keys = set(base_sd.keys())
    ckpt_keys = set(sd.keys())

    # 1) Params that are the same across both dicts (intersection).
    #    (Optional: filter to ones with matching shapes too.)
    common_keys = base_keys & ckpt_keys
    common_same_shape = {k for k in common_keys if base_sd[k].shape == sd[k].shape}

    print(f"[load_model] common keys: {len(common_keys)}")
    print(f"[load_model] common keys (same shape): {len(common_same_shape)}")
    # Print the actual sets (sorted for readability)
    print("[load_model] common keys (same shape):")
    # print(sorted(common_same_shape))

    # 2) Keys containing 'expert' that exist in the checkpoint but NOT in the base model
    expert_not_in_base = {k for k in ckpt_keys - base_keys if "expert" in k}

    print(f"[load_model] 'expert' keys in checkpoint but not in base_model: {len(expert_not_in_base)}")
    # print("[load_model] expert_not_in_base:")
    # print(sorted(expert_not_in_base))

    # 3) "expert" keys in base_model but NOT in checkpoint/common_keys
    expert_in_base_not_common = {k for k in (base_keys - common_keys) if "expert" in k}
    print(
        f"[load_model] 'expert' keys in base_model but not in checkpoint/common_keys: {len(expert_in_base_not_common)}"
    )
    # print("[load_model] expert_in_base_not_common:")
    # print(sorted(expert_in_base_not_common))

    # Load weights (strict=False so missing/unexpected are allowed)
    incompatible = model.load_state_dict(sd, strict=False)

    # # Extra helpful debug (optional)
    # if incompatible.missing_keys:
    #     print(f"[load_model] missing keys (first 50): {incompatible.missing_keys[:50]}")
    # if incompatible.unexpected_keys:
    #     print(f"[load_model] unexpected keys (first 50): {incompatible.unexpected_keys[:50]}")

    return model.to(device)


async def evaluator_worker(
    name: str,
    config,
    jobs_q: asyncio.Queue[MinerEvalJob],
    aggregator: MinerScoreAggregator,
    device: torch.device,
    base_model: nn.Module,
    tokenizer,
    combinded_seed: str,
    max_eval_batches: int = EVAL_MAX_BATCHES,
    rank: int | None = None,
):
    import gc

    while True:
        job = await jobs_q.get()
        if job is None:  # type: ignore
            jobs_q.task_done()
            logger.debug(f"{name}: shutdown signal received.")
            break

        try:
            snap = torch.cuda.memory_snapshot()
            # snap is a big Python structure; you can print summary-ish info:
            logger.info("segments:", len(snap))

            # Clear memory before loading
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"{name}: Evaluating hotkey={job.hotkey}")

            # Load model (potentially blocking) in a thread

            model = await asyncio.to_thread(load_model_from_path, job.model_path, base_model, device)

            logger.info(
                f"{name}: Evaluating",
                hotkey=job.hotkey,
                base_hash=get_model_hash(base_model.state_dict()),
                merged_hash=get_model_hash(model.state_dict()),
            )

            eval_dataloader = await asyncio.to_thread(
                get_dataloader, config=config, tokenizer=tokenizer, seed=combinded_seed, rank=0, world_size=10
            )

            metrics = await asyncio.to_thread(
                evaluate_model, job.step, model, eval_dataloader, device, max_eval_batches, rank
            )

            # choose a primary score (here 'accuracy'); adjust if your evaluate_model returns other keys
            score = float(metrics.get("val_loss", 100))
            aggregator.add_score(job.uid, job.hotkey, score)
            logger.info(f"{name}: uid={job.uid} score={score:.4f}")

            # Explicit cleanup
            del eval_dataloader, model, metrics
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.memory._dump_snapshot("cuda_snapshot.pickle")

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"{name}: OOM for uid={job.uid}")
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.exception(f"{name}: Evaluation failed for uid={job.uid}: {e}")
        finally:
            jobs_q.task_done()


async def run_evaluation(
    config, step, device, miners, score_aggregator, base_model: nn.Module, tokenizer, combinded_seed
):
    # Device & dataloader (MOCK). Replace eval_dataloader with a real one.
    miners_q: asyncio.Queue[MinerEvalJob] = asyncio.Queue()

    # Enqueue miners
    for m in miners:
        await miners_q.put(m)

    # Spin up evaluator workers
    eval_workers = [
        asyncio.create_task(
            evaluator_worker(
                f"evaluator-{i+1}", config, miners_q, score_aggregator, device, base_model, tokenizer, combinded_seed
            )
        )
        for i in range(EVAL_WORKERS)
    ]

    # Wait for all miners to be processed
    await miners_q.join()

    # Signal evaluator workers to stop
    for _ in eval_workers:
        await miners_q.put(None)

    await asyncio.gather(*eval_workers)
