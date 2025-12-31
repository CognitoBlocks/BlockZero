from __future__ import annotations

import copy
import asyncio
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from mycelia.shared.app_logging import structlog
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.evaluate import evaluate_model
from mycelia.shared.validation_integration import get_unpredictable_validation_dataset
from mycelia.shared.expert_evaluation_integration import (
    evaluate_model_with_expert_metrics,
    get_expert_group_name,
)
from mycelia.validator.aggregator import MinerScoreAggregator

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


def load_model_from_path(path: str, base_model, device: torch.device) -> nn.Module:
    sd = torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
    copy.deepcopy(base_model).load_state_dict(sd, strict=False)
    return base_model.to(device)


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
    subtensor=None,  # Add subtensor for unpredictable validation
    use_unpredictable: bool = True,  # Flag to enable/disable unpredictable validation
):
    import gc

    while True:
        job = await jobs_q.get()
        if job is None:  # type: ignore
            jobs_q.task_done()
            logger.debug(f"{name}: shutdown signal received.")
            break

        try:
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"{name}: Evaluating hotkey={job.hotkey}")

            # Load model (potentially blocking) in a thread
            model = await asyncio.to_thread(load_model_from_path, job.model_path, base_model, device)

            # Use unpredictable validation if enabled and subtensor available
            if use_unpredictable and subtensor is not None:
                logger.info(f"{name}: Using unpredictable validation")
                validation_dataset, validation_seed = await asyncio.to_thread(
                    get_unpredictable_validation_dataset,
                    config=config,
                    subtensor=subtensor,
                    tokenizer=tokenizer,
                    samples_per_validation=max_eval_batches * config.task.data.per_device_train_batch_size,
                )

                # Convert to DataLoader
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                eval_dataloader = DataLoader(
                    validation_dataset,
                    batch_size=config.task.data.per_device_train_batch_size,
                    collate_fn=data_collator,
                    shuffle=False,
                )

                logger.info(
                    f"{name}: Unpredictable validation dataset ready",
                    cycle=validation_seed.cycle_number,
                    dataset_size=len(validation_dataset),
                )
            else:
                # Fallback to old method
                logger.info(f"{name}: Using legacy validation (fallback)")
                eval_dataloader = await asyncio.to_thread(
                    get_dataloader, config=config, tokenizer=tokenizer, seed=combinded_seed, rank=0, world_size=10
                )

            with torch.inference_mode():
                # Use expert-specific evaluation for better domain assessment
                expert_group_id = config.task.expert_group_id
                logger.info(
                    f"{name}: Evaluating with expert-specific metrics",
                    expert_group=get_expert_group_name(expert_group_id),
                )

                metrics = await asyncio.to_thread(
                    evaluate_model_with_expert_metrics,
                    step=job.step,
                    model=model,
                    eval_dataloader=eval_dataloader,
                    device=device,
                    expert_group_id=expert_group_id,
                    max_eval_batches=max_eval_batches,
                    rank=rank,
                    collect_predictions=False,  # Disable for speed (optional)
                )

            # Use composite score for ranking (lower is better, like loss)
            # Falls back to val_loss if composite_score is not available
            score = float(metrics.get("composite_score", metrics.get("val_loss", 100)))
            aggregator.add_score(job.uid, job.hotkey, score)

            # Log expert-specific metrics if available
            expert_group_name = get_expert_group_name(expert_group_id)
            logger.info(
                f"{name}: uid={job.uid} score={score:.4f}",
                expert_group=expert_group_name,
                val_loss=metrics.get("val_loss"),
                diversity=metrics.get("expert_diversity_score"),
                active_ratio=metrics.get("experts_active_ratio"),
                composite_score=metrics.get("composite_score"),
            )

            # Explicit cleanup
            del eval_dataloader, model, metrics
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"{name}: OOM for uid={job.uid}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.exception(f"{name}: Evaluation failed for uid={job.uid}: {e}")
        finally:
            jobs_q.task_done()


async def run_evaluation(
    config,
    step,
    device,
    miners,
    score_aggregator,
    base_model: nn.Module,
    tokenizer,
    combinded_seed,
    subtensor=None,  # Add subtensor for unpredictable validation
    use_unpredictable: bool = True,  # Enable unpredictable validation by default
):
    # Device & dataloader (MOCK). Replace eval_dataloader with a real one.
    miners_q: asyncio.Queue[MinerEvalJob] = asyncio.Queue()

    # Enqueue miners
    for m in miners:
        await miners_q.put(m)

    logger.info(
        "Starting evaluation",
        n_miners=len(miners),
        use_unpredictable=use_unpredictable,
        has_subtensor=subtensor is not None,
    )

    # Spin up evaluator workers
    eval_workers = [
        asyncio.create_task(
            evaluator_worker(
                f"evaluator-{i+1}",
                config,
                miners_q,
                score_aggregator,
                device,
                base_model,
                tokenizer,
                combinded_seed,
                subtensor=subtensor,
                use_unpredictable=use_unpredictable,
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
