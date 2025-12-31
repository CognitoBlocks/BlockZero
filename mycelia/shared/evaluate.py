from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm
from typing import Optional

from mycelia.shared.app_logging import structlog
from mycelia.shared.expert_metrics import compute_expert_metrics, metrics_to_dict

logger = structlog.getLogger(__name__)

tqdm(disable=True, total=0)


@torch.no_grad
def evaluate_model(
    step: int,
    model: nn.Module,
    eval_dataloader,
    device: torch.device,
    max_eval_batches: int | None = 50,
    rank: int | None = None,
    num_experts: Optional[int] = None,
    collect_expert_metrics: bool = True,
) -> dict[str, float]:
    """
    Run a lightweight eval pass and return scalar metrics including expert-level analysis.

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
    rank : Optional[int]
        Process rank for distributed training.
    num_experts : Optional[int]
        Number of experts in the MoE model. Auto-detected if None.
    collect_expert_metrics : bool
        Whether to collect and compute expert-level metrics (default: True).

    Returns
    -------
    Dict[str, float]
        Comprehensive metrics including val_loss, expert utilization, routing quality,
        load balancing, and composite_score for miner ranking.
    """
    logger.info("evaluate model", step=step, collect_expert_metrics=collect_expert_metrics)
    model.eval()
    loss_sum: float = 0.0
    aux_loss_sum: float = 0.0
    router_logits_collection = [] if collect_expert_metrics else None

    # Auto-detect num_experts from model config if not provided
    if num_experts is None and collect_expert_metrics:
        if hasattr(model, 'config') and hasattr(model.config, 'num_experts'):
            num_experts = model.config.num_experts
        else:
            # Default fallback
            num_experts = 8
            logger.warning("Could not auto-detect num_experts, using default", num_experts=num_experts)

    with torch.no_grad():
        for batch_step, batch in enumerate(iterable=eval_dataloader):
            device_batch = {}
            for key in batch.keys():
                device_batch[key] = batch[key].to(device)

            # Autocast only on CUDA
            autocast_enabled = torch.cuda.is_available()
            with torch.amp.autocast("cuda" if autocast_enabled else "cpu", dtype=torch.float16, enabled=autocast_enabled):
                outputs = model(**device_batch, output_router_logits=collect_expert_metrics)

                if not torch.isnan(outputs.loss):
                    loss_sum += float(outputs.loss.item())

                aux_loss_sum += (
                    float(outputs.aux_loss.item())
                    if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None
                    else 0
                )

                # Collect router logits for expert metrics
                if collect_expert_metrics and hasattr(outputs, "router_logits") and outputs.router_logits is not None:
                    # router_logits can be a tuple of tensors (one per MoE layer)
                    if isinstance(outputs.router_logits, tuple):
                        for logits in outputs.router_logits:
                            if logits is not None:
                                router_logits_collection.append(logits.cpu())
                    else:
                        router_logits_collection.append(outputs.router_logits.cpu())

                del outputs

            del device_batch

            if max_eval_batches is not None and batch_step >= max_eval_batches:
                break

        logger.info("eval loss", loss_sum=loss_sum, aux_loss_sum=aux_loss_sum, batch_step=batch_step)

    # Compute basic metrics
    avg_val_loss = (loss_sum - aux_loss_sum) / (batch_step + 1) if batch_step >= 0 else 0.0
    avg_aux_loss = aux_loss_sum / (batch_step + 1) if batch_step >= 0 else 0.0

    # Compute expert metrics if enabled
    if collect_expert_metrics and num_experts is not None:
        expert_metrics = compute_expert_metrics(
            val_loss=avg_val_loss,
            val_aux_loss=avg_aux_loss,
            router_logits_list=router_logits_collection if router_logits_collection else None,
            num_experts=num_experts,
        )
        result = metrics_to_dict(expert_metrics)
        logger.info("expert metrics computed",
                    composite_score=result["composite_score"],
                    diversity=result["expert_diversity_score"],
                    active_ratio=result["experts_active_ratio"])
    else:
        # Fallback to basic metrics only
        result = {
            "val_loss": avg_val_loss,
            "val_aux_loss": avg_aux_loss,
        }

    return result
