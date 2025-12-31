"""
Integration Module for Expert-Specific Evaluation

Connects expert-specific metrics to the validation pipeline.
"""
from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from mycelia.shared.app_logging import structlog
from mycelia.shared.evaluate import evaluate_model
from mycelia.shared.expert_specific_metrics import (
    get_expert_metrics_computer,
    ExpertGroup,
)

logger = structlog.get_logger(__name__)


def evaluate_model_with_expert_metrics(
    step: int,
    model: nn.Module,
    eval_dataloader,
    device: torch.device,
    expert_group_id: int,
    max_eval_batches: int | None = 50,
    rank: int | None = None,
    collect_predictions: bool = True,
) -> dict[str, float]:
    """
    Evaluate model with expert group-specific metrics.

    Args:
        step: Training step
        model: Model to evaluate
        eval_dataloader: Validation data
        device: Computation device
        expert_group_id: Expert group ID (0=Math, 1=Agentic, 2=Planning)
        max_eval_batches: Max batches to evaluate
        rank: Process rank
        collect_predictions: Whether to collect and analyze predictions

    Returns:
        Dictionary of metrics including domain-specific ones
    """
    # Get base metrics from standard evaluation
    base_metrics = evaluate_model(
        step=step,
        model=model,
        eval_dataloader=eval_dataloader,
        device=device,
        max_eval_batches=max_eval_batches,
        rank=rank,
        collect_expert_metrics=True,
    )

    # Add expert group identifier
    base_metrics["expert_group_id"] = expert_group_id

    # If predictions collection is disabled, return base metrics
    if not collect_predictions:
        logger.info("Expert metrics: Using base metrics only (prediction collection disabled)")
        return base_metrics

    # Collect predictions for domain-specific analysis
    try:
        predictions, references = collect_model_predictions(
            model=model,
            eval_dataloader=eval_dataloader,
            device=device,
            max_batches=min(10, max_eval_batches) if max_eval_batches else 10,
        )

        if predictions and references:
            # Compute expert-specific metrics
            metrics_computer = get_expert_metrics_computer(expert_group_id)
            expert_metrics = metrics_computer.compute_metrics(
                predictions=predictions,
                references=references,
                base_metrics=base_metrics,
            )

            logger.info(
                "Expert-specific metrics computed",
                expert_group=ExpertGroup(expert_group_id).name,
                composite_score=expert_metrics.get("composite_score", base_metrics.get("composite_score", 0.0)),
            )

            return expert_metrics
        else:
            logger.warning("Failed to collect predictions, using base metrics")
            return base_metrics

    except Exception as e:
        logger.warning(f"Error computing expert-specific metrics: {e}, using base metrics")
        return base_metrics


def collect_model_predictions(
    model: nn.Module,
    eval_dataloader,
    device: torch.device,
    max_batches: int = 10,
    tokenizer=None,
) -> tuple[list[str], list[str]]:
    """
    Collect model predictions and references for analysis.

    Args:
        model: Model to evaluate
        eval_dataloader: Data loader
        device: Computation device
        max_batches: Maximum batches to collect
        tokenizer: Tokenizer for decoding (optional)

    Returns:
        (predictions, references) as lists of strings
    """
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= max_batches:
                break

            # Move batch to device
            device_batch = {k: v.to(device) for k, v in batch.items()}

            # Generate predictions
            try:
                # Use generate if available
                if hasattr(model, 'generate'):
                    input_ids = device_batch.get('input_ids')
                    attention_mask = device_batch.get('attention_mask')

                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=128,
                        do_sample=False,  # Greedy for consistency
                        pad_token_id=model.config.pad_token_id if hasattr(model.config, 'pad_token_id') else 0,
                    )

                    # Decode if tokenizer available
                    if tokenizer is not None:
                        pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        ref_texts = tokenizer.batch_decode(device_batch['input_ids'], skip_special_tokens=True)

                        predictions.extend(pred_texts)
                        references.extend(ref_texts)
                else:
                    # Fallback: just use input as reference
                    logger.warning("Model doesn't have generate(), skipping prediction collection")
                    break

            except Exception as e:
                logger.warning(f"Error collecting predictions: {e}")
                continue

    logger.info(f"Collected {len(predictions)} predictions for expert metric analysis")
    return predictions, references


def get_expert_group_name(expert_group_id: int) -> str:
    """Get human-readable name for expert group."""
    try:
        return ExpertGroup(expert_group_id).name
    except ValueError:
        return f"UNKNOWN_{expert_group_id}"


def get_expert_group_description(expert_group_id: int) -> str:
    """Get description of expert group specialization."""
    descriptions = {
        0: "Mathematical reasoning, theorem proving, numerical computation",
        1: "Tool use, function calling, ReAct reasoning, API interaction",
        2: "Multi-step planning with uncertainty decay modeling",
        3: "General testing/dummy group",
    }
    return descriptions.get(expert_group_id, "Unknown expert group")


# Example usage in validator:
"""
from mycelia.shared.expert_evaluation_integration import evaluate_model_with_expert_metrics

# In evaluation loop:
metrics = evaluate_model_with_expert_metrics(
    step=step,
    model=miner_model,
    eval_dataloader=validation_dataloader,
    device=device,
    expert_group_id=config.task.expert_group_id,  # 0, 1, or 2
    max_eval_batches=50,
)

# Metrics now include domain-specific measurements:
# - Math: numerical_accuracy, equation_validity, etc.
# - Agentic: tool_selection_accuracy, api_call_validity, etc.
# - Planning: uncertainty_calibration, recovery_capability, etc.

logger.info("Evaluation complete",
            expert_group=get_expert_group_name(config.task.expert_group_id),
            composite_score=metrics["composite_score"],
            val_loss=metrics["val_loss"])
"""
