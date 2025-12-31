"""
Expert-specific metrics for MoE model evaluation and miner scoring.

This module provides comprehensive metrics for evaluating Mixture of Experts (MoE) models,
including expert utilization, routing quality, load balancing effectiveness, and composite scoring.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExpertMetrics:
    """Container for expert-level metrics."""

    # Basic metrics
    val_loss: float
    val_aux_loss: float

    # Expert utilization metrics
    expert_usage_counts: Dict[int, int]  # expert_id -> count
    expert_usage_percentages: Dict[int, float]  # expert_id -> percentage
    expert_diversity_score: float  # How evenly distributed across experts (0-1)

    # Routing quality metrics
    routing_entropy: float  # Higher = more diverse routing
    routing_confidence: float  # Average max probability in routing decisions
    load_balance_loss: float  # Auxiliary load balancing loss

    # Performance metrics
    experts_active_ratio: float  # Fraction of experts actually used
    avg_tokens_per_expert: float  # Average load per expert

    # Composite score
    composite_score: float


def compute_expert_utilization(
    router_logits_list: List[torch.Tensor],
    num_experts: int,
) -> Tuple[Dict[int, int], Dict[int, float], float]:
    """
    Compute expert utilization metrics from router logits.

    Args:
        router_logits_list: List of router logits tensors from each MoE layer
        num_experts: Total number of experts in the model

    Returns:
        Tuple of (usage_counts, usage_percentages, diversity_score)
    """
    expert_counts = {i: 0 for i in range(num_experts)}
    total_tokens = 0

    for router_logits in router_logits_list:
        if router_logits is None:
            continue

        # router_logits shape: [batch_size, seq_len, num_experts]
        # Get the expert with highest logit for each token
        expert_indices = torch.argmax(router_logits, dim=-1)  # [batch_size, seq_len]

        # Count expert usage
        for expert_id in range(num_experts):
            count = (expert_indices == expert_id).sum().item()
            expert_counts[expert_id] += count
            total_tokens += count

    # Compute percentages
    expert_percentages = {}
    if total_tokens > 0:
        expert_percentages = {
            expert_id: count / total_tokens
            for expert_id, count in expert_counts.items()
        }
    else:
        expert_percentages = {expert_id: 0.0 for expert_id in range(num_experts)}

    # Compute diversity score (using normalized entropy)
    # Perfect diversity = 1.0, all tokens to one expert = 0.0
    percentages = np.array(list(expert_percentages.values()))
    percentages = percentages[percentages > 0]  # Remove zeros for log

    if len(percentages) > 0:
        entropy = -np.sum(percentages * np.log(percentages))
        max_entropy = np.log(num_experts)  # Maximum possible entropy
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        diversity_score = 0.0

    return expert_counts, expert_percentages, float(diversity_score)


def compute_routing_quality(
    router_logits_list: List[torch.Tensor],
) -> Tuple[float, float]:
    """
    Compute routing quality metrics.

    Args:
        router_logits_list: List of router logits tensors from each MoE layer

    Returns:
        Tuple of (routing_entropy, routing_confidence)
    """
    all_entropies = []
    all_confidences = []

    for router_logits in router_logits_list:
        if router_logits is None:
            continue

        # Compute softmax probabilities
        probs = torch.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]

        # Routing entropy (per token)
        token_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [batch_size, seq_len]
        all_entropies.append(token_entropy.mean().item())

        # Routing confidence (max probability per token)
        max_probs = probs.max(dim=-1)[0]  # [batch_size, seq_len]
        all_confidences.append(max_probs.mean().item())

    routing_entropy = float(np.mean(all_entropies)) if all_entropies else 0.0
    routing_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0

    return routing_entropy, routing_confidence


def compute_load_balance_metrics(
    expert_usage_counts: Dict[int, int],
    num_experts: int,
) -> float:
    """
    Compute load balancing effectiveness.

    Args:
        expert_usage_counts: Count of tokens assigned to each expert
        num_experts: Total number of experts

    Returns:
        Load balance score (0-1, higher is better)
    """
    counts = np.array([expert_usage_counts.get(i, 0) for i in range(num_experts)])
    total = counts.sum()

    if total == 0:
        return 0.0

    # Perfect balance would be equal distribution
    expected_per_expert = total / num_experts

    # Compute coefficient of variation (lower is better)
    mean = counts.mean()
    std = counts.std()

    if mean > 0:
        cv = std / mean
        # Convert to 0-1 score where 1 is perfect balance
        # CV of 0 = perfect balance = score 1.0
        # CV of 1 = high variance = score ~0.5
        balance_score = 1.0 / (1.0 + cv)
    else:
        balance_score = 0.0

    return float(balance_score)


def compute_composite_score(
    val_loss: float,
    diversity_score: float,
    load_balance_score: float,
    routing_confidence: float,
    experts_active_ratio: float,
    loss_weight: float = 0.5,
    diversity_weight: float = 0.2,
    balance_weight: float = 0.15,
    confidence_weight: float = 0.1,
    active_weight: float = 0.05,
) -> float:
    """
    Compute composite score combining multiple metrics.

    Lower is better (like loss). This score can be used to rank miners.

    Args:
        val_loss: Validation loss (lower is better)
        diversity_score: Expert diversity (0-1, higher is better)
        load_balance_score: Load balancing quality (0-1, higher is better)
        routing_confidence: Routing confidence (0-1, higher is better)
        experts_active_ratio: Fraction of experts used (0-1, higher is better)
        loss_weight: Weight for validation loss
        diversity_weight: Weight for diversity score
        balance_weight: Weight for load balance score
        confidence_weight: Weight for routing confidence
        active_weight: Weight for active experts ratio

    Returns:
        Composite score (lower is better)
    """
    # Normalize val_loss to 0-1 range (assuming typical range 0-10)
    # Use sigmoid-like transformation
    normalized_loss = val_loss / (val_loss + 1.0)

    # Invert "higher is better" metrics so lower composite score is better
    diversity_penalty = 1.0 - diversity_score
    balance_penalty = 1.0 - load_balance_score
    confidence_penalty = 1.0 - routing_confidence
    active_penalty = 1.0 - experts_active_ratio

    composite = (
        loss_weight * normalized_loss +
        diversity_weight * diversity_penalty +
        balance_weight * balance_penalty +
        confidence_weight * confidence_penalty +
        active_weight * active_penalty
    )

    return float(composite)


def compute_expert_metrics(
    val_loss: float,
    val_aux_loss: float,
    router_logits_list: Optional[List[torch.Tensor]],
    num_experts: int,
) -> ExpertMetrics:
    """
    Compute comprehensive expert metrics from evaluation results.

    Args:
        val_loss: Validation loss
        val_aux_loss: Auxiliary load balancing loss
        router_logits_list: List of router logits tensors from MoE layers (can be None)
        num_experts: Total number of experts in the model

    Returns:
        ExpertMetrics object with all computed metrics
    """
    # Handle case where router logits are not available
    if router_logits_list is None or len(router_logits_list) == 0:
        return ExpertMetrics(
            val_loss=val_loss,
            val_aux_loss=val_aux_loss,
            expert_usage_counts={i: 0 for i in range(num_experts)},
            expert_usage_percentages={i: 0.0 for i in range(num_experts)},
            expert_diversity_score=0.0,
            routing_entropy=0.0,
            routing_confidence=0.0,
            load_balance_loss=val_aux_loss,
            experts_active_ratio=0.0,
            avg_tokens_per_expert=0.0,
            composite_score=val_loss,  # Fall back to just val_loss
        )

    # Compute expert utilization
    usage_counts, usage_percentages, diversity_score = compute_expert_utilization(
        router_logits_list, num_experts
    )

    # Compute routing quality
    routing_entropy, routing_confidence = compute_routing_quality(router_logits_list)

    # Compute load balancing
    load_balance_score = compute_load_balance_metrics(usage_counts, num_experts)

    # Compute derived metrics
    active_experts = sum(1 for count in usage_counts.values() if count > 0)
    experts_active_ratio = active_experts / num_experts if num_experts > 0 else 0.0

    total_tokens = sum(usage_counts.values())
    avg_tokens_per_expert = total_tokens / num_experts if num_experts > 0 else 0.0

    # Compute composite score
    composite = compute_composite_score(
        val_loss=val_loss,
        diversity_score=diversity_score,
        load_balance_score=load_balance_score,
        routing_confidence=routing_confidence,
        experts_active_ratio=experts_active_ratio,
    )

    return ExpertMetrics(
        val_loss=val_loss,
        val_aux_loss=val_aux_loss,
        expert_usage_counts=usage_counts,
        expert_usage_percentages=usage_percentages,
        expert_diversity_score=diversity_score,
        routing_entropy=routing_entropy,
        routing_confidence=routing_confidence,
        load_balance_loss=val_aux_loss,
        experts_active_ratio=experts_active_ratio,
        avg_tokens_per_expert=avg_tokens_per_expert,
        composite_score=composite,
    )


def metrics_to_dict(metrics: ExpertMetrics) -> Dict:
    """
    Convert ExpertMetrics to a dictionary for logging/storage.

    Args:
        metrics: ExpertMetrics object

    Returns:
        Dictionary representation
    """
    return {
        "val_loss": metrics.val_loss,
        "val_aux_loss": metrics.val_aux_loss,
        "expert_diversity_score": metrics.expert_diversity_score,
        "routing_entropy": metrics.routing_entropy,
        "routing_confidence": metrics.routing_confidence,
        "load_balance_loss": metrics.load_balance_loss,
        "experts_active_ratio": metrics.experts_active_ratio,
        "avg_tokens_per_expert": metrics.avg_tokens_per_expert,
        "composite_score": metrics.composite_score,
        "expert_usage_counts": metrics.expert_usage_counts,
        "expert_usage_percentages": metrics.expert_usage_percentages,
    }
