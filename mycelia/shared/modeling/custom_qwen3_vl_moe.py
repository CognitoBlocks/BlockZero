"""
Custom Qwen3-VL-MoE implementation with expert group assignment.

This replaces custom_qwen3_next.py with a vision-capable MoE architecture.
Uses transformers.models.qwen3_vl_moe (NOT qwen3_vl which is non-MoE).
"""

from __future__ import annotations

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoePreTrainedModel,
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeTextMLP,
    Qwen3VLMoeTextDecoderLayer,
    Qwen3VLMoeTextModel,
    Qwen3VLMoeVisionModel,
)

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import MinerConfig
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.helper import *

logger = structlog.get_logger(__name__)


class TopKRouter(nn.Module):
    """
    Top-k expert router with masking for expert group assignment.

    During training, applies a mask to router_logits to force all tokens
    to route to assigned experts only. This ensures 100% utilization
    of trainable experts and proper gradient flow.
    """

    def __init__(self, config, available_experts=None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Store the allowed indices as a buffer (so it moves to GPU automatically)
        if available_experts is not None:
            self.register_buffer("allowed_ids", torch.as_tensor(available_experts).long())
        else:
            self.allowed_ids = None

    def _mask_routing_weights(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Zero-out routing weights for experts not present in this group.
        (Legacy method, kept for backward compatibility)
        """
        if self.available_experts is None:
            return x

        mask_1d = torch.zeros(x.size(dim), dtype=torch.bool, device=x.device)
        mask_1d[self.available_experts.to(x.device)] = True

        # Broadcast mask across all other dims
        shape = [1] * x.ndim
        shape[dim] = x.size(dim)
        mask = mask_1d.view(shape).to(dtype=x.dtype)

        mask_bool = mask.to(torch.bool)  # True = keep
        fill = x.min() - 2  # scalar, computed BEFORE masking
        x_masked = x.masked_fill(~mask_bool, fill)
        return x_masked

    def forward(self, hidden_states):
        # 1. Raw logits from the gate weight
        router_logits = self.weight(hidden_states)

        # 2. HARD ESFT MASKING
        if self.training and hasattr(self, 'allowed_ids') and self.allowed_ids is not None:
            # We create a new tensor to avoid in-place modification issues
            masked_logits = torch.full_like(router_logits, -1e4)  # Use a very large negative

            # Scatter 0.0 only to your specific expert indices
            # If your ids are [5, 6, 7, 8, 9, 10], ONLY these will have non-infinite scores
            masked_logits.scatter_(
                1,
                self.allowed_ids.unsqueeze(0).expand(router_logits.size(0), -1),
                router_logits.gather(1, self.allowed_ids.unsqueeze(0).expand(router_logits.size(0), -1))
            )

            router_logits = masked_logits

        # 3. Softmax and Top-K
        # Because every other expert is -10000, Top-K is FORCED to pick your assigned experts
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        # Normalization
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        return router_logits, routing_weights.to(hidden_states.dtype), selected_experts


@torch._dynamo.disable
def _compute_overlap(expert_hit, available_experts):
    expert_hit_set = set(expert_hit.detach().cpu().flatten().tolist())
    available_experts_set = set(available_experts.tolist())
    return torch.tensor(sorted(expert_hit_set.intersection(available_experts_set))).view(-1, 1)


class SparseMoeBlock(Qwen3VLMoeTextSparseMoeBlock):
    """
    Custom sparse MoE block with expert group assignment.

    Replaces the standard Qwen3VLMoeTextSparseMoeBlock with our custom router
    that supports expert group assignment for distributed training.
    """

    def __init__(
        self,
        config,
        layer_id: int,
    ):
        super().__init__(config)

        # Determine which experts are allowed in this group
        if config.expert_group_assignment is not None:
            if config.group_ids is None:
                group_ids = config.expert_group_assignment.keys()
            else:
                group_ids = config.group_ids

            allowed_expert_id = []
            for group_id in group_ids:
                allowed_expert_id += [
                    my_expert_id for my_expert_id, org_expert_id in config.expert_group_assignment[group_id][layer_id]
                ]
        else:
            allowed_expert_id = list(range(config.num_experts))

        self.available_experts = torch.as_tensor([int(k) for k in allowed_expert_id])

        # DEBUG: Verify expert alignment for first layer
        if layer_id == 0:
            logger.info(
                f"Layer {layer_id} expert alignment",
                ids=self.available_experts.tolist(),
                group_ids=list(group_ids) if config.expert_group_assignment is not None else None,
            )

        # Replace standard router with our TopKRouter
        self.gate = TopKRouter(config, self.available_experts)

        # Only create experts for this group (memory efficiency)
        # Ensure the keys are identical to available_experts (string versions)
        self.experts = nn.ModuleDict(
            {str(i): Qwen3VLMoeTextMLP(config, intermediate_size=config.moe_intermediate_size) for i in allowed_expert_id}
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, routing_weights, selected_experts = self.gate(hidden_states)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        qualified_expert_set = _compute_overlap(expert_hit, self.available_experts)

        for expert_idx in qualified_expert_set:
            expert_layer = self.experts[str(expert_idx.item())]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits


class DecoderLayer(Qwen3VLMoeTextDecoderLayer):
    """
    Custom decoder layer using our SparseMoeBlock.

    Replaces MoE layers with our custom implementation while keeping
    attention layers unchanged.
    """

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        layer_idx: int,
    ):
        super().__init__(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = SparseMoeBlock(
                config,
                layer_id=layer_idx,
            )


class CustomQwen3VLMoeTextModel(Qwen3VLMoeTextModel):
    """
    Custom text model using our decoder layers.

    Only the language model layers are customized - vision encoder remains standard.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])


class CustomQwen3VLMoeForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    """
    Custom Qwen3-VL-MoE model with expert group assignment.

    This is the main model class that replaces CustomQwen3NextForCausalLM.
    Key differences:
    - Uses ForConditionalGeneration instead of ForCausalLM
    - Has both visual and language_model components
    - Supports multimodal (image+text) inputs
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace language model with our custom version
        self.language_model = CustomQwen3VLMoeTextModel(config.text_config)
        # Vision model stays standard (no need for expert routing in vision encoder)


def get_moe_model_config(
    config: MinerConfig, topk: int, group_ids: list | None, expert_manager: ExpertManager
) -> PretrainedConfig:
    """
    Create MoE configuration for Qwen3-VL-MoE model.

    Args:
        config: Miner configuration with model settings
        topk: Number of experts to activate per token
        group_ids: List of expert group IDs to load (e.g., ["math", "vision"])
        expert_manager: Manager handling expert group assignments

    Returns:
        Qwen3VLMoeConfig configured for distributed MoE training

    Note:
        This replaces the custom_qwen3_next.get_moe_model_config function.
        Main difference: operates on config.text_config instead of config directly.
    """
    # Get the base config from qwen3-vl-moe model
    base_config = AutoConfig.from_pretrained(config.model.model_path)

    # The language model config is nested under text_config
    text_config = base_config.text_config

    # Full/partial dependent configuration
    text_config.num_experts_per_tok = int(topk)
    text_config.group_ids = group_ids  # in list, cause you may load a partial model that contains multiple group id

    # Merge our subnet config to the base config
    text_config.n_group = config.moe.num_worker_groups
    text_config.max_position_embeddings = config.task.data.sequence_length
    text_config.num_experts = (
        expert_manager.num_experts
    )  # this stays the same regardless of full/partial cause we keep the same router size either case
    # Enable router logits output and auxiliary loss for load balancing
    text_config.output_router_logits = get_nested_attr(config, "moe.aux_load_balance", True)
    text_config.router_aux_loss_coef = get_nested_attr(config, "moe.router_aux_loss_coef", 1.0)
    text_config.expert_group_assignment = expert_manager.expert_group_assignment

    # Update the base config with modified text_config
    base_config.text_config = text_config

    return base_config
