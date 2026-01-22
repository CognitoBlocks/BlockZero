from __future__ import annotations

from typing import Callable

import bittensor
import torch
from torch import nn

from mycelia.shared.app_logging import structlog
from mycelia.shared.chain import fetch_model_from_chain
from mycelia.shared.checkpoint_helper import load_checkpoint
from mycelia.shared.checkpoints import ModelCheckpoint
from mycelia.shared.checkpoints import select_best_checkpoint
from mycelia.shared.config import MinerConfig, ValidatorConfig
from mycelia.shared.expert_manager import (
    ExpertAssignments,
    ExpertManager,
    get_layer_expert_id,
)
from mycelia.shared.helper import get_nested_attr
from mycelia.shared.modeling.mycelia import get_base_model

logger = structlog.get_logger(__name__)


def replace_experts_for_training(model, moe_config, group_ids, expert_manager):
    """
    Replace quantized expert layers with trainable bfloat16 PartialExperts.

    This allows training experts while keeping the base model in 4-bit quantized form.
    Weights are stored in bfloat16 for better numerical stability with Qwen3-VL models
    (which were pre-trained in bf16). BF16 prevents overflow when dequantizing 4-bit weights.

    The pre-trained model uses Qwen3VLMoeTextExperts (3D tensors with ALL experts).
    We replace it with PartialExperts (compact 3D tensors with only assigned experts).
    """
    from mycelia.shared.modeling.custom_qwen3_vl_moe import PartialExperts

    if group_ids is None:
        group_ids = list(expert_manager.expert_group_assignment.keys())

    replaced_count = 0

    for layer_idx, layer in enumerate(model.language_model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            moe_block = layer.mlp
            old_experts = moe_block.experts

            # Collect allowed expert IDs for this layer
            allowed_expert_ids = []
            for gid in group_ids:
                if layer_idx in expert_manager.expert_group_assignment.get(gid, {}):
                    allowed_expert_ids.extend([
                        eid for eid, _ in expert_manager.expert_group_assignment[gid][layer_idx]
                    ])

            if not allowed_expert_ids:
                continue

            # Create new PartialExperts with only the assigned experts
            new_experts = PartialExperts(
                moe_config.text_config,
                allowed_expert_ids
            ).to(dtype=torch.bfloat16, device=model.device)

            # Copy weights from old experts to new PartialExperts
            with torch.no_grad():
                # Check if old_experts is Qwen3VLMoeTextExperts (has gate_up_proj as 3D tensor)
                if hasattr(old_experts, 'gate_up_proj') and old_experts.gate_up_proj.dim() == 3:
                    # Old experts use 3D tensors: (num_experts, dim, dim)
                    for expert_id in allowed_expert_ids:
                        compact_idx = new_experts.remap_expert_id(expert_id)
                        if compact_idx < 0:
                            continue

                        # Dequantize and copy gate_up_proj
                        old_gate_up = old_experts.gate_up_proj[expert_id]
                        if hasattr(old_gate_up, 'dequantize'):
                            new_experts.gate_up_proj.data[compact_idx] = old_gate_up.dequantize().to(torch.bfloat16)
                        else:
                            new_experts.gate_up_proj.data[compact_idx] = old_gate_up.to(torch.bfloat16)

                        # Dequantize and copy down_proj
                        old_down = old_experts.down_proj[expert_id]
                        if hasattr(old_down, 'dequantize'):
                            new_experts.down_proj.data[compact_idx] = old_down.dequantize().to(torch.bfloat16)
                        else:
                            new_experts.down_proj.data[compact_idx] = old_down.to(torch.bfloat16)

                        replaced_count += 1
                else:
                    # Old experts use ModuleDict (fallback for compatibility)
                    for expert_id in allowed_expert_ids:
                        compact_idx = new_experts.remap_expert_id(expert_id)
                        old_expert_key = str(expert_id)

                        if old_expert_key not in old_experts:
                            continue

                        old_expert = old_experts[old_expert_key]

                        # Copy gate_proj and up_proj into gate_up_proj
                        if hasattr(old_expert, 'gate_proj') and hasattr(old_expert, 'up_proj'):
                            gate_w = old_expert.gate_proj.weight
                            up_w = old_expert.up_proj.weight
                            if hasattr(gate_w, 'dequantize'):
                                gate_w = gate_w.dequantize()
                            if hasattr(up_w, 'dequantize'):
                                up_w = up_w.dequantize()
                            # gate_up_proj shape: (2 * intermediate, hidden) - concatenate gate and up
                            new_experts.gate_up_proj.data[compact_idx] = torch.cat(
                                [gate_w, up_w], dim=0
                            ).to(torch.bfloat16)

                        # Copy down_proj
                        if hasattr(old_expert, 'down_proj'):
                            down_w = old_expert.down_proj.weight
                            if hasattr(down_w, 'dequantize'):
                                down_w = down_w.dequantize()
                            new_experts.down_proj.data[compact_idx] = down_w.to(torch.bfloat16)

                        replaced_count += 1

            new_experts.requires_grad_(True)
            moe_block.experts = new_experts

    # Freeze all non-expert parameters
    for name, param in model.named_parameters():
        if 'experts' not in name:
            param.requires_grad_(False)

    logger.info(f"Replaced {replaced_count} experts with trainable bfloat16 PartialExperts (no GradScaler needed)")
    return model


def grad_hook(name):
    def h(grad):
        if grad is not None and not torch.isfinite(grad).all():
            print("❌ grad NaN/Inf at", name)
            raise RuntimeError(name)
        return grad

    return h


def freeze_parameters(
    model: nn.Module,
    expert_manager: ExpertManager,
    expert_group_id: int,
) -> list[str]:
    """
    Disable gradients for parameters that satisfy `predicate`.

    Args:
        model: torch.nn.Module
        predicate: function (name, parameter) -> bool
                   return True to freeze the parameter

    Returns:
        List of parameter names that were frozen
    """

    for name, param in model.named_parameters():
        layer_id, expert_id = get_layer_expert_id(name)

        if layer_id is not None and expert_id is not None:
            allowed_experts = {
                allowed_expert_id
                for allowed_expert_id, _ in expert_manager.expert_group_assignment[expert_group_id].get(layer_id, [])
            }
            param.requires_grad_(expert_id in allowed_experts)
        else:
            param.requires_grad_(False)

        # if param.requires_grad:
        #     param.register_hook(grad_hook(name))

    return model


def get_model_from_checkpoint(
    rank: int, config: MinerConfig | ValidatorConfig, expert_manager: ExpertManager
) -> tuple[nn.Module, ModelCheckpoint]:
    resume = False
    latest_checkpoint_path = None

    logger.info(
        "Get base model for checkpoint",
        group_ids=[config.task.exp.group_id] if config.role == "miner" else None,
        partial=(config.role == "miner"),
    )
    # get base model
    model = get_base_model(
        config,
        expert_manager=expert_manager,
        group_ids=[config.task.exp.group_id] if config.role == "miner" else None,
        partial=(config.role == "miner"),
    ).to(config.model.device)

    # load from checkpoint
    if get_nested_attr(config, "ckpt.resume_from_ckpt", False):
        latest_checkpoint= select_best_checkpoint(
            primary_dir=config.ckpt.validator_checkpoint_path,
            secondary_dir=config.ckpt.checkpoint_path,
            resume = config.ckpt.resume_from_ckpt,
        )

        if resume and latest_checkpoint.path:
            load_checkpoint(
                config=config,
                checkpoint_path=latest_checkpoint.path,
                model=model,
                rank=rank,
                device=config.model.device,
            )
        else:
            logger.info("Tried to resume from checkpoint, but no checkpoint found.")

    model = model.to(config.model.device)
    model.gradient_checkpointing_enable()
    return model, latest_checkpoint


def load_model(
    rank: int,
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    current_checkpoint: ModelCheckpoint | None = None,
) -> tuple[nn.Module, dict]:
    """
    Main entry point used by miners (and potentially validator itself).
    1) Ask the chain for an active validator endpoint.
    2) If available, ping and fetch current model.
    3) Else, initialize a default model.
    """
    # download new model from chain into file

    if current_checkpoint is None:
        current_checkpoint = select_best_checkpoint(
            primary_dir=config.ckpt.validator_checkpoint_path,
            secondary_dir=config.ckpt.checkpoint_path,
        )

    fetch_model_from_chain(current_model_meta=current_checkpoint, config=config, subtensor=subtensor, wallet=wallet, expert_group_ids=[config.task.exp.group_id])
    return get_model_from_checkpoint(rank=rank, config=config, expert_manager=expert_manager)
