from __future__ import annotations

from typing import Callable

import bittensor
import torch
from torch import nn

from mycelia.shared.app_logging import structlog
from mycelia.shared.chain import fetch_model_from_chain
from mycelia.shared.checkpoint import ModelMeta, load_checkpoint, start_model_from
from mycelia.shared.config import MinerConfig, ValidatorConfig
from mycelia.shared.expert_manager import (
    ExpertAssignments,
    ExpertManager,
    get_layer_expert_id,
)
from mycelia.shared.helper import get_nested_attr
from mycelia.shared.modeling.mycelia import get_base_model

logger = structlog.get_logger(__name__)


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
    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        layer_id, expert_id = get_layer_expert_id(name)

        if layer_id is not None and expert_id is not None:
            allowed_experts = {
                allowed_expert_id
                for allowed_expert_id, _ in expert_manager.expert_group_assignment[expert_group_id].get(layer_id, [])
            }
            should_train = expert_id in allowed_experts
            param.requires_grad_(should_train)
            if should_train:
                trainable_count += 1
            else:
                frozen_count += 1
        else:
            param.requires_grad_(False)
            frozen_count += 1

    logger.info(f"freeze_parameters: {trainable_count} trainable, {frozen_count} frozen for group {expert_group_id}")
    
    if trainable_count == 0:
        logger.warning("WARNING: No trainable parameters! Check expert_group_assignment matches model layers.")
    
    return model


def get_model_from_checkpoint(
    rank: int, config: MinerConfig | ValidatorConfig, expert_manager: ExpertManager
) -> tuple[nn.Module, ModelMeta]:
    resume = False
    latest_checkpoint_path = None

    logger.info(
        "Get base model for checkpoint",
        group_ids=[config.task.expert_group_id] if config.role == "miner" else None,
        partial=(config.role == "miner"),
    )
    # get base model
    model = get_base_model(
        config,
        expert_manager=expert_manager,
        group_ids=[config.task.expert_group_id] if config.role == "miner" else None,
        partial=(config.role == "miner"),
    )
    
    # Handle device - auto-detect best available
    # Skip .to() for quantized models (device_map handles placement)
    use_quantization = get_nested_attr(config, "model.use_quantization", False)
    
    device = config.model.device
    if device == "auto" or (device == "cuda" and not torch.cuda.is_available()):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Auto-detected device: {device}")
    
    if not use_quantization:
        model = model.to(device)

    # load from checkpoint
    if get_nested_attr(config, "ckpt.resume_from_ckpt", False):
        resume, model_version, latest_checkpoint_path = start_model_from(
            rank,
            config,
            primary_ckpt_path=config.ckpt.validator_checkpoint_path,
            secondary_ckpt_path=config.ckpt.checkpoint_path,
        )

        if resume and latest_checkpoint_path:
            load_checkpoint(
                config=config,
                checkpoint_path=latest_checkpoint_path,
                model=model,
                rank=rank,
                device=device,
            )
        else:
            logger.info("Tried to resume from checkpoint, but no checkpoint found.")

    if not use_quantization:
        model = model.to(device)
    
    # Only enable gradient checkpointing on CUDA (memory optimization)
    # On MPS/CPU, it can cause "no grad" issues
    if torch.cuda.is_available() and not use_quantization:
        model.gradient_checkpointing_enable()
    return model, model_version


def load_model(
    rank: int,
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    current_model_meta: ModelMeta | None = None,
) -> tuple[nn.Module, dict]:
    """
    Main entry point used by miners (and potentially validator itself).
    1) Ask the chain for an active validator endpoint.
    2) If available, ping and fetch current model.
    3) Else, initialize a default model.
    """
    # download new model from chain into file

    if current_model_meta is None:
        _, current_model_meta, _ = start_model_from(
            rank,
            config,
            primary_ckpt_path=config.ckpt.validator_checkpoint_path,
            secondary_ckpt_path=config.ckpt.checkpoint_path,
        )

    fetch_model_from_chain(current_model_meta=current_model_meta, config=config, subtensor=subtensor, wallet=wallet)
    return get_model_from_checkpoint(rank=rank, config=config, expert_manager=expert_manager)
