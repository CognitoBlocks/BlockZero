from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import MinerConfig, ValidatorConfig
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.helper import *
from mycelia.shared.modeling.custom_qwen3_next import (
    CustomQwen3NextForCausalLM,
    get_moe_model_config,
)

logger = structlog.get_logger(__name__)


def _replace_experts_for_training(model, moe_config, group_ids, expert_manager):
    """
    Replace quantized expert layers with trainable fp16 versions.
    
    This allows training experts while keeping the base model in 4-bit quantized form.
    Only the assigned experts are created (memory efficient).
    """
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextMLP
    
    # Get layer assignments for this group
    if group_ids is None:
        group_ids = list(expert_manager.expert_group_assignment.keys())
    
    replaced_count = 0
    
    # Iterate through decoder layers
    for layer_idx, layer in enumerate(model.model.layers):
        # Check if this layer has an MoE block
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            moe_block = layer.mlp
            
            # Determine which experts this miner should train
            allowed_expert_ids = []
            for gid in group_ids:
                if layer_idx in expert_manager.expert_group_assignment.get(gid, {}):
                    allowed_expert_ids.extend([
                        eid for eid, _ in expert_manager.expert_group_assignment[gid][layer_idx]
                    ])
            
            if not allowed_expert_ids:
                continue
                
            # Create new fp16 trainable experts for assigned IDs
            new_experts = nn.ModuleDict()
            for expert_id in allowed_expert_ids:
                # Create fresh fp16 expert
                new_expert = Qwen3NextMLP(
                    moe_config, 
                    intermediate_size=moe_config.moe_intermediate_size
                ).to(dtype=torch.float16, device=model.device)
                
                # Copy weights from quantized version if available
                old_expert_key = str(expert_id)
                if old_expert_key in moe_block.experts:
                    old_expert = moe_block.experts[old_expert_key]
                    # Dequantize and copy weights
                    with torch.no_grad():
                        for (name, new_param), (_, old_param) in zip(
                            new_expert.named_parameters(), 
                            old_expert.named_parameters()
                        ):
                            if hasattr(old_param, 'dequantize'):
                                new_param.copy_(old_param.dequantize().to(torch.float16))
                            else:
                                new_param.copy_(old_param.to(torch.float16))
                
                new_expert.requires_grad_(True)
                new_experts[old_expert_key] = new_expert
                replaced_count += 1
            
            # Replace the experts dict
            moe_block.experts = new_experts
    
    # Freeze all base model parameters
    for name, param in model.named_parameters():
        if 'experts' not in name:
            param.requires_grad_(False)
    
    logger.info(f"Replaced {replaced_count} experts with trainable fp16 versions")
    return model


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def get_base_model(
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    group_ids: list | None = None,
    state_dicts: list = [],
    partial=False,
) -> nn.Module | None:
    """
    Load base model with role-specific optimizations.

    Validators: Load with 4-bit quantization + Unsloth for memory efficiency
    Miners: Load standard model for training
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    topk = config.moe.partial_topk if partial else config.moe.full_topk
    moe_config = get_moe_model_config(config, topk, group_ids, expert_manager)

    is_validator = config.role == "validator"
    use_quantization = get_nested_attr(config, "model.use_quantization", False)
    use_unsloth = get_nested_attr(config, "model.use_unsloth", False) and is_validator

    # === QUANTIZED PATH ===
    if use_quantization:
        is_miner = config.role == "miner"
        logger.info(f"Loading with 4-bit quantization for {'miner (trainable experts)' if is_miner else 'validator'}")

        # Try Unsloth first for validators (fastest, inference-only)
        if use_unsloth and not is_miner:
            try:
                from unsloth import FastLanguageModel

                model, _ = FastLanguageModel.from_pretrained(
                    model_name=config.model.model_path,
                    max_seq_length=moe_config.max_position_embeddings,
                    dtype=torch.float16,
                    load_in_4bit=True,
                    device_map="auto",
                )
                FastLanguageModel.for_inference(model)
                logger.info("✓ Loaded with Unsloth optimizations")
                return model
            except Exception as e:
                logger.warning(f"Unsloth failed, falling back to BitsAndBytes: {e}")

        # BitsAndBytes quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        max_memory = get_nested_attr(config, "model.max_memory", None)
        if max_memory is None:
            max_memory = {0: "46GB", "cpu": "100GB"}

        # Load pretrained model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        
        # For miners: replace expert layers with trainable fp16 versions
        if is_miner:
            logger.info("Replacing experts with trainable fp16 versions...")
            model = _replace_experts_for_training(model, moe_config, group_ids, expert_manager)
            logger.info("✓ Experts replaced - base model frozen (4-bit), experts trainable (fp16)")
        else:
            logger.info("✓ Loaded with BitsAndBytes quantization (inference)")
        
        return model

    # === STANDARD PATH (Miners) ===
    # Check if we should use CPU offloading (for memory-constrained systems)
    use_cpu_offload = get_nested_attr(config, "model.cpu_offload", False)
    
    if use_cpu_offload:
        # Initialize on CPU first to avoid GPU OOM during model creation
        logger.info("Loading with CPU offloading enabled - model will be on CPU")
        with torch.device("cpu"):
            model = CustomQwen3NextForCausalLM(moe_config)
        # Keep on CPU - will be slow but won't OOM
    else:
        model = CustomQwen3NextForCausalLM(moe_config)

    if len(state_dicts) > 0:
        merged_stated_dict, missing = merge_state_dicts_with_priority(state_dicts, model)
        assert len(missing) == 0
        model.load_state_dict(merged_stated_dict, strict=True)

    if model is not None and get_nested_attr(config, "model.torch_compile", False):
        model = torch.compile(model)

    return model


def get_base_tokenizer(config: MinerConfig | ValidatorConfig):
    """
    Load the tokenizer for `config.model.model_path`.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, use_fast=True)
    # tokenizer.pad_token = "</s>"
    return tokenizer


def merge_state_dicts_with_priority(
    state_dicts: list[dict[str, torch.Tensor]],
    model: torch.nn.Module | None = None,
) -> tuple[OrderedDict, list[str] | None]:
    """
    Merge a list of state_dicts where earlier dicts have *higher* priority.
    Unexpected keys (not present in the model) are removed automatically.

    Args:
        state_dicts: list of state dicts, in priority order.
                     state_dicts[0] has highest priority, state_dicts[-1] lowest.
        model: optional model, used to filter out unexpected keys
               and check for missing keys.

    Returns:
        merged_state_dict: OrderedDict with cleaned + merged parameters.
        missing_keys: keys that the model expects but are not in merged
    """
    if not state_dicts:
        raise ValueError("state_dicts must be a non-empty list")

    merged = OrderedDict()

    # Build merged dict: earlier dicts override later ones.
    for sd in reversed(state_dicts):
        for k, v in sd.items():
            if k not in merged:
                merged[k] = v

    # If no model provided, return as is
    if model is None:
        return merged, None

    # Filter out unexpected keys
    model_keys = set(model.state_dict().keys())
    cleaned = OrderedDict((k, v) for k, v in merged.items() if k in model_keys)

    # Compute missing keys
    missing_keys = sorted(model_keys - set(cleaned.keys()))

    return cleaned, missing_keys
