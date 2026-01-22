from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import MinerConfig, ValidatorConfig
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.helper import *
from mycelia.shared.modeling.custom_qwen3_vl_moe import (
    CustomQwen3VLMoeForConditionalGeneration,
    get_moe_model_config,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------
# Model Loading Functions (split for clarity and maintainability)
# ---------------------------------------------------------------------

def _load_model_with_unsloth(config, moe_config) -> nn.Module | None:
    """
    Load model with Unsloth optimizations (validators only, inference).

    Unsloth provides fastest inference for quantized models but doesn't
    support training. Returns None if Unsloth fails to load.
    """
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
        logger.info("Loaded with Unsloth optimizations")
        return model
    except Exception as e:
        logger.warning(f"Unsloth import/load failed, falling back to BitsAndBytes: {e}")
        return None


def _load_model_quantized(config, moe_config, group_ids, expert_manager, is_miner: bool) -> nn.Module:
    """
    Load model with BitsAndBytes 4-bit quantization.

    For miners: replaces expert layers with trainable bfloat16 versions.
    For validators: returns frozen quantized model for inference.
    """
    from transformers import BitsAndBytesConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

    # Import here to avoid circular import
    from mycelia.shared.model import replace_experts_for_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    max_memory = get_nested_attr(config, "model.max_memory", None)
    if max_memory is None:
        max_memory = {0: "46GB", "cpu": "100GB"}

    # Set device_map to single GPU to avoid memory spikes
    device_map_setting = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_map_dict = {"": device_map_setting} if device_map_setting != "auto" else "auto"

    logger.info("Downloading and quantizing model (this may take a while for large models)...")

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        config.model.model_path,
        quantization_config=bnb_config,
        device_map=device_map_dict,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    logger.info("Model downloaded and quantized to 4-bit")

    # For miners: replace expert layers with trainable bfloat16 versions
    if is_miner:
        logger.info("Replacing experts with trainable bfloat16 versions...")
        model = replace_experts_for_training(model, moe_config, group_ids, expert_manager)
        logger.info("Experts replaced - base model frozen (4-bit), experts trainable (bfloat16)")
    else:
        logger.info("Loaded with BitsAndBytes quantization (inference)")

    return model


def _load_model_standard(config, moe_config, state_dicts: list) -> nn.Module:
    """
    Load model in standard full-precision mode (no quantization).

    Uses bfloat16 on Ampere+ GPUs, float16 otherwise.
    Optionally loads from checkpoint state_dicts.
    """
    load_on_cpu = get_nested_attr(config, "model.load_on_cpu", False)

    # Check GPU support for bfloat16 (Ampere+ GPUs: 30xx, 40xx, A-series)
    use_bf16 = False
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability(0)
        use_bf16 = device_capability[0] >= 8  # Ampere (8.0) or newer

    dtype = torch.bfloat16 if use_bf16 else torch.float16
    logger.info(f"Loading model in {'BF16' if use_bf16 else 'FP16'} (standard path, no quantization)")

    if load_on_cpu:
        logger.info("Loading on CPU (load_on_cpu=True) - will be slower but avoids OOM")
        with torch.device("cpu"):
            model = CustomQwen3VLMoeForConditionalGeneration(moe_config)
    else:
        model = CustomQwen3VLMoeForConditionalGeneration(moe_config)

    model = model.to(dtype=dtype)

    if len(state_dicts) > 0:
        merged_state_dict, missing = merge_state_dicts_with_priority(state_dicts, model)
        assert len(missing) == 0
        model.load_state_dict(merged_state_dict, strict=True)
        model = model.to(dtype=dtype)

    if get_nested_attr(config, "model.torch_compile", False):
        model = torch.compile(model)

    return model


def get_base_model(
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    group_ids: list | None = None,
    state_dicts: list = [],
    partial=False,
) -> nn.Module | None:
    """
    Load base model with role-specific optimizations.

    Dispatches to appropriate loader based on config:
    - Validators with quantization: try Unsloth first, then BitsAndBytes
    - Miners with quantization: BitsAndBytes with trainable experts
    - Standard path: full-precision model for training
    """
    topk = config.moe.partial_topk if partial else config.moe.full_topk
    moe_config = get_moe_model_config(config, topk, group_ids, expert_manager)

    is_validator = config.role == "validator"
    is_miner = config.role == "miner"
    use_quantization = get_nested_attr(config, "model.use_quantization", False)
    use_unsloth_requested = get_nested_attr(config, "model.use_unsloth", False)
    use_unsloth = use_unsloth_requested and is_validator

    # Warn if miner requested unsloth (not supported for training)
    if use_unsloth_requested and is_miner:
        logger.warning("use_unsloth=True is ignored for miners (Unsloth is inference-only, not compatible with training)")

    # === QUANTIZED PATH ===
    if use_quantization:
        logger.info(f"Loading with 4-bit quantization for {'miner (trainable experts)' if is_miner else 'validator'}")

        # Try Unsloth first for validators (fastest, inference-only)
        if use_unsloth:
            model = _load_model_with_unsloth(config, moe_config)
            if model is not None:
                return model

        # Fall back to BitsAndBytes quantization
        return _load_model_quantized(config, moe_config, group_ids, expert_manager, is_miner)

    # === STANDARD PATH ===
    return _load_model_standard(config, moe_config, state_dicts)


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
