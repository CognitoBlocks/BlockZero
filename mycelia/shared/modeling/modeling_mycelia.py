"""
MoE model utilities: convert dense models to MoE-style, define custom sparse blocks,
and provide helpers for partial (grouped) expert execution.

Contents
--------
- DenseBlock: wraps DeepseekV3 MLP to behave like a "dense" block while preserving API
- myceliaSparseMoeBlock: sparse MoE block that activates only a subset of experts (by group)
- CustomMoE: an DeepseekV3 model variant that interleaves MoE and dense blocks
- get_base_model: load base LLaMA or OLMo, optionally convert to MoE
- get_base_tokenizer: load tokenizer (HF login via env var if provided)
- dense_model_to_moe: transform a dense model into DeepseekV3 parameter layout
- get_layer_expert_id: parse layer/expert indices from parameter names
- partial_moe: drop experts outside the current group and rebuild a partial model

Assumptions
-----------
- Parameter names use DeepseekV3/transformers conventions (e.g., "model.layers.{i}.mlp.experts.{e}").
- Experts are identified by the "experts.{expert_id}" path segment.
- Your `Config` provides fields used by `get_base_model` and `get_base_tokenizer`.
"""

from __future__ import annotations

import os
import re
import copy
import logging
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    OlmoForCausalLM,
    AutoModelForCausalLM,
    PretrainedConfig,
)

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.utils.deprecation import deprecate_kwarg
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from mycelia.shared.modeling.modeling_custom_deepseek import CustomDeekSeekMoE, get_moe_model_config
from mycelia.shared.config import MinerConfig, ValidatorConfig
from mycelia.shared.expert_manager import get_layer_expert_id
from mycelia.shared.app_logging import structlog
from mycelia.shared.helper import *

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def get_base_model(
    config: MinerConfig,
    expert_group_assignment: Dict[int, Dict[int, List[int]]] | None = None,
    noise: bool = False,
    full=False,
) -> Optional[LlamaForCausalLM | OlmoForCausalLM]:
    """
    Load a base Causal LM by `config.model.model_path` and optionally convert to MoE.

    Returns
    -------
    Optional[nn.Module]
        A Hugging Face causal LM (LLaMA or OLMo), possibly converted to OLMoE.
    """
    model = None

    if config.model.foundation:
        model_config = AutoConfig.from_pretrained(
            config.model.model_path, trust_remote_code=True
        )  # TODO: need miner agreement
        moe_config = get_moe_model_config(
            config, config.moe.full_topk if full else config.moe.partial_topk, org_model_config=model_config
        )
        model = CustomDeekSeekMoE(config, moe_config, expert_group_assignment=expert_group_assignment)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_path, trust_remote_code=True
        )  # TODO: need miner agreement to trust remote code

    if model is not None and get_nested_attr(config, "model.torch_compile", False):
        model = torch.compile(model)

    return model


def get_base_tokenizer(config: MinerConfig | ValidatorConfig):
    """
    Load the tokenizer for `config.model.model_path`.

    Notes
    -----
    * If `HF_TOKEN` is set in the environment, we call `huggingface_hub.login` with it.
      Otherwise we rely on cached credentials (e.g., `huggingface-cli login`).
    * Sets `pad_token` to `"</s>"` for causal LM padding compatibility.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, use_fast=True)
    # tokenizer.pad_token = "</s>"
    return tokenizer


def noise_injection(weight: torch.Tensor, noise_ratio: float = 0.5, init_std: float = 0.02) -> torch.Tensor:
    """
    Randomly replace a fraction of weights with Gaussian noise.

    Parameters
    ----------
    weight : Tensor
        Source tensor (modified in-place).
    noise_ratio : float
        Fraction of elements to replace with noise.
    init_std : float
        Std for the injected Gaussian noise.

    Returns
    -------
    Tensor
        The modified `weight`.
    """
    mask = torch.FloatTensor(weight.size()).uniform_() < noise_ratio
    mask = mask.to(weight.device)
    rand_weight = torch.nn.init.normal_(copy.deepcopy(weight), mean=0.0, std=init_std)
    weight[mask] = rand_weight[mask]
    return weight


def derive_model_from_shared_expert(
    config: MinerConfig,
    model: nn.Module,
    topk: int,
    noise: bool = False,
    noise_std: Optional[float] = None,
    expert_group_assignment: Dict[int, list] | None = None,
):

    gate_mat_pat = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")

    state_dict = model.state_dict()
    moe_config = get_moe_model_config(config, topk, org_model_config=model.config)

    # 1) Find per-layer top expert index from gate.weight
    # TODO: change top expert selection based on sigmoid gate weight & may need to select multiple expert
    layer_top_expert = {}
    for k, v in state_dict.items():
        m = gate_mat_pat.match(k)
        if not m:
            continue

        layer = int(m.group(1))
        gate_w = v
        try:
            scores = gate_w.sum(dim=1)  # [num_experts]

        except Exception as ex:
            raise RuntimeError(
                f"Failed to compute per-expert sums for {k} " f"(shape={tuple(getattr(gate_w, 'shape', []))})."
            ) from ex

        try:
            # torch.Tensor argmax
            top_idx = int(scores.argmax().item())
        except Exception:
            # Fallback for non-torch arrays
            top_idx = int(scores.argmax())

        layer_top_expert[layer] = top_idx

    # logger.info("top experts", layer_top_expert)

    # 2) Build new state dict, copying everything except experts that we drop/overwrite
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        layer, expert_id = get_layer_expert_id(k)

        # no layer -> directly copy old
        if layer is None:
            new_sd[k] = v
            continue

        # no expert -> if layer is even -> skip, else directly copy
        if expert_id is None:
            if (layer + 1) % 2 == 0:
                if "mlp.gate.weight" in k:
                    gate_sd = TopkRouter(moe_config).state_dict()
                    new_sd[k] = gate_sd["weight"]
                    new_sd[k.replace("weight", "e_score_correction_bias")] = gate_sd["e_score_correction_bias"]
                    continue
                else:
                    new_sd[k] = v
                    continue

            elif "gate.weight" not in k and "shared" not in k:
                new_sd[k] = v
                continue
            else:
                continue

        if expert_id >= config.moe.num_experts:
            continue

        if layer not in layer_top_expert:
            new_sd[k] = v
            continue

        top_eid = layer_top_expert[layer]
        # We keep the slot but replace its weight with the top expert's weight
        src_key = k.replace(f"expert.{expert_id}", f"expert.{top_eid}")
        if (layer + 1) % 2 == 0:
            src_w = state_dict[src_key]
            try:
                new_sd[k] = src_w.clone()
            except AttributeError:
                new_sd[k] = src_w

        elif expert_id == 0:
            new_sd[k.replace(f"experts.{expert_id}.", "")] = state_dict[src_key].clone()

    # Start from a base OLMoE config, then copy overlapping fields from the dense config
    model = CustomDeekSeekMoE(config, moe_config, expert_group_assignment=expert_group_assignment)

    # TODO: strict should be true
    _missing, _unexpected = model.load_state_dict(new_sd, strict=False)  # will raise if mismatch
    return model


def dense_model_to_moe(
    config: MinerConfig,
    dense_model: nn.Module,
    topk: int,
    noise: bool = False,
    noise_std: Optional[float] = None,
    expert_group_assignment: Dict[int, list] | None = None,
) -> CustomDeekSeekMoE:
    """
    Convert a dense transformer model to an OLMoE-structured model by:
      * Injecting a router gate and expanding MLP weights into expert shards.
      * Optionally injecting initialization noise into expert parameters.
      * Mapping certain naming differences (e.g., post_feedforward -> input).

    Parameters
    ----------
    dense_model : nn.Module
        A pretrained dense Causal LM (e.g., LLaMA).
    num_experts : int
        Number of experts to create per MoE layer.
    topk : int
        Number of experts to route each token to (num_experts_per_tok).
    noise : bool
        Whether to inject noise when cloning dense weights into expert slots.
    noise_std : Optional[float]
        Std of the initialization noise (default 0.02).
    interleave : bool
        If True, convert every other MLP layer to MoE (even indices).

    Returns
    -------
    CustomMoE
        An OLMoE-structured model initialized from the dense weights.
    """
    mlp_layer_name = {"w1": "gate_proj", "w2": "up_proj", "w3": "down_proj"}
    layer_name_mapping = {"post_feedforward": "input"}  # compatibility tweak

    sd = dense_model.state_dict()

    hidden_size = dense_model.config.hidden_size
    intermediate_size = dense_model.config.intermediate_size

    moe_sd: Dict[str, torch.Tensor] = {}

    # Some models may lack q/k normalization; synthesize if needed later.
    has_layer_norm = "k_norm" in dense_model.model.layers[0].self_attn.__dir__()

    for key in list(sd.keys()):
        # Try to infer the layer index from the parameter name
        if "layers." in key:
            start = key.find("layers.") + len("layers.")
            end = key.find(".", start)
            layer_index = int(key[start:end])
        else:
            layer_index = None

        # Convert target MLP layers into MoE (interleave == even layers or all if interleave=False)
        if mlp_layer_name["w1"] in key and (
            (config.moe.interleave and layer_index is not None and (layer_index + 1) % 2 == 0)
            or not config.moe.interleave
        ):
            layer_prefix = key[: key.find("mlp.") + len("mlp.")]
            layer_suffix = key[key.find("mlp.") + len("mlp.") :]

            # Router gate weights (E x H)
            moe_sd[layer_prefix + "gate.weight"] = torch.zeros(
                (config.moe.num_experts, hidden_size), device=sd[key].device
            )

            if noise and noise_std is not None:
                moe_sd[layer_prefix + "gate.weight"] = moe_sd[layer_prefix + "gate.weight"].normal_(std=noise_std)

            # Gather dense MLP weights we will replicate per-expert
            expert_suffix = {
                "w1": layer_suffix,
                "w2": layer_suffix.replace(mlp_layer_name["w1"], mlp_layer_name["w2"]),
                "w3": layer_suffix.replace(mlp_layer_name["w1"], mlp_layer_name["w3"]),
            }

            if intermediate_size is None:
                intermediate_size = sd[layer_prefix + expert_suffix["w1"]].shape[0]

            expert_layers = {}
            for suf in expert_suffix.values():
                name = layer_prefix + suf
                expert_layers[suf] = sd.pop(name)

            # Replicate dense weights into each expert (optionally with noise)
            for expert_id in range(config.moe.num_experts):
                for suf in expert_suffix.values():
                    src = expert_layers[suf]
                    dst_name = f"{layer_prefix}experts.{expert_id}.{suf}"
                    if noise:
                        moe_sd[dst_name] = noise_injection(src.clone(), init_std=noise_std or 0.02)
                    else:
                        moe_sd[dst_name] = src.clone()

        # Pass-through for everything else (with minor renaming)
        elif key in sd:
            new_key = key
            for old, new in layer_name_mapping.items():
                new_key = new_key.replace(old, new)
            moe_sd[new_key] = sd.pop(key)

            # # If attention lacks q/k norms, synthesize them when we see o_proj
            if not has_layer_norm and "self_attn.o_proj.weight" in new_key:
                moe_sd[new_key.replace("o_proj", "q_norm")] = torch.ones(hidden_size, device=moe_sd[new_key].device)
                moe_sd[new_key.replace("o_proj", "k_norm")] = torch.ones(hidden_size, device=moe_sd[new_key].device)

        else:
            # already processed
            pass

    # Start from a base OLMoE config, then copy overlapping fields from the dense config
    moe_config = get_moe_model_config(config, topk, org_model_config=dense_model.config)

    model = CustomDeekSeekMoE(config, moe_config, expert_group_assignment=expert_group_assignment)

    _missing, _unexpected = model.load_state_dict(moe_sd, strict=True)  # will raise if mismatch
    return model


def partial_moe(
    config: MinerConfig,
    moe_model: CustomDeekSeekMoE,
    my_group_id: int,
    expert_group_assignment: Dict[int, Dict[int, List[int]]],
) -> CustomDeekSeekMoE:
    """
    Build a partial MoE that retains only experts for `my_group_id`.

    Parameters
    ----------
    moe_model : CustomMoE
        A full OLMoE-structured model.
    my_group_id : int
        Group to retain.
    expert_group_assignment : Dict[int, Dict[int, List[int]]]
        Mapping: layer_id -> (group_id -> list of expert ids).

    Returns
    -------
    CustomMoE
        A partial model with only this group's experts instantiated/loaded.
    """
    sd = moe_model.state_dict()

    # Remove parameters of experts not owned by this group
    for k in list(sd.keys()):
        layer_id, expert_id = get_layer_expert_id(k)
        if expert_id is not None and expert_id not in expert_group_assignment[layer_id][my_group_id]:
            del sd[k]

    topk = getattr(moe_model.config, "num_experts_per_tok", 2)
    moe_config = get_moe_model_config(config, topk, moe_model.config)

    partial = CustomDeekSeekMoE(
        config, moe_config, my_group_id=my_group_id, expert_group_assignment=expert_group_assignment, partial=True
    )

    if partial is not None and get_nested_attr(config, "model.torch_compile", False):
        partial = torch.compile(partial)

    partial.load_state_dict(sd, strict=True)  # partial by design
    return partial
