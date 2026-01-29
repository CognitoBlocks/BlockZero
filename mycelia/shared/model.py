from __future__ import annotations

import time
import traceback
from pathlib import Path

import bittensor
import torch
from torch import nn

from mycelia.shared.app_logging import structlog
from mycelia.shared.chain import (
    SignedModelHashChainCommit,
    WorkerChainCommit,
    get_chain_commits,
)
from mycelia.shared.checkpoint_helper import compile_full_state_dict_from_path, load_checkpoint
from mycelia.shared.checkpoints import (
    ChainCheckpoints,
    ModelCheckpoint,
    build_chain_checkpoints,
    build_chain_checkpoints_from_previous_phase,
    delete_old_checkpoints,
    select_best_checkpoint,
)
from mycelia.shared.client import download_model
from mycelia.shared.config import MinerConfig, ValidatorConfig, WorkerConfig
from mycelia.shared.cycle import PhaseNames, get_blocks_from_previous_phase_from_api
from mycelia.shared.expert_manager import (
    ExpertManager,
    get_layer_expert_id,
)
from mycelia.shared.helper import get_model_hash, get_nested_attr
from mycelia.shared.modeling.mycelia import get_base_model
from mycelia.shared.schema import verify_message

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
    resume = get_nested_attr(config, "ckpt.resume_from_ckpt", False)
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
    if resume:
        latest_checkpoint = select_best_checkpoint(
            primary_dir=config.ckpt.validator_checkpoint_path,
            secondary_dir=config.ckpt.checkpoint_path,
            resume=config.ckpt.resume_from_ckpt,
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

    fetch_model_from_chain_validator(
        current_model_meta=current_checkpoint,
        config=config,
        subtensor=subtensor,
        wallet=wallet,
        expert_group_ids=[config.task.exp.group_id],
    )

    return get_model_from_checkpoint(rank=rank, config=config, expert_manager=expert_manager)


def fetch_model_from_chain_validator(
    current_model_meta: ModelCheckpoint | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    expert_group_ids: list[int | str],
) -> dict | None:
    """
    Fetches a model from the chain validator if it's has the right commit format from the previous phase commits (validator_commit_1 & validator_commit_2) and newer than the current model.
    """
    chain_checkpoints = build_chain_checkpoints_from_previous_phase(config=config, subtensor=subtensor, for_role ="validator")

    # --- Filter to only newer than current model ---
    if current_model_meta is not None: 
        chain_checkpoints = ChainCheckpoints(
            checkpoints=[ckpt for ckpt in chain_checkpoints.checkpoints if ckpt > current_model_meta]
        )
        
    should_download = len(chain_checkpoints.checkpoints) > 0

    logger.info(
        "Fetching model from chain",
        should_download=should_download,
        chain_checkpoints=chain_checkpoints,
        current_model_meta=current_model_meta,
    )

    # --- Download model if available ---
    if should_download and chain_checkpoints:
        download_success = False
        retries = 0
        max_retries = 3
        base_delay_s = 5  # backoff base

        while (not download_success) and (retries < max_retries):
            for chain_checkpoint in chain_checkpoints.checkpoints:
                logger.info(f"Downloading from chain: uid = {chain_checkpoint.uid}", chain_checkpoint=chain_checkpoint)

                # Resolve URL if not provided; fall back to ip/port + default route
                # Best-effort defaults; customize if your API differs
                protocol = getattr(getattr(config, "miner", object()), "protocol", "http")
                if chain_checkpoint.ip and chain_checkpoint.port:
                    url = f"{protocol}://{chain_checkpoint.ip}:{chain_checkpoint.port}/get-checkpoint"
                else:
                    logger.warning("Skipping meta without URL or ip:port: %s", chain_checkpoint)
                    continue

                out_folder = Path(config.ckpt.validator_checkpoint_path) / (
                    f"uid_{chain_checkpoint.uid}_hotkey_{chain_checkpoint.hotkey}_globalver_{chain_checkpoint.global_ver}"
                )

                out_folder.mkdir(parents=True, exist_ok=True)

                for expert_group_id in expert_group_ids:
                    if isinstance(expert_group_id, int):
                        out_file = f"model_expgroup_{expert_group_id}.pt"
                    elif expert_group_id == "shared":
                        out_file = "model_shared.pt"
                    else:
                        logger.warning("Invalid expert_group_id, skipping:", expert_group_id=expert_group_id)
                        continue

                    out_path = out_folder / out_file
                    try:
                        download_model(
                            url=url,
                            my_hotkey=wallet.hotkey,  # type: ignore
                            target_hotkey_ss58=chain_checkpoint.hotkey,
                            block=subtensor.block,
                            expert_group_id=expert_group_id,
                            token=getattr(config.cycle, "token", ""),
                            out_dir=out_path,
                        )

                        chain_checkpoint.path = out_folder
                        validated = chain_checkpoint.validate()

                        if not validated:
                            logger.warning(
                                "❌ Downloaded checkpoint failed validation",
                                out_path=out_path,
                                current_model_version=current_model_meta.global_ver
                                if current_model_meta
                                else None,
                                current_model_hash=current_model_meta.model_hash if current_model_meta else None,
                            )
                            continue
                        # If download + verification succeed, consider it a success
                        download_success = validated

                        current_model_version = chain_checkpoint.global_ver
                        current_model_hash = chain_checkpoint.model_hash
                        
                        logger.info(
                            "✅ Downloaded checkpoint (verified)",
                            out_path=out_path,
                            current_model_version=current_model_version,
                            current_model_hash=current_model_hash,
                            validation_success=validated,
                        )

                        delete_old_checkpoints(
                            checkpoint_path=Path(config.ckpt.validator_checkpoint_path),
                            topk=config.ckpt.checkpoint_topk,
                        )

                        return chain_checkpoint
                    except Exception as e:
                        logger.warning("Download failed", url, e)
                        traceback.print_exc()

            if not download_success:
                retries += 1
                if retries < max_retries:
                    delay = base_delay_s * (2 ** (retries - 1))
                    logger.info("Retrying", delay=delay, retries=retries + 1, max_retries=max_retries)
                    time.sleep(delay)

        if not download_success:
            logger.error(f"❌ All download attempts failed after {retries} retries.")

            return None
