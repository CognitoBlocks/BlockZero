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
    ChainCheckpoint,
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
from mycelia.shared.validator_selection import (
    ValidatorBlacklist,
    ValidatorSelector,
    VerifiedModelDownloader,
    DownloadResult,
    DownloadErrorType,
    create_verified_downloader,
)

logger = structlog.get_logger(__name__)

# Module-level blacklist instance for persistence across download calls within a session.
# This ensures validators that fail hash verification remain deprioritized
# without requiring external state management.
# Note: For full persistence across restarts, serialize blacklist.to_dict() to disk.
_validator_blacklist: ValidatorBlacklist | None = None


def get_validator_blacklist() -> ValidatorBlacklist:
    """
    Get or create the module-level validator blacklist.

    The blacklist persists for the duration of the process,
    tracking validators that have served incorrect models.
    """
    global _validator_blacklist
    if _validator_blacklist is None:
        _validator_blacklist = ValidatorBlacklist()
    return _validator_blacklist


def reset_validator_blacklist() -> None:
    """
    Reset the validator blacklist. Useful for testing or manual intervention.
    """
    global _validator_blacklist
    _validator_blacklist = None


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
    use_verified_download: bool = True,
) -> ChainCheckpoint | None:
    """
    Fetches a model from a verified validator with robust selection and hash verification.

    This function implements the full validator selection and model download pipeline:
    1. Builds chain checkpoints from previous phase validator commits
    2. Filters to checkpoints newer than current model
    3. Uses canonical validator selection (highest version, consensus hash)
    4. Downloads with hash verification
    5. Blacklists validators that serve incorrect models
    6. Retries with different validators on failure

    Args:
        current_model_meta: Current local model checkpoint (for version comparison)
        config: Worker configuration
        subtensor: Bittensor subtensor for chain interaction
        wallet: Wallet for signing download requests
        expert_group_ids: List of expert group IDs to download
        use_verified_download: If True, uses robust verified downloader with blacklist.
                               If False, falls back to legacy download logic.

    Returns:
        ChainCheckpoint of the successfully downloaded model, or None if all attempts failed.

    Failure paths:
        - No validators available: Returns None, logs warning
        - All validators blacklisted: Returns None, logs warning with blacklist state
        - Hash mismatch: Blacklists validator, retries with next candidate
        - Network error: Records failure, retries with backoff
        - Max retries exceeded: Returns None, logs error
    """
    # Build chain checkpoints from the previous phase's validator commits
    # These represent validators that have committed model hashes to chain
    chain_checkpoints = build_chain_checkpoints_from_previous_phase(
        config=config, subtensor=subtensor, for_role="validator"
    )

    # Filter to only checkpoints newer than our current model
    # Why: No need to download if we already have the latest version
    chain_checkpoints = ChainCheckpoints(
        checkpoints=[ckpt for ckpt in chain_checkpoints.checkpoints if ckpt > current_model_meta]
    )
    should_download = len(chain_checkpoints.checkpoints) > 0

    logger.info(
        "Fetching model from chain",
        should_download=should_download,
        num_candidates=len(chain_checkpoints.checkpoints),
        current_model_version=current_model_meta.global_ver if current_model_meta else None,
    )

    if not should_download or not chain_checkpoints:
        # No newer models available - this is normal, not an error
        return None

    # Get current block for deterministic operations
    current_block = subtensor.block

    if use_verified_download:
        # Use the robust verified downloader with blacklist support
        return _fetch_with_verified_downloader(
            chain_checkpoints=chain_checkpoints,
            config=config,
            wallet=wallet,
            current_block=current_block,
            expert_group_ids=expert_group_ids,
        )
    else:
        # Legacy download path (kept for backwards compatibility)
        return _fetch_legacy(
            chain_checkpoints=chain_checkpoints,
            config=config,
            subtensor=subtensor,
            wallet=wallet,
            current_model_meta=current_model_meta,
            expert_group_ids=expert_group_ids,
        )


def _fetch_with_verified_downloader(
    chain_checkpoints: ChainCheckpoints,
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    current_block: int,
    expert_group_ids: list[int | str],
) -> ChainCheckpoint | None:
    """
    Download model using the verified downloader with hash verification and blacklist.

    This implements the robust download flow:
    1. Select canonical validator (highest version, most consensus)
    2. Download model files
    3. Verify hash matches chain commitment
    4. On mismatch: blacklist validator, retry with next candidate
    5. On success: return checkpoint with verified path

    The blacklist persists across calls within this session, ensuring
    Byzantine validators remain deprioritized.
    """
    # Get the persistent blacklist for this session
    blacklist = get_validator_blacklist()

    # Clean up expired blacklist entries to prevent unbounded growth
    blacklist.cleanup_expired(current_block)

    # Create the verified downloader with our blacklist
    downloader = create_verified_downloader(
        blacklist=blacklist,
        max_retries=5,  # More retries than legacy to handle Byzantine validators
    )

    # Attempt download with verification
    result: DownloadResult = downloader.download_and_verify(
        chain_checkpoints=chain_checkpoints,
        config=config,
        wallet=wallet,
        current_block=current_block,
        expert_group_ids=expert_group_ids,
        out_folder=Path(config.ckpt.validator_checkpoint_path),
    )

    if not result.success:
        # Log detailed failure information for debugging
        logger.error(
            "Verified download failed",
            error_type=result.error_type.value if result.error_type else "unknown",
            error_message=result.error_message,
            blacklist_size=len(blacklist.failures),
        )
        return None

    # Download succeeded with verified hash
    logger.info(
        "Model downloaded and hash verified successfully",
        validator_hotkey=result.checkpoint.hotkey[:16] + "..." if result.checkpoint else "unknown",
        path=str(result.local_path),
        hash=result.actual_hash[:24] + "..." if result.actual_hash else "unknown",
    )

    # Update checkpoint path for downstream use
    # Note: We create a new checkpoint rather than mutating to avoid side effects
    verified_checkpoint = ChainCheckpoint(
        uid=result.checkpoint.uid,
        ip=result.checkpoint.ip,
        port=result.checkpoint.port,
        hotkey=result.checkpoint.hotkey,
        signed_model_hash=result.checkpoint.signed_model_hash,
        model_hash=result.actual_hash,  # Use verified hash
        global_ver=result.checkpoint.global_ver,
        expert_group=result.checkpoint.expert_group,
        path=result.local_path,
        signature_verified=True,
        hash_verified=True,  # We verified this ourselves
    )

    # Clean up old checkpoints to manage disk space
    delete_old_checkpoints(
        checkpoint_path=Path(config.ckpt.validator_checkpoint_path),
        topk=config.ckpt.checkpoint_topk,
    )

    return verified_checkpoint


def _fetch_legacy(
    chain_checkpoints: ChainCheckpoints,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    current_model_meta: ModelCheckpoint | None,
    expert_group_ids: list[int | str],
) -> ChainCheckpoint | None:
    """
    Legacy download path without robust validator selection.

    Kept for backwards compatibility. Prefer use_verified_download=True.

    Note: This path does NOT blacklist validators or verify hashes rigorously.
    It may retry the same failing validator multiple times.
    """
    download_success = False
    retries = 0
    max_retries = 3
    base_delay_s = 5

    while (not download_success) and (retries < max_retries):
        for chain_checkpoint in chain_checkpoints.checkpoints:
            logger.info(
                f"Downloading from chain: uid = {chain_checkpoint.uid}",
                chain_checkpoint=chain_checkpoint,
            )

            protocol = getattr(getattr(config, "miner", object()), "protocol", "http")
            if chain_checkpoint.ip and chain_checkpoint.port:
                url = f"{protocol}://{chain_checkpoint.ip}:{chain_checkpoint.port}/get-checkpoint"
            else:
                # Skip: validator has no network info, cannot download
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
                        my_hotkey=wallet.hotkey,
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
                            "Downloaded checkpoint failed validation",
                            out_path=out_path,
                            current_model_version=current_model_meta.global_ver if current_model_meta else None,
                            current_model_hash=current_model_meta.model_hash if current_model_meta else None,
                        )
                        continue

                    download_success = validated

                    logger.info(
                        "Downloaded checkpoint (verified)",
                        out_path=out_path,
                        current_model_version=chain_checkpoint.global_ver,
                        current_model_hash=chain_checkpoint.model_hash,
                        validation_success=validated,
                    )

                    delete_old_checkpoints(
                        checkpoint_path=Path(config.ckpt.validator_checkpoint_path),
                        topk=config.ckpt.checkpoint_topk,
                    )

                    return chain_checkpoint
                except Exception as e:
                    logger.warning("Download failed", url=url, error=str(e))
                    traceback.print_exc()

        if not download_success:
            retries += 1
            if retries < max_retries:
                delay = base_delay_s * (2 ** (retries - 1))
                logger.info("Retrying", delay=delay, retries=retries + 1, max_retries=max_retries)
                time.sleep(delay)

    if not download_success:
        logger.error(f"All download attempts failed after {retries} retries.")
        return None

    return None
