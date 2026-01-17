from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import traceback
from pathlib import Path

import bittensor
from pydantic import BaseModel, ConfigDict, Field

from mycelia.shared.app_logging import structlog
from mycelia.shared.checkpoints import (
    ModelCheckpoint,
    ChainCheckpoints,
    build_chain_checkpoints,
    delete_old_checkpoints,
)
from mycelia.shared.client import download_model
from mycelia.shared.config import WorkerConfig
from mycelia.shared.schema import verify_message

logger = structlog.get_logger(__name__)

# Global lock for subtensor WebSocket access to prevent concurrent recv calls
_subtensor_lock = threading.Lock()


# --- Info gather ---
def get_active_validator_info() -> dict | None:
    raise NotImplementedError


def get_active_miner_info():
    raise NotImplementedError


# --- Status structure and submission (for miner validator communication)---
class WorkerChainCommit(BaseModel):
    ip: str
    port: int
    active: bool
    stake: float
    validator_permit: bool


class ValidatorChainCommit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    signed_model_hash: str | None = Field(default=None, alias="m")
    model_hash: str | None = Field(default=None, alias="h")
    global_ver: int | None = Field(default=None, alias="v")
    expert_group: int | None = Field(default=None, alias="e")
    miner_seed: int | None = Field(default=None, alias="s")
    block: int | None = Field(default=None, alias="b")


class MinerChainCommit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    block: int = Field(alias="b")
    expert_group: int | None = Field(default=None, alias="e")
    signed_model_hash: str | None = Field(default=None, alias="m")
    model_hash: str | None = Field(default=None, alias="h")
    global_ver: int | None = Field(default=0, alias="v")
    inner_opt: int | None = Field(default=0, alias="i")


def commit_status(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    subtensor: bittensor.Subtensor,
    status: ValidatorChainCommit | MinerChainCommit,
) -> None:
    """
    Commit the worker status to chain.

    If encrypted=False:
        - Uses subtensor.set_commitment (plain metadata, immediately visible).

    If encrypted=True:
        - Timelock-encrypts the status JSON using Drand.
        - Stores it via the Commitments pallet so it will be revealed later
          when the target Drand round is reached.

    Assumes:
        - config.chain.netuid: subnet netuid
        - config.chain.timelock_rounds_ahead: how many Drand rounds in the future
          you want the data to be revealed (fallback to 200 if missing).
    """
    # Serialize status first; same input for both plain + encrypted paths
    data_dict = status.model_dump(by_alias=True)

    data = json.dumps(data_dict)

    success = subtensor.set_commitment(wallet=wallet, netuid=config.chain.netuid, data=data, raise_error=False)
    
    if not success: 
        logger.warning("Failed to commit status to chain", status=data_dict)
    else:
        logger.info("Committed status to chain", status=data_dict)
    
    return data_dict


def get_chain_commits(
    config: WorkerConfig, subtensor: bittensor.Subtensor, wait_to_decrypt: bool = False
) -> tuple[WorkerChainCommit, bittensor.Neuron]:
    all_commitments = subtensor.get_all_commitments(netuid=config.chain.netuid)
    metagraph = subtensor.metagraph(netuid=config.chain.netuid)

    parsed = []

    for hotkey, commit in all_commitments.items():
        uid = metagraph.hotkeys.index(hotkey)

        try:
            status_dict = json.loads(commit)

            chain_commit = (
                ValidatorChainCommit.model_validate(status_dict)
                if "miner_seed" in status_dict or "s" in status_dict
                else MinerChainCommit.model_validate(status_dict)
            )

        except Exception as e:
            chain_commit = None

        parsed.append((chain_commit, metagraph.neurons[uid]))

    return parsed


# --- setup chain worker ---
def setup_chain_worker(config):
    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)
    subtensor = bittensor.Subtensor(network=config.chain.network)
    serve_axon(
        config=config,
        wallet=wallet,
        subtensor=subtensor,
    )
    return wallet, subtensor


def serve_axon(config: WorkerConfig, wallet: bittensor.Wallet, subtensor: bittensor.Subtensor):
    axon = bittensor.Axon(wallet=wallet, external_port=config.chain.port, ip=config.chain.ip)
    axon.serve(netuid=348, subtensor=subtensor)


# --- Chain weight submission ---
def submit_weight() -> str:
    raise NotImplementedError


# --- Get model from chain ---
# def scan_chain_for_new_model(
#     current_model_meta: ModelCheckpoint | None,
#     config: WorkerConfig,
#     subtensor: bittensor.Subtensor,
# ) -> tuple[bool, list[dict]]:
#     commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)
#     chain_checkpoints = build_chain_checkpoints(current_model_meta=current_model_meta, commits=commits)

#     download_meta = []
#     for ckpt in chain_checkpoints.checkpoints:
#         download_meta.append(
#             {
#                 "uid": ckpt._extra("uid"),
#                 "ip": ckpt._extra("ip"),
#                 "port": ckpt._extra("port"),
#                 "model_hash": ckpt.model_hash,
#                 "global_ver": ckpt.global_ver,
#                 "target_hotkey_ss58": ckpt._extra("target_hotkey_ss58"),
#             }
#         )

#     should_download = len(download_meta) > 0
#     return should_download, download_meta


def fetch_model_from_chain(
    current_model_meta: ModelCheckpoint | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    expert_group_ids: list[int | str],
) -> dict | None:
    
    chain_checkpoints = build_chain_checkpoints(commits=get_chain_commits(config, subtensor))
    chain_checkpoints = ChainCheckpoints(checkpoints=[ckpt for ckpt in chain_checkpoints.checkpoints if ckpt > current_model_meta])
    should_download = len(chain_checkpoints.checkpoints) > 0

    logger.info("Fetching model from chain", should_download=should_download, chain_checkpoints=chain_checkpoints, current_model_meta=current_model_meta)

    if should_download and chain_checkpoints:
        download_success = False
        retries = 0
        max_retries = 3
        base_delay_s = 5  # backoff base

        while (not download_success) and (retries < max_retries):
            for chain_checkpoint in chain_checkpoints:
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
                        # If download_model doesn't raise, consider it a success
                        download_success = True
                        current_model_version = chain_checkpoint.global_ver
                        current_model_hash = chain_checkpoint.model_hash
                        logger.info(
                            "✅ Downloaded checkpoint",
                            out_path=out_path,
                            current_model_version=current_model_version,
                            current_model_hash=current_model_hash,
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
