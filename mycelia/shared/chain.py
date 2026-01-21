from __future__ import annotations

import json
import threading

import bittensor
from pydantic import BaseModel, ConfigDict, Field

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import WorkerConfig

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


class SignedModelHashChainCommit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    signed_model_hash: str | None = Field(default=None, alias="m")


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
    block: int | None = Field(default=None, alias="b")
    expert_group: int | None = Field(default=None, alias="e")
    signed_model_hash: str | None = Field(default=None, alias="m")
    model_hash: str | None = Field(default=None, alias="h")
    global_ver: int | None = Field(default=0, alias="v")
    inner_opt: int | None = Field(default=0, alias="i")


def commit_status(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    subtensor: bittensor.Subtensor,
    status: ValidatorChainCommit | MinerChainCommit | SignedModelHashChainCommit,
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
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wait_to_decrypt: bool = False,
    block: int | None = None,
    signature_commit: bool = False,
) -> tuple[WorkerChainCommit, bittensor.Neuron]:
    all_commitments = subtensor.get_all_commitments(
        netuid=config.chain.netuid, block=block,
    )
    metagraph = subtensor.metagraph(netuid=config.chain.netuid, block=block)

    parsed = []

    for hotkey, commit in all_commitments.items():
        uid = metagraph.hotkeys.index(hotkey)

        try:
            status_dict = json.loads(commit)

            if signature_commit:
                chain_commit = SignedModelHashChainCommit.model_validate(status_dict)
            else:
                chain_commit = (
                    ValidatorChainCommit.model_validate(status_dict)
                    if "miner_seed" in status_dict or "s" in status_dict  # TODO: fix this check
                    else MinerChainCommit.model_validate(status_dict)
                )

        except Exception:
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
    axon.serve(netuid=config.chain.netuid, subtensor=subtensor)


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
