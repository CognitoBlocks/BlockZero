from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import bittensor
import requests
from pydantic import BaseModel

from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import (
    MinerChainCommit,
    ValidatorChainCommit,
    WorkerChainCommit,
    get_chain_commits,
    serve_axon,
)
from mycelia.shared.config import MinerConfig, ValidatorConfig, WorkerConfig
from mycelia.shared.helper import h256_int, parse_dynamic_filename
from mycelia.validator.evaluator import MinerEvalJob

configure_logging()
logger = structlog.get_logger(__name__)

class PhaseResponseLite(BaseModel):
    phase_name: str
    phase_start_block: int
    phase_end_block: int
class PhaseResponse(BaseModel):
    block: int
    cycle_length: int  # how long is one cycle
    cycle_index: int  # which cycle are we in
    cycle_block_index: int  # how far in block are we into a cycle
    phase_name: str  # what is the name of the current phase
    phase_index: int  # what is the id of the phase
    phase_start_block: int  # the start block of the phase
    phase_end_block: int  # the end block of the phase
    blocks_into_phase: int  # how far in block are we in the current phase
    blocks_remaining_in_phase: int  # how manuy block left in the phase


@dataclass
class PhaseNames:
    distribute: str = "Distribute"  # miner download from validator
    train: str = "Train"  # miner trian
    miner_commit_1: str = "MinerCommit1"  # miner commit signed_model_hash and  vlaidators commit seed
    miner_commit_2: str = "MinerCommit2"  # miner commit model_hash
    submission: str = "Submission"  # submit model
    validate: str = "Validate"  # validator validate
    merge: str = "Merge"  # validator merge
    validator_commit_1: str = "ValidatorCommit1"  # validator commit signed_model_hash
    validator_commit_2: str = "ValidatorCommit2"  # validator commit model_hash


def wait_till(config: MinerConfig, phase_name: PhaseNames, poll_fallback_block: int = 3):
    ready = False
    
    first_print = True
    while not ready:
        ready, blocks_till, phase_response = should_act(config, phase_name, retry_blocks=poll_fallback_block)
        if ready is False and blocks_till > 0:
            sleep_sec = min(blocks_till, max(poll_fallback_block, blocks_till * 0.9)) * 12

            check_time = datetime.now() + timedelta(seconds=sleep_sec)
            check_time_str = check_time.strftime("%H:%M:%S")

            expect_time = datetime.now() + timedelta(seconds=blocks_till * 12)
            expect_time_str = expect_time.strftime("%H:%M:%S")

            if first_print: logger.info(f"<{phase_name}> to begin in {blocks_till} blocks, at {expect_time_str}")
            first_print = False
            time.sleep(sleep_sec)

    logger.info(f"<{phase_name}> has started, {phase_response.blocks_into_phase} blocks passed, {phase_response.blocks_remaining_in_phase} blocks left in phase.")
    return phase_response


def check_phase_expired(subtensor: bittensor.Subtensor, phase_response: PhaseResponse) -> bool:
    current_block = subtensor.block
    blocks_remaining = phase_response.phase_end_block - current_block
    if current_block > phase_response.phase_end_block:
        logger.warning(
            f"<{phase_response.phase_name}> phase couldnt complete on time",
            current_block=current_block,
            phase_end_block=phase_response.phase_end_block,
            diff=blocks_remaining,
        )
        return True
    
    if blocks_remaining >= 0:
        logger.info(
            f"<{phase_response.phase_name}> phase completed on time",
            current_block=current_block,
            phase_end_block=phase_response.phase_end_block,
            blocks_remaining=blocks_remaining,
        )
    
    return False


def should_act(config: MinerConfig, phase_name: PhaseNames, retry_blocks: int) -> tuple[bool, int, int]:
    phase_response: PhaseResponse | None = get_phase_from_api(config)

    if phase_response is None:
        should_submit = False
    else:
        should_submit = phase_response.phase_name == phase_name

    blocks_till_next_phase = get_blocks_until_next_phase_from_api(config)

    if blocks_till_next_phase is None:
        blocks_till = retry_blocks
    else:
        blocks_till = blocks_till_next_phase[phase_name][2]

    return should_submit, blocks_till, phase_response


def search_model_submission_destination(
    wallet: bittensor.Wallet, config: MinerConfig, subtensor: bittensor.Subtensor
) -> bittensor.Axon:
    
    validator_miner_assignment = get_validator_miner_assignment(config, subtensor)

    assigned_validator_hotkey = None
    for validator, miners in validator_miner_assignment.items():
        if wallet.hotkey.ss58_address in miners:
            assigned_validator_hotkey = validator
            break

    if assigned_validator_hotkey is None:
        return None

    metagraph = subtensor.metagraph(netuid=config.chain.netuid)
    uid = metagraph.hotkeys.index(assigned_validator_hotkey)

    logger.info('assigned_validator_hotkey', assigned_validator_hotkey = assigned_validator_hotkey, uid = uid)

    return metagraph.axons[uid]


def assign_miners_to_validators(
    validators: dict[str, Any],  # {validator_id: seed}
    miners: list[str],
) -> dict[str, list[str]]:
    n_v = len(validators)
    n_m = len(miners)

    if n_v == 0:
        raise ValueError("No validators provided")

    # --- 0) Combined seed (hash of all validator seeds)
    combined_seed_str = "".join(str(validators[v]) for v in sorted(validators.keys()))
    combined_seed = hashlib.sha256(combined_seed_str.encode()).hexdigest()

    # --- 1) Balanced capacities
    base = n_m // n_v
    rem = n_m % n_v
    v_ids = list(validators.keys())

    ranked_for_bonus = sorted(
        v_ids,
        key=lambda vid: h256_int("cap_bonus", validators[vid], combined_seed),
        reverse=True,
    )
    capacities = {vid: base for vid in v_ids}
    for vid in ranked_for_bonus[:rem]:
        capacities[vid] += 1

    # --- 2) Deterministic miner order seeded by combined validator seed
    miners_sorted = sorted(miners, key=lambda mid: h256_int("miner_order", mid, combined_seed))

    # --- 3) Preference per miner (based on validator seed + combined seed)
    def validator_prefs(mid: str) -> list[str]:
        return sorted(
            v_ids,
            key=lambda vid: h256_int("preference", mid, validators[vid], combined_seed),
            reverse=True,
        )

    # --- 4) Assign miners evenly, respecting capacities
    assignment: dict[str, list[str]] = {vid: [] for vid in v_ids}
    for mid in miners_sorted:
        prefs = validator_prefs(mid)
        for vid in prefs:
            if capacities[vid] > 0:
                assignment[vid].append(mid)
                capacities[vid] -= 1
                break
        else:
            # Should never happen if capacities sum == len(miners)
            assignment[prefs[-1]].append(mid)

    return assignment


def get_combined_validator_seed(config: WorkerConfig, subtensor: bittensor.Subtensor) -> str:
    """
    Deterministically combine validator seeds into a single hex string.

    We sort validator IDs so the result is independent of dict iteration order.
    """
    commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)

    validator_seeds = get_validator_seed_from_commit(config, commits)
    if not validator_seeds:
        raise ValueError("No validators provided")

    combined_seed_str = "".join(str(validator_seeds[v]) for v in sorted(validator_seeds.keys()))
    return hashlib.sha256(combined_seed_str.encode()).hexdigest()


def get_validator_miner_assignment(config: WorkerConfig, subtensor: bittensor.Subtensor):
    commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)
    validator_seeds = get_validator_seed_from_commit(config, commits)
    miners = get_miners_from_commit(config, commits)

    validator_miner_assignment = assign_miners_to_validators(validator_seeds, miners)
    logger.info("global validator_miner_assignment", validator_miner_assignment)
    return validator_miner_assignment


def get_validator_seed_from_commit(config, commits):
    validator_seeds: dict[str, int] = {
        neuron.hotkey: commit.miner_seed
        for commit, neuron in commits
        if isinstance(commit, ValidatorChainCommit)
        and getattr(commit, "expert_group", None) == config.task.exp.group_id
    }
    return validator_seeds


def get_miners_from_commit(config, commits):
    miners: list[str] = [
        neuron.hotkey
        for commit, neuron in commits
        if isinstance(commit, MinerChainCommit) and getattr(commit, "expert_group", None) == config.task.exp.group_id
    ]

    return miners


def get_phase_from_api(config: WorkerConfig) -> PhaseResponse | None:
    """
    Determine current phase based on block schedule.

    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    base_url = f"http://{config.cycle.owner_ip}:{config.cycle.owner_port}"
    url = f"{base_url}/get_phase"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return PhaseResponse(**resp.json())

    except requests.exceptions.HTTPError as e:
        r = e.response
        status = r.status_code if r is not None else None
        body_snippet = r.text[:500] if (r is not None and r.text) else ""
        logger.exception("HTTPError calling %s (status=%s). Body (first 500 chars): %r", url, status, body_snippet)
        return None

    except requests.exceptions.RequestException as e:
        logger.exception("RequestException calling %s: %s", url, e)
        return None

    except (ValueError, TypeError) as e:
        # ValueError: JSON decode problems
        # TypeError: PhaseResponse(**...) got unexpected/missing fields
        logger.exception("Bad response payload from %s: %s", url, e)
        return None


def get_blocks_until_next_phase_from_api(config: WorkerConfig) -> dict[str, tuple[int, int, int]] | None:
    """
    Determine current phase based on block schedule.

    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    base_url = f"http://{config.cycle.owner_ip}:{config.cycle.owner_port}"
    url = f"{base_url}/blocks_until_next_phase"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.HTTPError as e:
        # HTTP error responses (4xx/5xx)
        r = e.response
        status = r.status_code if r is not None else None
        body_snippet = r.text[:500] if (r is not None and r.text) else ""
        logger.exception("HTTPError calling %s (status=%s). Body (first 500 chars): %r", url, status, body_snippet)
        return None

    except requests.exceptions.RequestException as e:
        # Connection errors, timeouts, DNS, etc.
        logger.exception("RequestException calling %s: %s", url, e)
        return None

    except ValueError as e:
        # JSON decoding failed
        logger.exception("Invalid JSON from %s: %s", url, e)
        return None


def get_blocks_from_previous_phase_from_api(config: WorkerConfig) -> PhaseResponse | None:
    """
    Determine current phase based on block schedule.

    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    base_url = f"http://{config.cycle.owner_ip}:{config.cycle.owner_port}"
    url = f"{base_url}/previous_phase_blocks"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.HTTPError as e:
        r = e.response
        status = r.status_code if r is not None else None
        body_snippet = r.text[:500] if (r is not None and r.text) else ""
        logger.exception("HTTPError calling %s (status=%s). Body (first 500 chars): %r", url, status, body_snippet)
        return None

    except requests.exceptions.RequestException as e:
        logger.exception("RequestException calling %s: %s", url, e)
        return None

    except ValueError as e:
        # JSON decoding failed
        logger.exception("Invalid JSON from %s: %s", url, e)
        return None


def load_submission_files(folder: str = "miner_submission"):
    """
    Scans a folder for .pt files and returns:
        { filename: {parsed key/values} }
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    files_dict = {}
    for file_name in folder_path.glob("*.pt"):
        if file_name.name.startswith("._tmp"):
            continue
        meta = parse_dynamic_filename(file_name.name)
        if meta is None:
            continue
        files_dict[file_name.name] = meta

    return files_dict


def gather_validation_job(config: ValidatorConfig, subtensor: bittensor.Subtensor, step: int) -> list[MinerEvalJob]:
    validator_miner_assignment = get_validator_miner_assignment(config, subtensor)

    miner_assignment = validator_miner_assignment.get(config.chain.hotkey_ss58, [])

    logger.info("assigned_miners", miner_assignment=miner_assignment)

    miner_submission_files = load_submission_files(str(config.ckpt.miner_submission_path))
    previous_phase_range = get_blocks_from_previous_phase_from_api(config)[PhaseNames.submission]

    hotkeys = subtensor.metagraph(netuid=config.chain.netuid).hotkeys
    miner_jobs = []
    qualifying_hotkeys: set[str] = set()
    unexpected_submissions = []
    for file_name, submission_meta in miner_submission_files.items():
        is_assigned = submission_meta["hotkey"] in miner_assignment
        in_previous_phase = previous_phase_range[0] <= submission_meta["block"] <= previous_phase_range[1]
        if is_assigned and in_previous_phase:
            logger.info("Found qualifying submission file", file_name=file_name, submission_meta=submission_meta)
            qualifying_hotkeys.add(submission_meta["hotkey"])
            miner_jobs.append(
                MinerEvalJob(
                    uid=hotkeys.index(submission_meta["hotkey"]),
                    hotkey=submission_meta["hotkey"],
                    model_path=config.ckpt.miner_submission_path / file_name,
                    step=step,
                )
            )
        else:
            if not in_previous_phase:
                reason = f"out_of_phase_range block {previous_phase_range[0]} - {previous_phase_range[1]}"
            
            elif not is_assigned:
                reason = "in phase but unassigned miner"

            else:
                reason = "unknown"

            unexpected_submissions.append(
                {
                    "file_name": file_name,
                    "hotkey": submission_meta["hotkey"],
                    "block": submission_meta["block"],
                    "reason": reason,
                }
            )

    missing_hotkeys = [hotkey for hotkey in miner_assignment if hotkey not in qualifying_hotkeys]
    if missing_hotkeys:
        logger.warning(
            "Missing miner submissions",
            missing_hotkeys=missing_hotkeys,
            assigned_count=len(miner_assignment),
            received_count=len(qualifying_hotkeys),
        )
    if unexpected_submissions:
        logger.warning(
            "Unexpected miner submissions",
            unexpected_count=len(unexpected_submissions),
            unexpected_submissions=unexpected_submissions,
        )

    return miner_jobs
