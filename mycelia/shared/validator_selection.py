"""
Robust Validator Selection with Blacklist and Hash Verification

This module implements:
1. ValidatorBlacklist: Tracks validators that have served mismatched models
2. ValidatorSelector: Canonical, deterministic validator selection
3. VerifiedModelDownloader: Downloads models with hash verification and retry logic

All logic is deterministic and compatible with decentralized consensus.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import bittensor
from pydantic import BaseModel, ConfigDict, Field

from mycelia.shared.app_logging import structlog
from mycelia.shared.checkpoint_helper import compile_full_state_dict_from_path
from mycelia.shared.checkpoints import ChainCheckpoint, ChainCheckpoints
from mycelia.shared.client import download_model
from mycelia.shared.helper import get_model_hash, h256_int

logger = structlog.get_logger(__name__)


class ValidatorFailureReason(Enum):
    """
    Reasons a validator may be blacklisted or deprioritized.
    Each reason has a different severity affecting how long the validator is excluded.
    """
    HASH_MISMATCH = "hash_mismatch"           # Model hash did not match reported hash
    CONNECTION_FAILED = "connection_failed"   # Failed to connect/respond
    NO_MODEL_HASH = "no_model_hash"           # Did not provide valid model hash
    DOWNLOAD_FAILED = "download_failed"       # Download failed (network/IO error)
    INVALID_RESPONSE = "invalid_response"     # Response was malformed


class DownloadErrorType(Enum):
    """
    Explicit error types for download failures.
    Avoids brittle string parsing for error classification.
    """
    HASH_MISMATCH = "hash_mismatch"
    NETWORK_ERROR = "network_error"
    HASH_COMPUTATION_FAILED = "hash_computation_failed"
    INVALID_EXPERT_GROUP = "invalid_expert_group"
    NO_VALIDATORS = "no_validators"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"


@dataclass
class ValidatorFailure:
    """
    Record of a single validator failure.
    Stored to enable deterministic blacklist decisions.
    """
    hotkey: str
    reason: ValidatorFailureReason
    block: int        # Block number when failure occurred (deterministic)
    details: str = "" # Additional context for debugging


class ValidatorBlacklist(BaseModel):
    """
    Tracks validators that have failed hash verification or other checks.

    Design decisions:
    - Uses block-based expiry for determinism across nodes (time-based expiry is non-deterministic)
    - Failures are stored by hotkey for O(1) lookup
    - Priority downgrade accumulates with repeated failures
    - Non-destructive: validators can recover after expiry blocks pass

    Blacklist behavior:
    - HASH_MISMATCH: Most severe, 100 block exclusion per failure
    - CONNECTION_FAILED: Mild, 10 block exclusion
    - DOWNLOAD_FAILED: Moderate, 25 block exclusion
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Map of hotkey -> list of failures
    # Use Field with default_factory to avoid shared mutable default
    failures: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    # Block penalty per failure reason (how many blocks to exclude per failure)
    BLOCK_PENALTIES: dict[str, int] = {
        ValidatorFailureReason.HASH_MISMATCH.value: 100,      # Severe: potential Byzantine behavior
        ValidatorFailureReason.CONNECTION_FAILED.value: 10,   # Transient network issues
        ValidatorFailureReason.NO_MODEL_HASH.value: 50,       # Missing data, possibly malicious
        ValidatorFailureReason.DOWNLOAD_FAILED.value: 25,     # Network/IO issues
        ValidatorFailureReason.INVALID_RESPONSE.value: 50,    # Malformed response
    }

    # Maximum accumulated penalty blocks (cap to prevent permanent bans)
    MAX_PENALTY_BLOCKS: int = 500

    def record_failure(
        self,
        hotkey: str,
        reason: ValidatorFailureReason,
        block: int,
        details: str = "",
    ) -> None:
        """
        Record a validator failure for blacklist consideration.

        Args:
            hotkey: Validator's hotkey (SS58 address)
            reason: Why the validator failed
            block: Current block number for deterministic expiry
            details: Additional context for debugging
        """
        if hotkey not in self.failures:
            self.failures[hotkey] = []

        # Only store deterministic fields (no timestamp for reproducibility)
        failure_record = {
            "reason": reason.value,
            "block": block,
            "details": details,
        }

        self.failures[hotkey].append(failure_record)

        logger.warning(
            "Validator failure recorded",
            hotkey=hotkey[:16] + "...",
            reason=reason.value,
            block=block,
            details=details,
            total_failures=len(self.failures[hotkey]),
        )

    def get_penalty_blocks(self, hotkey: str, current_block: int) -> int:
        """
        Calculate how many blocks a validator should be excluded for.

        Returns 0 if the validator is not blacklisted or penalty has expired.
        Uses block-based calculation for determinism.
        """
        if hotkey not in self.failures:
            return 0

        total_penalty = 0
        for failure in self.failures[hotkey]:
            failure_block = failure["block"]
            reason = failure["reason"]
            penalty = self.BLOCK_PENALTIES.get(reason, 10)

            # Calculate remaining penalty: penalty - (current_block - failure_block)
            blocks_since_failure = current_block - failure_block
            remaining = penalty - blocks_since_failure

            if remaining > 0:
                total_penalty += remaining

        # Cap the penalty to prevent permanent exclusion
        return min(total_penalty, self.MAX_PENALTY_BLOCKS)

    def is_blacklisted(self, hotkey: str, current_block: int) -> bool:
        """
        Check if a validator is currently blacklisted.

        A validator is blacklisted if their accumulated penalty > 0.
        """
        return self.get_penalty_blocks(hotkey, current_block) > 0

    def get_priority_score(self, hotkey: str, current_block: int) -> int:
        """
        Get a priority score for sorting validators.

        Higher score = higher priority (should be selected first).
        Validators with no failures get score 0 (highest).
        Validators with failures get negative scores based on penalty.
        """
        penalty = self.get_penalty_blocks(hotkey, current_block)
        return -penalty  # Negative so higher penalty = lower priority

    def cleanup_expired(self, current_block: int) -> int:
        """
        Remove fully expired failures to prevent unbounded growth.

        Returns number of failures removed.
        """
        removed = 0
        for hotkey in list(self.failures.keys()):
            # Keep only failures that still contribute penalty
            original_count = len(self.failures[hotkey])
            self.failures[hotkey] = [
                f for f in self.failures[hotkey]
                if (current_block - f["block"]) < self.BLOCK_PENALTIES.get(f["reason"], 10)
            ]
            removed += original_count - len(self.failures[hotkey])

            # Remove empty entries
            if not self.failures[hotkey]:
                del self.failures[hotkey]

        if removed > 0:
            logger.info("Cleaned up expired blacklist entries", removed=removed)

        return removed

    def get_failure_count(self, hotkey: str) -> int:
        """Get total number of recorded failures for a validator."""
        return len(self.failures.get(hotkey, []))

    def get_hash_mismatch_count(self, hotkey: str) -> int:
        """Get number of hash mismatch failures (most severe)."""
        if hotkey not in self.failures:
            return 0
        return sum(
            1 for f in self.failures[hotkey]
            if f["reason"] == ValidatorFailureReason.HASH_MISMATCH.value
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize blacklist to dict for persistence.
        Only contains deterministic fields.
        """
        return {"failures": self.failures}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidatorBlacklist":
        """
        Deserialize blacklist from dict (e.g., loaded from disk).
        """
        return cls(failures=data.get("failures", {}))


@dataclass
class ValidatorCandidate:
    """
    A validator candidate for model download with associated metadata.
    """
    checkpoint: ChainCheckpoint
    priority_score: int = 0      # From blacklist (0 = clean, negative = has failures)
    hash_group_size: int = 0     # How many validators report same hash (consensus)

    @property
    def hotkey(self) -> str:
        return self.checkpoint.hotkey

    @property
    def model_hash(self) -> str:
        return self.checkpoint.model_hash

    @property
    def global_ver(self) -> int:
        return self.checkpoint.global_ver or 0

    def selection_key(self) -> tuple:
        """
        Deterministic key for sorting validators.

        Selection criteria (in order of priority):
        1. Higher global version (most up-to-date model)
        2. Larger hash consensus group (more validators agree)
        3. Higher priority score (fewer failures)
        4. Deterministic tiebreaker using hotkey hash
        """
        return (
            self.global_ver,
            self.hash_group_size,
            self.priority_score,
            # Deterministic tiebreaker: hash of hotkey for consistent ordering
            h256_int("validator_tiebreak", self.hotkey),
        )


class ValidatorSelector:
    """
    Implements canonical, deterministic validator selection.

    Selection algorithm:
    1. Filter out inactive validators (no response, no valid hash)
    2. Filter out blacklisted validators
    3. Group remaining validators by their reported model hash
    4. Select the hash with highest global version
    5. Among validators with that hash, choose deterministically
       (highest stake, lowest penalty, then hash-based tiebreaker)
    """

    def __init__(self, blacklist: ValidatorBlacklist | None = None):
        self.blacklist = blacklist or ValidatorBlacklist()

    def filter_active_validators(
        self,
        checkpoints: ChainCheckpoints,
        current_block: int,
    ) -> list[ChainCheckpoint]:
        """
        Filter to only active, verified validators.

        A validator is considered INACTIVE if:
        - It has no model_hash (did not commit to chain)
        - It has no signed_model_hash (signature missing)
        - It is expired or not responding
        - Its checkpoint validation failed

        Returns list of active checkpoints, excluding inactive ones.
        """
        active = []

        for ckpt in checkpoints.checkpoints:
            # Skip: No model hash provided
            if not ckpt.model_hash:
                logger.debug(
                    "Skipping validator: no model_hash",
                    hotkey=ckpt.hotkey[:16] + "..." if ckpt.hotkey else "unknown",
                    uid=ckpt.uid,
                )
                continue

            # Skip: No signed hash (signature required for verification)
            if not ckpt.signed_model_hash:
                logger.debug(
                    "Skipping validator: no signed_model_hash",
                    hotkey=ckpt.hotkey[:16] + "..." if ckpt.hotkey else "unknown",
                    uid=ckpt.uid,
                )
                continue

            # Skip: Checkpoint is expired
            if ckpt.expired():
                logger.debug(
                    "Skipping validator: checkpoint expired",
                    hotkey=ckpt.hotkey[:16] + "..." if ckpt.hotkey else "unknown",
                    uid=ckpt.uid,
                )
                continue

            # Skip: Missing network info (cannot download from)
            if not ckpt.ip or not ckpt.port:
                logger.debug(
                    "Skipping validator: missing ip/port",
                    hotkey=ckpt.hotkey[:16] + "..." if ckpt.hotkey else "unknown",
                    uid=ckpt.uid,
                )
                continue

            active.append(ckpt)

        logger.info(
            "Filtered active validators",
            total=len(checkpoints.checkpoints),
            active=len(active),
            inactive=len(checkpoints.checkpoints) - len(active),
        )

        return active

    def filter_non_blacklisted(
        self,
        checkpoints: list[ChainCheckpoint],
        current_block: int,
        exclude_hotkeys: set[str] | None = None,
    ) -> list[ChainCheckpoint]:
        """
        Filter out blacklisted validators.

        Args:
            checkpoints: List of active checkpoints
            current_block: Current block for penalty calculation
            exclude_hotkeys: Additional hotkeys to exclude (e.g., already tried)

        Returns list of non-blacklisted checkpoints.
        """
        exclude_hotkeys = exclude_hotkeys or set()
        non_blacklisted = []

        for ckpt in checkpoints:
            # Skip: Explicitly excluded (already tried and failed this round)
            if ckpt.hotkey in exclude_hotkeys:
                logger.debug(
                    "Skipping validator: explicitly excluded",
                    hotkey=ckpt.hotkey[:16] + "..." if ckpt.hotkey else "unknown",
                )
                continue

            # Skip: Currently blacklisted
            if self.blacklist.is_blacklisted(ckpt.hotkey, current_block):
                penalty = self.blacklist.get_penalty_blocks(ckpt.hotkey, current_block)
                logger.debug(
                    "Skipping validator: blacklisted",
                    hotkey=ckpt.hotkey[:16] + "..." if ckpt.hotkey else "unknown",
                    penalty_blocks=penalty,
                )
                continue

            non_blacklisted.append(ckpt)

        logger.info(
            "Filtered non-blacklisted validators",
            before=len(checkpoints),
            after=len(non_blacklisted),
            blacklisted=len(checkpoints) - len(non_blacklisted),
        )

        return non_blacklisted

    def group_by_hash(
        self,
        checkpoints: list[ChainCheckpoint],
    ) -> dict[str, list[ChainCheckpoint]]:
        """
        Group validators by the model hash they report.

        Returns dict mapping model_hash -> list of checkpoints reporting that hash.
        """
        groups: dict[str, list[ChainCheckpoint]] = {}

        for ckpt in checkpoints:
            if not ckpt.model_hash:
                continue

            if ckpt.model_hash not in groups:
                groups[ckpt.model_hash] = []
            groups[ckpt.model_hash].append(ckpt)

        logger.debug(
            "Grouped validators by hash",
            num_hashes=len(groups),
            group_sizes={h[:16]: len(v) for h, v in groups.items()},
        )

        return groups

    def select_canonical_hash(
        self,
        hash_groups: dict[str, list[ChainCheckpoint]],
    ) -> str | None:
        """
        Select the canonical model hash based on:
        1. Highest global version (most up-to-date)
        2. If tied, largest consensus group (most validators agree)
        3. If still tied, deterministic hash-based selection

        Returns the selected model_hash or None if no valid groups.
        """
        if not hash_groups:
            return None

        def hash_priority(model_hash: str) -> tuple:
            checkpoints = hash_groups[model_hash]

            # Get highest global version in this group
            max_version = max(ckpt.global_ver or 0 for ckpt in checkpoints)

            # Group size (consensus strength)
            group_size = len(checkpoints)

            # Deterministic tiebreaker
            tiebreaker = h256_int("hash_selection", model_hash)

            return (max_version, group_size, tiebreaker)

        # Select hash with highest priority
        selected_hash = max(hash_groups.keys(), key=hash_priority)

        priority = hash_priority(selected_hash)
        logger.info(
            "Selected canonical hash",
            hash=selected_hash[:24] + "...",
            global_ver=priority[0],
            consensus_size=priority[1],
        )

        return selected_hash

    def select_validator(
        self,
        checkpoints: list[ChainCheckpoint],
        current_block: int,
    ) -> ChainCheckpoint | None:
        """
        Select a single validator deterministically from a list.

        Selection criteria:
        1. Lower penalty score (fewer failures)
        2. Deterministic tiebreaker using hotkey hash

        Returns the selected checkpoint or None if list is empty.
        """
        if not checkpoints:
            return None

        def selection_key(ckpt: ChainCheckpoint) -> tuple:
            # Lower penalty = higher priority (negate for sorting)
            penalty = self.blacklist.get_priority_score(ckpt.hotkey, current_block)

            # Deterministic tiebreaker
            tiebreaker = h256_int("validator_select", ckpt.hotkey, current_block)

            return (penalty, tiebreaker)

        # Select validator with highest priority (max because penalty is already negative)
        selected = max(checkpoints, key=selection_key)

        logger.info(
            "Selected validator for download",
            hotkey=selected.hotkey[:16] + "...",
            uid=selected.uid,
            global_ver=selected.global_ver,
            ip=selected.ip,
        )

        return selected

    def select_canonical_validator(
        self,
        chain_checkpoints: ChainCheckpoints,
        current_block: int,
        exclude_hotkeys: set[str] | None = None,
        expert_group_id: int | None = None,
    ) -> tuple[ChainCheckpoint | None, str | None]:
        """
        Main entry point: Select the canonical validator for model download.

        Algorithm:
        1. Filter to active validators only
        2. Filter out blacklisted validators
        3. Group by model hash
        4. Select canonical hash (highest version)
        5. From validators with canonical hash, select one deterministically

        Args:
            chain_checkpoints: All chain checkpoints from previous phase
            current_block: Current block number
            exclude_hotkeys: Hotkeys to skip (e.g., already tried this round)
            expert_group_id: Filter to specific expert group if provided

        Returns:
            Tuple of (selected_checkpoint, canonical_hash) or (None, None) if none available
        """
        # Step 1: Filter to active validators
        active = self.filter_active_validators(chain_checkpoints, current_block)
        if not active:
            logger.warning("No active validators available")
            return None, None

        # Step 2: Filter out blacklisted validators
        eligible = self.filter_non_blacklisted(active, current_block, exclude_hotkeys)
        if not eligible:
            logger.warning(
                "All active validators are blacklisted or excluded",
                active_count=len(active),
                exclude_count=len(exclude_hotkeys or set()),
            )
            return None, None

        # Step 3: Group by hash
        hash_groups = self.group_by_hash(eligible)
        if not hash_groups:
            logger.warning("No validators with valid model hash")
            return None, None

        # Step 4: Select canonical hash
        canonical_hash = self.select_canonical_hash(hash_groups)
        if not canonical_hash:
            logger.warning("Could not determine canonical hash")
            return None, None

        # Step 5: Select validator from canonical hash group
        candidates = hash_groups[canonical_hash]
        selected = self.select_validator(candidates, current_block)

        return selected, canonical_hash


@dataclass
class DownloadResult:
    """Result of a model download attempt."""
    success: bool
    checkpoint: ChainCheckpoint | None = None
    local_path: Path | None = None
    expected_hash: str | None = None
    actual_hash: str | None = None
    error_type: DownloadErrorType | None = None
    error_message: str | None = None

    @property
    def hash_matched(self) -> bool:
        if not self.success:
            return False
        return self.expected_hash == self.actual_hash


class VerifiedModelDownloader:
    """
    Downloads models with hash verification and automatic retry on mismatch.

    Flow:
    1. Select canonical validator
    2. Download model
    3. Compute local hash (over ALL expert groups for full verification)
    4. Compare to reported hash
    5. If mismatch: blacklist validator, retry with different validator
    6. Continue until success or no validators remain

    Note: This downloader is designed for off-chain miner use.
    It uses time.sleep() for backoff which is appropriate for miners
    but should not be used in consensus-critical or async contexts.
    """

    def __init__(
        self,
        selector: ValidatorSelector,
        max_retries: int = 5,
        base_delay_s: float = 2.0,
        # Dependency injection for testability
        download_fn: Callable | None = None,
        hash_fn: Callable | None = None,
    ):
        self.selector = selector
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        # Allow injection for unit testing
        self._download_fn = download_fn or download_model
        self._hash_fn = hash_fn or get_model_hash

    def compute_local_hash(
        self,
        path: Path,
        expert_groups: list[int | str],
    ) -> str | None:
        """
        Compute hash of downloaded model file(s).

        Hashes ALL expert groups together to ensure full verification.
        A validator could serve correct shared model but incorrect expert group,
        so we must hash everything.

        Uses same hashing algorithm as chain commits for consistency.
        """
        try:
            # Convert expert_groups to format expected by compile_full_state_dict_from_path
            # Filter to only int groups for now (shared is handled separately)
            int_groups = [g for g in expert_groups if isinstance(g, int)]

            # Compile state dict from all expert groups for complete hash
            state = compile_full_state_dict_from_path(path, expert_groups=int_groups if int_groups else None)
            return self._hash_fn(state, hex=True)
        except Exception as e:
            logger.error("Failed to compute model hash", path=str(path), error=str(e))
            return None

    def download_and_verify(
        self,
        chain_checkpoints: ChainCheckpoints,
        config: Any,
        wallet: bittensor.Wallet,
        current_block: int,
        expert_group_ids: list[int | str],
        out_folder: Path,
    ) -> DownloadResult:
        """
        Download model with verification and retry logic.

        This is the main entry point for robust model download.

        Args:
            chain_checkpoints: Available validator checkpoints
            config: Worker config
            wallet: Wallet for signing requests
            current_block: Current block number
            expert_group_ids: Expert groups to download
            out_folder: Where to save downloaded model

        Returns:
            DownloadResult with success status and details
        """
        tried_validators: set[str] = set()
        attempt = 0

        while attempt < self.max_retries:
            attempt += 1

            # Select validator (excluding already-tried ones)
            selected, canonical_hash = self.selector.select_canonical_validator(
                chain_checkpoints=chain_checkpoints,
                current_block=current_block,
                exclude_hotkeys=tried_validators,
            )

            if not selected:
                logger.error(
                    "No eligible validators remaining",
                    attempt=attempt,
                    tried_count=len(tried_validators),
                )
                return DownloadResult(
                    success=False,
                    error_type=DownloadErrorType.NO_VALIDATORS,
                    error_message="No eligible validators remaining after exclusions",
                )

            tried_validators.add(selected.hotkey)

            logger.info(
                "Attempting download from validator",
                attempt=attempt,
                hotkey=selected.hotkey[:16] + "...",
                uid=selected.uid,
                expected_hash=canonical_hash[:24] + "..." if canonical_hash else None,
            )

            # Attempt download
            try:
                result = self._attempt_download(
                    checkpoint=selected,
                    config=config,
                    wallet=wallet,
                    current_block=current_block,
                    expert_group_ids=expert_group_ids,
                    out_folder=out_folder,
                    expected_hash=canonical_hash,
                )

                if result.success:
                    return result

                # Handle failure based on explicit error type
                if result.error_type == DownloadErrorType.HASH_MISMATCH:
                    # Hash mismatch: blacklist with severe penalty
                    self.selector.blacklist.record_failure(
                        hotkey=selected.hotkey,
                        reason=ValidatorFailureReason.HASH_MISMATCH,
                        block=current_block,
                        details=f"Expected {canonical_hash[:16] if canonical_hash else 'None'}, got {result.actual_hash[:16] if result.actual_hash else 'None'}",
                    )
                elif result.error_type == DownloadErrorType.HASH_COMPUTATION_FAILED:
                    # Could not compute hash - might be corrupt download
                    self.selector.blacklist.record_failure(
                        hotkey=selected.hotkey,
                        reason=ValidatorFailureReason.INVALID_RESPONSE,
                        block=current_block,
                        details=result.error_message or "Hash computation failed",
                    )
                else:
                    # Other failure: record as download failed
                    self.selector.blacklist.record_failure(
                        hotkey=selected.hotkey,
                        reason=ValidatorFailureReason.DOWNLOAD_FAILED,
                        block=current_block,
                        details=result.error_message or "Unknown error",
                    )

            except Exception as e:
                logger.warning(
                    "Download attempt failed with exception",
                    attempt=attempt,
                    error=str(e),
                )
                self.selector.blacklist.record_failure(
                    hotkey=selected.hotkey,
                    reason=ValidatorFailureReason.CONNECTION_FAILED,
                    block=current_block,
                    details=str(e),
                )

            # Exponential backoff before retry
            # Note: This uses time.sleep() which is appropriate for miner-side code
            # but should not be used in consensus-critical paths
            if attempt < self.max_retries:
                delay = self.base_delay_s * (2 ** (attempt - 1))
                logger.info(
                    "Retrying after delay",
                    delay_s=delay,
                    attempt=attempt,
                    max_retries=self.max_retries,
                )
                time.sleep(delay)

        return DownloadResult(
            success=False,
            error_type=DownloadErrorType.MAX_RETRIES_EXCEEDED,
            error_message=f"All {self.max_retries} download attempts failed",
        )

    def _attempt_download(
        self,
        checkpoint: ChainCheckpoint,
        config: Any,
        wallet: bittensor.Wallet,
        current_block: int,
        expert_group_ids: list[int | str],
        out_folder: Path,
        expected_hash: str | None,
    ) -> DownloadResult:
        """
        Single download attempt with hash verification.

        Separates:
        - Network IO (download)
        - Filesystem write (via download_model)
        - Hash computation and verification (over ALL expert groups)

        Note: We do NOT mutate checkpoint.path to avoid coupling issues.
        The path is returned via DownloadResult only.
        """
        # Construct download URL
        protocol = getattr(getattr(config, "miner", object()), "protocol", "http")
        url = f"{protocol}://{checkpoint.ip}:{checkpoint.port}/get-checkpoint"

        # Prepare output directory
        version_folder = out_folder / (
            f"uid_{checkpoint.uid}_hotkey_{checkpoint.hotkey}_globalver_{checkpoint.global_ver}"
        )
        version_folder.mkdir(parents=True, exist_ok=True)

        # Download each expert group
        for expert_group_id in expert_group_ids:
            if isinstance(expert_group_id, int):
                out_file = f"model_expgroup_{expert_group_id}.pt"
            elif expert_group_id == "shared":
                out_file = "model_shared.pt"
            else:
                logger.warning("Invalid expert_group_id, skipping", expert_group_id=expert_group_id)
                return DownloadResult(
                    success=False,
                    checkpoint=checkpoint,
                    error_type=DownloadErrorType.INVALID_EXPERT_GROUP,
                    error_message=f"Invalid expert_group_id: {expert_group_id}",
                )

            out_path = version_folder / out_file

            try:
                # Network IO + Filesystem write
                self._download_fn(
                    url=url,
                    my_hotkey=wallet.hotkey,
                    target_hotkey_ss58=checkpoint.hotkey,
                    block=current_block,
                    expert_group_id=expert_group_id,
                    token=getattr(config.cycle, "token", ""),
                    out_dir=out_path,
                )
            except Exception as e:
                return DownloadResult(
                    success=False,
                    checkpoint=checkpoint,
                    error_type=DownloadErrorType.NETWORK_ERROR,
                    error_message=f"Download failed: {str(e)}",
                )

        # Hash verification - compute hash over ALL downloaded expert groups
        # This prevents attacks where validator serves correct shared but wrong expert
        actual_hash = self.compute_local_hash(version_folder, expert_groups=expert_group_ids)

        if actual_hash is None:
            return DownloadResult(
                success=False,
                checkpoint=checkpoint,
                local_path=version_folder,
                expected_hash=expected_hash,
                error_type=DownloadErrorType.HASH_COMPUTATION_FAILED,
                error_message="Failed to compute hash of downloaded model",
            )

        # Compare hashes
        if expected_hash and actual_hash != expected_hash:
            logger.error(
                "Hash mismatch detected - validator served incorrect model",
                validator_hotkey=checkpoint.hotkey[:16] + "...",
                expected=expected_hash[:24] + "...",
                actual=actual_hash[:24] + "...",
            )
            return DownloadResult(
                success=False,
                checkpoint=checkpoint,
                local_path=version_folder,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
                error_type=DownloadErrorType.HASH_MISMATCH,
                error_message=f"Hash mismatch: expected {expected_hash}, got {actual_hash}",
            )

        logger.info(
            "Model downloaded and verified successfully",
            validator_hotkey=checkpoint.hotkey[:16] + "...",
            hash=actual_hash[:24] + "...",
            path=str(version_folder),
        )

        return DownloadResult(
            success=True,
            checkpoint=checkpoint,
            local_path=version_folder,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
        )


def create_verified_downloader(
    blacklist: ValidatorBlacklist | None = None,
    max_retries: int = 5,
) -> VerifiedModelDownloader:
    """
    Factory function to create a configured VerifiedModelDownloader.

    Args:
        blacklist: Existing blacklist to use (or creates new one)
        max_retries: Maximum download attempts before giving up

    Returns:
        Configured VerifiedModelDownloader instance
    """
    blacklist = blacklist or ValidatorBlacklist()
    selector = ValidatorSelector(blacklist=blacklist)
    return VerifiedModelDownloader(
        selector=selector,
        max_retries=max_retries,
    )
