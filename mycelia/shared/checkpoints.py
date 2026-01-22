from __future__ import annotations

import os
import time
from collections import Counter
from functools import total_ordering
from pathlib import Path
from typing import Any

import bittensor
import fsspec
from fsspec.generic import GenericFileSystem
from pydantic import BaseModel, ConfigDict, Field

from mycelia.shared.app_logging import structlog
from mycelia.shared.checkpoint_helper import compile_full_state_dict_from_path
from mycelia.shared.expert_manager import get_layer_expert_id
from mycelia.shared.helper import get_model_hash, parse_dynamic_filename
from mycelia.shared.schema import sign_message, verify_message

from mycelia.shared.cycle import PhaseNames, get_blocks_from_previous_phase_from_api
from mycelia.shared.config import MinerConfig, ValidatorConfig, WorkerConfig
from mycelia.shared.chain import (
    SignedModelHashChainCommit,
    WorkerChainCommit,
    get_chain_commits,
)
from mycelia.shared.cycle import PhaseNames, get_blocks_from_previous_phase_from_api

logger = structlog.get_logger(__name__)


def _normalize_hash(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        return value.lower()
    return str(value).lower()


def _hash_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value
    try:
        return bytes.fromhex(value)
    except ValueError:
        return value.encode()


@total_ordering
class ModelCheckpoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True, extra="allow")

    signed_model_hash: str | None = Field(default=None, alias="h")
    model_hash: str | None = None
    global_ver: int | None = None
    expert_group: int | None = None

    inner_opt: int | None = None
    path: Path | None = None  # path to folder or file
    role: str | None = None  # [miner, validator]
    place: str = "local"  # [local / onchain]

    signature_required: bool = False
    signature_verified: bool = False

    hash_required: bool = False
    hash_verified: bool = False

    expert_group_check_required: bool = False
    expert_group_verified: bool = False

    def __eq__(self, other: object) -> bool:
        try:
            other_global_ver = other.global_ver  # type: ignore[attr-defined]
            other_inner_opt = other.inner_opt  # type: ignore[attr-defined]
            other_model_hash = getattr(other, "model_hash", None)
        except AttributeError:
            return NotImplemented
        return (
            self.global_ver == other_global_ver
            and self.inner_opt == other_inner_opt
            and self.model_hash == other_model_hash
        )

    def __lt__(self, other: ModelCheckpoint) -> bool:
        try:
            other_global_ver = other.global_ver  # type: ignore[attr-defined]
            other_inner_opt = other.inner_opt  # type: ignore[attr-defined]
        except AttributeError:
            return NotImplemented

        # Compare by global_ver first
        self_global_ver = self.global_ver if isinstance(self.global_ver, int) else -1
        other_global_ver = other_global_ver if isinstance(other_global_ver, int) else -1
        if self_global_ver != other_global_ver:
            return self_global_ver < other_global_ver

        # Then compare by inner_opt
        self_inner_opt = self.inner_opt if isinstance(self.inner_opt, int) else -1
        other_inner_opt = other_inner_opt if isinstance(other_inner_opt, int) else -1
        return self_inner_opt < other_inner_opt

    def _extra(self, key: str, default: Any | None = None) -> Any | None:
        if self.model_extra and key in self.model_extra:
            return self.model_extra[key]
        return default

    def expired(self) -> bool:
        return False
        # if self.place == "local" and self.path is not None and not self.path.exists():
        #     return True

        # expires_at = self._extra("expires_at")
        # if isinstance(expires_at, (int, float)):
        #     return time.time() >= expires_at

        # expires_at_block = self._extra("expires_at_block")
        # current_block = self._extra("current_block")
        # if isinstance(expires_at_block, int) and isinstance(current_block, int):
        #     return current_block >= expires_at_block

        # return False

    def hash_model(self) -> str:
        if self.path is None:
            raise ValueError("path is required to hash a model")

        if self.expert_group is None:
            raise ValueError("expert_group is required to hash a model")
        
        state = compile_full_state_dict_from_path(self.path, expert_groups=[self.expert_group])
        self.model_hash = get_model_hash(state, hex=True)
        self.hash_verified = True
        return self.model_hash

    def sign_hash(self, wallet: bittensor.Wallet) -> str:
        if self.model_hash is None:
            self.hash_model()

        self.signed_model_hash = sign_message(
            wallet.hotkey,
            self.model_hash,
        )
        self.signature_verified = True
        return self.signed_model_hash

    def verify_hash(self) -> bool:
        if self.path is None:
            logger.warning("Hash verification failed: missing checkpoint path", checkpoint=self)
            self.hash_verified = False
            return False

        state = compile_full_state_dict_from_path(self.path, expert_groups=[self.expert_group])
        if not state:
            logger.warning(
                "Hash verification failed: empty state dict",
                checkpoint_path=self.path,
                expert_group=self.expert_group,
            )
            self.hash_verified = False
            return False
        expected_hash = get_model_hash(state, hex=True)

        if expected_hash is None:
            logger.warning(
                "Hash verification failed: unable to compute expected hash",
                checkpoint_path=self.path,
                expert_group=self.expert_group,
            )
            self.hash_verified = False
            return False

        self.hash_verified = self.model_hash == expected_hash
        if not self.hash_verified:
            logger.info(
                "Hash verification mismatch",
                checkpoint_path=self.path,
                expert_group=self.expert_group,
                expected_hash=expected_hash,
                provided_hash=self.model_hash,
            )

        return self.hash_verified

    def verify_signature(self) -> bool:
        if self.signed_model_hash is None or self.model_hash is None or self.hotkey is None:
            logger.warning(
                "Signature verification failed: missing signed hash, model hash, or hotkey",
                signed_model_hash_present=self.signed_model_hash,
                model_hash_present=self.model_hash,
                hotkey_present=self.hotkey,
                checkpoint=self,
            )
            self.signature_verified = False
            return False

        self.signature_verified = verify_message(
            self.hotkey, message=self.model_hash, signature_hex=self.signed_model_hash
        )
        if not self.signature_verified:
            logger.info(
                "Signature verification failed",
                hotkey=self.hotkey,
                model_hash=self.model_hash,
                signed_model_hash=self.signed_model_hash,
            )

        return self.signature_verified

    def verify_expert_group(self) -> bool:
        if not self.expert_group_check_required:
            self.expert_group_verified = True
            return True

        expected = self._extra("expected_expert_group")
        if expected is None:
            expected = self.expert_group

        if expected is None:
            self.expert_group_verified = False
            return self.expert_group_verified

        if self.expert_group != expected:
            self.expert_group_verified = False
            return self.expert_group_verified

        expert_group_assignment = self._extra("expert_group_assignment")
        if expert_group_assignment is None:
            self.expert_group_verified = True
            return self.expert_group_verified

        if self.path is None:
            self.expert_group_verified = False
            return self.expert_group_verified

        state_dict = compile_full_state_dict_from_path(self.path, expert_groups=[self.expert_group])
        if not state_dict:
            self.expert_group_verified = False
            return self.expert_group_verified

        allowed_layers = expert_group_assignment.get(expected, {})
        for name in state_dict.keys():
            layer_id, expert_id = get_layer_expert_id(name)
            if layer_id is None or expert_id is None:
                continue

            mappings = allowed_layers.get(layer_id, [])
            allowed_expert_ids = {my_expert_id for my_expert_id, _ in mappings}
            if expert_id not in allowed_expert_ids:
                self.expert_group_verified = False
                return self.expert_group_verified

        self.expert_group_verified = True
        return self.expert_group_verified

    def validate(self) -> bool:
        self.verify_hash()
        self.verify_signature()
        # self.verify_expert_group()
        return self.validated()
    
    def validated(self) -> bool:
        if self.expired():
            logger.info("checkpoint expired", ckpt=self)
            return False
        if self.signature_required and not self.signature_verified:
            logger.info("checkpoint signature not verified", ckpt=self)
            return False
        if self.hash_required and not self.hash_verified:
            logger.info("checkpoint hash not verified", ckpt=self)
            return False
        # if self.expert_group_check_required and not self.expert_group_verified:
        #     logger.info("checkpoint expert group not verified", ckpt=self)
        #     return False

        return True

    def active(self) -> bool:
        # if self.expired():
        #     return False

        # if self.role == "validator" and not self.validated():
        #     return False

        return True


class ChainCheckpoint(ModelCheckpoint):
    uid: int | None = Field(default=None, alias="h")
    ip: str | None = None
    port: int | None = None
    hotkey: str | None = None

    def __init__(self, **data: Any):
        data.setdefault("place", "onchain")
        super().__init__(**data)

    def get_signed_hash_commit(self) -> dict[str, Any] | None:
        if self.signed_model_hash is None:
            return None
        return self.commit_signature()

    def get_hash_commit(self) -> dict[str, Any] | None:
        if self.model_hash is None:
            return None
        return self.commit_hash()

    def priority(self) -> tuple[int, int, int, int, int]:
        return (
            1 if self.active() else 0,
            1 if self.signature_verified else 0,
            1 if self.hash_verified else 0,
            self.global_ver,
            self.inner_opt,
        )


class ChainCheckpoints(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoints: list[ChainCheckpoint]

    def __len__(self) -> int:
        return len(self.checkpoints)

    def get(self, hotkey: str) -> ChainCheckpoint | None:
        for ckpt in self.checkpoints:
            if ckpt.hotkey == hotkey:
                return ckpt  
            
        return None

    def filter_checkpoints(self, for_role: str = "validator") -> ChainCheckpoints:
        # filter out incomplete checkpoints
        filtered = []
        for ckpt in self.checkpoints:
            if (
                ckpt.signed_model_hash is None
                or ckpt.model_hash is None
                or ckpt.global_ver is None
                or ckpt.expert_group is None
                or (ckpt.miner_seed is None and for_role == "validator")
                or ckpt.uid is None
                or ckpt.ip is None
                or ckpt.port is None
                or ckpt.hotkey is None
            ):
                logger.info("filtering out checkpoint due to missing field", ckpt=ckpt)
                continue

            filtered.append(ckpt)

        if not filtered:
            return ChainCheckpoints(checkpoints=[])

        # keep only checkpoints with the highest global_ver
        version_filtered = []
        max_model_checkpoint = max(filtered) if filtered else None
        for ckpt in filtered:
            if max_model_checkpoint and ckpt >= max_model_checkpoint:
                version_filtered.append(ckpt)
            else:
                logger.info("filtering out checkpoint due to outdated", ckpt_ver=ckpt.global_ver, max_global_ver = max_model_checkpoint.global_ver)

        if not version_filtered:
            return ChainCheckpoints(checkpoints=[])

        # select majority model_hash
        hash_counts = Counter([ckpt.model_hash for ckpt in version_filtered if ckpt.model_hash])
        if not hash_counts:
            return ChainCheckpoints(checkpoints=[])

        majority_hash, _count = hash_counts.most_common(1)[0]

        majority_filtered = []
        for ckpt in version_filtered:
            if ckpt.model_hash == majority_hash:
                majority_filtered.append(ckpt)
            else:
                logger.info("filtering out checkpoint due to non-major hash", majority_hash=majority_hash, ckpt_model_hash = ckpt.model_hash)

        return ChainCheckpoints(checkpoints = majority_filtered)

    def renew(self) -> None:
        before = len(self.checkpoints)
        self.checkpoints = [ckpt for ckpt in self.checkpoints if not ckpt.expired()]
        after = len(self.checkpoints)
        if after != before:
            logger.info("removed expired chain checkpoints", before=before, after=after)

    def get_signed_hash_commit(self) -> dict[str, Any] | None:
        ordered = sorted(self.checkpoints, key=lambda ckpt: ckpt.priority(), reverse=True)
        for ckpt in ordered:
            commit = ckpt.get_signed_hash_commit()
            if commit is not None:
                return commit
        return None

    def get_hash_commit(self) -> dict[str, Any] | None:
        ordered = sorted(self.checkpoints, key=lambda ckpt: ckpt.priority(), reverse=True)
        for ckpt in ordered:
            commit = ckpt.get_hash_commit()
            if commit is not None:
                return commit
        return None


class ModelCheckpoints(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoints: list[ModelCheckpoint]

    def ordered(self) -> list[ModelCheckpoint]:
        return sorted(
            self.checkpoints,
            key=lambda ckpt: (1 if ckpt.active() else 0, ckpt.global_ver, ckpt.inner_opt),
            reverse=True,
        )


class Checkpoints(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    local: ModelCheckpoints
    chain: ChainCheckpoints

    def ordered(self) -> list[ModelCheckpoint]:
        return sorted(
            [*self.local.checkpoints, *self.chain.checkpoints],
            key=lambda ckpt: (1 if ckpt.active() else 0, ckpt.global_ver, ckpt.inner_opt),
            reverse=True,
        )

    def download(self) -> ModelCheckpoint | None:
        downloader = self.model_extra.get("downloader") if self.model_extra else None
        if downloader is None:
            raise ValueError("downloader callback is required for Checkpoints.download")

        ordered_chain = sorted(self.chain.checkpoints, key=lambda ckpt: ckpt.priority(), reverse=True)
        for chain_ckpt in ordered_chain:
            if not chain_ckpt.active():
                continue

            try:
                local_path = downloader(chain_ckpt)
            except Exception as exc:
                logger.warning("download failed", error=str(exc))
                continue

            if local_path is None:
                continue

            local_ckpt = ModelCheckpoint(
                signed_model_hash=chain_ckpt.signed_model_hash,
                model_hash=chain_ckpt.model_hash,
                global_ver=chain_ckpt.global_ver,
                expert_group=chain_ckpt.expert_group,
                inner_opt=chain_ckpt.inner_opt,
                path=Path(local_path),
                role=chain_ckpt.role,
                place="local",
                signature_required=chain_ckpt.signature_required,
                hash_required=chain_ckpt.hash_required,
                expert_group_check_required=chain_ckpt.expert_group_check_required,
            )

            if local_ckpt.hash_required and local_ckpt.model_hash is not None:
                local_ckpt.verify_hash(local_ckpt.model_hash)

            if local_ckpt.signature_required and local_ckpt.signed_model_hash is not None:
                local_ckpt.verify_signature(local_ckpt.signed_model_hash)

            if local_ckpt.expert_group_check_required:
                local_ckpt.verify_expert_group()

            if local_ckpt.active():
                self.local.checkpoints.append(local_ckpt)
                return local_ckpt

        return None


def build_local_checkpoint(path: Path, role: str = "miner") -> ModelCheckpoint | None:
    if path.name.startswith(".tmp_") or "yaml" in path.name.lower():
        return None

    meta = parse_dynamic_filename(str(path))
    if meta is None:
        return None

    return ModelCheckpoint(
        global_ver=int(meta.get("globalver", 0)),
        inner_opt=int(meta.get("inneropt", 0)),
        path=path,
        role=role,
        place="local",
    )


def build_local_checkpoints(ckpt_dir: Path, role: str = "miner") -> ModelCheckpoints:
    fs, root = fsspec.core.url_to_fs(str(ckpt_dir))
    checkpoints: list[ModelCheckpoint] = []

    for entry in fs.ls(root, detail=False):
        path = Path(entry)
        if path.name.startswith(".tmp_"):
            continue
        if "yaml" in path.name.lower():
            continue

        meta = parse_dynamic_filename(str(path))
        if meta is None:
            continue

        checkpoints.append(
            ModelCheckpoint(
                global_ver=int(meta.get("globalver", 0)),
                inner_opt=int(meta.get("inneropt", 0)),
                path=path,
                role=role,
                place="local",
            )
        )

    return ModelCheckpoints(checkpoints=checkpoints)


def build_chain_checkpoints(
    signed_hash_chain_commits: list[tuple[Any, Any]],
    hash_chain_commits: list[tuple[Any, Any]],
    for_role: str = "validator",
) -> ChainCheckpoints:
    """
    Build chain checkpoints by joining signed-hash commits with hash commits.
    """
    signed_by_hotkey: dict[str, str] = {}
    for commit, neuron in signed_hash_chain_commits:
        try:
            hotkey = getattr(neuron, "hotkey", None)
            signed = getattr(commit, "signed_model_hash", None)
            if hotkey and signed:
                signed_by_hotkey[hotkey] = signed
        except Exception:
            logger.info("Cannot read signed hash commit", commit=commit)

    checkpoints: list[ChainCheckpoint] = []
    for commit, neuron in hash_chain_commits:
        try:
            hotkey = getattr(neuron, "hotkey", None)
            checkpoints.append(
                ChainCheckpoint(
                    signed_model_hash=signed_by_hotkey.get(hotkey) if hotkey else None,
                    model_hash=getattr(commit, "model_hash", None),
                    global_ver=getattr(commit, "global_ver", None),
                    expert_group=getattr(commit, "expert_group", None),
                    miner_seed=getattr(commit, "miner_seed", None),
                    inner_opt=getattr(commit, "inner_opt", None),
                    uid=getattr(neuron, "uid", None),
                    ip=getattr(neuron.axon_info, "ip", None),
                    port=getattr(neuron.axon_info, "port", None),
                    hotkey=hotkey,
                    signature_required=True,
                    hash_required=True,
                    expert_group_check_required=True,
                )
            )
        except Exception:
            logger.info("Cannot append commit", commit=commit)

    filtered_checkpoints = ChainCheckpoints(checkpoints=checkpoints).filter_checkpoints(for_role=for_role)

    if len(filtered_checkpoints) == 0:
        return ChainCheckpoints(checkpoints=[])
    return filtered_checkpoints

def build_chain_checkpoints_from_previous_phase(
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    for_role: str = "validator",
) -> ChainCheckpoints:
    # --- Validate type ---
    if for_role == "miner":
        phase_name_1 = PhaseNames.miner_commit_1
        phase_name_2 = PhaseNames.miner_commit_2
        
    elif for_role == "validator":
        phase_name_1 = PhaseNames.validator_commit_1
        phase_name_2 = PhaseNames.validator_commit_2

    else:
        raise ValueError(f"Invalid type: {for_role}. Must be 'miner' or 'validator'.")

    # --- Get block ranges for previous phases ---
    previous_phase_range = get_blocks_from_previous_phase_from_api(config)

    if previous_phase_range is not None: 
        commit_1_end_block = previous_phase_range[phase_name_1][1] + 1
        commit_2_end_block = previous_phase_range[phase_name_2][1] + 1

        # --- Get commits from chain at the right blocks ---
        signed_hash_chain_commits: tuple[SignedModelHashChainCommit, bittensor.Neuron] = get_chain_commits(
            config, subtensor, block=commit_1_end_block
        )
        hash_chain_commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(
            config, subtensor, block=commit_2_end_block
        )

    else:
        signed_hash_chain_commits = []
        hash_chain_commits = []

    # --- Build chain checkpoints ---
    return build_chain_checkpoints(
        signed_hash_chain_commits=signed_hash_chain_commits,
        hash_chain_commits=hash_chain_commits,
        for_role=for_role,
    )

def delete_old_checkpoints(checkpoint_path: str | Path, topk: int) -> list[str]:
    """
    Deletes old checkpoints, keeping only the top 'k' most recent ones.
    """
    fs = GenericFileSystem()
    sorted_ckpt_files = build_local_checkpoints(checkpoint_path).ordered()

    ckpt_deleted = []
    for model_meta in sorted_ckpt_files[topk:]:
        fs.rm(str(model_meta.path), recursive=True)
        ckpt_deleted.append(str(model_meta.path))
    return ckpt_deleted


def delete_old_checkpoints_by_hotkey(folder_path: Path) -> list[str]:
    """
    Deletes all non-latest submission files coming from the same hotkey.
    Keeps only the file with the highest block number per hotkey.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    submissions_by_hotkey: dict[str, list[tuple[int, Path]]] = {}
    for file_path in folder_path.glob("*.pt"):
        meta = parse_dynamic_filename(file_path.name)
        if "hotkey" not in meta or "block" not in meta:
            print(f"Skipping malformed filename: {file_path.name}")
            continue

        hotkey = meta["hotkey"]
        block = meta["block"]

        if hotkey not in submissions_by_hotkey:
            submissions_by_hotkey[hotkey] = []
        submissions_by_hotkey[hotkey].append((block, file_path))

    deleted_files = []
    for _, entries in submissions_by_hotkey.items():
        entries.sort(key=lambda x: x[0], reverse=True)

        for _, file_path in entries[2:]:
            try:
                os.remove(file_path)
                deleted_files.append(file_path.name)
            except Exception as exc:
                print(f"Failed to delete {file_path.name}: {exc}")

    if deleted_files:
        logger.info("Deleted outdated submissions", count=len(deleted_files), files=deleted_files)
    else:
        logger.info("No outdated submissions found.")

    return deleted_files


def select_best_checkpoint(
    primary_dir: Path, secondary_dir: Path | None = None, resume: bool = True
) -> ModelCheckpoint | None:
    if not resume:
        return None

    primary = build_local_checkpoints(primary_dir, role="miner")

    if secondary_dir is None:
        combined = Checkpoints(
            local=primary,
            chain=ChainCheckpoints(checkpoints=[]),
        )
        ordered = combined.ordered()
        return ordered[0] if ordered else None

    secondary = build_local_checkpoints(secondary_dir, role="validator")
    combined_local = ModelCheckpoints(checkpoints=[*primary.checkpoints, *secondary.checkpoints])

    combined = Checkpoints(
        local=combined_local,
        chain=ChainCheckpoints(checkpoints=[]),
    )

    for ckpt in combined.ordered():
        if ckpt.active():
            return ckpt

    return None
