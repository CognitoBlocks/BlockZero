from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import fsspec
from pydantic import BaseModel, ConfigDict, Field

from mycelia.shared.app_logging import structlog
from mycelia.shared.checkpoint import compile_full_state_dict_from_path
from mycelia.shared.helper import get_model_hash, parse_dynamic_filename
from mycelia.shared.schema import construct_block_message, construct_model_message, sign_message, verify_message

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


class ModelCheckpoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True, extra="allow")

    signed_model_hash: str | None = Field(default=None, alias="h")
    model_hash: str | None = None
    global_ver: int = 0
    expert_group: int | None = None

    inner_opt: int = 0
    path: Path | None = None  # path to folder or file
    role: str | None = None  # [miner, validator]
    place: str = "local"  # [local / onchain]

    signature_required: bool = False
    signature_verified: bool = False
    hash_required: bool = False
    hash_verified: bool = False
    expert_group_check_required: bool = False
    expert_group_verified: bool = False

    validated: bool = False  # for validator only

    def _extra(self, key: str, default: Any | None = None) -> Any | None:
        if self.model_extra and key in self.model_extra:
            return self.model_extra[key]
        return default

    def _infer_expert_group_from_path(self) -> int | None:
        if self.path is None:
            return None

        if self.path.is_file():
            name = self.path.name
            if name.startswith("model_expgroup_") and name.endswith(".pt"):
                try:
                    return int(name[len("model_expgroup_") : -len(".pt")])
                except ValueError:
                    return None
            return None

        candidates = list(self.path.glob("model_expgroup_*.pt"))
        if len(candidates) == 1:
            name = candidates[0].name
            try:
                return int(name[len("model_expgroup_") : -len(".pt")])
            except ValueError:
                return None
        return None

    def _build_signature_message(self) -> bytes:
        explicit_msg = self._extra("message")
        if isinstance(explicit_msg, (bytes, bytearray)):
            return bytes(explicit_msg)

        target_hotkey_ss58 = self._extra("target_hotkey_ss58")
        block = self._extra("block")

        if self.path is not None and target_hotkey_ss58 is not None and block is not None:
            return construct_model_message(self.path, target_hotkey_ss58, block)

        if self.model_hash is None and self.path is not None:
            self.hash_model()

        if self.model_hash is None:
            raise ValueError("model_hash is required to build a signature message")

        message = _hash_bytes(self.model_hash)
        if target_hotkey_ss58 is not None and block is not None:
            message += construct_block_message(target_hotkey_ss58, block)
        return message

    def expired(self) -> bool:
        if self.place == "local" and self.path is not None and not self.path.exists():
            return True

        expires_at = self._extra("expires_at")
        if isinstance(expires_at, (int, float)):
            return time.time() >= expires_at

        expires_at_block = self._extra("expires_at_block")
        current_block = self._extra("current_block")
        if isinstance(expires_at_block, int) and isinstance(current_block, int):
            return current_block >= expires_at_block

        return False

    def hash_model(self) -> str:
        if self.path is None:
            raise ValueError("path is required to hash a model")

        state = compile_full_state_dict_from_path(self.path)
        self.model_hash = get_model_hash(state, hex=True)
        self.hash_verified = True
        return self.model_hash

    def sign_hash(self) -> str:
        signer = self._extra("signer")
        if signer is None:
            raise ValueError("signer is required to sign the hash")

        message = self._build_signature_message()
        self.signed_model_hash = sign_message(signer, message)
        self.signature_verified = True
        return self.signed_model_hash

    def commit_signature(self) -> dict[str, Any]:
        if self.signed_model_hash is None:
            self.sign_hash()

        return {
            "h": self.signed_model_hash,
            "v": self.global_ver,
            "e": self.expert_group,
            "i": self.inner_opt,
        }

    def commit_hash(self) -> dict[str, Any]:
        if self.model_hash is None:
            self.hash_model()

        return {
            "h": self.model_hash,
            "v": self.global_ver,
            "e": self.expert_group,
            "i": self.inner_opt,
        }

    def verify_hash(self, hash: str | bytes | None) -> bool:
        expected = _normalize_hash(hash or self.model_hash)
        if expected is None:
            self.hash_verified = False
            return False

        if self.path is None:
            self.hash_verified = False
            return False

        actual = _normalize_hash(get_model_hash(compile_full_state_dict_from_path(self.path), hex=True))
        self.hash_verified = actual == expected
        if self.hash_verified:
            self.model_hash = expected
        return self.hash_verified

    def verify_signature(self, signature: str | None) -> bool:
        sig = signature or self.signed_model_hash
        origin_hotkey_ss58 = self._extra("origin_hotkey_ss58")
        if sig is None or origin_hotkey_ss58 is None:
            self.signature_verified = False
            return False

        try:
            message = self._build_signature_message()
        except Exception as exc:
            logger.warning("failed to build signature message", error=str(exc))
            self.signature_verified = False
            return False

        self.signature_verified = verify_message(origin_hotkey_ss58, message, sig)
        return self.signature_verified

    def verify_expert_group(self, signature: str | None = None) -> bool:
        if not self.expert_group_check_required:
            self.expert_group_verified = True
            return True

        inferred = self._infer_expert_group_from_path()
        if inferred is not None and self.expert_group is None:
            self.expert_group = inferred

        expected = self._extra("expected_expert_group")
        if expected is None:
            expected = self.expert_group

        if expected is None:
            self.expert_group_verified = False
        else:
            self.expert_group_verified = self.expert_group == expected
        return self.expert_group_verified

    def active(self) -> bool:
        if self.expired():
            return False

        if self.signature_required and not self.signature_verified:
            return False
        if self.hash_required and not self.hash_verified:
            return False
        if self.expert_group_check_required and not self.expert_group_verified:
            return False
        if self.role == "validator" and not self.validated:
            return False

        return True


class ValidatorChainCheckpoint(ModelCheckpoint):
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


class ValidatorChainCheckpoints(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoints: list[ValidatorChainCheckpoint]

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
    chain: ValidatorChainCheckpoints

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


def select_best_checkpoint(
    primary_dir: Path, secondary_dir: Path | None = None, resume: bool = True
) -> ModelCheckpoint | None:
    
    if not resume:
        return None
    
    primary = build_local_checkpoints(primary_dir, role="miner")

    if secondary_dir is None:
        combined = Checkpoints(
            local=primary,
            chain=ValidatorChainCheckpoints(checkpoints=[]),
        )
        ordered = combined.ordered()
        return ordered[0] if ordered else None

    secondary = build_local_checkpoints(secondary_dir, role="validator")
    combined_local = ModelCheckpoints(checkpoints=[*primary.checkpoints, *secondary.checkpoints])

    combined = Checkpoints(
        local=combined_local,
        chain=ValidatorChainCheckpoints(checkpoints=[]),
    )

    for ckpt in combined.ordered():
        if ckpt.active():
            return ckpt

    return None
