# --- Authorizer --- 
import base64
import fnmatch
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Set

import bittensor as bt
import hivemind
import torch
import torch.nn as nn
from hivemind.averaging import DecentralizedAverager
from hivemind.proto import auth_pb2
from mycelia.shared.cycle import get_init_peer_id

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import ValidatorConfig
from mycelia.shared.expert_manager import get_layer_expert_id
from mycelia.shared.schema import sign_message, verify_message

logger = structlog.get_logger(__name__)


def get_init_peer_ids(config: ValidatorConfig):
    return [get_init_peer_id(config)]


def connect_with_peers(config, wallet, subtensor):
    initial_peer_ids: list[str] = get_init_peer_id(config)

    authorizer = HotkeyAuthorizer(
        my_hotkey=wallet.hotkey,
        max_time_skew_s=30.0,
        subtensor=subtensor,
        config = config,
    )

    dht = hivemind.DHT(start=True, initial_peers=initial_peer_ids, authorizer=authorizer)
    logger.info('accessible multiaddrs', '\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
    return dht

# --- expert group selection helpers ---
def names_for_expert(
    model: nn.Module, eid, expert_name_fmt: str, include_buffers: bool
) -> list[tuple[str, torch.Tensor]]:
    """Collect all tensors whose names start with the expert module prefix."""
    prefix = expert_name_fmt.format(eid=eid)
    out = []
    for name, tensor in iter_named_grads(model):
        if name.startswith(prefix + ".") or name == prefix:
            out.append((name, tensor))
    return out


def iter_named_grads(model: nn.Module):
    """
    Yield (name, grad_tensor) for all model parameters that have gradients.
    """
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            p.grad = torch.zeros_like(p)

        yield n, p.grad


def name_selected(name, include_globs, exclude_globs):
    inc_ok = (not include_globs) or any(fnmatch.fnmatch(name, pat) for pat in include_globs)
    exc_ok = not any(fnmatch.fnmatch(name, pat) for pat in exclude_globs)
    return inc_ok and exc_ok


def select_tensors(model, include_globs=(), exclude_globs=()):
    # deterministic order across peers: sort by name!
    chosen = []
    for name, tensor in sorted(iter_named_grads(model), key=lambda kv: kv[0]):
        if name_selected(name, include_globs, exclude_globs):
            chosen.append(tensor)
    return chosen


# --- packaging gradient buff ---
def build_buff_from_params(params):
    numels = [p.numel() for p in params]
    offsets = [0]
    for n in numels[:-1]:
        offsets.append(offsets[-1] + n)
    total = sum(numels)
    flat_grad = torch.zeros(total, device="cpu")  # or cuda

    return {"params": params, "numels": numels, "offsets": offsets, "buff": flat_grad}


def pack_grads(buff_meta):
    with torch.no_grad():
        for p, off, n in zip(buff_meta["params"], buff_meta["offsets"], buff_meta["numels"], strict=False):
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            buff_meta["buff"][off : off + n].copy_(g.view(-1))


def unpack_to_grads(buff_meta):
    with torch.no_grad():
        for p, off, n in zip(buff_meta["params"], buff_meta["offsets"], buff_meta["numels"], strict=False):
            view = buff_meta["buff"][off : off + n].view_as(p).to(p.device)
            if p.grad is None:
                p.grad = view.clone()
            else:
                p.grad.copy_(view)


# --- getting averager ---
def build_grad_buff_from_model(
    model: nn.Module,
    expert_group_assignment: dict[int, dict[int, list[int]]],
) -> dict[str | int, dict]:
    """
    Returns:
      - group_averagers: dict[group_id] -> DecentralizedAverager averaging *all* experts in that group
      - non_expert_averager: DecentralizedAverager averaging all non-expert tensors
    Notes:
      - All peers that should meet in the same group MUST use the same prefix (we derive it from group_id).
      - We sort tensor names to keep a deterministic order across peers.
    """
    # 1) Index tensors by name and prepare expert buckets
    all_named = list(iter_named_grads(model))
    all_named.sort(key=lambda kv: kv[0])  # deterministic order
    name_to_tensor = dict(all_named)
    expert_group_to_names = {group_id: [] for group_id in list(expert_group_assignment.keys())}

    for name, _ in name_to_tensor.items():
        layer_id, expert_id = get_layer_expert_id(name)
        if layer_id and expert_id is not None:
            for group_id, layer_to_expert_ids in expert_group_assignment.items():
                if expert_id in [a for a, b in layer_to_expert_ids[layer_id]]:
                    expert_group_to_names[group_id].append(name)

    # 2) Build gradient buffer per expert group
    group_buff_metas: dict[str | int, Any] = {}
    for group_id in expert_group_to_names.keys():
        tensors_for_group = [name_to_tensor[name] for name in expert_group_to_names[group_id]]
        if len(tensors_for_group) == 0:
            logger.warning(
                "No tensors found for expert group",
                group_id=group_id,
            )
        group_buff_metas[group_id] = build_buff_from_params(params=tensors_for_group)
        logger.info(
            f"Built expert group grad buffer - {group_id}",
            tensor_count=len(tensors_for_group),
        )

    expert_owned_names = [name for names in expert_group_to_names.values() for name in names]
    non_expert_names = [n for n, _t in all_named if n not in expert_owned_names]
    non_expert_tensors = [name_to_tensor[n] for n in non_expert_names]
    group_buff_metas["shared"] = build_buff_from_params(non_expert_tensors)
    logger.info(
        "Built shared grad buffer",
        tensor_count=len(non_expert_tensors),
        total_param_count=len(all_named),
    )

    return group_buff_metas


def build_averagers_from_buff(
    group_buff_metas: dict[int | str, dict[str, torch.Tensor]],
    dht: hivemind.DHT,
    prefix_base: str = "expert_averaging",
    target_group_size: int = 4,
    min_group_size: int = 2,
    averaging_alpha: float = 1.0,
) -> dict[str | int, DecentralizedAverager]:
    """
    Returns:
      - group_averagers: dict[group_id] -> DecentralizedAverager averaging *all* experts in that group
      - non_expert_averager: DecentralizedAverager averaging all non-expert tensors
    Notes:
      - All peers that should meet in the same group MUST use the same prefix (we derive it from group_id).
      - We sort tensor names to keep a deterministic order across peers.
    """

    group_averagers: dict[str | int, DecentralizedAverager] = {}
    for group_id, buff_meta in group_buff_metas.items():
        prefix = f"{prefix_base}/group{group_id}"
        group_averagers[group_id] = DecentralizedAverager(
            averaged_tensors=[buff_meta["buff"]],
            dht=dht,
            start=True,
            prefix=prefix,
            target_group_size=target_group_size,
            min_group_size=1,
            allreduce_timeout=120,
            client_mode=False,
        )
        logger.info(
            f"build hivemind averager - {group_id}",
            prefix=prefix,
            mode=group_averagers[group_id].mode,
            client_mode=group_averagers[group_id].client_mode,
            total_size = group_averagers[group_id].total_size
        )

    return group_averagers


# --- Authrizer ---

def _canon_request_bytes(method: str, payload_bytes: bytes, t_ms: int, nonce: bytes, caller_peer_id: Optional[str]) -> bytes:
    """
    Canonical request message to sign.

    Include caller_peer_id if you want signatures to be non-transferable across peer IDs.
    If you don't have it at signing time, pass None and it will be omitted.
    """
    parts = [
        b"HM_REQ_V1",
        method.encode("utf-8"),
        payload_bytes,
        str(t_ms).encode("ascii"),
        nonce,
    ]
    if caller_peer_id is not None:
        parts.append(caller_peer_id.encode("utf-8"))
    return b"|".join(parts)


def _canon_response_bytes(request_nonce: bytes, response_bytes: bytes) -> bytes:
    return b"|".join([b"HM_RESP_V1", request_nonce, response_bytes])


# ---------------------------
# Hotkey authorizer
# ---------------------------
class HotkeyAuthorizer:
    """
    DHT request/response authorizer that only accepts messages signed by allowed hotkeys (SS58).

    You pass this into: hivemind.DHT(..., authorizer=HotkeyAuthorizer(...))
    """

    def __init__(self, my_hotkey: bt.Wallet.hotkey, subtensor: bt.Subtensor, config, max_time_skew_s: float = 30):
        self.my_hotkey: bt.Keypair = my_hotkey
        self.max_time_skew_s: float = max_time_skew_s
        self._seen_nonces: set[bytes] = None
        self.subtensor: bt.Subtensor = subtensor
        self.config = config

    def __post_init__(self):
        if self._seen_nonces is None:
            self._seen_nonces = set()

    def get_allowed_hotkey(self):
        metagraph = self.subtensor.metagraph(netuid = self.config.chain.netuid)
        allowed_validator = [] 
        for hotkey, T in zip(metagraph.hotkeys, metagraph.validator_trust):
            if T > 0:
                allowed_validator.append(hotkey)

        return allowed_validator + ['5DoHdXfDYraqPzkLjrXGMZxvGXYdDYhuC8tGbQdb4zvz2LbH']
        
    @property
    def my_hotkey_ss58(self) -> str:
        return self.my_hotkey.ss58_address

    # ---- Core API (names may vary slightly by hivemind version) ----
    def sign_request(self, method: str, request_without_auth: bytes, caller_peer_id: Optional[str] = None) -> auth_pb2.RequestAuthInfo:
        nonce = os.urandom(16)
        t_ms = int(time.time() * 1000)

        msg = _canon_request_bytes(method, request_without_auth, t_ms, nonce, caller_peer_id)
        sig_b64url = sign_message(self.my_hotkey, msg)

        return auth_pb2.RequestAuthInfo(
            # we store SS58 string as bytes
            service_public_key=self.my_hotkey_ss58.encode("utf-8"),
            time=t_ms / 1000.0,   # hivemind uses float seconds in some versions; keep compatible
            nonce=nonce,
            signature=sig_b64url.encode("utf-8"),
        )

    def validate_request(
        self,
        method: str,
        request_without_auth: bytes,
        auth: Optional[auth_pb2.RequestAuthInfo],
        remote_peer_id: Optional[str] = None,  # pass str(remote_id) if available
    ) -> bool:
        if auth is None:
            return False

        try:
            signer_ss58 = auth.service_public_key.decode("utf-8")
            sig_b64url = auth.signature.decode("utf-8")
        except Exception:
            return False


        if signer_ss58 not in self.get_allowed_hotkey():
            return False

        # time skew
        now_s = time.time()
        if abs(now_s - float(auth.time)) > self.max_time_skew_s:
            return False

        # replay protection
        if auth.nonce in self._seen_nonces:
            return False
        self._seen_nonces.add(auth.nonce)

        t_ms = int(float(auth.time) * 1000)
        msg = _canon_request_bytes(method, request_without_auth, t_ms, auth.nonce, remote_peer_id)

        return verify_message(signer_ss58, msg, sig_b64url)

    def sign_response(self, request_auth: auth_pb2.RequestAuthInfo, response_without_auth: bytes) -> auth_pb2.ResponseAuthInfo:
        msg = _canon_response_bytes(request_auth.nonce, response_without_auth)
        sig_b64url = sign_message(self.my_hotkey, msg)

        return auth_pb2.ResponseAuthInfo(
            nonce=request_auth.nonce,
            signature=sig_b64url.encode("utf-8"),
        )

    def validate_response(
        self,
        request_auth: auth_pb2.RequestAuthInfo,
        response_without_auth: bytes,
        auth: Optional[auth_pb2.ResponseAuthInfo],
    ) -> bool:
        if auth is None or auth.nonce != request_auth.nonce:
            return False

        msg = _canon_response_bytes(request_auth.nonce, response_without_auth)
        sig_b64url = auth.signature.decode("utf-8", errors="ignore")

        # Because ResponseAuthInfo doesn’t include responder pubkey, we verify
        # against any allowed hotkey. If you need “must be signed by peer X”,
        # you’ll need responder identity in the response (protocol extension) or
        # verify at a higher layer.
        for ss58 in self.get_allowed_hotkey():
            if verify_message(ss58, msg, sig_b64url):
                return True
        return False