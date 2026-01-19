import contextlib
import hashlib
import mimetypes
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import bittensor
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.checkpoints import delete_old_checkpoints_by_hotkey, select_best_checkpoint, build_chain_checkpoints_from_previous_phase
from mycelia.shared.config import ValidatorConfig, parse_args
from mycelia.shared.cycle import (
    PhaseNames,
    PhaseResponseLite,
    get_blocks_from_previous_phase_from_api,
    get_blocks_until_next_phase_from_api,
    get_phase_from_api,
    get_validator_miner_assignment,
)
from mycelia.shared.helper import parse_dynamic_filename
from mycelia.shared.schema import (
    construct_block_message,
    construct_model_message,
    verify_message,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
configure_logging()
logger = structlog.get_logger(__name__)
app = FastAPI(title="Checkpoint Sender", version="1.0.0")

# ---- Settings via environment variables ----
AUTH_TOKEN = os.getenv("AUTH_TOKEN")  # optional bearer token for auth


# ---- Helpers ----
def require_auth(authorization: str | None) -> None:
    if not AUTH_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


def file_response_for(path: Path, download_name: str | None = None) -> FileResponse:
    if not path.exists() or not path.is_file():
        logger.info("file response for, path dosent exist", path.exists(), path.is_file())
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # Default to binary; guess if we can
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    stat = path.stat()
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    headers = {
        # Encourage efficient delivery; Starlette handles Range requests automatically for FileResponse
        "Cache-Control": "no-store",
        "Last-Modified": last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT"),
        # Nginx/Envoy tip to avoid buffering big files
        "X-Accel-Buffering": "no",
    }

    return FileResponse(
        path=path,
        media_type=media_type,
        filename=download_name or path.name,  # sets Content-Disposition: attachment; filename="..."
        headers=headers,
    )


# Optional: centralize where uploads go (env var overrides default)
def get_upload_root() -> Path:
    root = os.getenv("CHECKPOINT_INBOX", "/var/lib/validator/checkpoints/incoming")
    p = Path(root).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

class ValidatorStateCache:
    def __init__(
        self,
        config: ValidatorConfig,
        subtensor: bittensor.Subtensor,
    ) -> None:
        self._config = config
        self._subtensor = subtensor
        self._blocks_until_cache = None
        self._previous_phase_cache = None
        self._assignment_cache = None
        self._assignment_cache_block = None
        self._chain_checkpoints_cache = None
        self._chain_checkpoints_cache_block = None
        self._phase_cache = None
        self._phase_cache_block = None

    # ---- Private logic ----
    def _should_refresh_for_validation(self, _last_cache_block: int | None = None) -> tuple[bool, PhaseResponseLite | None]:
        current_phase = self.get_phase()
        
        if current_phase is None:
            return False, None

        should_refresh = False
        if current_phase.phase_name != PhaseNames.submission:
            # it is the submission phase
            cycle_length = self._get_cycle_length_from_blocks_until_cache()
            if (
                self._assignment_cache is not None
                and _last_cache_block is not None
                and cycle_length is not None
                and _last_cache_block + cycle_length < self._subtensor.block
            ):
                # the assignment cache is stale cause it is more than a cycle ago
                should_refresh = True

        elif _last_cache_block != current_phase.phase_start_block:
            # it is not the submission phase
            should_refresh = True

        return should_refresh, current_phase
    
    # ---- Public getters ----
    def get_phase(self):
        block = self._subtensor.block
        if (
            self._phase_cache is None
            or self._phase_cache_block is None
            or block > self._phase_cache.phase_end_block
        ):
            current_phase = self.get_current_phase_locally()
            
            if current_phase is None:
                # refresh
                self._phase_cache = get_phase_from_api(self._config)
                self._phase_cache_block = block

            else:
                self._phase_cache = current_phase

        return self._phase_cache

    def get_blocks_until_next_phase(self):
        block = self._subtensor.block

        max_end_block = None
        if self._blocks_until_cache:
            max_end_block = max(end for _, end, _ in self._blocks_until_cache.values())

        if self._blocks_until_cache is None or (max_end_block is not None and block > max_end_block):
            # refresh
            blocks_until = get_blocks_until_next_phase_from_api(self._config)
            if blocks_until is not None:
                self._blocks_until_cache = blocks_until


        return self._blocks_until_cache

    def _get_cycle_length_from_blocks_until_cache(self):
        blocks_until = self.get_blocks_until_next_phase()
        if not blocks_until:
            return None

        return sum((end - start + 1) for start, end, _ in blocks_until.values())

    def get_current_phase_locally(self):
        blocks_until = self.get_blocks_until_next_phase()
        if blocks_until is None:
            return None

        block = self._subtensor.block
        for phase_name, (start, end, _) in blocks_until.items():
            if start <= block <= end:
                return PhaseResponseLite(
                    phase_name=phase_name,
                    phase_start_block=start,
                    phase_end_block=end,
                )

        return None

    def get_assignment(self):
        should_refresh, current_phase = self._should_refresh_for_validation(_last_cache_block=self._assignment_cache_block)
        if should_refresh:
            self._assignment_cache = get_validator_miner_assignment(self._config, self._subtensor)
            self._assignment_cache_block = current_phase.phase_start_block

        return self._assignment_cache
    
    def get_chain_checkpoints(self):
        should_refresh, current_phase = self._should_refresh_for_validation(_last_cache_block=self._chain_checkpoints_cache_block)

        if should_refresh:
            self._chain_checkpoints_cache = build_chain_checkpoints_from_previous_phase(
                config=self._config,
                subtensor=self._subtensor,
                for_role="miner",
            )
            self._chain_checkpoints_cache_block = current_phase.phase_start_block

        return self._chain_checkpoints_cache

# ---- Schemas ----
class SendBody(BaseModel):
    # Optional override; if not provided, uses CHECKPOINT_PATH
    path: str | None = None
    # Optional download file name (e.g., "model-v1.ckpt")
    download_name: str | None = None


@app.on_event("startup")
async def _startup():
    configure_logging()  # <— configure ONCE
    structlog.get_logger(__name__).info("startup_ok")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        "HTTPException %s %s -> %s | detail=%r",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# ---- Routes ----
@app.get("/")
async def index(request: Request):
    return JSONResponse(
        {
            "name": config.chain.hotkey_ss58,
            "service": "Checkpoint Sender",
            "version": "0.0.0",
            "endpoints": {
                "GET /ping": "Health check",
                "GET /checkpoint": "Download the configured checkpoint",
                "POST /send-checkpoint": {
                    "body": {"path": "optional string", "download_name": "optional string"},
                    "desc": "Download checkpoint (optionally overriding path/name)",
                },
            },
            "auth": "Set AUTH_TOKEN env var to require 'Authorization: Bearer <token>'",
        }
    )


@app.get("/ping")
async def ping():
    """Health check."""
    print("ping")
    logger = structlog.get_logger(__name__)
    logger.info("ping")  # <— this will now print
    return {"status": "ok"}


@app.get("/status")
async def status():
    # respond to what is the current status of validator, the newest model id
    pass



# miners / client get model from validator
@app.get("/get-checkpoint")
async def get_checkpoint(
    authorization: str | None = Header(default=None),
    target_hotkey_ss58: str = Form(None, description="Receiver's hotkey"),
    origin_hotkey_ss58: str = Form(None, description="Sender's hotkey"),
    origin_block: int = Form(None, description="The block that the message was sent."),  # insecure, do not use this field for validation, TODO: change it to block hash?
    signature: str = Form(None, description="Signed message"),
    expert_group_id: int | str | None = Form(None, description="List of expert groups to fetch"),
):
    """GET to download the configured checkpoint immediately."""
    require_auth(authorization)

    logger.info("<download> Received get checkpoint request - verifying signature", from_hk=origin_hotkey_ss58, to_hk=target_hotkey_ss58, origin_block=origin_block, signature=signature)
    verify_message(
        origin_hotkey_ss58=origin_hotkey_ss58,
        message=construct_block_message(
            target_hotkey_ss58=target_hotkey_ss58,  # TODO: assert hotkey is valid within the metagraph
            block=origin_block,  # TODO: change to block hash and assert it is not too far from current
        ),
        signature_hex=signature,
    )

    latest_checkpoint = select_best_checkpoint(primary_dir=config.ckpt.checkpoint_path)

    if not latest_checkpoint or not latest_checkpoint.path:
        raise HTTPException(status_code=500, detail="CHECKPOINT_PATH env var is not set")

    logger.info(
        "<download> Received get checkpoint request",
        from_hk=origin_hotkey_ss58,
        block=origin_block,
        expert_group_id=expert_group_id,
    )

    if expert_group_id is not None:
        if expert_group_id == "shared":
            ckpt_path = latest_checkpoint.path / "model_shared.pt"
        else:
            ckpt_path = latest_checkpoint.path / f"model_expgroup_{expert_group_id}.pt"

    else:
        ckpt_path = latest_checkpoint.path / "model.pt"

    if not ckpt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {ckpt_path.name}",
        )

    logger.info("<download> Preparing file response for path", path=Path(ckpt_path))
    result = file_response_for(Path(ckpt_path), f"step{latest_checkpoint.global_ver}")

    return result

# miners submit checkpoint
# @app.post("/submit-checkpoint-permit")
def validate_submission_phase_and_assignment(
    config: ValidatorConfig,
    subtensor: bittensor.Subtensor,
    origin_block: int,
    target_hotkey_ss58: str | None,
    origin_hotkey_ss58: str | None,
    authorization: str | None = Header(default=None),
) -> None:
    
    # check if the submission is to this validator
    if target_hotkey_ss58 != config.chain.hotkey_ss58:
        raise HTTPException(status_code=403, detail="Submission target hotkey does not match this validator")

    # check if it is now the submission phase
    phase = validator_state_cache.get_phase()
    if phase is None or phase.phase_name != PhaseNames.submission:
        raise HTTPException(status_code=409, detail="Submissions are only accepted during submission phase")

    # check of the origin block is within the submission phase
    if not (phase.phase_start_block <= origin_block <= phase.phase_end_block):
        raise HTTPException(status_code=400, detail="Origin block is not within the submission phase")

    # check if miner is assigned to this validator
    assignment = validator_state_cache.get_assignment()
    assigned_miners = assignment.get(config.chain.hotkey_ss58, [])
    if origin_hotkey_ss58 not in assigned_miners:
        raise HTTPException(status_code=403, detail="Miner is not assigned to this validator")

    # check if the miner has already submitted during this phase
    for path in Path(config.ckpt.miner_submission_path).glob(f"hotkey_{origin_hotkey_ss58}_block_*.pt"):
        meta = parse_dynamic_filename(path.name)
        block = meta.get("block")
        if isinstance(block, int) and phase.phase_start_block <= block <= phase.phase_end_block:
            raise HTTPException(status_code=409, detail=f"Miner already submitted during this phase, existing path {path}, existing range {phase.phase_start_block}-{phase.phase_end_block}")

# miners submit checkpoint
@app.post("/submit-checkpoint")
async def submit_checkpoint(
    authorization: str | None = Header(default=None),
    origin_block: int = Form(None, description="The block that the message was sent."),
    target_hotkey_ss58: str = Form(None, description="Receiver's hotkey"),
    origin_hotkey_ss58: str = Form(None, description="Sender's hotkey"),
    signature: str = Form(None, description="Signed message"),
    file: UploadFile = File(..., description="The checkpoint file, e.g. model.pt"),
):
    """
    POST a checkpoint to the validator.

    Accepts multipart/form-data with fields:
      - step (int, required)
      - checksum_sha256 (str, optional)
      - file (UploadFile, required)
    """

    logger.info("<submit> Received checkpoint submission request", from_hk=origin_hotkey_ss58, to_hk=target_hotkey_ss58, block=origin_block, signature=signature)
    require_auth(authorization)

    validate_submission_phase_and_assignment(
        config=config,
        origin_block=origin_block,
        subtensor=subtensor,
        target_hotkey_ss58=target_hotkey_ss58,
        origin_hotkey_ss58=origin_hotkey_ss58,
    )

    # Basic filename safety (avoid path tricks). We'll still rename it server-side.
    original_name = file.filename or ""
    if not SAFE_NAME.match(Path(original_name).name):
        raise HTTPException(status_code=400, detail="Unsafe filename")

    # Restrict to common checkpoint extensions (adjust if you use others)
    allowed_exts = {".pt", ".bin", ".ckpt", ".safetensors", ".tar"}
    ext = Path(original_name).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

    # Stream write + compute SHA256
    block = subtensor.block
    model_name = f"hotkey_{origin_hotkey_ss58}_block_{block}.pt"
    hasher = hashlib.sha256()
    bytes_written = 0
    dest_path = config.ckpt.miner_submission_path / model_name
    tmp_path = dest_path.with_name(f".tmp_{dest_path.name}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.unlink(missing_ok=True)
    try:
        async with aiofiles.open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MiB
                if not chunk:
                    break
                hasher.update(chunk)
                await out.write(chunk)
                bytes_written += len(chunk)

        os.replace(tmp_path, dest_path)
    except Exception as e:
        # Clean up partial writes
        with contextlib.suppress(Exception):
            tmp_path.unlink(missing_ok=True)
        logger.exception("Failed to store uploaded checkpoint")
        raise HTTPException(status_code=500, detail="Failed to store file") from e
    finally:
        await file.close()

    computed = hasher.hexdigest()

    logger.info(f"<submit> Stored checkpoint at {dest_path} ({bytes_written} bytes) sha256={computed}")

    # get chain checkpoint for model hash, signature, and expert group verification
    chain_checkpoints = validator_state_cache.get_chain_checkpoints()
    logger.info(f"<submit> Fetched {len(chain_checkpoints)} chain checkpoints for validation.", chain_checkpoints=chain_checkpoints.checkpoints, origin_hotkey_ss58=origin_hotkey_ss58)
    chain_checkpoint = chain_checkpoints.get(origin_hotkey_ss58)
    if chain_checkpoint is None:
        raise HTTPException(status_code=400, detail="No checkpoint found on chain for this miner")
    chain_checkpoint.path = dest_path
    validated = chain_checkpoint.validate() and verify_message(
        origin_hotkey_ss58=origin_hotkey_ss58,
        message=chain_checkpoint.model_hash
        + construct_block_message(target_hotkey_ss58, block=origin_block),
        signature_hex=signature,
    ) # checkpoint validation and package signature validation
    
    if not validated:
        # delete the invalid checkpoint
        with contextlib.suppress(Exception):
            dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Submitted checkpoint does not match chain checkpoint")

    logger.info(f"<submit> Verified submission at {dest_path}.")

    delete_old_checkpoints_by_hotkey(config.ckpt.miner_submission_path)

    return {
        "status": "ok",
        "stored_path": str(dest_path),
        "bytes": bytes_written,
        "sha256": computed,
    }


if __name__ == "__main__":
    args = parse_args()

    global config
    global subtensor
    global validator_state_cache

    if args.path:
        config = ValidatorConfig.from_path(args.path)
    else:
        config = ValidatorConfig()

    config.write()

    subtensor = bittensor.Subtensor(network=config.chain.network)

    validator_state_cache = ValidatorStateCache(config=config, subtensor=subtensor)
        
    if validator_state_cache is None:
        validator_state_cache = ValidatorStateCache(config=config, subtensor=subtensor)

    uvicorn.run(app, host=config.chain.ip, port=config.chain.port)
