import os
import torch
import fsspec

from pathlib import Path
import re
import os

from copy import deepcopy
from mycelia.shared.config import MinerConfig, ValidatorConfig
from fsspec.generic import GenericFileSystem
from torchdata.stateful_dataloader import StatefulDataLoader
from mycelia.shared.app_logging import structlog
from mycelia.shared.helper import parse_dynamic_filename

logger = structlog.getLogger(__name__)

def start_model_from(rank: int, config: MinerConfig) -> Tuple(bool, int | None, int | None, str | Path):

    validator_ckpt_found, validator_version, latest_validator_ckpt = get_resume_info(rank, config, config.miner.validator_checkpoint_path)
    miner_ckpt_found, miner_version, latest_miner_ckpt = get_resume_info(rank, config, config.ckpt.checkpoint_path)

    if miner_ckpt_found:
        miner_ckpt_meta = parse_dynamic_filename(latest_miner_ckpt)
    else:    
        return validator_ckpt_found, {'gloablopt': validator_version, 'inneropt': 0}, latest_validator_ckpt
    
    if not validator_ckpt_found:
        return miner_ckpt_found, {'gloablopt': miner_ckpt_meta['globalopt'], 'inneropt': miner_version}, latest_miner_ckpt/'model.pt'
    
    if miner_ckpt_meta['globalopt'] >= validator_version:
        return miner_ckpt_found, {'gloablopt': miner_ckpt_meta['globalopt'], 'inneropt': miner_version}, latest_miner_ckpt/'model.pt'
    else:
        return validator_ckpt_found, {'gloablopt': miner_ckpt_meta['globalopt'], 'inneropt': 0}, latest_validator_ckpt


def get_resume_info(rank: int, config: MinerConfig | ValidatorConfig, path : Path | None = None) -> tuple[bool, dict, str | None]:
    """
    Retrieves the resume information for a given rank and checkpoint configuration.

    Args:
        rank (int): The rank of the process.
        ckpt_config (Config): The configuration object for the checkpoint.

    Returns:
        tuple[bool, int, str | None]: A tuple containing a boolean indicating success,
        the checkpoint step, and an optional string message.
    """
    """
    Check if we should resume from a checkpoint, if yes return the path to the checkpoint, otherwise return None
    """
    if config.ckpt.resume_from_ckpt is None:
        return False, {}, None

    elif isinstance(config.ckpt.resume_from_ckpt, bool):
        # Using fsspec to list directory contents
        try:
            if path == None:
                path = config.ckpt.checkpoint_path
            
            ckpt_files = get_sorted_checkpoints(path)
            
        except FileNotFoundError:
            logger.info(f"rank {rank}: Checkpoint path {config.ckpt.checkpoint_path} not found, starting from scratch")
            return False, {}, None

        if len(ckpt_files) == 0:
            logger.info(f"rank {rank}: No checkpoints found in {config.ckpt.checkpoint_path}, starting from scratch")
            return False, {}, None

        latest_ckpt = ckpt_files[0][0]
        version = ckpt_files[0][1]
        logger.info(f"rank {rank}: Latest checkpoint found in {latest_ckpt}")
        return True, version, latest_ckpt


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    rank: int,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    inner_optimizer: torch.optim.Optimizer | None = None,
    outer_optimizer: torch.optim.Optimizer | None = None,
    inner_scaler: torch.amp.GradScaler | None = None,
    outer_scaler: torch.amp.GradScaler | None = None,
    loss: float | None = None,
    data_loader: StatefulDataLoader | None = None,
    save_global_state: bool = True,
) -> None:
    """
    Saves the current model checkpoint.

    Returns:
        None
    """
    # === save model, optimizer ===
    checkpoint = {
        "model_state_dict": {k: v.detach().to("cpu", non_blocking=True) for k, v in model.state_dict().items()},
        "loss": loss,
    }

    target = os.path.join(checkpoint_path, f"model.pt")
    if not os.path.exists(target):
        with fsspec.open(target, "wb") as f:
            torch.save(checkpoint, f)

    # === save optimizer ===
    if inner_optimizer is not None:
        opt_checkpoint = {
            "optimizer_state_dict": inner_optimizer.state_dict(),
        }

        target = os.path.join(checkpoint_path, f"inner_optimizer.pt")
        if not os.path.exists(target):
            with fsspec.open(target, "wb") as f:
                torch.save(opt_checkpoint, f)

    if outer_optimizer is not None:
        opt_checkpoint = {
            "optimizer_state_dict": outer_optimizer.state_dict(),
        }
        target = os.path.join(checkpoint_path, f"outer_optimizer.pt")
        if not os.path.exists(target):
            with fsspec.open(target, "wb") as f:
                torch.save(opt_checkpoint, f)

    # === save dataloader ===
    if data_loader is not None:
        rank_state_dict = {}
        rank_state_dict["data_loader"] = data_loader.state_dict()

        target = os.path.join(checkpoint_path, f"dataloader_rank{rank}.pt")
        if not os.path.exists(target):
            with fsspec.open(target, "wb") as f:
                torch.save(rank_state_dict, f)

        del rank_state_dict

    if not save_global_state:
        return

    # === save global state ===
    global_state_dict = {
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "loss": loss if loss is not None else 0,
    }

    if inner_scaler is not None:
        global_state_dict["inner_scaler_state_dict"] = inner_scaler.state_dict()

    if outer_scaler is not None:
        global_state_dict["outer_scaler_state_dict"] = outer_scaler.state_dict()

    target = os.path.join(checkpoint_path, "global_state.pt")
    if not os.path.exists(target):
        with fsspec.open(target, "wb") as f:
            torch.save(global_state_dict, f)

    logger.info(f"Checkpoint saved to {checkpoint_path}")

    del checkpoint
    del opt_checkpoint
    del global_state_dict


def load_optimizer(checkpoint_path, optimizer):
    def _get_name_to_id(optimizer_state_dict):
        param_name = [pid for g in optimizer_state_dict["param_groups"] for pid in g["param_names"]]
        param_id = [pid for g in optimizer_state_dict["param_groups"] for pid in g["params"]]
        param_name_to_id = {name: pid for name, pid in zip(param_name, param_id)}
        return param_name_to_id

    def _get_name_to_param(optimizer_state_dict):
        """
        Build a mapping: parameter_name -> optimizer_state (for that param).
        Works regardless of the optimizer’s internal param IDs.
        """
        state = optimizer_state_dict["state"]
        name_to_id = _get_name_to_id(optimizer_state_dict)
        name_to_state = {param_name: state[pid] for param_name, pid in name_to_id.items()}
        return name_to_state

    def _update_state_dict(optimizer_state_dict, name_to_param):
        """
        Build a *loadable* optimizer.state_dict() for `target_optimizer` such that:
        - Params that belong to the requested experts get their merged state.
        - All other params are left without state (the optimizer will re-init them).
        """

        optimizer_state_dict["state"] = {}

        target_name_to_id = _get_name_to_id(optimizer_state_dict)

        for pid, name in target_name_to_id.items():
            st = name_to_param.get(name, None)
            if st is not None:
                optimizer_state_dict["state"][pid] = deepcopy(st)  # avoid aliasing

        return optimizer_state_dict

    full_name_to_param = {}
    model_files = fsspec.open_files(checkpoint_path, mode="rb")

    if len(model_files) == 0:
        return

    for f in model_files:
        with f as fh:
            state_dict = torch.load(fh, map_location=torch.device("cpu"))
            full_name_to_param = full_name_to_param | _get_name_to_param(state_dict["optimizer_state_dict"])

    optimizer.load_state_dict(_update_state_dict(optimizer.state_dict(), full_name_to_param))


def load_checkpoint(
    checkpoint_path: str,
    config: MinerConfig,
    rank: int | None,
    device: torch.device,
    model: torch.nn.Module | None = None,
    inner_optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    outer_optimizer: torch.optim.Optimizer | None = None,
    inner_scaler: torch.amp.GradScaler | None = None,
    outer_scaler: torch.amp.GradScaler | None = None,
    data_loader: StatefulDataLoader | None = None,
) -> float:
    """Load the model and optimizer state from a checkpoint folder

    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to load
        optimizer: the optimizer to load
        scheduler: the scheduler to load
        outer_optimizer: the outer optimizer to load
        data_loader: the data loader to load

    Returns:
        loss: the loss from the checkpoint
    """

    if model is not None:
        full_state_dict = {}
        model_files = fsspec.open_files(os.path.join(checkpoint_path, "model*.pt"), mode="rb")
        model_files += fsspec.open_files(os.path.join(checkpoint_path, "model.pt"), mode="rb")
        for f in model_files:
            with f as fh:
                state_dict = torch.load(fh, map_location=torch.device("cpu"))
                full_state_dict = full_state_dict | state_dict["model_state_dict"]
                logger.info(f"rank {rank}: loaded checkpoint {f} loss {state_dict['loss']:.5f}")

        model.load_state_dict(full_state_dict, strict=True)
        model.to(device)

    if inner_optimizer is not None:
        load_optimizer(os.path.join(checkpoint_path, "inner_optimizer*.pt"), inner_optimizer)

    if outer_optimizer is not None:
        load_optimizer(os.path.join(checkpoint_path, "outer_optimizer*.pt"), outer_optimizer)

    if data_loader is not None:
        with fsspec.open(os.path.join(checkpoint_path, f"dataloader_rank{rank}.pt"), "rb") as f:
            rank_state_dict = torch.load(f, map_location=torch.device("cpu"))
        data_loader.load_state_dict(rank_state_dict["data_loader"])

    with fsspec.open(os.path.join(checkpoint_path, f"global_state.pt"), "rb") as f:
        global_state_dict = torch.load(f, map_location=torch.device("cpu"))

    if scheduler is not None:
        scheduler.load_state_dict(global_state_dict["scheduler"])
        inner_optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]

        # Push optimizer tensors to the same device as the model
        for st in outer_optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)

    if inner_scaler is not None:
        inner_scaler.load_state_dict(global_state_dict["inner_scaler_state_dict"])

    if outer_scaler is not None:
        outer_scaler.load_state_dict(global_state_dict["outer_scaler_state_dict"])

    return global_state_dict["loss"]

def get_sorted_checkpoints(checkpoint_path: str):
    fs, root = fsspec.core.url_to_fs(checkpoint_path)

    ckpt_files = {}
    for f in fs.ls(root, detail=False):
        meta = parse_dynamic_filename(f)
        if meta is None:
            continue

        # ensure both fields exist and are numeric
        meta["globalopt"] = int(meta.get("globalopt") or 0)
        meta["inneropt"] = int(meta.get("inneropt") or 0)
        ckpt_files[f] = meta

    # sort descending by globalopt, then inneropt
    sorted_ckpt_files = sorted(
        ckpt_files.items(),
        key=lambda item: (-item[1]["globalopt"], -item[1]["inneropt"]),
    )

    return sorted_ckpt_files



def delete_old_checkpoints(checkpoint_path: str, topk: int) -> list[str]:
    """
    Deletes old checkpoints, keeping only the top 'k' most recent ones.

    Args:
        checkpoint_path (str): The path to the checkpoint directory.
        topk (int): The number of recent checkpoints to keep.

    Returns:
        list[str]: A list of deleted checkpoint filenames.
    """
    fs = GenericFileSystem()
    sorted_ckpt_files = get_sorted_checkpoints(checkpoint_path)
     
    ckpt_deleted = []
    for ckpt_file, order in sorted_ckpt_files[:-topk]:
        fs.rm(ckpt_file, recursive=True)
        ckpt_deleted.append(ckpt_file)
    return ckpt_deleted


def delete_old_checkpoints_by_hotkey(folder_path : Path):
    """
    Deletes all non-latest submission files coming from the same hotkey.
    Keeps only the file with the highest block number per hotkey.

    Requires: parse_dynamic_filename(filename: str) -> dict
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    # Step 1: Group files by hotkey
    submissions_by_hotkey = {}
    for file_path in folder_path.glob("*.pt"):
        meta = parse_dynamic_filename(file_path.name)
        if "hotkey" not in meta or "block" not in meta:
            print(f"⚠️ Skipping malformed filename: {file_path.name}")
            continue

        hotkey = meta["hotkey"]
        block = meta["block"]

        # Track the latest submission per hotkey
        if hotkey not in submissions_by_hotkey:
            submissions_by_hotkey[hotkey] = []
        submissions_by_hotkey[hotkey].append((block, file_path))

    # Step 2: For each hotkey, keep only the highest block file
    deleted_files = []
    for hotkey, entries in submissions_by_hotkey.items():
        # Sort by block number descending (latest first)
        entries.sort(key=lambda x: x[0], reverse=True)

        # Keep the first (latest) one, delete the rest
        for block, file_path in entries[1:]:
            try:
                os.remove(file_path)
                deleted_files.append(file_path.name)
            except Exception as e:
                print(f"❌ Failed to delete {file_path.name}: {e}")

    # Step 3: Log result
    if deleted_files:
        logger.info(f"🧹 Deleted {len(deleted_files)} outdated submission(s):", deleted_files)
        for f in deleted_files:
            print(f"   - {f}")
    else:
        logger.info("✅ No outdated submissions found.")

    return deleted_files