import datetime
import gc
import os
import time

import bittensor
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

from mycelia.miner.train_helper import free_cuda_models, get_status
from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import setup_chain_worker
from mycelia.shared.checkpoint_helper import (
    ModelMeta,
    delete_old_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from mycelia.shared.checkpoints import (
    select_best_checkpoint,
)
from mycelia.shared.config import MinerConfig, parse_args
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.evaluate import evaluate_model
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.helper import get_model_hash, get_nested_attr
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.model import freeze_parameters, load_model
from mycelia.shared.modeling.mycelia import get_base_tokenizer

configure_logging()
logger = structlog.get_logger(__name__)
torch.autograd.set_detect_anomaly(True)

# this is for local DP only
def init_process(local_rank: int, config: MinerConfig, world_size: int, fn: callable, backend: str = "nccl") -> None:
    """
    Initializes the process for distributed training.

    Args:
        rank (int): The rank of the process.
        world_size (int): The total number of processes.
        fn (callable): The function to run for the process.
        backend (str): The backend to use for distributed training.

    Returns:
        None
    """
    if local_rank == 0:
        print(config)

    if world_size > 1:
        os.environ["MASTER_ADDR"] = config.local_par.ip_address
        os.environ["MASTER_PORT"] = str(config.local_par.port)

        dist.init_process_group(
            backend,
            rank=local_rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=3600),
            device_id=(
                torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
                if local_rank < world_size
                else None
            ),
        )

    fn(local_rank, world_size, config)


def setup_training(
    config,
    rank: int,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    current_model_meta: ModelMeta,
) -> tuple[
    torch.nn.Module,  # model
    torch.optim.Optimizer,  # inner_optimizer
    torch.amp.GradScaler,  # inner_scaler
    torch.optim.lr_scheduler.LRScheduler,  # scheduler
    "ExpertManager",  # em
    StatefulDataLoader,
    dict,  # current model version
]:
    """
    Build model(s), experts layout, optimizers, scheduler, scaler, and optionally resume from a checkpoint.

    Args:
        config: Training/config object with attributes used here (e.g., lr, outer_lr, warmup_steps, etc.).
        rank (int): Process rank.
        device (str | torch.device): Device for the local model (e.g., "cuda:0").

    Returns:
        model (nn.Module): Local (possibly partial) MoE model placed on `device`.
        global_model (nn.Module): Deep-copied global model on CPU, kept in sync with `model`.
        inner_optimizer (Optimizer): Optimizer for `model`.
        outer_optimizer (Optimizer): Optimizer for `global_model`.
        scaler (torch.cuda.amp.GradScaler): GradScaler (enabled iff `config.model.precision == "fp16-mixed"`).
        scheduler (LRScheduler): LR scheduler attached to `inner_optimizer`.
        start_step (int): Step to resume from (0 if starting fresh).
        expert_groups (Sequence[Sequence[int]]): Grouping returned by `create_expert_groups`; typically a list
            (or other sequence) of groups where each group lists the ranks/experts belonging to it.
        group_ids (int): This rank’s group id from `create_expert_groups`.
        expert_manager (ExpertManager): The instantiated ExpertManager for this model/rank.

    Notes:
        - Param group layouts are taken from the *target* optimizers created here.
        - If `resume_from_ckpt` is set and a checkpoint is found, model/opt/scheduler/scaler states are restored
          before syncing `global_model` from `model`.
    """
    logger.info("(0) Setup training")

    # === model & Experts manager ===
    logger.info(f"init - model and expert manager")
    expert_manager = ExpertManager(config)
    model, model_checkpoint = load_model(rank, config, expert_manager, subtensor, wallet)
    model = model.to(device)
    model = freeze_parameters(model=model, expert_manager=expert_manager, expert_group_id=config.task.exp.group_id)

    # === optimizers ===
    logger.info(f"init - optimizer")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"trainable params: {len(trainable_params)} / total: {sum(1 for _ in model.parameters())}")
    if len(trainable_params) == 0:
        sample_names = [name for name, _ in list(model.named_parameters())[:8]]
        logger.warning(
            "No trainable parameters found; check expert_group_id and get_layer_expert_id() matching.",
            expert_group_id=config.task.exp.group_id,
            sample_param_names=sample_names,
        )
    inner_optimizer = torch.optim.AdamW(trainable_params, lr=config.opt.lr, weight_decay=0.1, betas=(0.9, 0.95))

    # === scheduler === (for inner optimizer)
    logger.info(f"init - scheduler")
    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=config.sched.warmup_steps,
        num_training_steps=config.sched.total_steps,
    )

    # === scaler ===
    logger.info(f"init - inner scaler")
    precision = get_nested_attr(config, "model.precision", "fp16-mixed")
    if precision == "bf16-mixed" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        logger.warning("BF16 not supported on this device; falling back to fp16-mixed")
        precision = "fp16-mixed"
    inner_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False# (precision == "fp16-mixed"),
    )

    # === dataloader ===
    logger.info(f"init - train dataloader")
    train_dataloader = get_dataloader(config, rank=rank, world_size=config.task.exp.data.world_size, tokenizer=tokenizer)

    # === load checkpoint (if any) ===
    logger.info(f"init - load checkpoint")
    resume = False

    if get_nested_attr(config, "ckpt.resume_from_ckpt", False):
        latest_checkpoint = select_best_checkpoint(config.ckpt.checkpoint_path, resume=config.ckpt.resume)

    if get_nested_attr(config, "resume_from_ckpt", False) and resume and latest_checkpoint.path is not None:
        _ = load_checkpoint(
            config=config,
            checkpoint_path=latest_checkpoint.path,
            inner_optimizer=inner_optimizer,
            scheduler=scheduler,
            inner_scaler=inner_scaler,
            rank=rank,
            device=device,
            data_loader=train_dataloader,
        )

    logger.info(f"setup_training: success!")
    return (
        model,
        inner_optimizer,
        inner_scaler,
        scheduler,
        expert_manager,
        train_dataloader,
        model_checkpoint,
    )


def sum_model_gradients(model):
    """
    Returns the sum of absolute gradients of all model parameters.
    Assumes backward() has already been called.
    """
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total += param.grad.abs().sum().item()
        return total


def train_worker(rank: int, world_size: int, config: MinerConfig) -> None:
    """
    The worker function for training in a distributed setting.

    Args:
        rank (int): The rank of the process.
        world_size (int): The total number of processes.
        config (Config): The configuration object for the training.

    Returns:
        None
    """
    eval_rref = None
    if rank == 0:
        config.write()

    # === set logging ===
    metric_logger = MetricLogger(config, rank)

    # === set up chain worker ===
    wallet, subtensor = setup_chain_worker(config)

    # === mis ===
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = get_base_tokenizer(config)

    eval_dataloader = get_dataloader(
        config,
        rank=config.local_par.world_size,
        world_size=config.local_par.world_size + 1,
        tokenizer=tokenizer,
    )

    # === set up training ===
    (
        model,
        inner_optimizer,
        inner_scaler,
        scheduler,
        expert_manager,
        train_dataloader,
        current_model_meta,
    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta=None)

    # === training ===
    precision = get_nested_attr(config, "model.precision", "fp16-mixed")
    if precision == "bf16-mixed" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        logger.warning("BF16 not supported on this device; falling back to fp16-mixed")
        precision = "fp16-mixed"
    amp_enabled = precision in ("fp16-mixed", "bf16-mixed")
    autocast_dtype = torch.float16 if precision == "fp16-mixed" else torch.bfloat16
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    training_time = 0
    total_training_time = 0
    training_start_time = None

    inner_optimizer.zero_grad()
    try:
        for step, batch in enumerate(
            iterable=train_dataloader,
            start=max(0, current_model_meta.inner_opt) * config.local_par.gradient_accumulation_steps,
        ):
            # for each step, we run 1 backward
            # for each inner_opt_step, we run local optimization; gradient_accumulation_steps = 1 real step
            # for each global_opt_interval number of inner_opt_step, we synchronise weight from different ddp worker, and then run global optimization

            inner_opt_step = step // config.local_par.gradient_accumulation_steps
            is_inner_optimizer_step = step % config.local_par.gradient_accumulation_steps == 0
            is_start_step = step == current_model_meta.inner_opt * config.local_par.gradient_accumulation_steps
            current_model_meta.inner_opt = inner_opt_step

            # === Training and inner optimization ===
            if is_inner_optimizer_step:
                logger.info(
                    "(1) Start epoch training",
                    step=step,
                    inner_opt_step=inner_opt_step,
                    is_inner_optimizer_step=is_inner_optimizer_step,
                    gradient_accumulation_steps=config.local_par.gradient_accumulation_steps,
                    current_model_meta=current_model_meta,
                )
            if (
                not is_start_step
            ):  # skip training when it is the start step, so that we can benchamrk the original model first
                model.train()
                if training_start_time is None:
                    training_start_time = time.time()
                batch_device = {}
                for key in batch.keys():
                    batch_device[key] = batch[key].to(device)
                labels = batch_device.get("labels")
                if labels is not None:
                    logger.info("detected none label")
                    valid_labels = labels.ne(-100)
                    num_valid = int(valid_labels.sum().item())
                    if num_valid == 0:
                        logger.warning("Skipping batch with no valid labels", step=step)
                        continue

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(**batch_device)

                    loss = outputs.loss / config.local_par.gradient_accumulation_steps
                    # aux_loss = outputs.aux_loss / config.local_par.gradient_accumulation_steps if outputs.aux_loss is not None else torch.tensor(0)
                    aux_loss = torch.tensor(0)

                if not torch.isfinite(loss):
                    logits = outputs.logits
                    logits_finite = torch.isfinite(logits)
                    logits_finite_ratio = float(logits_finite.float().mean().item())
                    logits_min = float(logits.min().item())
                    logits_max = float(logits.max().item())
                    label_min = None
                    label_max = None
                    if labels is not None and num_valid > 0:
                        label_min = int(labels[valid_labels].min().item())
                        label_max = int(labels[valid_labels].max().item())
                    logger.error(
                        "Non-finite loss detected",
                        loss=float(loss.item()) if loss.numel() == 1 else None,
                        logits_min=logits_min,
                        logits_max=logits_max,
                        logits_finite_ratio=logits_finite_ratio,
                        label_min=label_min,
                        label_max=label_max,
                        num_valid_labels=num_valid if labels is not None else None,
                        precision=precision,
                        step=step,
                    )
                    raise RuntimeError("Non-finite loss detected; see logs for details.")
                logger.info("batch loss", loss)

                loss_batch += loss.item()
                aux_loss_batch += aux_loss.item()

                inner_scaler.scale(loss).backward()

                grad_total = sum_model_gradients(model)
                sample_grads = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        p_norm = param.norm().item()
                        grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                        sample_grads.append((name, grad_norm, p_norm))
                        if len(sample_grads) >= 5:
                            break

                # === Aggressively free intermediate tensors ===
                del loss, aux_loss, batch_device, outputs
                gc.collect()

            # === inner optimizer ===
            if not is_start_step and is_inner_optimizer_step:
                old_model_hash = get_model_hash(model.state_dict(), hex = True)

                for n, p in model.named_parameters():
                    if p.grad is None or torch.isnan(p.grad.sum()):
                        continue
                    # dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size)
                inner_scaler.unscale_(optimizer=inner_optimizer)

                grad_norm = clip_grad_norm_(
                    [p for p in model.parameters() if p.grad is not None and not torch.isnan(p.grad.sum())], 1.0
                )

                scale_before = inner_scaler.get_scale() if inner_scaler.is_enabled() else None
                step_result = inner_scaler.step(inner_optimizer)
                step_skipped = inner_scaler.is_enabled() and step_result is None

                logger.info(
                        "GradScaler for optimizer step",
                        grad_norm=grad_norm,
                        grad_sum = sum_model_gradients(model),
                        scale_before=scale_before,
                        skipped = step_skipped
                )

                inner_scaler.update()
                if inner_scaler.is_enabled():
                    logger.info(
                        "Scaler updated",
                        scale_after=inner_scaler.get_scale(),
                    )

                scheduler.step()
                inner_optimizer.zero_grad()

                training_time = time.time() - training_start_time
                total_training_time += training_time
                training_start_time = None

                # === Clear memory after optimizer step ===
                gc.collect()
                torch.cuda.empty_cache()

                new_model_hash = get_model_hash(model.state_dict(), hex = True)
                logger.info(f"Updated model", old_model_hash=old_model_hash, new_model_hash=new_model_hash)

            # === Log metric ===
            if (
                is_inner_optimizer_step
                and inner_opt_step % max(round(config.local_par.global_opt_interval * 0.02), 1) == 0
            ):
                logger.info("(2) Logging step", loss_batch=loss_batch, aux_loss_batch=aux_loss_batch)
                metrics = get_status(
                    config=config,
                    model=model,
                    step=step,
                    inner_opt_step=inner_opt_step,
                    training_time=training_time,
                    total_training_time=total_training_time,
                    inner_optimizer=inner_optimizer,
                    loss_batch=loss_batch,
                    aux_loss_batch=aux_loss_batch,
                )
                metric_logger.log(metrics, print_log=False)

            # === local validation and log metric ===
            if is_inner_optimizer_step and inner_opt_step % config.log.metric_interval == 0:
                logger.info("(3) Local evaluation")

                val_metric = evaluate_model(
                    rank=rank, step=inner_opt_step, model=model, eval_dataloader=train_dataloader, device=device
                )

                metrics = (
                    get_status(
                        config=config,
                        model=model,
                        step=step,
                        inner_opt_step=inner_opt_step,
                        training_time=training_time,
                        total_training_time=total_training_time,
                        inner_optimizer=inner_optimizer,
                        loss_batch=loss_batch,
                        aux_loss_batch=aux_loss_batch,
                    )
                    | val_metric
                )

                metric_logger.log(metrics)

                logger.info("reached barrier, waiting for partial validation and metric logging to complete")
                # dist.barrier(device_ids=[rank])

            # === save checkpoint ===
            if (
                is_inner_optimizer_step
                and config.ckpt.checkpoint_interval is not None
                and inner_opt_step % config.ckpt.checkpoint_interval == 0
            ):
                logger.info("(4) Saving checkpoint")

                ckpt_path = os.path.join(
                    config.ckpt.checkpoint_path,
                    f"globalver_{current_model_meta.global_ver}_inneropt_{inner_opt_step}",
                )

                save_checkpoint(
                    checkpoint_path=ckpt_path,
                    model=model,
                    inner_optimizer=inner_optimizer,
                    scheduler=scheduler,
                    loss=loss_batch.item(),
                    inner_scaler=inner_scaler,
                    data_loader=train_dataloader,
                    save_global_state=rank == 0,
                    rank=rank,
                    save_model_by_expert_group=True,
                    expert_manager=expert_manager,
                )

                if config.ckpt.checkpoint_topk is not None:
                    ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                    if ckpt_deleted:
                        logger.info(f"Deleted old checkpoints: {ckpt_deleted}")

                logger.info("reached barrier, waiting for complete checkpoint saving")
                # dist.barrier(device_ids=[rank])

            # === reload model ===
            if is_inner_optimizer_step:
                logger.info("(5) Reload Model")

                newest_checkpoint = select_best_checkpoint(
                    primary_dir=config.ckpt.validator_checkpoint_path,
                    secondary_dir=config.ckpt.checkpoint_path,
                )

                if newest_checkpoint > current_model_meta:
                    logger.info(
                        "Should reload model",
                        newest_checkpoint=newest_checkpoint,
                        current_model_meta=current_model_meta,
                    )
                    # dist.barrier(device_ids=[rank])  # make sure everything is saved and everyone is ready to load
                    logger.info("freeing cuda memory")
                    free_cuda_models(models=[model], optimizers=[inner_optimizer], devices=[device])
                    logger.info(
                        "restarting model",
                        current_model_meta=current_model_meta,
                        largest_avail_model=select_best_checkpoint(
                            primary_dir=config.ckpt.validator_checkpoint_path,
                            secondary_dir=config.ckpt.checkpoint_path,
                        )
                    )
                    (
                        model,
                        inner_optimizer,
                        inner_scaler,
                        scheduler,
                        expert_manager,
                        train_dataloader,
                        current_model_version,
                    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta)
                else:
                    logger.info(
                        "No need to reload model",
                        newest_checkpoint=newest_checkpoint,
                        current_model_meta=current_model_meta,
                    )

            # === Clean up ===
            if is_inner_optimizer_step:
                logger.info("(6) Clean up")
                loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
                aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Clean up completed")

        logger.info("used up train dataloader")

    except Exception:
        logger.error("Quit training", exc_info=True)
        # dist.destroy_process_group()
        torch.cuda.synchronize()
        metric_logger.close()

        if rank == 0:
            torch.save(model.state_dict(), "mycelia_final.pt")


def run_distributed_training() -> None:
    """
    Runs the distributed training process.

    Returns:
        None
    """
    args = parse_args()

    if args.path:
        config = MinerConfig.from_path(args.path)
    else:
        config = MinerConfig()

    if config.local_par.world_size > 1:
        mp.spawn(
            init_process,
            args=(config, config.local_par.world_size, train_worker),
            nprocs=config.local_par.world_size,
        )
    else:
        init_process(0, config, config.local_par.world_size, train_worker)


if __name__ == "__main__":
    run_distributed_training()
