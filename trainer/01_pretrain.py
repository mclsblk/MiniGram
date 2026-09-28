"""TOML-driven pretraining. SFT/GRPO migrate to the shared APIs separately."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset.data_utils import PretrainDataset
from trainer.config_utils import apply_cli_overrides, load_train_config, CLI_FIELDS
from trainer.ddp_utils import (
    barrier, build_distributed_sampler, build_worker_seed_fn, cleanup_distributed,
    init_distributed, maybe_no_sync, rank0_log, reduce_metrics, set_sampler_epoch, wrap_ddp,
)
from trainer.train_utils import (
    build_amp, build_model_config, create_model, get_lr, get_param, get_remaining_time,
    load_checkpoint, load_tokenizer, log, log_train_metrics, save_checkpoint, set_seed,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniGram TOML pretraining")
    parser.add_argument("--config", required=True, help="Training TOML file")
    parser.add_argument("--resume_from", help="Training checkpoint .pth file")
    parser.add_argument("--device", help="Override runtime.device (auto, cpu, cuda:0)")
    parser.add_argument("--save_dir", help="Override output.save_dir, relative to current directory")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args(argv)


def validate_progress(progress, steps_per_epoch, epochs, accumulation_steps):
    fields = ("epoch", "epoch_step", "micro_step", "optimizer_step")
    if set(progress) != set(fields) or any(type(progress[k]) is not int or progress[k] < 0 for k in fields):
        raise ValueError("Invalid checkpoint progress")
    epoch, step = progress["epoch"], progress["epoch_step"]
    if epoch > epochs or step >= steps_per_epoch or (epoch == epochs and step != 0):
        raise ValueError("Checkpoint next-batch position is outside the training schedule")
    if step % accumulation_steps:
        raise ValueError("Checkpoint is not at an accumulation boundary")
    if progress["micro_step"] != epoch * steps_per_epoch + step:
        raise ValueError("Checkpoint micro_step and next-batch position disagree")
    max_updates = epoch * ((steps_per_epoch + accumulation_steps - 1) // accumulation_steps) + step // accumulation_steps
    if progress["optimizer_step"] > max_updates:
        raise ValueError("Checkpoint optimizer_step exceeds completed accumulation windows")


def train(config, resume_from, ddp_state):
    data, train_cfg, runtime, output = (config[key] for key in ("data", "train", "runtime", "output"))
    device = ddp_state.device
    info = lambda msg: rank0_log(ddp_state, msg, log)
    set_seed(train_cfg["seed"] + ddp_state.rank, deterministic=False)
    info(f"device={device}, dtype={train_cfg['dtype']}, rank={ddp_state.rank}/{ddp_state.world_size}")
    if device.type != "cuda" and train_cfg["dtype"] != "fp32":
        info("Non-CUDA execution uses FP32; configured autocast dtype applies only on CUDA.")
    if device.type == "cuda" and train_cfg["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("train.dtype=bf16 is unsupported on this CUDA device")

    tokenizer = load_tokenizer(data.get("tokenizer_path", data.get("tokenizer_name")))
    model_options = dict(config["model"])
    # SDPA also supports CPU. Keep the effective model config independent of device overrides.
    model_options.setdefault("flash_attention", True)
    lm_config = build_model_config(model_options, tokenizer)
    resolved = copy.deepcopy(config)
    resolved["model"] = json.loads(json.dumps(lm_config.to_dict()))
    dataset = PretrainDataset(data["data_path"], tokenizer, max_length=data["max_length"])
    sampler = build_distributed_sampler(dataset, ddp_state, shuffle=False, drop_last=False,
                                        seed=train_cfg["seed"])
    generator = torch.Generator().manual_seed(train_cfg["seed"] + ddp_state.rank)
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], sampler=sampler, shuffle=False, drop_last=False,
        num_workers=data["num_workers"], pin_memory=device.type == "cuda",
        worker_init_fn=build_worker_seed_fn(train_cfg["seed"], ddp_state.rank), generator=generator,
    )
    steps_per_epoch = len(loader)
    if not steps_per_epoch:
        raise ValueError("Training dataloader is empty")
    total_steps = train_cfg["epochs"] * steps_per_epoch
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    accumulation = train_cfg["accumulation_steps"]
    progress = {"epoch": 0, "epoch_step": 0, "micro_step": 0, "optimizer_step": 0}
    model = create_model(lm_config, tokenizer, prepare_mapping=not resume_from).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                            weight_decay=train_cfg["weight_decay"])
    autocast_ctx, scaler = build_amp(train_cfg["dtype"], device.type)
    resume_config = {"train": train_cfg, "data": data, "world_size": ddp_state.world_size,
                     "steps_per_epoch": steps_per_epoch}
    if resume_from:
        saved = load_checkpoint(resume_from, model, optimizer, scaler)
        if saved["train_config"] != resume_config:
            raise ValueError("Resume training configuration mismatch")
        progress = saved["step"]
        validate_progress(progress, steps_per_epoch, train_cfg["epochs"], accumulation)
        del saved
        info(f"Resuming at {progress}")
    if runtime["use_compile"]:
        model = torch.compile(model)
        info("torch.compile enabled")
    model = wrap_ddp(model, ddp_state)
    info(f"Model parameters: {get_param(model)['total_params_human']}")
    info("Effective model config: " + json.dumps(resolved["model"], ensure_ascii=False, sort_keys=True))
    if ddp_state.is_main:
        Path(output["save_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(output["save_dir"]) / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started_at = time.time()
    start_micro = progress["micro_step"]
    start_epoch, start_batch = progress["epoch"], progress["epoch_step"]
    pending_save = False

    def save():
        if ddp_state.is_main:
            filename = output["save_name"] + ".pth"
            save_checkpoint(Path(output["save_dir"]) / "checkpoint" / filename, model,
                            optimizer=optimizer, scaler=scaler, step=dict(progress), train_config=resume_config)
            save_checkpoint(Path(output["save_dir"]) / filename, model)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        set_sampler_epoch(sampler, epoch)
        skip = start_batch if epoch == start_epoch else 0
        for batch_idx, (input_ids, labels) in enumerate(loader):
            if batch_idx < skip:
                continue
            progress["micro_step"] += 1
            micro_step = progress["micro_step"]
            lr = get_lr(micro_step, total_steps, train_cfg["learning_rate"], warmup_steps, train_cfg["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            mask = labels.ne(-100)  # Pretrain only: SFT prompt masking has different semantics.
            window_start = (batch_idx // accumulation) * accumulation
            window_size = min(accumulation, steps_per_epoch - window_start)
            should_step = batch_idx + 1 == window_start + window_size
            with maybe_no_sync(model, ddp_state, should_step):
                with autocast_ctx():
                    outputs = model(input_ids=input_ids, labels=labels, attention_mask=mask, use_cache=False)
                    total_loss = outputs.loss + outputs.aux_loss
                scaler.scale(total_loss / window_size).backward()
            if should_step:
                scaler.unscale_(optimizer)
                if train_cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    progress["optimizer_step"] += 1
                optimizer.zero_grad(set_to_none=True)
            last_batch = batch_idx + 1 == steps_per_epoch
            progress["epoch"] = epoch + 1 if last_batch else epoch
            progress["epoch_step"] = 0 if last_batch else batch_idx + 1
            if micro_step % output["log_interval"] == 0 or micro_step == total_steps:
                metrics = reduce_metrics({"loss": total_loss, "logits_loss": outputs.loss,
                                          "aux_loss": outputs.aux_loss}, ddp_state)
                eta = get_remaining_time(micro_step, total_steps, start_micro, started_at)
                if ddp_state.is_main:
                    log_train_metrics(f"epoch[{epoch + 1}/{train_cfg['epochs']}] micro[{micro_step}/{total_steps}] "
                                      f"optimizer[{progress['optimizer_step']}]", metrics, lr, eta)
            pending_save |= micro_step % output["save_interval"] == 0 or micro_step == total_steps
            if pending_save and should_step:
                save()
                pending_save = False
            del outputs, total_loss, input_ids, labels, mask
        info(f"Epoch {epoch + 1} complete.")
    if start_micro == total_steps:
        save()
    barrier(ddp_state)
    info("Training complete.")


def main(argv=None):
    args = parse_args(argv)
    overrides = {key: getattr(args, key) for key in CLI_FIELDS}
    config = apply_cli_overrides(load_train_config(args.config, "pretrain"), overrides)
    try:
        requested = config["runtime"]["device"]
        ddp_state = init_distributed(SimpleNamespace(device=None if requested == "auto" else requested))
        rank0_log(ddp_state, f"Config: {Path(args.config).expanduser().resolve()}; CLI overrides: "
                  + json.dumps({k: v for k, v in overrides.items() if v is not None}), log)
        train(config, args.resume_from, ddp_state)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
