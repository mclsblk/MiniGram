"""TOML-driven SFT initialized from a new-model checkpoint."""
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

from dataset.data_utils import SFTDataset
from trainer.config_utils import apply_cli_overrides, load_train_config, CLI_FIELDS
from trainer.ddp_utils import (
    barrier, build_distributed_sampler, build_worker_seed_fn, cleanup_distributed,
    init_distributed, maybe_no_sync, rank0_log, reduce_metrics, set_sampler_epoch, wrap_ddp,
)
from trainer.train_utils import (
    build_amp, create_model_from_checkpoint, get_lr, get_param, get_remaining_time,
    load_tokenizer, log, log_train_metrics, save_checkpoint, set_seed, validate_progress,
    restore_training_state,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniGram TOML supervised fine-tuning")
    parser.add_argument("--config", required=True, help="Training TOML file")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--init_from", help="New-model checkpoint used to initialize SFT")
    source.add_argument("--resume_from", help="SFT training checkpoint used to resume")
    parser.add_argument("--device", help="Override runtime.device (auto, cpu, cuda:0)")
    parser.add_argument("--save_dir", help="Override output.save_dir, relative to current directory")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args(argv)


def train(config, init_from, resume_from, ddp_state):
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
    source = Path(resume_from or init_from).expanduser().resolve()
    model, saved = create_model_from_checkpoint(source, tokenizer)
    lm_config = model.config
    if data["max_length"] > lm_config.max_length:
        raise ValueError("data.max_length exceeds checkpoint model.max_length")
    resolved = copy.deepcopy(config)
    resolved["model"] = json.loads(json.dumps(lm_config.to_dict()))
    resolved["resume_from" if resume_from else "init_from"] = str(source)
    dataset = SFTDataset(
        data["data_path"], tokenizer, max_length=data["max_length"],
        train_on_prompt=data["train_on_prompt"], return_attention_mask=True,
    )
    sampler = build_distributed_sampler(dataset, ddp_state, shuffle=True, drop_last=True,
                                        seed=train_cfg["seed"])
    generator = torch.Generator().manual_seed(train_cfg["seed"] + ddp_state.rank)
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], sampler=sampler, shuffle=sampler is None, drop_last=True,
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
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                            weight_decay=train_cfg["weight_decay"])
    autocast_ctx, scaler = build_amp(train_cfg["dtype"], device.type)
    resume_config = {"stage": "sft", "train": train_cfg, "data": data, "world_size": ddp_state.world_size,
                     "steps_per_epoch": steps_per_epoch}
    if resume_from:
        if saved.get("train_config") != resume_config:
            raise ValueError("Resume training configuration mismatch")
        restore_training_state(saved, optimizer, scaler)
        progress = saved["step"]
        validate_progress(progress, steps_per_epoch, train_cfg["epochs"], accumulation)
        info(f"Resuming at {progress}")
    del saved
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
        generator.manual_seed(train_cfg["seed"] + epoch)
        skip = start_batch if epoch == start_epoch else 0
        for batch_idx, (input_ids, labels, mask) in enumerate(loader):
            if batch_idx < skip:
                continue
            progress["micro_step"] += 1
            micro_step = progress["micro_step"]
            lr = get_lr(micro_step, total_steps, train_cfg["learning_rate"], warmup_steps, train_cfg["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            mask = mask.to(device, non_blocking=device.type == "cuda")
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
    config = apply_cli_overrides(load_train_config(args.config, "sft"), overrides)
    try:
        requested = config["runtime"]["device"]
        ddp_state = init_distributed(SimpleNamespace(device=None if requested == "auto" else requested))
        rank0_log(ddp_state, f"Config: {Path(args.config).expanduser().resolve()}; CLI overrides: "
                  + json.dumps({k: v for k, v in overrides.items() if v is not None}), log)
        train(config, args.init_from, args.resume_from, ddp_state)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
