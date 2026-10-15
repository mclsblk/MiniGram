"""Single-process TOML LoRA training with structured adapter exports."""
import argparse
import copy
from itertools import islice
import json
import os
from pathlib import Path
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset.data_utils import SFTDataset
from model.lora import apply_lora, mark_only_lora_as_trainable
from trainer.config_utils import CLI_FIELDS, apply_cli_overrides, load_train_config
from trainer.ddp_utils import build_worker_seed_fn
from trainer.lora_utils import load_lora_adapter, read_lora_adapter, save_lora_adapter
from trainer.train_utils import (
    build_amp, create_model_from_checkpoint, get_lr, get_param, get_remaining_time,
    load_tokenizer, log, log_train_metrics, set_seed,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniGram TOML LoRA (single process, adapter export only)")
    parser.add_argument("--config", required=True, help="LoRA TOML file")
    parser.add_argument("--init_from", required=True, help="New-model checkpoint used as the frozen base")
    parser.add_argument("--adapter_path", help="Optional structured adapter used for fresh training")
    parser.add_argument("--device", help="Override runtime.device (auto, cpu, cuda:0)")
    parser.add_argument("--save_dir", help="Override output.save_dir, relative to current directory")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args(argv)


def train(config, init_from, adapter_path=None):
    data, train_cfg, runtime, output, lora = (
        config[key] for key in ("data", "train", "runtime", "output", "lora")
    )
    requested = runtime["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("LoRA supports only CPU and CUDA devices")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if train_cfg["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("train.dtype=bf16 is unsupported on this CUDA device")
    set_seed(train_cfg["seed"], deterministic=False)
    log(f"device={device}, dtype={train_cfg['dtype']}")
    if device.type != "cuda" and train_cfg["dtype"] != "fp32":
        log("Non-CUDA execution uses FP32; configured autocast dtype applies only on CUDA.")

    source = Path(init_from).expanduser().resolve()
    save_dir = Path(output["save_dir"]).expanduser().resolve()
    export_path = (save_dir / (output["save_name"] + ".pth")).resolve()
    resolved_path = (save_dir / "resolved_config.json").resolve()
    if source in (export_path, resolved_path):
        raise ValueError("LoRA outputs must not overwrite the base checkpoint")
    tokenizer = load_tokenizer(data.get("tokenizer_path", data.get("tokenizer_name")))
    model, saved = create_model_from_checkpoint(source, tokenizer)
    del saved
    if data["max_length"] > model.config.max_length:
        raise ValueError("data.max_length exceeds checkpoint model.max_length")
    replaced = apply_lora(model, lora["target_modules"], r=lora["r"],
                          alpha=lora["alpha"], dropout=lora["dropout"])
    if adapter_path:
        load_lora_adapter(read_lora_adapter(adapter_path), model, config=lora)
        log(f"Initialized adapter from {Path(adapter_path).expanduser().resolve()} (fresh optimizer)")
    mark_only_lora_as_trainable(model)
    model = model.to(device=device, dtype=torch.float32)
    log(f"Applied LoRA to {replaced} modules; trainable parameters: {get_param(model)['trainable_params_human']}")

    dataset = SFTDataset(data["data_path"], tokenizer, max_length=data["max_length"],
                         train_on_prompt=data["train_on_prompt"], return_attention_mask=True)
    generator = torch.Generator().manual_seed(train_cfg["seed"])
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=data["num_workers"], pin_memory=device.type == "cuda",
        worker_init_fn=build_worker_seed_fn(train_cfg["seed"], 0), generator=generator,
    )
    if not len(loader):
        raise ValueError("Training dataloader is empty; lower batch_size or provide more data")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    autocast_ctx, scaler = build_amp(train_cfg["dtype"], device.type)
    total_steps = train_cfg["epochs"] * len(loader)
    if train_cfg["max_steps"]:
        total_steps = min(total_steps, train_cfg["max_steps"])
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    accumulation = train_cfg["accumulation_steps"]
    resolved = copy.deepcopy(config)
    resolved.update(model=model.config.to_dict(), init_from=str(source))
    if adapter_path:
        resolved["adapter_path"] = str(Path(adapter_path).expanduser().resolve())
    save_dir.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = optimizer_step = 0
    started_at = time.time()
    pending_save = False

    for epoch in range(train_cfg["epochs"]):
        epoch_steps = min(len(loader), total_steps - micro_step)
        if not epoch_steps:
            break
        generator.manual_seed(train_cfg["seed"] + epoch)
        for batch_idx, (input_ids, labels, mask) in enumerate(islice(loader, epoch_steps)):
            micro_step += 1
            lr = get_lr(micro_step, total_steps, train_cfg["learning_rate"], warmup_steps, train_cfg["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            input_ids, labels, mask = (
                tensor.to(device, non_blocking=device.type == "cuda") for tensor in (input_ids, labels, mask)
            )
            window_start = (batch_idx // accumulation) * accumulation
            window_size = min(accumulation, epoch_steps - window_start)
            should_step = batch_idx + 1 == window_start + window_size
            with autocast_ctx():
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=mask, use_cache=False)
                total_loss = outputs.loss + outputs.aux_loss
            if not torch.isfinite(total_loss):
                raise ValueError("Non-finite LoRA training loss")
            scaler.scale(total_loss / window_size).backward()
            if should_step:
                scaler.unscale_(optimizer)
                if train_cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, train_cfg["grad_clip"])
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    optimizer_step += 1
                optimizer.zero_grad(set_to_none=True)
            if micro_step % output["log_interval"] == 0 or micro_step == total_steps:
                log_train_metrics(
                    f"epoch[{epoch + 1}/{train_cfg['epochs']}] micro[{micro_step}/{total_steps}] "
                    f"optimizer[{optimizer_step}]",
                    {"loss": total_loss.detach().item(), "logits_loss": outputs.loss.detach().item(),
                     "aux_loss": outputs.aux_loss.detach().item()},
                    lr, get_remaining_time(micro_step, total_steps, 0, started_at),
                )
            pending_save |= micro_step % output["save_interval"] == 0 or micro_step == total_steps
            if pending_save and should_step:
                save_lora_adapter(export_path, model, lora, source)
                pending_save = False
            del outputs, total_loss, input_ids, labels, mask
        log(f"Epoch {epoch + 1} complete.")
    log(f"LoRA complete: exported {export_path} (adapter only, no resume state)")


def main(argv=None):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("RANK", "0")) != 0:
        raise ValueError("LoRA supports single-process execution only")
    args = parse_args(argv)
    overrides = {key: getattr(args, key) for key in CLI_FIELDS}
    config = apply_cli_overrides(load_train_config(args.config, "lora"), overrides)
    log(f"Config: {Path(args.config).expanduser().resolve()}; CLI overrides: "
        + json.dumps({key: value for key, value in overrides.items() if value is not None}))
    train(config, args.init_from, args.adapter_path)


if __name__ == "__main__":
    main()
