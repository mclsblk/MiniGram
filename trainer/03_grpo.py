"""Single-process TOML GRPO initialized from a new-model checkpoint."""
import argparse
import copy
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

from dataset.data_utils import GRPODataset
from trainer.config_utils import CLI_FIELDS, apply_cli_overrides, load_train_config
from trainer.ddp_utils import build_worker_seed_fn
from trainer.reward_utils import compute_rewards
from trainer.rl_utils import compute_grpo_loss, compute_per_token_logps, group_normalize_rewards
from trainer.rollout_engine import TorchRolloutEngine
from trainer.train_utils import (
    build_amp, create_model_from_checkpoint, get_lr, get_param, get_remaining_time,
    load_tokenizer, log, log_train_metrics, save_checkpoint, set_seed,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniGram TOML GRPO (single process, model export only)")
    parser.add_argument("--config", required=True, help="GRPO TOML file")
    parser.add_argument("--init_from", required=True, help="New-model checkpoint used for policy and reference")
    parser.add_argument("--device", help="Override runtime.device (auto, cpu, cuda:0)")
    parser.add_argument("--save_dir", help="Override output.save_dir, relative to current directory")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args(argv)


def train(config, init_from):
    data, train_cfg, runtime, output, grpo, reward_cfg = (
        config[key] for key in ("data", "train", "runtime", "output", "grpo", "reward")
    )
    requested = runtime["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("GRPO supports only CPU and CUDA devices")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if train_cfg["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("train.dtype=bf16 is unsupported on this CUDA device")
    set_seed(train_cfg["seed"], deterministic=False)
    log(f"device={device}, dtype={train_cfg['dtype']}")
    if device.type != "cuda" and train_cfg["dtype"] != "fp32":
        log("Non-CUDA execution uses FP32; configured autocast dtype applies only on CUDA.")

    tokenizer = load_tokenizer(data.get("tokenizer_path", data.get("tokenizer_name")))
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    source = Path(init_from).expanduser().resolve()
    policy, saved = create_model_from_checkpoint(source, tokenizer)
    del saved
    if data["max_length"] + grpo["max_new_tokens"] > policy.config.max_length:
        raise ValueError("Prompt and completion limits exceed checkpoint model.max_length")
    reference = copy.deepcopy(policy).to(device).eval().requires_grad_(False) if grpo["beta"] > 0 else None
    policy = policy.to(device).eval()
    log(f"Model parameters: {get_param(policy)['total_params_human']}; dropout disabled")
    reward_model, reward_tokenizer = None, None
    reward_source = reward_cfg.get("model_path", reward_cfg.get("model_name"))
    if reward_source is not None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_source).to(device)
        reward_model.eval().requires_grad_(False)
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_source)
        if reward_tokenizer.pad_token_id is None and reward_tokenizer.eos_token_id is not None:
            reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id
        log(f"Loaded reward model from {reward_source}")

    dataset = GRPODataset(data["data_path"], tokenizer, max_length=data["max_length"])
    generator = torch.Generator().manual_seed(train_cfg["seed"])
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=False,
        num_workers=data["num_workers"], pin_memory=device.type == "cuda",
        worker_init_fn=build_worker_seed_fn(train_cfg["seed"], 0), generator=generator,
    )
    if not len(loader):
        raise ValueError("Training dataloader is empty")
    optimizer = optim.AdamW(policy.parameters(), lr=train_cfg["learning_rate"],
                            weight_decay=train_cfg["weight_decay"])
    autocast_ctx, scaler = build_amp(train_cfg["dtype"], device.type)
    engine = TorchRolloutEngine(policy, tokenizer, autocast_ctx=autocast_ctx)
    total_rollouts = train_cfg["epochs"] * len(loader)
    if train_cfg["max_steps"]:
        total_rollouts = min(total_rollouts, train_cfg["max_steps"])
    total_micro = total_rollouts * grpo["num_iterations"]
    warmup_steps = int(total_micro * train_cfg["warmup_ratio"])
    accumulation = train_cfg["accumulation_steps"]
    save_dir = Path(output["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(config)
    resolved.update(model=policy.config.to_dict(), init_from=str(source))
    (save_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    log("Effective model config: " + json.dumps(resolved["model"], ensure_ascii=False, sort_keys=True))
    export_path = save_dir / (output["save_name"] + ".pth")
    optimizer.zero_grad(set_to_none=True)
    started_at = time.time()
    rollout_step = micro_step = optimizer_step = 0
    pending_save = False

    for epoch in range(train_cfg["epochs"]):
        generator.manual_seed(train_cfg["seed"] + epoch)
        epoch_rollouts = min(len(loader), total_rollouts - rollout_step)
        if not epoch_rollouts:
            break
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= epoch_rollouts:
                break
            rollout_step += 1
            prompts = batch["prompt"]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=data["max_length"], add_special_tokens=False,
            ).to(device)
            result = engine.rollout(
                inputs["input_ids"], inputs["attention_mask"],
                grpo["num_generations"], grpo["max_new_tokens"],
            )
            n_keep = result.completion_ids.size(1)
            ref_logps = None
            if reference is not None:
                with torch.no_grad(), autocast_ctx():
                    ref_logps, _ = compute_per_token_logps(
                        reference, result.output_ids, n_keep, attention_mask=result.attention_mask,
                    )
            rewards = compute_rewards(
                prompts, result.completions, references=batch["reference"],
                model=reward_model, tokenizer=reward_tokenizer, max_length=reward_cfg["max_length"],
            ).to(device)
            advantages = group_normalize_rewards(rewards, grpo["num_generations"])
            window_start = (batch_idx // accumulation) * accumulation
            window_end = min(window_start + accumulation, epoch_rollouts)
            # Count actual prompt samples, including the final smaller DataLoader batch.
            window_samples = min(window_end * train_cfg["batch_size"], len(dataset)) - (
                window_start * train_cfg["batch_size"]
            )
            loss_weight = len(prompts) / window_samples
            should_step = batch_idx + 1 == window_end

            for iteration in range(grpo["num_iterations"]):
                micro_step += 1
                lr = get_lr(micro_step, total_micro, train_cfg["learning_rate"],
                            warmup_steps, train_cfg["min_lr"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                with autocast_ctx():
                    current_logps, aux_loss = compute_per_token_logps(
                        policy, result.output_ids, n_keep, attention_mask=result.attention_mask,
                        compute_aux_loss=True,
                    )
                grpo_loss, mean_kl = compute_grpo_loss(
                    current_logps, result.old_per_token_logps, ref_logps,
                    advantages, result.completion_mask, grpo["beta"], grpo["epsilon"],
                )
                total_loss = grpo_loss + aux_loss
                if not torch.isfinite(total_loss):
                    raise ValueError("Non-finite total GRPO/MoE loss")
                scaler.scale(total_loss * loss_weight).backward()
                if should_step:
                    scaler.unscale_(optimizer)
                    if train_cfg["grad_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), train_cfg["grad_clip"])
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= previous_scale:
                        optimizer_step += 1
                    optimizer.zero_grad(set_to_none=True)
                if micro_step % output["log_interval"] == 0 or micro_step == total_micro:
                    log_train_metrics(
                        f"epoch[{epoch + 1}/{train_cfg['epochs']}] rollout[{rollout_step}/{total_rollouts}] "
                        f"iteration[{iteration + 1}/{grpo['num_iterations']}] "
                        f"micro[{micro_step}/{total_micro}] optimizer[{optimizer_step}]",
                        {"loss": total_loss.detach().item(), "grpo_loss": grpo_loss.detach().item(),
                         "aux_loss": aux_loss.detach().item(), "reward": rewards.mean().item(),
                         "reward_std": rewards.std(unbiased=False).item(), "kl": mean_kl.detach().item(),
                         "avg_len": result.completion_mask.sum(dim=1).float().mean().item()},
                        lr, get_remaining_time(micro_step, total_micro, 0, started_at),
                    )
                pending_save |= micro_step % output["save_interval"] == 0 or micro_step == total_micro
                if pending_save and should_step and iteration + 1 == grpo["num_iterations"]:
                    save_checkpoint(export_path, policy)
                    pending_save = False
                del current_logps, aux_loss, grpo_loss, mean_kl, total_loss
            del result, ref_logps, rewards, advantages, inputs
        log(f"Epoch {epoch + 1} complete.")
    log(f"GRPO complete: exported {export_path} (model only, no resume state)")


def main(argv=None):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("RANK", "0")) != 0:
        raise ValueError("GRPO supports single-process execution only")
    args = parse_args(argv)
    overrides = {key: getattr(args, key) for key in CLI_FIELDS}
    config = apply_cli_overrides(load_train_config(args.config, "grpo"), overrides)
    log(f"Config: {Path(args.config).expanduser().resolve()}; CLI overrides: "
        + json.dumps({k: v for k, v in overrides.items() if v is not None}))
    train(config, args.init_from)


if __name__ == "__main__":
    main()
