import argparse
import os
import sys
import time

import torch
from torch import optim
from torch.utils.data import DataLoader

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.data_utils import GRPODataset
from trainer.reward_utils import compute_rewards
from trainer.rl_utils import compute_grpo_loss, compute_per_token_logps, group_normalize_rewards
from trainer.rollout_engine import TorchRolloutEngine
from trainer.train_utils import (
    build_amp,
    get_lr,
    get_param,
    get_remaining_time,
    init_model,
    log,
    log_train_metrics,
    save_checkpoint,
    save_model_only,
    set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="MiniGram GRPO training (single-GPU v1)")

    # data/tokenizer
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="Path to JSON/JSONL GRPO data file")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="Tokenizer path/name for AutoTokenizer")
    parser.add_argument("--max_prompt_len", type=int, default=512, help="Maximum prompt length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum rollout completion length")

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="Enable Mixture of Experts layers")
    parser.add_argument("--use_engrams", type=int, default=0, choices=[0, 1], help="Enable engram blocks")
    parser.add_argument("--engram_vocab_size", type=int, default=1024, help="Engram block vocab size (if use_engrams=1)")

    # rollout/grpo
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions per prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="Rollout sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Rollout top-p sampling")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO clip epsilon")

    # train
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--min_lr", type=float, default=3e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1], help="Enable torch.compile")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after N micro-batches when > 0")
    parser.add_argument("--reward_model_path", type=str, default="default", help="Reward model path (default uses simple heuristics)")

    # init/output
    parser.add_argument("--init_from", type=str, default="./out/minigram_sft.pth", help="SFT checkpoint used for policy and reference")
    parser.add_argument("--save_dir", type=str, default="./out")
    parser.add_argument("--save_name", type=str, default="minigram_grpo")
    parser.add_argument("--save_interval", type=int, default=100, help="Save every N micro-batches")
    parser.add_argument("--log_interval", type=int, default=1, help="Log every N micro-batches")

    # runtime
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0 or cpu")

    return parser.parse_args()


def _build_output_attention_mask(prompt_attention_mask, completion_ids, num_generations):
    repeated_prompt_mask = prompt_attention_mask.repeat_interleave(num_generations, dim=0)
    completion_attention_mask = torch.ones_like(completion_ids)
    return torch.cat([repeated_prompt_mask, completion_attention_mask], dim=1)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in device else "cpu"
    set_seed(args.seed, deterministic=False)
    log(f"Using device={device}, dtype={args.dtype}")

    policy_model, tokenizer = init_model(
        args,
        device=device,
        device_type=device_type,
        checkpoint_path=args.init_from,
        use_cache=False,
    )
    ref_model, _ = init_model(
        args,
        device=device,
        device_type=device_type,
        checkpoint_path=args.init_from,
        use_cache=False,
    )
    ref_model.eval().requires_grad_(False)

    reward_model, reward_tokenizer = None, None
    if args.reward_model_path != "default":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        reward_model = AutoModelForCausalLM.from_pretrained(args.reward_model_path).to(device)
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
        reward_model.eval().requires_grad_(False)
        log(f"Loaded reward model from {args.reward_model_path}")

    if args.use_compile == 1:
        policy_model = torch.compile(policy_model)
        log("torch.compile enabled")

    param_info = get_param(policy_model)
    log(
        "Model size: "
        f"total={param_info['total_params_human']} ({param_info['total_params']}) "
        f"trainable={param_info['trainable_params_human']} ({param_info['trainable_params']}) "
    )

    tokenizer.padding_side = "left"
    train_ds = GRPODataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_prompt_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
    )
    if len(train_loader) == 0:
        raise ValueError("Training dataloader is empty. Lower batch_size or provide more data.")

    optimizer = optim.AdamW(policy_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    autocast_ctx, scaler = build_amp(args.dtype, device_type)
    rollout_engine = TorchRolloutEngine(policy_model, tokenizer, autocast_ctx=autocast_ctx)

    steps_per_epoch = len(train_loader)
    planned_steps = args.epochs * steps_per_epoch
    total_steps = min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    policy_model.train()
    optimizer.zero_grad(set_to_none=True)
    global_start = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            global_step += 1
            lr = get_lr(global_step, total_steps, args.learning_rate, warmup_steps, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            prompts = batch["prompt"]
            references = batch["reference"]
            prompt_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_len,
                add_special_tokens=False,
            ).to(device)

            rollout_result = rollout_engine.rollout(
                prompt_ids=prompt_inputs["input_ids"],
                attention_mask=prompt_inputs["attention_mask"],
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            output_ids = rollout_result.output_ids
            completion_ids = rollout_result.completion_ids
            completion_mask = rollout_result.completion_mask.to(device)
            old_per_token_logps = rollout_result.old_per_token_logps.to(device)
            output_attention_mask = _build_output_attention_mask(
                prompt_inputs["attention_mask"],
                completion_ids,
                args.num_generations,
            )

            with autocast_ctx():
                per_token_logps = compute_per_token_logps(
                    policy_model,
                    output_ids,
                    completion_ids.size(1),
                    attention_mask=output_attention_mask,
                )

            with torch.no_grad():
                ref_per_token_logps = compute_per_token_logps(
                    ref_model,
                    output_ids,
                    completion_ids.size(1),
                    attention_mask=output_attention_mask,
                )

            rewards = compute_rewards(
                prompts,
                rollout_result.completions,
                references=references,
                model=reward_model,
                tokenizer=reward_tokenizer,
            ).to(device)
            advantages = group_normalize_rewards(rewards, args.num_generations)
            grpo_loss, mean_kl = compute_grpo_loss(
                per_token_logps=per_token_logps,
                old_per_token_logps=old_per_token_logps,
                ref_per_token_logps=ref_per_token_logps,
                advantages=advantages,
                completion_mask=completion_mask,
                beta=args.beta,
                epsilon=args.epsilon,
            )

            loss = grpo_loss / args.accumulation_steps
            scaler.scale(loss).backward()

            should_step = ((batch_idx + 1) % args.accumulation_steps == 0) or (
                (batch_idx + 1) == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_interval == 0 or global_step == total_steps:
                remaining_time = get_remaining_time(
                    global_step=global_step,
                    total_steps=total_steps,
                    start_step=0,
                    start_time=global_start,
                )
                log_train_metrics(
                    prefix=f"epoch[{epoch + 1}/{args.epochs}]({global_step}/{total_steps})",
                    metrics={
                        "loss": float(grpo_loss.detach().item()),
                        "reward": float(rewards.mean().detach().item()),
                        "reward_std": float(rewards.std(unbiased=False).detach().item()),
                        "kl": float(mean_kl.detach().item()),
                        "avg_len": float(completion_mask.sum(dim=1).mean().detach().item()),
                    },
                    lr=lr,
                    eta_seconds=remaining_time,
                )

            if global_step % args.save_interval == 0 or global_step == total_steps:
                save_model_only(args.save_dir, model=policy_model, name=args.save_name, dtype=policy_model.dtype)
                save_checkpoint(
                    args.save_dir,
                    model=policy_model,
                    name=args.save_name,
                    optimizer=optimizer,
                    step={"epoch": epoch, "epoch_step": batch_idx + 1},
                    model_dtype=policy_model.dtype,
                )

            del output_ids, completion_ids, completion_mask, old_per_token_logps
            del per_token_logps, ref_per_token_logps, rewards, advantages, grpo_loss, mean_kl, loss

        log(f"Epoch {epoch + 1} complete.")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    save_model_only(args.save_dir, model=policy_model, name=args.save_name, dtype=policy_model.dtype)
    log("GRPO training complete.")


if __name__ == "__main__":
    main()
