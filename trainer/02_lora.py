import argparse
import os
import sys
import time

import torch
from torch import optim
from torch.utils.data import DataLoader

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.data_utils import SFTDataset
from model.lora import apply_lora, mark_only_lora_as_trainable
from trainer.ddp_utils import (
    barrier,
    build_distributed_sampler,
    build_worker_seed_fn,
    cleanup_distributed,
    get_model_dtype,
    init_distributed,
    maybe_no_sync,
    rank0_log,
    reduce_metrics,
    set_sampler_epoch,
    wrap_ddp,
)
from trainer.lora_utils import (
    load_lora_adapter,
    load_lora_checkpoint,
    lora_config,
    save_lora_checkpoint,
    save_lora_model_only,
)
from trainer.train_utils import (
    build_amp,
    get_lr,
    get_param,
    get_remaining_time,
    init_model,
    log,
    log_train_metrics,
    set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_targets(value):
    targets = [item.strip() for item in value.split(",") if item.strip()]
    if not targets:
        raise ValueError("--lora_target_modules must not be empty.")
    return targets


def parse_args():
    parser = argparse.ArgumentParser(description="MiniGram post-SFT LoRA training")

    parser.add_argument("--data_path", type=str, default="../dataset/lora_exam.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="../model")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_on_prompt", type=int, default=0, choices=[0, 1])

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_engrams", type=int, default=0, choices=[0, 1])
    parser.add_argument("--engram_vocab_size", type=int, default=1024)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_steps", type=int, default=0)

    parser.add_argument("--init_from", type=str, default="./out/minigram_sft.pth")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./out")
    parser.add_argument("--save_name", type=str, default="minigram_lora")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    ddp_state = init_distributed(args)
    if ddp_state.is_main:
        os.makedirs(args.save_dir, exist_ok=True)
    barrier(ddp_state)

    device = ddp_state.device
    device_type = ddp_state.device_type
    set_seed(args.seed + ddp_state.rank, deterministic=False)

    def log_info(msg):
        rank0_log(ddp_state, msg, log)

    log_info(
        f"Using device={device}, dtype={args.dtype}, "
        f"ddp={ddp_state.enabled}, rank={ddp_state.rank}/{ddp_state.world_size}"
    )

    targets = parse_targets(args.lora_target_modules)
    config = lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, targets)

    model, tokenizer = init_model(
        args,
        device=device,
        device_type=device_type,
        checkpoint_path=args.init_from,
        use_cache=False,
    )
    log_info(f"Loaded SFT base weights from {args.init_from}")

    replaced = apply_lora(model, targets, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    log_info(f"Applied LoRA to {replaced} modules: {','.join(targets)}")

    if args.adapter_path and not args.resume_from:
        load_lora_adapter(args.adapter_path, model, config=config)
        log_info(f"Loaded initial LoRA adapter from {args.adapter_path}")

    mark_only_lora_as_trainable(model)

    train_ds = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_on_prompt=bool(args.train_on_prompt),
    )
    pin_memory = device_type == "cuda"
    train_sampler = build_distributed_sampler(
        train_ds,
        ddp_state,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed + ddp_state.rank)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=build_worker_seed_fn(args.seed, ddp_state.rank),
        generator=loader_generator,
    )
    if len(train_loader) == 0:
        raise ValueError("Training dataloader is empty. Lower batch_size or provide more data.")

    if args.use_compile == 1:
        model = torch.compile(model)
        log_info("torch.compile enabled")
    model = wrap_ddp(model, ddp_state)

    param_info = get_param(model)
    log_info(
        "Model size: "
        f"total={param_info['total_params_human']} ({param_info['total_params']}) "
        f"trainable={param_info['trainable_params_human']} ({param_info['trainable_params']}) "
    )

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    autocast_ctx, scaler = build_amp(args.dtype, device_type)

    steps_per_epoch = len(train_loader)
    planned_steps = args.epochs * steps_per_epoch
    total_steps = min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    start_epoch = 0
    resume_step = 0
    global_step = 0
    if args.resume_from:
        _, start_epoch, resume_step = load_lora_checkpoint(args.resume_from, model, optimizer, config=config)
        global_step = (start_epoch * steps_per_epoch) + resume_step
        log_info(
            "Resumed LoRA checkpoint "
            f"{args.resume_from} (epoch={start_epoch}, global_step={global_step}, epoch_step={resume_step})"
        )

    start_step = global_step
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        set_sampler_epoch(train_sampler, epoch)
        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < resume_step:
                continue
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            global_step += 1
            lr = get_lr(global_step, total_steps, args.learning_rate, warmup_steps, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            input_ids, labels = batch
            input_ids = input_ids.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            should_step = ((batch_idx + 1) % args.accumulation_steps == 0) or (
                (batch_idx + 1) == len(train_loader)
            )
            with maybe_no_sync(model, ddp_state, should_step):
                with autocast_ctx():
                    outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
                    aux_loss = getattr(outputs, "aux_loss", None)
                    if aux_loss is None:
                        aux_loss = outputs.loss.new_zeros(())
                    elif not torch.is_tensor(aux_loss):
                        aux_loss = outputs.loss.new_tensor(float(aux_loss))
                    logits_loss = outputs.loss
                    total_loss = logits_loss + aux_loss

                loss = total_loss / args.accumulation_steps
                scaler.scale(loss).backward()

            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_interval == 0 or global_step == total_steps:
                reduced_metrics = reduce_metrics(
                    {
                        "loss": total_loss,
                        "logits_loss": logits_loss,
                        "aux_loss": aux_loss,
                    },
                    ddp_state,
                )
                remaining_time = get_remaining_time(global_step, total_steps, start_step, start_time)
                if ddp_state.is_main:
                    log_train_metrics(
                        prefix=f"epoch[{epoch + 1}/{args.epochs}]({global_step}/{total_steps})",
                        metrics=reduced_metrics,
                        lr=lr,
                        eta_seconds=remaining_time,
                    )

            if ddp_state.is_main and (global_step % args.save_interval == 0 or global_step == total_steps):
                model_dtype = get_model_dtype(model)
                save_lora_model_only(args.save_dir, model, args.save_name, config, args.init_from, dtype=model_dtype)
                save_lora_checkpoint(
                    args.save_dir,
                    model,
                    args.save_name,
                    optimizer,
                    {"epoch": epoch, "epoch_step": batch_idx + 1},
                    config,
                    args.init_from,
                    dtype=model_dtype,
                )

            del outputs, loss, logits_loss, aux_loss, total_loss, input_ids, labels

        log_info(f"Epoch {epoch + 1} complete.")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if ddp_state.is_main:
        save_lora_model_only(args.save_dir, model, args.save_name, config, args.init_from, dtype=get_model_dtype(model))
        log("LoRA training complete.")
    barrier(ddp_state)
    cleanup_distributed()


if __name__ == "__main__":
    main()
