import argparse
import os
import sys
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.data_utils import PretrainDataset
from model.model_minigram import MiniGramConfig, MiniGramForCausalLM
from trainer.train_utils import (
    build_amp,
    get_param,
    get_lr,
    load_checkpoint,
    log,
    save_checkpoint,
    save_model_only,
    set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="MiniGram pretraining (single-GPU v1)")

    # data/tokenizer
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t.jsonl", help="Path to JSON/JSONL pretraining data file")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="Tokenizer path/name for AutoTokenizer")
    parser.add_argument("--max_length", type=int, default=340, help="Sequence length")

    # model
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="Enable Mixture of Experts (MoE) layers")
    parser.add_argument("--use_engrams", type=int, default=0, choices=[0, 1], help="Enable engram blocks")
    parser.add_argument("--engram_vocab_size", type=int, default=1024, help="Engram block vocab size (if use_engrams=1)")

    # train
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1], help="Enable torch.compile")

    # output/resume
    parser.add_argument("--save_dir", type=str, default="./out")
    parser.add_argument("--save_name", type=str, default="minigram_pretrain")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save every N micro-batches")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N micro-batches")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to .pth checkpoint")

    # runtime
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0 or cpu")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in device else "cpu"
    set_seed(args.seed, deterministic=False)

    log(f"Using device={device}, dtype={args.dtype}")
    # TODO: add DDP support in a future pass.

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds = PretrainDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    pin_memory = device_type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        # shuffle=True,
        # drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    lm_config = MiniGramConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_engrams=bool(args.use_engrams),
        use_cache=False,
        flash_attention=True if device_type == "cuda" else False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        engram_vocab_size=args.engram_vocab_size if args.use_engrams else None,
        use_moe=args.use_moe
    )
    model = MiniGramForCausalLM(lm_config).to(device)

    if args.use_compile == 1:
        model = torch.compile(model)
        log("torch.compile enabled")
    param_info = get_param(model)
    log(
        "Model size: "
        f"total={param_info['total_params_human']} ({param_info['total_params']}) "
        f"trainable={param_info['trainable_params_human']} ({param_info['trainable_params']}) "
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    autocast_ctx, scaler = build_amp(args.dtype, device_type)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    start_epoch = 0
    resume_step = 0
    global_step = 0

    if args.resume_from:
        state, start_epoch, resume_step = load_checkpoint(args.resume_from, model, optimizer=optimizer, map_location="cpu")
        start_epoch = int(state.get("epoch", 0))
        global_step = start_epoch * steps_per_epoch + resume_step
        log(
            "Resumed from checkpoint "
            f"{args.resume_from} (epoch={start_epoch}, global_step={global_step}, epoch_step={resume_step})"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < resume_step:
                continue

            global_step += 1
            lr = get_lr(global_step, total_steps, args.learning_rate, warmup_steps, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            input_ids, labels = batch
            input_ids = input_ids.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

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

            should_step = ((batch_idx + 1) % args.accumulation_steps == 0) or (
                (batch_idx + 1) == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            logits_loss_value = float(logits_loss.detach().item())
            aux_loss_value = float(aux_loss.detach().item())
            total_loss_value = logits_loss_value + aux_loss_value

            current_time = time.time()
            used_time = current_time - global_start
            used_steps = global_step - (start_epoch * steps_per_epoch) - resume_step
            remaining_steps = max(0, total_steps - global_step)
            remaining_time = remaining_steps * (used_time / used_steps)

            if (global_step % args.log_interval == 0 or global_step == total_steps) and batch_idx > 0:
                log(
                    f"epoch[{epoch + 1}/{args.epochs}]({global_step}/{total_steps}) "
                    f"loss={total_loss_value:.4f} logits_loss={logits_loss_value:.4f} "
                    f"aux_loss={aux_loss_value:.4f} lr={lr:.7f} "
                    f"eta={remaining_time / 60:.2f}min"
                )

            if (global_step % args.save_interval == 0 or global_step == total_steps) and batch_idx > 0:
                save_model_only(args.save_dir, model=model, name=args.save_name, dtype=model.dtype)
                save_checkpoint(args.save_dir, model=model, name=args.save_name, optimizer=optimizer,
                                step={"epoch": epoch, "epoch_step": batch_idx + 1}, model_dtype=model.dtype)
            del outputs, loss, logits_loss, aux_loss, total_loss, input_ids, labels
        log(f"Epoch {epoch + 1} complete.")
    save_model_only(args.save_dir, model=model,name=args.save_name)
    log("Training complete.")


if __name__ == "__main__":
    main()
