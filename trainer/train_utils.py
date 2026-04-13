import math
import os
import random
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_lr(step, total_steps, base_lr, warmup_steps, min_lr) -> float:
    if total_steps <= 0:
        return base_lr

    step = max(1, step)
    warmup_steps = max(0, warmup_steps)

    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * float(step) / float(warmup_steps)

    if total_steps <= warmup_steps:
        return min_lr

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def build_amp(dtype: str, device_type: str):
    dtype = dtype.lower()
    if device_type != "cuda" or dtype == "fp32":
        return nullcontext, torch.amp.GradScaler(enabled=False)

    if dtype not in {"bf16", "fp16"}:
        raise ValueError(f"Unsupported dtype: {dtype}. Expected one of: fp32, bf16, fp16.")

    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    def autocast_ctx():
        return torch.amp.autocast(device_type=device_type, dtype=amp_dtype)

    scaler = torch.amp.GradScaler(enabled=(dtype == "fp16"))
    return autocast_ctx, scaler


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _unwrap_model(model):
    if hasattr(model, "module"):
        model = model.module
    model = getattr(model, "_orig_mod", model)
    return model


def _human_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def get_param(model):
    raw_model = _unwrap_model(model)
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_human": _human_count(total_params),
        "trainable_params_human": _human_count(trainable_params)
    }


def save_checkpoint(path, model, optimizer, scaler, epoch, step, args, best_loss=None):
    save_dir = os.path.dirname(path) or "."
    os.makedirs(save_dir, exist_ok=True)
    raw_model = _unwrap_model(model)
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "step": step,
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "best_loss": best_loss,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location)
    raw_model = _unwrap_model(model)
    raw_model.load_state_dict(state["model"], strict=True)

    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    return state
