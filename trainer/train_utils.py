import json
import math
from pathlib import Path
import os
import random
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer
from model.model_minigram import MiniGramConfig, MiniGramForCausalLM
from trainer.ddp_utils import unwrap_model
from trainer.config_utils import MODEL_FIELDS
from model.token_compression import CompressedTokenMapper, build_compressed_token_map


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

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
    now = datetime.now().strftime("%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def get_remaining_time(global_step: int, total_steps: int, start_step: int, start_time: float) -> float:
    used_steps = max(1, global_step - start_step)
    used_time = (datetime.now() - datetime.fromtimestamp(start_time)).total_seconds()
    remaining_steps = max(0, total_steps - global_step)
    return remaining_steps * (used_time / used_steps)


def log_train_metrics(prefix: str, metrics: dict, lr: float, eta_seconds: float) -> None:
    metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    log(f"{prefix} {metric_text} lr={lr:.7f} eta={max(0.0, eta_seconds) / 60:.2f}min")


def _human_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def get_param(model):
    raw_model = unwrap_model(model)
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_human": _human_count(total_params),
        "trainable_params_human": _human_count(trainable_params)
    }


def load_tokenizer(source):
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define PAD or EOS")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define BOS and EOS")
    return tokenizer


def build_model_config(model_options, tokenizer):
    unknown = set(model_options) - set(MODEL_FIELDS)
    if unknown:
        raise ValueError(f"Unknown model fields: {sorted(unknown)}")
    config = MiniGramConfig(
        **model_options, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("model.hidden_size: must be divisible by model.num_attention_heads")
    if config.num_attention_heads % config.num_kv_heads:
        raise ValueError("model.num_attention_heads: must be divisible by model.num_kv_heads")
    if (config.hidden_size // config.num_attention_heads) % 2:
        raise ValueError("model.hidden_size: attention head dimension must be even for RoPE")
    if not 1 <= config.num_expert_per_token <= config.num_experts:
        raise ValueError("model.num_expert_per_token: must be between 1 and model.num_experts")
    return config


def create_model(config, tokenizer, prepare_mapping=True):
    model = MiniGramForCausalLM(config)
    if prepare_mapping and any(isinstance(layer.token_mapper, CompressedTokenMapper)
           for layer in model.model.engrams.values()):
        if not hasattr(tokenizer, "backend_tokenizer"):
            raise ValueError("Compressed Engram requires a fast tokenizer with backend_tokenizer")
        token_map, _ = build_compressed_token_map(tokenizer)
        model.set_engram_token_map(token_map)
    return model


def save_checkpoint(path, model, optimizer=None, scaler=None, step=None, train_config=None):
    """One .pth file; omit optimizer for a model-only export."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = unwrap_model(model)
    state = {"config": raw.config.to_dict(),
             "model": {k: v.detach().cpu() for k, v in raw.state_dict().items()}}
    if optimizer is not None:
        state.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict() if scaler is not None else None,
                     step=step, train_config=train_config)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not {"config", "model"} <= state.keys():
        raise ValueError("Expected a .pth containing config and model; old weights are unsupported")
    return state


def restore_training_state(state, optimizer, scaler=None):
    """Restore optimizer/scaler from an already read training checkpoint."""
    if not {"optimizer", "scaler", "step", "train_config"} <= state.keys():
        raise ValueError("Resuming requires a training checkpoint, not a model-only export")
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])


def _load_model_state(state, model):
    raw = unwrap_model(model)
    if json.loads(json.dumps(state["config"])) != json.loads(json.dumps(raw.config.to_dict())):
        raise ValueError("Checkpoint model config mismatch")
    raw.load_state_dict(state["model"], strict=True)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Strictly load matching new-model weights, optionally restoring training state."""
    state = _read_checkpoint(path)
    _load_model_state(state, model)
    if optimizer is not None:
        restore_training_state(state, optimizer, scaler)
    return state


def create_model_from_checkpoint(path, tokenizer):
    """Use saved architecture and mappings; return model and state for optional resume."""
    state = _read_checkpoint(path)
    config = MiniGramConfig(**state["config"])
    expected = {"vocab_size": len(tokenizer), "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(f"Tokenizer {name} does not match checkpoint configuration")
    model = create_model(config, tokenizer, prepare_mapping=False)
    _load_model_state(state, model)
    return model, state


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


