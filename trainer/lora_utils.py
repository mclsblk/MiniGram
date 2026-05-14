import os

import torch

from model.lora import load_lora_state_dict, lora_state_dict


def lora_config(r, alpha, dropout, target_modules):
    return {
        "r": int(r),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_modules": list(target_modules),
    }


def validate_lora_config(expected, actual):
    if expected != actual:
        raise ValueError(f"LoRA config mismatch. expected={expected}, actual={actual}")


def _save(payload, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def save_lora_model_only(path, model, name, config, base_checkpoint=None, dtype=None):
    save_path = os.path.join(path or ".", f"{name}.pth")
    _save(
        {
            "adapter": lora_state_dict(model, dtype=dtype),
            "lora_config": config,
            "base_checkpoint": base_checkpoint,
        },
        save_path,
    )
    return save_path


def save_lora_checkpoint(path, model, name, optimizer, step, config, base_checkpoint=None, dtype=None):
    save_path = os.path.join(path or ".", "checkpoint", f"{name}.pth")
    _save(
        {
            "adapter": lora_state_dict(model, dtype=dtype),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "lora_config": config,
            "base_checkpoint": base_checkpoint,
        },
        save_path,
    )
    return save_path


def load_lora_adapter(path, model, config=None, map_location="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location)
    if "lora_config" in state and config is not None:
        validate_lora_config(config, state["lora_config"])
    load_lora_state_dict(model, state["adapter"] if "adapter" in state else state, strict=True)
    return state


def load_lora_checkpoint(path, model, optimizer=None, config=None, map_location="cpu"):
    state = load_lora_adapter(path, model, config=config, map_location=map_location)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    step = state.get("step", {})
    return state, int(step.get("epoch", 0)), int(step.get("epoch_step", 0))
