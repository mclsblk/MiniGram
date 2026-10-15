"""Structured adapter I/O; the base model stays in its original checkpoint."""
import json
import os
from pathlib import Path

import torch

from model.lora import load_lora_state_dict, lora_state_dict
from trainer.config_utils import validate_lora_config


def read_lora_adapter(path):
    state = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=True)
    required = {"adapter", "lora_config", "base_checkpoint", "base_config"}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("Expected a structured LoRA adapter with base metadata; old adapters are unsupported")
    validate_lora_config(state["lora_config"])
    if not isinstance(state["base_checkpoint"], str) or not state["base_checkpoint"].strip():
        raise ValueError("LoRA adapter must record its base checkpoint path")
    if not isinstance(state["base_config"], dict) or not state["base_config"]:
        raise ValueError("LoRA adapter must record its base model configuration")
    adapter = state["adapter"]
    if not isinstance(adapter, dict) or not adapter or any(
        not isinstance(key, str) or not torch.is_tensor(value) for key, value in adapter.items()
    ):
        raise ValueError("LoRA adapter must contain a nonempty tensor state_dict")
    return state


def load_lora_adapter(state, model, config=None):
    """Apply an already read adapter after injecting its LoRA layers."""
    if config is not None and state["lora_config"] != config:
        raise ValueError("LoRA configuration does not match the adapter")
    expected = json.loads(json.dumps(model.config.to_dict()))
    if json.loads(json.dumps(state["base_config"])) != expected:
        raise ValueError("LoRA base configuration does not match the model")
    load_lora_state_dict(model, state["adapter"], strict=True)


def save_lora_adapter(path, model, config, base_checkpoint):
    path = Path(path).expanduser().resolve()
    base_checkpoint = Path(base_checkpoint).expanduser().resolve()
    temporary = path.with_suffix(path.suffix + ".tmp")
    if base_checkpoint in (path, temporary):
        raise ValueError("Adapter output must not overwrite the base checkpoint")
    state = {
        "adapter": lora_state_dict(model, dtype=torch.float32), "lora_config": config,
        "base_checkpoint": str(base_checkpoint), "base_config": model.config.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
