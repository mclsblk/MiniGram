#!/usr/bin/env python3
"""Inspect saved MiniGram metadata and tensor shapes without creating a model."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import torch


LORA_RE = re.compile(r"(.+)\.lora_([AB])\.weight")
CONFIG_FIELDS = (
    "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_kv_heads",
    "intermediate_size", "max_length", "bos_token_id", "eos_token_id", "pad_token_id",
    "use_moe", "num_experts", "num_expert_per_token", "residual_variant", "residual_channels",
    "residual_low_rank", "use_engrams", "engram_variant", "engram_n_layer_list", "engram_overrides",
)


def inspect_checkpoint(payload, path):
    if not isinstance(payload, dict):
        raise ValueError("Expected a structured MiniGram checkpoint; bare weights are unsupported")
    metadata = {}
    if {"config", "model"} <= payload.keys():
        source, config_source, kind = "model", "config", "model_export"
        training = {"optimizer", "scaler", "step", "train_config"}
        if training & payload.keys():
            if not training <= payload.keys():
                raise ValueError("Incomplete training checkpoint: optimizer, scaler, step and train_config are required")
            kind = "training_checkpoint"
            metadata = {key: payload[key] for key in ("step", "train_config")}
    elif {"adapter", "lora_config", "base_checkpoint", "base_config"} <= payload.keys():
        source, config_source, kind = "adapter", "base_config", "lora_adapter"
        if not isinstance(payload["base_checkpoint"], str) or not payload["base_checkpoint"].strip():
            raise ValueError("Adapter base_checkpoint must be a nonempty path")
        lora = payload["lora_config"]
        if not isinstance(lora, dict) or not {"r", "alpha", "dropout", "target_modules"} <= lora.keys():
            raise ValueError("Adapter lora_config must contain r, alpha, dropout and target_modules")
        metadata = {key: payload[key] for key in ("base_checkpoint", "lora_config")}
    else:
        raise ValueError("Expected config/model or a structured adapter with base metadata; old formats are unsupported")
    config, weights = payload[config_source], payload[source]
    if not isinstance(config, dict) or not config:
        raise ValueError(f"{config_source} must be a nonempty saved configuration")
    if not isinstance(weights, dict) or not weights or any(
        not isinstance(key, str) or not torch.is_tensor(value) for key, value in weights.items()
    ):
        raise ValueError(f"{source} must be a nonempty tensor state_dict")
    dtype_counts, dtype_elements = Counter(), Counter()
    mapping_buffers = {}
    modules = defaultdict(dict)
    for name, tensor in weights.items():
        dtype = str(tensor.dtype).removeprefix("torch.")
        dtype_counts[dtype] += 1
        dtype_elements[dtype] += tensor.numel()
        if name.startswith("model.engrams.") and name.endswith(
            (".token_mapper.token_map", ".token_mapper.compressed_vocab_size")
        ):
            mapping_buffers[name] = {"shape": list(tensor.shape), "dtype": dtype}
        if source == "adapter":
            match = LORA_RE.fullmatch(name)
            if match is None or tensor.ndim != 2:
                raise ValueError(f"Expected a 2D LoRA A/B weight: {name}")
            modules[match[1]][match[2]] = list(tensor.shape)
    if any(set(parts) != {"A", "B"} for parts in modules.values()):
        raise ValueError("Adapter must contain both A and B for each LoRA module")
    return {
        "path": str(path), "checkpoint_kind": kind, "state_source": source,
        "config_source": config_source, "config": config, "metadata": metadata,
        "tensor_count": len(weights), "tensor_elements": sum(dtype_elements.values()),
        "dtypes": {dtype: {"tensor_count": dtype_counts[dtype], "tensor_elements": dtype_elements[dtype]}
                   for dtype in sorted(dtype_counts)},
        "count_note": "Counts state_dict entries, including buffers and duplicated shared weights; not distinct model parameters.",
        "engram_mapping_buffers": mapping_buffers,
        "lora_modules": [{"module": name, "A": parts["A"], "B": parts["B"]}
                         for name, parts in sorted(modules.items())],
        "inspection_note": "Metadata and shapes only; no model was instantiated and no external base was loaded.",
    }


def print_report(info):
    print(f"Path: {info['path']}")
    print(f"Kind: {info['checkpoint_kind']}; weights: {info['state_source']}")
    print(f"State tensors: {info['tensor_count']:,}; elements: {info['tensor_elements']:,}")
    print(info["count_note"])
    print("Dtypes:")
    for dtype, counts in info["dtypes"].items():
        print(f"  {dtype}: {counts['tensor_count']:,} tensors, {counts['tensor_elements']:,} elements")
    print(f"\nSaved {info['config_source']} (full configuration available with --json):")
    for key in CONFIG_FIELDS:
        if key in info["config"]:
            print(f"  {key}: {json.dumps(info['config'][key], ensure_ascii=False, default=str)}")
    print(f"\nEmbedded Engram mapping buffers: {len(info['engram_mapping_buffers'])}")
    for name, details in info["engram_mapping_buffers"].items():
        print(f"  {name}: shape={details['shape']}, dtype={details['dtype']}")
    if info["metadata"]:
        print("\nMetadata:")
        print(json.dumps(info["metadata"], indent=2, ensure_ascii=False, default=str))
    if info["lora_modules"]:
        print(f"\nLoRA modules: {len(info['lora_modules'])}")
        for module in info["lora_modules"]:
            print(f"  {module['module']}: A={module['A']}, B={module['B']}")
    print("\n" + info["inspection_note"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="New-model export, training checkpoint or structured LoRA adapter")
    parser.add_argument("--json", action="store_true", help="Print the full saved configuration and tensor summary as JSON")
    args = parser.parse_args(argv)
    path = Path(args.path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    info = inspect_checkpoint(payload, path)
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    else:
        print_report(info)


if __name__ == "__main__":
    main()
