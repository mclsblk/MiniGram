#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Mapping

import torch


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
ENGRAM_EMBED_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.engram\.embeddings\.(\d+)\.weight$")
LORA_RE = re.compile(r"(.+)\.lora_([AB])\.weight$")


def human_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def shape_of(value: Any):
    return list(value.shape) if torch.is_tensor(value) else None


def canonical_key(key: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        while key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def is_tensor_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(isinstance(k, str) and torch.is_tensor(v) for k, v in value.items())


def load_checkpoint(path: str, unsafe_load: bool = False) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=not unsafe_load)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        if unsafe_load:
            raise
        raise RuntimeError(
            "Failed to load with torch.load(..., weights_only=True). "
            "If this is a trusted checkpoint that needs pickle loading, rerun with --unsafe-load."
        ) from exc


def extract_state_dict(payload: Any):
    if is_tensor_state_dict(payload):
        return payload, "root"

    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a dict-like checkpoint, got {type(payload).__name__}.")

    preferred_keys = ("model", "state_dict", "model_state_dict", "adapter")
    for key in preferred_keys:
        value = payload.get(key)
        if is_tensor_state_dict(value):
            return value, key

    candidates = []
    for key, value in payload.items():
        if is_tensor_state_dict(value):
            tensor_count = sum(torch.is_tensor(v) for v in value.values())
            candidates.append((tensor_count, key, value))
    if candidates:
        _, key, value = max(candidates, key=lambda item: item[0])
        return value, key

    raise ValueError("Could not find a tensor state_dict in this checkpoint.")


def infer_attention_candidates(hidden_size: int | None, kv_out: int | None):
    if not hidden_size or not kv_out:
        return []

    candidates = []
    for head_dim in range(1, hidden_size + 1):
        if hidden_size % head_dim != 0 or kv_out % head_dim != 0:
            continue
        num_attention_heads = hidden_size // head_dim
        num_kv_heads = kv_out // head_dim
        if num_attention_heads < num_kv_heads:
            continue
        if num_attention_heads % num_kv_heads != 0:
            continue
        candidates.append(
            {
                "num_attention_heads": num_attention_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }
        )

    preferred_head_dims = [64, 128, 32, 80, 96, 40, 48, 56, 72, 88, 112, 160, 192, 256]

    def score(item):
        head_dim = item["head_dim"]
        heads = item["num_attention_heads"]
        try:
            dim_rank = preferred_head_dims.index(head_dim)
        except ValueError:
            dim_rank = len(preferred_head_dims) + abs(head_dim - 64)
        return (dim_rank, abs(heads - 8), heads)

    return sorted(candidates, key=score)


def infer_lora(state_dict: Mapping[str, torch.Tensor], payload: Any):
    lora_modules = defaultdict(dict)
    for key, value in state_dict.items():
        match = LORA_RE.match(key)
        if not match or not torch.is_tensor(value):
            continue
        module_name, side = match.groups()
        lora_modules[module_name][side] = shape_of(value)

    if not lora_modules:
        return None

    modules = []
    ranks = []
    for module_name, parts in sorted(lora_modules.items()):
        a_shape = parts.get("A")
        b_shape = parts.get("B")
        rank = None
        in_features = None
        out_features = None
        if a_shape and len(a_shape) == 2:
            rank = a_shape[0]
            in_features = a_shape[1]
        if b_shape and len(b_shape) == 2:
            rank = b_shape[1] if rank is None else rank
            out_features = b_shape[0]
        if rank is not None:
            ranks.append(rank)
        modules.append(
            {
                "module": module_name,
                "rank": rank,
                "in_features": in_features,
                "out_features": out_features,
                "lora_A": a_shape,
                "lora_B": b_shape,
            }
        )

    config = payload.get("lora_config") if isinstance(payload, Mapping) else None
    return {
        "module_count": len(modules),
        "ranks": sorted(set(ranks)),
        "target_modules": sorted({name.rsplit(".", 1)[-1] for name in lora_modules}),
        "config": config,
        "modules": modules,
    }


def infer_engrams(state_dict: Mapping[str, torch.Tensor], hidden_size: int | None):
    engram_layers = sorted(
        {
            int(match.group(1))
            for key in state_dict
            for match in [re.search(r"(?:^|\.)layers\.(\d+)\.engram\.", key)]
            if match
        }
    )
    if not engram_layers:
        return None

    embeddings_by_layer = defaultdict(list)
    for key, value in state_dict.items():
        match = ENGRAM_EMBED_RE.search(key)
        if match and torch.is_tensor(value):
            layer_id = int(match.group(1))
            embedding_id = int(match.group(2))
            embeddings_by_layer[layer_id].append((embedding_id, list(value.shape)))

    layer_summaries = {}
    vocab_sizes = set()
    head_dims = set()
    total_heads = set()
    conv_kernel_sizes = set()
    memory_dims = set()

    for layer_id in engram_layers:
        prefix = f"model.layers.{layer_id}.engram."
        alt_prefix = f"layers.{layer_id}.engram."

        embeddings = sorted(embeddings_by_layer.get(layer_id, []), key=lambda item: item[0])
        for _, shape in embeddings:
            if len(shape) == 2:
                vocab_sizes.add(shape[0])
                head_dims.add(shape[1])
        if embeddings:
            total_heads.add(len(embeddings))

        memory_key = state_dict.get(prefix + "memory_key_proj.weight")
        if memory_key is None:
            memory_key = state_dict.get(alt_prefix + "memory_key_proj.weight")
        memory_dim = None
        if torch.is_tensor(memory_key) and memory_key.dim() == 2:
            memory_dim = int(memory_key.shape[1])
            memory_dims.add(memory_dim)

        conv = state_dict.get(prefix + "memory_conv.weight")
        if conv is None:
            conv = state_dict.get(alt_prefix + "memory_conv.weight")
        conv_kernel_size = None
        if torch.is_tensor(conv) and conv.dim() == 3:
            conv_kernel_size = int(conv.shape[2])
            conv_kernel_sizes.add(conv_kernel_size)

        layer_summaries[str(layer_id)] = {
            "embedding_count": len(embeddings),
            "embedding_shape": embeddings[0][1] if embeddings else None,
            "memory_dim": memory_dim,
            "conv_kernel_size": conv_kernel_size,
        }

    inferred_num_heads = None
    if len(total_heads) == 1:
        only_total = next(iter(total_heads))
        if only_total % 2 == 0:
            inferred_num_heads = {
                "assuming_default_ngram_list_[2,3]": only_total // 2,
                "note": "n_gram_list is not uniquely recoverable from weights.",
            }

    return {
        "use_engrams": True,
        "engram_n_layer_list": engram_layers,
        "engram_vocab_size": sorted(vocab_sizes) or None,
        "engram_embedding_head_dim": sorted(head_dims) or None,
        "total_memory_heads": sorted(total_heads) or None,
        "memory_dim": sorted(memory_dims) or None,
        "engram_conv_size": sorted(conv_kernel_sizes) or None,
        "engram_num_heads_guess": inferred_num_heads,
        "layers": layer_summaries,
        "hidden_size_matches_memory_proj": (
            bool(hidden_size and len(memory_dims) == 1 and next(iter(memory_dims)) >= hidden_size)
            if memory_dims
            else None
        ),
    }


def first_tensor_shape(state_dict: Mapping[str, torch.Tensor], suffixes: tuple[str, ...]):
    for suffix in suffixes:
        for key, value in state_dict.items():
            if key.endswith(suffix) and torch.is_tensor(value):
                return key, list(value.shape)
    return None, None


def infer_architecture(payload: Any, state_dict: Mapping[str, torch.Tensor], state_source: str):
    normalized_state = {canonical_key(k): v for k, v in state_dict.items() if torch.is_tensor(v)}
    tensor_items = list(normalized_state.items())

    total_params = sum(int(value.numel()) for _, value in tensor_items)
    dtypes = Counter(str(value.dtype).replace("torch.", "") for _, value in tensor_items)

    layer_ids = sorted(
        {
            int(match.group(1))
            for key in normalized_state
            for match in [LAYER_RE.search(key)]
            if match
        }
    )

    lm_head_key, lm_head_shape = first_tensor_shape(normalized_state, ("lm_head.weight",))
    token_key, token_shape = first_tensor_shape(normalized_state, ("model.token_embedding.weight", "token_embedding.weight"))

    vocab_size = None
    hidden_size = None
    if lm_head_shape and len(lm_head_shape) == 2:
        vocab_size, hidden_size = lm_head_shape
    elif token_shape and len(token_shape) == 2:
        vocab_size, hidden_size = token_shape

    q_key, q_shape = first_tensor_shape(normalized_state, ("attention.q_proj.weight",))
    k_key, k_shape = first_tensor_shape(normalized_state, ("attention.k_proj.weight",))
    v_key, v_shape = first_tensor_shape(normalized_state, ("attention.v_proj.weight",))
    out_key, out_shape = first_tensor_shape(normalized_state, ("attention.out_proj.weight",))

    if hidden_size is None and q_shape and len(q_shape) == 2:
        hidden_size = q_shape[1]

    kv_out = k_shape[0] if k_shape and len(k_shape) == 2 else None
    attention_candidates = infer_attention_candidates(hidden_size, kv_out)
    attention_guess = attention_candidates[0] if attention_candidates else None

    use_moe = any(".ffn.experts." in key for key in normalized_state)
    expert_ids = sorted(
        {
            int(match.group(1))
            for key in normalized_state
            for match in [re.search(r"\.ffn\.experts\.(\d+)\.", key)]
            if match
        }
    )
    moe_gate_key, moe_gate_shape = first_tensor_shape(normalized_state, ("ffn.gate.weight",))

    intermediate_size = None
    ffn_gate_key, ffn_gate_shape = first_tensor_shape(
        normalized_state,
        (
            "ffn.gate_proj.weight",
            "ffn.experts.0.gate_proj.weight",
        ),
    )
    if ffn_gate_shape and len(ffn_gate_shape) == 2:
        intermediate_size = ffn_gate_shape[0]

    norm_key, norm_shape = first_tensor_shape(normalized_state, ("model.norm.weight", "norm.weight"))

    lora = infer_lora(normalized_state, payload)
    engrams = infer_engrams(normalized_state, hidden_size)

    top_level_keys = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key == state_source:
                top_level_keys[key] = "state_dict"
            elif torch.is_tensor(value):
                top_level_keys[key] = {"shape": shape_of(value), "dtype": str(value.dtype).replace("torch.", "")}
            else:
                top_level_keys[key] = type(value).__name__

    checkpoint_kind = "state_dict"
    if isinstance(payload, Mapping) and "optimizer" in payload:
        checkpoint_kind = "training_checkpoint"
    if lora and state_source == "adapter":
        checkpoint_kind = "lora_adapter"

    metadata = {}
    if isinstance(payload, Mapping):
        for key in ("step", "epoch", "global_step", "config", "model_config", "lora_config", "base_checkpoint"):
            if key in payload:
                metadata[key] = payload[key]

    return {
        "path": None,
        "checkpoint_kind": checkpoint_kind,
        "state_source": state_source,
        "top_level_keys": top_level_keys or None,
        "metadata": metadata or None,
        "parameter_count": total_params,
        "parameter_count_human": human_count(total_params),
        "tensor_count": len(tensor_items),
        "dtypes": dict(sorted(dtypes.items())),
        "architecture": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": (max(layer_ids) + 1 if layer_ids else None),
            "layer_ids": layer_ids,
            "intermediate_size": intermediate_size,
            "attention": {
                "q_proj": {"key": q_key, "shape": q_shape},
                "k_proj": {"key": k_key, "shape": k_shape},
                "v_proj": {"key": v_key, "shape": v_shape},
                "out_proj": {"key": out_key, "shape": out_shape},
                "best_guess": attention_guess,
                "candidates": attention_candidates[:12],
                "note": (
                    "num_attention_heads and num_kv_heads are inferred from divisors of "
                    "hidden_size and k_proj output size; the checkpoint does not uniquely store them."
                    if attention_candidates
                    else None
                ),
            },
            "ffn": {
                "use_moe": use_moe,
                "num_experts": (max(expert_ids) + 1 if expert_ids else None),
                "expert_ids": expert_ids or None,
                "moe_gate": {"key": moe_gate_key, "shape": moe_gate_shape},
                "gate_proj": {"key": ffn_gate_key, "shape": ffn_gate_shape},
            },
            "engrams": engrams or {"use_engrams": False},
            "norm": {"key": norm_key, "shape": norm_shape},
            "embeddings": {
                "lm_head": {"key": lm_head_key, "shape": lm_head_shape},
                "token_embedding": {"key": token_key, "shape": token_shape},
                "tied_embedding_possible": bool(lm_head_shape and token_shape and lm_head_shape == token_shape),
            },
            "lora": lora,
        },
    }


def format_shape(shape):
    return "?" if shape is None else "x".join(str(item) for item in shape)


def print_report(info):
    arch = info["architecture"]
    attention = arch["attention"]
    ffn = arch["ffn"]
    engrams = arch["engrams"]
    lora = arch["lora"]

    print(f"Path: {info['path']}")
    print(f"Checkpoint kind: {info['checkpoint_kind']} (state source: {info['state_source']})")
    print(f"Parameters: {info['parameter_count_human']} ({info['parameter_count']})")
    print(f"Tensors: {info['tensor_count']}")
    print(f"DTypes: {', '.join(f'{k}={v}' for k, v in info['dtypes'].items())}")
    print()

    print("Architecture")
    print(f"  vocab_size: {arch['vocab_size']}")
    print(f"  hidden_size: {arch['hidden_size']}")
    print(f"  num_hidden_layers: {arch['num_hidden_layers']}")
    print(f"  intermediate_size: {arch['intermediate_size']}")

    guess = attention["best_guess"]
    if guess:
        print(
            "  attention guess: "
            f"num_attention_heads={guess['num_attention_heads']}, "
            f"num_kv_heads={guess['num_kv_heads']}, "
            f"head_dim={guess['head_dim']}"
        )
    else:
        print("  attention guess: unknown")
    print(f"  q_proj: {format_shape(attention['q_proj']['shape'])}")
    print(f"  k_proj: {format_shape(attention['k_proj']['shape'])}")
    print(f"  v_proj: {format_shape(attention['v_proj']['shape'])}")
    print(f"  out_proj: {format_shape(attention['out_proj']['shape'])}")

    if attention["candidates"]:
        candidates = attention["candidates"][:5]
        text = ", ".join(
            f"h={item['num_attention_heads']}/kv={item['num_kv_heads']}/d={item['head_dim']}"
            for item in candidates
        )
        print(f"  attention candidates: {text}")

    print(f"  use_moe: {ffn['use_moe']}")
    if ffn["use_moe"]:
        print(f"  num_experts: {ffn['num_experts']}")
        print(f"  moe_gate: {format_shape(ffn['moe_gate']['shape'])}")

    print(f"  use_engrams: {engrams.get('use_engrams', False)}")
    if engrams.get("use_engrams"):
        print(f"  engram_n_layer_list: {engrams.get('engram_n_layer_list')}")
        print(f"  engram_vocab_size: {engrams.get('engram_vocab_size')}")
        print(f"  total_memory_heads: {engrams.get('total_memory_heads')}")
        print(f"  engram_conv_size: {engrams.get('engram_conv_size')}")

    print(f"  lora: {bool(lora)}")
    if lora:
        print(f"  lora target_modules: {lora['target_modules']}")
        print(f"  lora ranks: {lora['ranks']}")

    if info.get("metadata"):
        print()
        print("Metadata")
        for key, value in info["metadata"].items():
            print(f"  {key}: {value}")

    print()
    print("Suggested args")
    args = []
    for key in ("hidden_size", "num_hidden_layers"):
        if arch.get(key) is not None:
            args.append(f"--{key} {arch[key]}")
    if guess:
        args.append(f"--num_attention_heads {guess['num_attention_heads']}")
        args.append(f"--num_kv_heads {guess['num_kv_heads']}")
    if ffn["use_moe"]:
        args.append("--use_moe 1")
    if engrams.get("use_engrams"):
        args.append("--use_engrams 1")
        vocab_sizes = engrams.get("engram_vocab_size")
        if vocab_sizes and len(vocab_sizes) == 1:
            args.append(f"--engram_vocab_size {vocab_sizes[0]}")
    print("  " + " ".join(args))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer MiniGram checkpoint structure from a .pth file without instantiating the model."
    )
    parser.add_argument("path", help="Path to a .pth checkpoint or state_dict.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Allow pickle loading for trusted checkpoints if weights_only loading fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.path):
        raise FileNotFoundError(args.path)

    payload = load_checkpoint(args.path, unsafe_load=args.unsafe_load)
    state_dict, state_source = extract_state_dict(payload)
    info = infer_architecture(payload, state_dict, state_source)
    info["path"] = os.path.abspath(args.path)

    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    else:
        print_report(info)


if __name__ == "__main__":
    main()
