import argparse
import os
import sys

import torch
from transformers import AutoTokenizer


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from model.lora import apply_lora, load_lora_state_dict, merge_lora_weights
from model.model_minigram import MiniGramConfig, MiniGramForCausalLM


def parse_targets(value):
    targets = [item.strip() for item in value.split(",") if item.strip()]
    if not targets:
        raise ValueError("--lora_target_modules must not be empty.")
    return targets


def dtype_from_name(name):
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def load_plain_model_state(path, map_location):
    state = torch.load(path, map_location=map_location)
    return state["model"] if isinstance(state, dict) and "model" in state else state


def resolve_existing_path(path, reference_path=None):
    if path is None:
        return None
    if os.path.isabs(path) or os.path.exists(path):
        return path

    candidates = [os.path.abspath(path)]
    if reference_path is not None:
        reference_dir = os.path.dirname(os.path.abspath(reference_path))
        candidates.append(os.path.normpath(os.path.join(reference_dir, path)))
        candidates.append(os.path.join(reference_dir, os.path.basename(path)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def default_output_path(lora_path):
    root, ext = os.path.splitext(lora_path)
    return f"{root}_merged{ext or '.pth'}"


def build_model(args, tokenizer, device):
    config = MiniGramConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        use_engrams=bool(args.use_engrams),
        engram_vocab_size=args.engram_vocab_size,
        max_length=args.max_embed_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        flash_attention=False,
        use_cache=False,
        use_moe=bool(args.use_moe),
        num_experts=args.num_experts,
        num_expert_per_token=args.num_expert_per_token,
    )
    return MiniGramForCausalLM(config).to(device)


def cpu_state_dict(model, dtype):
    payload = {}
    for key, value in model.state_dict().items():
        tensor = value.detach()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        payload[key] = tensor.cpu()
    return payload


def save_state_dict(path, state_dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a MiniGram LoRA adapter into a full .pth checkpoint.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA adapter .pth file.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to the base/SFT model .pth file. Defaults to adapter metadata when available.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Merged model output path. Defaults to '<lora_path>_merged.pth'.",
    )
    parser.add_argument("--tokenizer_path", type=str, default=os.path.join(ROOT_DIR, "model"))

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--max_embed_len", type=int, default=32768)
    parser.add_argument("--use_engrams", type=int, default=0, choices=[0, 1])
    parser.add_argument("--engram_vocab_size", type=int, default=1024)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_expert_per_token", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=8, help="Fallback rank for adapters without lora_config.")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="Fallback alpha for adapters without lora_config.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout is disabled for merge/inference.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Fallback target modules for adapters without lora_config.",
    )
    parser.add_argument("--output_dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output_path or default_output_path(args.lora_path)

    adapter_state = torch.load(args.lora_path, map_location=args.device)
    adapter_config = adapter_state.get("lora_config") if isinstance(adapter_state, dict) else None
    if adapter_config is None:
        adapter_config = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": parse_targets(args.lora_target_modules),
        }

    base_model_path = args.base_model_path
    if base_model_path is None and isinstance(adapter_state, dict):
        base_model_path = adapter_state.get("base_checkpoint")
    base_model_path = resolve_existing_path(base_model_path, reference_path=args.lora_path)
    if base_model_path is None:
        raise ValueError("Missing base model path. Pass --base_model_path or use an adapter with base_checkpoint metadata.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = build_model(args, tokenizer, args.device)
    model.load_state_dict(load_plain_model_state(base_model_path, args.device), strict=True)

    target_modules = adapter_config["target_modules"]
    replaced = apply_lora(
        model,
        target_modules,
        r=int(adapter_config["r"]),
        alpha=float(adapter_config["alpha"]),
        dropout=0.0,
    )
    adapter_payload = adapter_state["adapter"] if isinstance(adapter_state, dict) and "adapter" in adapter_state else adapter_state
    load_lora_state_dict(model, adapter_payload, strict=True)

    merged = merge_lora_weights(model)
    if merged != replaced:
        raise RuntimeError(f"LoRA merge count mismatch: applied={replaced}, merged={merged}")

    save_state_dict(output_path, cpu_state_dict(model, dtype_from_name(args.output_dtype)))
    print(f"Loaded base model: {base_model_path}")
    print(f"Loaded LoRA adapter: {args.lora_path}")
    print(f"Merged LoRA modules: {merged}")
    print(f"Saved merged model: {output_path}")


if __name__ == "__main__":
    main()
