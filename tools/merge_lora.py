"""Merge a structured LoRA adapter into a new MiniGram model checkpoint."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.lora import apply_lora, merge_lora_weights
from trainer.lora_utils import load_lora_adapter, read_lora_adapter
from trainer.train_utils import create_model_from_checkpoint, load_tokenizer, save_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True, help="Structured adapter .pth with base metadata")
    parser.add_argument("--base_model_path", help="Override the base checkpoint location saved in the adapter")
    parser.add_argument("--output_path", help="Defaults to <adapter_stem>_merged.pth next to the adapter")
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"), help="Tokenizer directory or name")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0; merge and export use FP32")
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    adapter_path = Path(args.lora_path).expanduser().resolve()
    state = read_lora_adapter(adapter_path)
    base_path = Path(args.base_model_path or state["base_checkpoint"]).expanduser().resolve()
    output_path = (Path(args.output_path).expanduser().resolve() if args.output_path else
                   adapter_path.with_name(adapter_path.stem + "_merged.pth"))
    temporary = output_path.with_suffix(output_path.suffix + ".tmp").resolve()
    if {output_path, temporary} & {adapter_path, base_path}:
        raise ValueError("Merged output and its temporary file must not overwrite either input")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Merge supports only CPU and CUDA devices")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tokenizer = load_tokenizer(args.tokenizer_path)
    model, saved = create_model_from_checkpoint(base_path, tokenizer)
    del saved
    model = model.to(device=device, dtype=torch.float32)
    config = state["lora_config"]
    replaced = apply_lora(model, config["target_modules"], r=config["r"],
                          alpha=config["alpha"], dropout=config["dropout"])
    load_lora_adapter(state, model)
    model.eval()
    del state
    merged = merge_lora_weights(model)
    if merged != replaced:
        raise RuntimeError(f"LoRA merge count mismatch: applied={replaced}, merged={merged}")
    save_checkpoint(output_path, model)
    print(f"Loaded base model: {base_path}")
    print(f"Loaded LoRA adapter: {adapter_path}")
    print(f"Merged LoRA modules: {merged}")
    print(f"Saved FP32 model checkpoint: {output_path}")


if __name__ == "__main__":
    main()
