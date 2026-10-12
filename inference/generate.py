"""One chat/raw inference entrypoint for new MiniGram .pth checkpoints."""
import argparse
import math
import os
from pathlib import Path
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import TextStreamer

from dataset.chat_utils import build_generation_prompt
from trainer.train_utils import create_model_from_checkpoint, load_tokenizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniGram checkpoint inference (chat or raw)")
    parser.add_argument("--checkpoint", required=True, help="New-model .pth containing config and model")
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"), help="Tokenizer directory or name")
    parser.add_argument("--mode", choices=("chat", "raw"), default="chat")
    parser.add_argument("--prompt", help="Generate once; omit for interactive input")
    parser.add_argument("--history_turns", type=int, default=0, help="Recent complete chat turns to retain")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 for greedy decoding")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--ignore_eos", action="store_true", help="Continue after EOS until the token limit")
    parser.add_argument("--echo", action="store_true", help="Also stream the encoded prompt")
    parser.add_argument("--show_tokens", action="store_true", help="Print generated token IDs and pieces")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:0")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.history_turns < 0:
        parser.error("max_new_tokens must be positive and history_turns must be nonnegative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature must be finite and nonnegative")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("top_p must be in (0, 1]")
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0:
        parser.error("repetition_penalty must be finite and positive")
    if args.mode == "raw" and args.history_turns:
        parser.error("history_turns is only supported in chat mode")
    return args


def sample_token(logits, input_ids, args):
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise ValueError("Model returned non-finite generation logits")
    if args.repetition_penalty != 1.0:
        seen = logits.gather(1, input_ids)
        seen = torch.where(seen < 0, seen * args.repetition_penalty, seen / args.repetition_penalty)
        logits = logits.scatter(1, input_ids, seen)
    if args.temperature > 0:
        logits = logits / args.temperature
    if not torch.isfinite(logits).all():
        raise ValueError("Sampling settings produced non-finite logits")
    if args.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    if args.top_p < 1.0:
        sorted_logits, indices = logits.sort(dim=-1, descending=True)
        remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > args.top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = sorted_logits.masked_fill(remove, -torch.inf)
        return indices.gather(1, torch.multinomial(logits.softmax(dim=-1), 1))
    return torch.multinomial(logits.softmax(dim=-1), 1)


@torch.no_grad()
def generate(model, tokenizer, text, history, args, device):
    prompt = build_generation_prompt(
        tokenizer, user_text=text, history=history, with_history=bool(args.history_turns),
    ) if args.mode == "chat" else text
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    if input_ids.size(1) == 0:
        raise ValueError("Prompt must encode to at least one token")
    prompt_limit = model.config.max_length - args.max_new_tokens
    if input_ids.size(1) > prompt_limit:
        print(f"Prompt truncated on the left to {prompt_limit} tokens.", file=sys.stderr)
        input_ids = input_ids[:, -prompt_limit:]
        attention_mask = attention_mask[:, -prompt_limit:]
    streamer = TextStreamer(tokenizer, skip_prompt=not args.echo, skip_special_tokens=args.mode == "chat")
    past = None
    next_input = input_ids
    generated = []
    try:
        streamer.put(input_ids.cpu())
        for _ in range(args.max_new_tokens):
            outputs = model(
                input_ids=next_input, attention_mask=attention_mask, past_key_values=past,
                use_cache=True, logits_to_keep=1, compute_aux_loss=False,
            )
            next_token = sample_token(outputs.logits[:, -1, :], input_ids, args)
            past = outputs.past_key_values
            del outputs
            token_id = next_token.item()
            generated.append(token_id)
            streamer.put(next_token.cpu())
            if not args.ignore_eos and token_id == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
            next_input = next_token
    finally:
        streamer.end()
    if args.show_tokens:
        print("Generated token IDs:", generated)
        for token_id in generated:
            print(f"  {token_id:6d} | {tokenizer.convert_ids_to_tokens(token_id)!r}")
    answer = tokenizer.decode(generated, skip_special_tokens=args.mode == "chat")
    return answer.strip() if args.mode == "chat" else answer


def main(argv=None):
    args = parse_args(argv)
    requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Inference supports CPU and CUDA devices")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tokenizer = load_tokenizer(args.tokenizer_path)
    model, saved = create_model_from_checkpoint(Path(args.checkpoint).expanduser().resolve(), tokenizer)
    del saved
    if args.max_new_tokens >= model.config.max_length:
        raise ValueError("max_new_tokens must be smaller than checkpoint model.max_length")
    model = model.to(device=device, dtype=torch.float32).eval()
    if args.prompt is not None:
        generate(model, tokenizer, args.prompt, [], args, device)
        return
    history = []
    print(f"MiniGram {args.mode}; type q, quit or exit to stop.")
    while True:
        try:
            text = input("User: " if args.mode == "chat" else "Prefix: ")
            if text.strip().lower() in ("q", "quit", "exit"):
                break
            print("Assistant: " if args.mode == "chat" else "Continuation: ", end="", flush=True)
            answer = generate(model, tokenizer, text, history, args, device)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if args.mode == "chat" and args.history_turns:
            history.extend([{"role": "user", "content": text.strip()},
                            {"role": "assistant", "content": answer}])
            history = history[-2 * args.history_turns:]


if __name__ == "__main__":
    main()
