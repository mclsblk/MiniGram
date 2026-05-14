import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_minigram import MiniGramConfig, MiniGramForCausalLM


def init_model(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load pretrained weights if specified
    if args.pretrained_model_path:
        config = MiniGramConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_engrams=bool(args.use_engrams),
            engram_vocab_size=args.engram_vocab_size,
            max_length=args.max_embed_len,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = MiniGramForCausalLM(config)

        state_dict = torch.load(args.pretrained_model_path, map_location=args.device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
    elif args.model_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
    else:
        raise ValueError("Must specify either --pretrained_model_path or --model_path to load model weights.")

    model.to(args.device)
    model.eval()
    return tokenizer, model


def generate_raw(args, tokenizer, model, prefix_text):
    model_inputs = tokenizer(prefix_text, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(args.device)
    
    # Truncate to model max context window
    if input_ids.size(1) + args.max_new_tokens > args.max_embed_len:
        input_ids = input_ids[:, -(args.max_embed_len - args.max_new_tokens):]
        print(f"[WARNING] Input truncated to fit context window, kept {input_ids.size(1)} tokens")

    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(args.device)

    if args.echo:
        streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)
    else:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    print(f"Input tokens: {input_ids.size(1)}, generating {args.max_new_tokens} tokens...")
    print("=" * 60)

    with torch.inference_mode():
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "streamer": streamer,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id if args.stop_at_eos else None,
            "do_sample": args.temperature > 0,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "use_cache": True,
        }
        answer = model.generate(**generate_kwargs)

    generated_tokens = answer[0, input_ids.size(1):]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    print("\n" + "=" * 60)
    print(f"Generated {len(generated_tokens)} tokens")

    if args.show_tokens:
        print("\nGenerated token ids:", generated_tokens.tolist())
        print("\nGenerated token list:")
        for tid in generated_tokens.tolist():
            print(f"  {tid:6d} | {repr(tokenizer.convert_ids_to_tokens(tid))}")

    return generated_text


TEST_PREFIXES = [
    "深度学习是",
    "The meaning of life is",
    "### 快速排序算法实现",
    "def quicksort(arr):",
    "公元2077年，人类首次",
    "在神经网络中，注意力机制",
    "\n",
]

def main():
    parser = argparse.ArgumentParser(description="MiniGram base model raw generation (for pretrained checkpoints)")
    parser.add_argument("--test", type=int, default=0, choices=[0, 1], help="Run automatic test loop or not")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="Tokenizer path/name")
    parser.add_argument("--pretrained_model_path", type=str, default="../model/minigram_pretrain.pth", help="Pretrained base model weights")
    parser.add_argument("--model_path", type=str, default=None, help="Hugging Face format model path")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--max_embed_len", type=int, default=32768, help="Model context window size")
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--use_engrams", type=int, default=0, choices=[0, 1])
    parser.add_argument("--engram_vocab_size", type=int, default=1024)
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature, 0 = greedy decode")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty (>1.0 reduces repeats)")
    parser.add_argument("--stop_at_eos", type=int, default=1, choices=[0, 1], help="Stop generation when EOS token appears")
    parser.add_argument("--echo", type=int, default=0, choices=[0, 1], help="Echo input prompt in output")
    parser.add_argument("--show_tokens", type=int, default=0, choices=[0, 1], help="Show detailed token information after generation")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")

    args = parser.parse_args()
    tokenizer, model = init_model(args)

    print("MiniGram Base Model Raw Generation")
    print("=" * 70)

    if args.test == 1:
        print("Running automatic test loop for all preset prefixes")
        print("=" * 70)
        
        for idx, prefix in enumerate(TEST_PREFIXES):
            print(f"\n\n{'='*70}")
            print(f"TEST #{idx} | Prefix: {repr(prefix[:80])}")
            print('='*70)  
            try:
                generate_raw(args, tokenizer, model, prefix)
            except Exception as e:
                print(f"[ERROR] Test #{idx} failed: {e}")
            print("\n" + "-" * 70)
    else:
        print("Running generation with custom input")
        print("Type q or quit to exit")
        while True:
            print("=" * 70)
            try:
                user_input = input("Enter your prompt: ")
                if user_input.strip().lower() in ["q", "quit"]:
                    print("\nExiting...\n")
                    break
            except KeyboardInterrupt:
                print("\nExiting...\n")
                break
            generate_raw(args, tokenizer, model, user_input)


if __name__ == "__main__":
    main()