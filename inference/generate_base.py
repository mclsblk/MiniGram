import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import argparse
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.chat_utils import build_generation_prompt
from model.model_minigram import MiniGramConfig, MiniGramForCausalLM


def init_model(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
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
        )
        model = MiniGramForCausalLM(config)
        state_dict = torch.load(args.pretrained_model_path, map_location=args.device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
    elif args.model_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
    else:
        raise ValueError("Must specify either --pretrained_model_path or --model_path")

    return tokenizer, model.to(args.device).eval()


def generate(args, tokenizer, model, conversation, history=None):
    prompt = build_generation_prompt(
        tokenizer=tokenizer,
        user_text=conversation["user"],
        history=history,
        with_history=bool(args.with_history),
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    input_ids = model_inputs["input_ids"]
    attn_mask = model_inputs.get("attention_mask")
    attention_mask = torch.ones_like(input_ids) if attn_mask is None else attn_mask

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print("Generated response:", end=" ", flush=True)
    with torch.inference_mode():
        generate_kwargs = {
            "input_ids": input_ids, "attention_mask": attention_mask, "max_length": args.max_length,
            "streamer": streamer, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
            "do_sample": True, "top_p": args.top_p, "temperature": args.temperature, "use_cache": True,
        }
        answer = model.generate(**generate_kwargs)
    generated_tokens = answer[0, input_ids.size(1):]
    conversation["answer"] = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return conversation

# python generate_base.py --pretrained_model_path ../trainer/out/minigram_sft.pth --hidden_size 768 --num_hidden_layers 7 --use_engrams 1 --engram_vocab_size 26003
def main():
    parser = argparse.ArgumentParser(description="MiniGram inference script")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="Tokenizer path/name for AutoTokenizer")
    parser.add_argument("--pretrained_model_path", type=str, default="../model/minigram_sft.pth", help="Path to pretrained model weights (PyTorch .pth file)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model (Hugging Face format). Overrides --pretrained_model_path if specified.")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--max_embed_len", type=int, default=32768, help="Maximum input embedding length (context window size)")
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--use_engrams", type=int, default=0, choices=[0, 1], help="Enable engram blocks")
    parser.add_argument("--engram_vocab_size", type=int, default=1024, help="Engram block vocab size (if use_engrams=1)")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.85, help="Top-p (nucleus) sampling probability")
    parser.add_argument("--max_length", type=int, default=340, help="Maximum generation length")
    parser.add_argument("--with_history", type=int, default=0, help="include how many previous conversation turns in the input prompt (0 for only current user query)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")

    args = parser.parse_args()
    tokenizer, model = init_model(args)

    history = []
    print("MiniGram Inference Chat. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting inference.")
            break
        conversation = {"user": user_input}
        conversation = generate(args, tokenizer, model, conversation, history)
        if args.with_history != 0:
            if args.with_history < len(history) // 2:
                history = history[-(args.with_history * 2):]  # Keep only the last N full turns (user + assistant)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": conversation["answer"]})


if __name__ == "__main__":
    main()
