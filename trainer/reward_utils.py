import re

import torch


def non_empty_reward(completion):
    return 0.5 if completion.strip() else -0.5


def length_reward(completion, min_chars=20, max_chars=800):
    length = len(completion.strip())
    return 0.5 if min_chars <= length <= max_chars else -0.5


def repetition_penalty(completion, n=3, cap=0.5):
    tokens = re.findall(r"\w+|[^\w\s]", completion.lower())
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


def reference_overlap_reward(completion, reference):
    if not reference:
        return 0.0
    completion_tokens = set(re.findall(r"\w+|[^\w\s]", completion.lower()))
    reference_tokens = set(re.findall(r"\w+|[^\w\s]", reference.lower()))
    if not completion_tokens or not reference_tokens:
        return 0.0
    overlap = len(completion_tokens & reference_tokens) / len(reference_tokens)
    return 0.5 * overlap


def model_reward(prompt, completion, model, tokenizer):
    eval_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    with torch.no_grad():
        score = model.get_score(tokenizer, eval_messages)
    return max(min(float(score), 3.0), -3.0)


def compute_rewards(prompts, completions, references=None, model=None, tokenizer=None):
    if references is None:
        references = [""] * len(completions)

    rewards = []
    num_generations = len(completions) // len(prompts)
    for prompt_idx, prompt in enumerate(prompts):
        for generation_idx in range(num_generations):
            idx = prompt_idx * num_generations + generation_idx
            completion = completions[idx]
            reference = references[prompt_idx] if prompt_idx < len(references) else ""
            reward = 0.0
            reward += non_empty_reward(completion)
            reward += length_reward(completion)
            reward -= repetition_penalty(completion)
            reward += reference_overlap_reward(completion, reference)
            if model is not None and tokenizer is not None:
                reward += model_reward(prompt, completion, model, tokenizer)
            rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float32)
