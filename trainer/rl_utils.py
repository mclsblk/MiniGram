import torch
import torch.nn.functional as F


def gather_logprobs(logits, target_ids):
    return torch.gather(
        F.log_softmax(logits, dim=-1),
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)


def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None):
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=n_keep + 1,
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, -n_keep:]
    return gather_logprobs(logits, target_ids)


def build_completion_mask(completion_ids, eos_token_id, pad_token_id=None):
    mask = torch.ones_like(completion_ids, dtype=torch.float32)

    if eos_token_id is not None:
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full(
            (completion_ids.size(0),),
            completion_ids.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        positions = torch.arange(completion_ids.size(1), device=completion_ids.device).unsqueeze(0)
        mask = (positions <= eos_idx.unsqueeze(1)).to(dtype=torch.float32)

    if pad_token_id is not None and pad_token_id != eos_token_id:
        mask = mask * (completion_ids != pad_token_id).to(dtype=torch.float32)

    return mask


def masked_mean(values, mask, dim=None, eps=1e-8):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


def group_normalize_rewards(rewards, num_generations, eps=1e-4):
    grouped_rewards = rewards.view(-1, num_generations)
    mean = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
    std = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_generations)
    return (rewards - mean) / (std + eps)


def compute_grpo_loss(
        per_token_logps, old_per_token_logps, ref_per_token_logps,
        advantages, completion_mask, beta, epsilon
):
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    per_token_kl_delta = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(per_token_kl_delta) - per_token_kl_delta - 1.0

    policy_objective = torch.min(
        ratio * advantages.unsqueeze(1),
        clipped_ratio * advantages.unsqueeze(1),
    )
    per_token_loss = -(policy_objective - beta * per_token_kl)
    loss = masked_mean(per_token_loss, completion_mask, dim=1).mean()
    mean_kl = masked_mean(per_token_kl, completion_mask)
    return loss, mean_kl
