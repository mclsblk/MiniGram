import torch
import torch.nn.functional as F


def gather_logprobs(logits, target_ids):
    return torch.gather(
        F.log_softmax(logits.float(), dim=-1), dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)


def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None, compute_aux_loss=False):
    """Completion logprobs and the auxiliary loss from the same forward."""
    if not 0 < n_keep < input_ids.size(1):
        raise ValueError("Logprob computation requires a prompt and a nonempty completion")
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        logits_to_keep=n_keep + 1, use_cache=False, compute_aux_loss=compute_aux_loss,
    )
    logits = outputs.logits[:, :-1, :]
    if attention_mask is not None:
        logits = logits.masked_fill(~attention_mask[:, -n_keep:, None].bool(), 0)
    logps = gather_logprobs(logits, input_ids[:, -n_keep:])
    return logps, outputs.aux_loss.float()


def masked_mean(values, mask, dim=None, eps=1e-8):
    mask = mask.to(dtype=values.dtype)
    return values.masked_fill(~mask.bool(), 0).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


def group_normalize_rewards(rewards, num_generations, eps=1e-4):
    rewards = rewards.float()
    if rewards.ndim != 1 or rewards.numel() == 0 or num_generations < 2 or rewards.numel() % num_generations:
        raise ValueError("Rewards must contain complete groups with at least two generations")
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite")
    grouped = rewards.view(-1, num_generations)
    advantages = (grouped - grouped.mean(dim=1, keepdim=True)) / (
        grouped.std(dim=1, unbiased=False, keepdim=True) + eps
    )
    return advantages.reshape(-1)


def compute_grpo_loss(
    per_token_logps, old_per_token_logps, ref_per_token_logps,
    advantages, completion_mask, beta, epsilon,
):
    valid = completion_mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("Every completion must contain at least one valid token")
    current = per_token_logps.float().masked_fill(~valid, 0)
    old = old_per_token_logps.detach().float().masked_fill(~valid, 0)
    advantages = advantages.detach().float()
    if not (torch.isfinite(current).all() and torch.isfinite(old).all()
            and torch.isfinite(advantages).all()):
        raise ValueError("Non-finite GRPO logprobs or advantages at valid positions")
    ratio = torch.exp(current - old)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    if beta > 0:
        if ref_per_token_logps is None:
            raise ValueError("A reference policy is required when beta > 0")
        reference = ref_per_token_logps.detach().float().masked_fill(~valid, 0)
        if not torch.isfinite(reference).all():
            raise ValueError("Non-finite reference logprobs at valid positions")
        delta = reference - current
        per_token_kl = torch.exp(delta) - delta - 1.0
    else:
        per_token_kl = torch.zeros_like(current)
    if not (torch.isfinite(ratio).all() and torch.isfinite(per_token_kl).all()):
        raise ValueError("Non-finite GRPO ratio or KL at valid positions")
    objective = torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))
    per_token_loss = -objective + beta * per_token_kl
    if not torch.isfinite(per_token_loss).all():
        raise ValueError("Non-finite GRPO objective")
    loss = masked_mean(per_token_loss, valid, dim=1).mean()
    return loss, masked_mean(per_token_kl, valid)
