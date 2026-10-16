from contextlib import nullcontext
from dataclasses import dataclass
from typing import List

import torch

from trainer.rl_utils import compute_per_token_logps


@dataclass
class RolloutResult:
    output_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_per_token_logps: torch.Tensor
    completions: List[str]


class TorchRolloutEngine:
    def __init__(self, model, tokenizer, autocast_ctx=None):
        self.model = model
        self.tokenizer = tokenizer
        self.autocast_ctx = autocast_ctx or nullcontext

    @torch.no_grad()
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens):
        if num_generations < 2 or max_new_tokens < 1:
            raise ValueError("Rollout requires at least two generations and a positive token limit")
        if self.model.training:
            raise ValueError("GRPO rollout requires an eval-mode policy with dropout disabled")
        if (prompt_ids.size(1) == 0 or not attention_mask.bool().any(dim=1).all()
                or not attention_mask[:, -1].bool().all()):
            raise ValueError("Rollout requires nonempty, left-padded prompts")
        output_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        output_mask = attention_mask.repeat_interleave(num_generations, dim=0)
        active = torch.ones(output_ids.size(0), dtype=torch.bool, device=output_ids.device)
        tokens, masks = [], []
        past = None
        next_input = output_ids
        for _ in range(max_new_tokens):
            with self.autocast_ctx():
                outputs = self.model(
                    input_ids=next_input, attention_mask=output_mask, past_key_values=past,
                    use_cache=True, logits_to_keep=1, compute_aux_loss=False,
                )
            # Finished rows are forced to PAD; sampled PAD on active rows remains a valid action.
            logits = outputs.logits[:, -1, :].float().masked_fill(~active[:, None], 0)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite rollout logits at active positions")
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
            next_token = next_token.masked_fill(~active, self.tokenizer.pad_token_id)
            token_mask = active.clone()
            tokens.append(next_token)
            masks.append(token_mask)
            output_ids = torch.cat([output_ids, next_token[:, None]], dim=1)
            output_mask = torch.cat([output_mask, token_mask[:, None].to(output_mask.dtype)], dim=1)
            active = active & (next_token != self.tokenizer.eos_token_id)
            past = outputs.past_key_values
            next_input = next_token[:, None]
            del outputs
            if not active.any():
                break
        completion_ids = torch.stack(tokens, dim=1)
        completion_mask = torch.stack(masks, dim=1)
        del past
        with self.autocast_ctx():
            old_logps, _ = compute_per_token_logps(
                self.model, output_ids, completion_ids.size(1), attention_mask=output_mask,
            )
        completions = [
            self.tokenizer.decode(ids[mask].tolist(), skip_special_tokens=True)
            for ids, mask in zip(completion_ids, completion_mask)
        ]
        return RolloutResult(
            output_ids=output_ids, attention_mask=output_mask,
            completion_ids=completion_ids, completion_mask=completion_mask,
            old_per_token_logps=old_logps.detach(), completions=completions,
        )
