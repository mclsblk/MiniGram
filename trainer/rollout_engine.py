from dataclasses import dataclass
from typing import List

import torch

from trainer.rl_utils import build_completion_mask, compute_per_token_logps


@dataclass
class RolloutResult:
    output_ids: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_per_token_logps: torch.Tensor
    completions: List[str]


class TorchRolloutEngine:
    def __init__(self, model, tokenizer, autocast_ctx=None):
        self.model = model
        self.tokenizer = tokenizer
        self.autocast_ctx = autocast_ctx

    def rollout(
        self,
        prompt_ids,
        attention_mask,
        num_generations,
        max_new_tokens,
        temperature=0.8,
        top_p=0.9,
    ):
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        prompt_len = prompt_ids.size(1)
        completion_ids = output_ids[:, prompt_len:]
        completion_mask = build_completion_mask(
            completion_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        repeated_prompt_mask = attention_mask.repeat_interleave(num_generations, dim=0)
        completion_attention_mask = torch.ones_like(completion_ids)
        output_attention_mask = torch.cat([repeated_prompt_mask, completion_attention_mask], dim=1)

        with torch.no_grad():
            if self.autocast_ctx is None:
                old_per_token_logps = compute_per_token_logps(
                    self.model,
                    output_ids,
                    completion_ids.size(1),
                    attention_mask=output_attention_mask,
                )
            else:
                with self.autocast_ctx():
                    old_per_token_logps = compute_per_token_logps(
                        self.model,
                        output_ids,
                        completion_ids.size(1),
                        attention_mask=output_attention_mask,
                    )

        if was_training:
            self.model.train()

        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(
            output_ids=output_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            old_per_token_logps=old_per_token_logps.detach(),
            completions=completions,
        )
