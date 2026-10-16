"""Tokenizer normalization and persistent compressed-ID lookup.

Engram owns hash algorithms and calls this module to prepare and install maps.
This module depends only on PyTorch and boundary validation, never on Engram.
"""

import torch
from torch import nn

from .validation import validate_token_map


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with (no clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


class CompressedTokenMapper(nn.Module):
    """Persistent fixed-size lookup; no tokenizer work in model forward."""

    def __init__(self, vocab_size):
        super().__init__()
        self.register_buffer("token_map", torch.full((vocab_size,), -1, dtype=torch.long))
        self.register_buffer("compressed_vocab_size", torch.tensor(0, dtype=torch.long))
        self.started = False

    def forward(self, input_ids):
        if not self.started:
            if self.compressed_vocab_size.item() == 0:
                raise RuntimeError("Engram token map is not ready; call set_engram_token_map or load prepared weights")
            self.started = True
        return self.token_map[input_ids.long()]

    @torch.no_grad()
    def set_mapping(self, lookup, compressed_size):
        self.token_map.copy_(lookup.to(self.token_map))
        self.compressed_vocab_size.fill_(compressed_size)


def prepare_token_map(token_map, vocab_size, pad_token_id, mappers):
    """Validate before mutation and derive the compressed vocabulary metadata."""
    lookup = torch.as_tensor(token_map, dtype=torch.long)
    validate_token_map(lookup, vocab_size, pad_token_id, any(mapper.started for mapper in mappers))
    compressed_size = int(lookup.max().item()) + 1
    pad_id = int(lookup[pad_token_id].item())
    return lookup, compressed_size, pad_id
