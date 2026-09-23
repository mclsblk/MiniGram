"""Engram configuration and temporal state, independent of model assembly.

    forward(input_ids, streams, state=None, token_mask=None) -> delta, new_state

Both streams and delta use [B, S, R, D]. Memory heads are ordered by ascending
n-gram order, then head index. Hashers produce head-local bucket IDs; stores
alone translate them to packed offsets. Qwen and DeepSeek land in stages 6
and 7; the builder must never substitute one preset for another.
"""

from dataclasses import asdict, dataclass, fields
import math

import torch
from torch import nn
import torch.nn.functional as F

from .common import RMSNorm
from .validation import (
    validate_engram_selection, validate_engram_options, validate_engram_parameters,
)


@dataclass(frozen=True)
class EngramSpec:
    ngram_orders: tuple[int, ...] | list[int]
    token_mapper: str
    hasher: str
    memory_store: str
    readout: str
    postprocessor: str
    insertion: str
    bucket_size: int
    num_heads: int
    head_dim: int
    hash_seed: int | None
    conv_kernel_size: int
    conv_dilation: int

    def to_dict(self):
        """Plain JSON-compatible fields, with no module or tensor objects."""
        values = asdict(self)
        values["ngram_orders"] = list(self.ngram_orders)
        return values


def resolve_engram_spec(variant, overrides, hidden_size):
    """Resolve and validate shallow configuration; no modules are built here."""
    presets = {
        "legacy": dict(ngram_orders=(2, 3), token_mapper="identity", hasher="legacy",
                       memory_store="separate", readout="legacy",
                       postprocessor="legacy_conv", insertion="after_attention"),
        "qwen": dict(ngram_orders=(2, 3), token_mapper="identity", hasher="qwen_xor",
                     memory_store="packed", readout="qwen_signed_sqrt",
                     postprocessor="causal_conv", insertion="before_attention"),
        "deepseek": dict(ngram_orders=(2, 3, 4), token_mapper="compressed", hasher="deepseek_xor",
                         memory_store="packed", readout="deepseek_signed_sqrt",
                         postprocessor="identity", insertion="before_attention"),
    }
    overrides = {} if overrides is None else overrides
    validate_engram_selection(variant, overrides, presets, {field.name for field in fields(EngramSpec)})
    values = {**presets[variant], "bucket_size": 1024, "num_heads": 4, **overrides}
    validate_engram_options(values)
    orders = values["ngram_orders"]
    values["ngram_orders"] = tuple(orders)
    values.setdefault("head_dim", math.ceil(hidden_size / (len(orders) * values["num_heads"])))
    processor = values["postprocessor"]
    values.setdefault("conv_kernel_size", {"identity": 1, "legacy_conv": 3, "causal_conv": 4}[processor])
    values.setdefault("conv_dilation", max(orders) if processor == "causal_conv" else 1)
    hasher = values["hasher"]
    values.setdefault("hash_seed", {"legacy": 17, "qwen_xor": 1234, "deepseek_xor": None}[hasher])
    validate_engram_parameters(values)
    return EngramSpec(**values)


@dataclass(frozen=True)
class EngramState:
    """Decode history: hash-domain token context [B, T] and conv input [B, T, R, D].

    Hash context includes the selected hasher's boundary markers. post_state is
    absent for identity processing. State belongs to the caller, not the module.
    """

    hash_tail: torch.Tensor | None = None
    post_state: torch.Tensor | None = None

    def reorder(self, beam_idx: torch.Tensor) -> "EngramState":
        def select(value):
            return None if value is None else value.index_select(0, beam_idx.to(value.device))

        return EngramState(select(self.hash_tail), select(self.post_state))


class IdentityTokenMapper(nn.Module):
    def forward(self, input_ids):
        return input_ids.long()


class LegacyHasher(nn.Module):
    """Original XOR addresses; missing prefix windows use reserved bucket zero.

    As in the original algorithm, token IDs (including EOS/padding) participate
    literally. token_mask does not reset or remove legacy windows.
    """

    def __init__(self, spec, layer_id):
        super().__init__()
        self.orders = spec.ngram_orders
        self.num_heads = spec.num_heads
        self.tail_size = max(self.orders) - 1
        self.bucket_sizes = (spec.bucket_size,) * (len(self.orders) * self.num_heads)
        self.modulus = spec.bucket_size - 1
        max_int = (1 << 31) - 1
        for n in self.orders:
            multipliers, offsets = [], []
            for head in range(self.num_heads):
                seed = spec.hash_seed + 10007 * (layer_id + 1) + 1543 * (n + 1) + 8191 * (head + 1)
                multipliers.append([
                    ((seed + 32771 * (pos + 1) + 65537 * (head + 1) * (pos + 1)) % max_int) * 2 + 1
                    for pos in range(n)
                ])
                offsets.append((seed * 48271 + 97 * (n + head + 1)) % max_int)
            self.register_buffer(f"multipliers_{n}", torch.tensor(multipliers, dtype=torch.long))
            self.register_buffer(f"offsets_{n}", torch.tensor(offsets, dtype=torch.long))

    def forward(self, token_ids, tail=None, token_mask=None):
        batch, length = token_ids.shape
        tail = token_ids.new_empty(batch, 0) if tail is None else tail.to(token_ids)
        context = torch.cat((tail, token_ids), dim=1)
        hashes = token_ids.new_zeros(batch, length, len(self.bucket_sizes))
        for order_index, n in enumerate(self.orders):
            full_hash = token_ids.new_zeros(batch, context.size(1), self.num_heads)
            if context.size(1) >= n:
                windows = context.unfold(1, n, 1)
                multipliers = getattr(self, f"multipliers_{n}")
                offsets = getattr(self, f"offsets_{n}")
                mixed = windows[:, :, 0, None] * multipliers[:, 0]
                for pos in range(1, n):
                    mixed = torch.bitwise_xor(mixed, windows[:, :, pos, None] * multipliers[:, pos])
                full_hash[:, n - 1:] = (mixed + offsets) % self.modulus + 1
            start = order_index * self.num_heads
            hashes[:, :, start:start + self.num_heads] = full_hash[:, tail.size(1):tail.size(1) + length]
        return hashes, context[:, -self.tail_size:].clone()


class SeparateMemoryStore(nn.Module):
    def __init__(self, bucket_sizes, head_dim, padding_idx=None):
        super().__init__()
        self.padding_idx = padding_idx
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, head_dim, padding_idx=padding_idx) for size in bucket_sizes
        ])

    def forward(self, bucket_ids):
        return torch.cat([table(bucket_ids[..., head]) for head, table in enumerate(self.embeddings)], dim=-1)

    @torch.no_grad()
    def reset_padding(self):
        if self.padding_idx is not None:
            for table in self.embeddings:
                table.weight[self.padding_idx].zero_()


class PackedMemoryStore(nn.Module):
    def __init__(self, bucket_sizes, head_dim, padding_idx=None):
        super().__init__()
        self.padding_idx = padding_idx
        offsets, total = [], 0
        for size in bucket_sizes:
            offsets.append(total)
            total += size
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(total, head_dim)
        self.reset_padding()

    def forward(self, bucket_ids):
        values = self.embedding(bucket_ids + self.offsets)
        if self.padding_idx is not None:
            # Packed tables have one padding row per head; masking also blocks gradients.
            values = values.masked_fill((bucket_ids == self.padding_idx).unsqueeze(-1), 0)
        return values.flatten(-2)

    @torch.no_grad()
    def reset_padding(self):
        if self.padding_idx is not None:
            self.embedding.weight.index_fill_(0, self.offsets + self.padding_idx, 0)


class LegacyReadout(nn.Module):
    def __init__(self, memory_dim, hidden_size):
        super().__init__()
        self.query_norm = RMSNorm(hidden_size)
        self.key_proj = nn.Linear(memory_dim, hidden_size, bias=False)
        self.value_proj = nn.Linear(memory_dim, hidden_size, bias=False)
        self.key_norm = RMSNorm(hidden_size)
        self.value_norm = RMSNorm(hidden_size)
        self.gate_bias = nn.Parameter(torch.tensor(-4.0))
        self.scale = hidden_size ** -0.5

    def forward(self, memory, streams):
        key = self.key_norm(self.key_proj(memory)).unsqueeze(2)
        value = self.value_norm(self.value_proj(memory)).unsqueeze(2)
        logits = (self.query_norm(streams) * key).sum(dim=-1, keepdim=True) * self.scale
        return torch.sigmoid(logits + self.gate_bias) * value


class IdentityPostProcessor(nn.Module):
    def forward(self, values, state=None, token_mask=None):
        return values, None


class LegacyCausalConv(nn.Module):
    """Original depthwise convolution of gated memory, plus its residual."""

    def __init__(self, hidden_size, kernel_size):
        super().__init__()
        self.tail_size = kernel_size - 1
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, values, state=None, token_mask=None):
        batch, length, _, width = values.shape
        if state is None:
            state = values.new_empty(batch, 0, 1, width)
        else:
            state = state.to(values)
        if length == 0:
            return values, state
        source = torch.cat((state, values), dim=1).squeeze(2)
        convolved = self.conv(F.pad(source.transpose(1, 2), (self.tail_size, 0))).transpose(1, 2)
        tail = source[:, -self.tail_size:] if self.tail_size else source[:, :0]
        return values + convolved[:, -length:].unsqueeze(2), tail.unsqueeze(2).clone()


class EngramLayer(nn.Module):
    def __init__(self, spec, token_mapper, hasher, memory_store, readout, postprocessor):
        super().__init__()
        self.before_attention = spec.insertion == "before_attention"
        self.token_mapper = token_mapper
        self.hasher = hasher
        self.memory_store = memory_store
        self.readout = readout
        self.postprocessor = postprocessor

    def forward(self, input_ids, streams, state=None, token_mask=None):
        state = EngramState() if state is None else state
        mapped = self.token_mapper(input_ids)
        bucket_ids, hash_tail = self.hasher(mapped, state.hash_tail, token_mask)
        memory = self.memory_store(bucket_ids)
        gated = self.readout(memory, streams)
        delta, post_state = self.postprocessor(gated, state.post_state, token_mask)
        return delta, EngramState(hash_tail, post_state)

    @torch.no_grad()
    def reset_special_parameters(self):
        """Preserve padding and zero-convolution initialization after HF post_init."""
        self.memory_store.reset_padding()
        if isinstance(self.postprocessor, LegacyCausalConv):
            nn.init.zeros_(self.postprocessor.conv.weight)


def build_engram_layers(config) -> nn.ModuleDict:
    """Build independent layer-index-keyed modules; disabled memory is empty.

    Only legacy components are available at this stage. Component substitutions
    that need Qwen/DeepSeek never silently use a legacy implementation instead.
    """
    if not config.use_engrams:
        return nn.ModuleDict()
    if config.engram_variant != "legacy":
        raise NotImplementedError(f"Engram preset {config.engram_variant!r} arrives in stages 6/7")
    # MiniGramConfig already resolved defaults and validated the combination.
    spec = EngramSpec(**config.engram_overrides)
    supported = {"token_mapper": ("identity",), "hasher": ("legacy",),
                 "readout": ("legacy",), "postprocessor": ("identity", "legacy_conv")}
    for name, choices in supported.items():
        if getattr(spec, name) not in choices:
            raise NotImplementedError(f"Engram component {name}={getattr(spec, name)!r} is not implemented yet")
    store_class = {"separate": SeparateMemoryStore, "packed": PackedMemoryStore}[spec.memory_store]
    layers = nn.ModuleDict()
    for layer_id in config.engram_n_layer_list:
        hasher = LegacyHasher(spec, layer_id)
        postprocessor = (LegacyCausalConv(config.hidden_size, spec.conv_kernel_size)
                         if spec.postprocessor == "legacy_conv" else IdentityPostProcessor())
        layers[str(layer_id)] = EngramLayer(
            spec, IdentityTokenMapper(), hasher,
            store_class(hasher.bucket_sizes, spec.head_dim, padding_idx=0),
            LegacyReadout(len(hasher.bucket_sizes) * spec.head_dim, config.hidden_size), postprocessor,
        )
    return layers
