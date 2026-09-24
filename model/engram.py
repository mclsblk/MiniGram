"""Engram configuration and temporal state, independent of model assembly.

    forward(input_ids, streams, state=None, token_mask=None) -> delta, new_state

Both streams and delta use [B, S, R, D]. Memory heads are ordered by ascending
n-gram order, then head index. Hashers produce head-local bucket IDs; stores
alone translate them to packed offsets. Legacy, Qwen and DeepSeek retain their
respective boundary and normalization formulas.
"""

from dataclasses import asdict, dataclass, fields
import math

import torch
from torch import nn
import torch.nn.functional as F

from .common import RMSNorm, QwenRMSNorm
from .token_compression import (
    CompressedTokenMapper, build_compressed_token_map, prepare_token_map,
)
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

    def forward(self, memory, streams, token_mask=None):
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


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed: int) -> torch.Tensor:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class QwenHasher(nn.Module):
    """Reference EOS-delimited XOR hashes with a distinct prime per memory head."""

    def __init__(self, spec, vocab_size, eos_token_id, memory_layer_index):
        super().__init__()
        self.orders = spec.ngram_orders
        self.num_heads = spec.num_heads
        self.tail_size = max(self.orders) - 1
        eos = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        self.register_buffer("eos_token_id", torch.tensor(eos, dtype=torch.long))
        total_heads = len(self.orders) * self.num_heads
        first_head = memory_layer_index * total_heads
        prime = _find_nth_prime_after(spec.bucket_size - 1, first_head + 1)
        sizes = [prime]
        for _ in range(total_heads - 1):
            prime = _find_nth_prime_after(prime, 1)
            sizes.append(prime)
        self.bucket_sizes = tuple(sizes)
        self.register_buffer("head_capacities", torch.tensor(sizes, dtype=torch.long))
        self.register_buffer("multipliers", _build_layer_multipliers(
            vocab_size, max(self.orders), memory_layer_index, spec.hash_seed,
        ))

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, self.eos_token_id)

    def forward(self, token_ids, tail=None, token_mask=None):
        batch, length = token_ids.shape
        if tail is None:
            tail = self.eos_token_id.expand(batch, self.tail_size)
        context = torch.cat((tail.to(token_ids), token_ids), dim=1)
        shifted = [self._shift_right_ignore_eos(context, shift) for shift in range(self.tail_size + 1)]
        blocks = []
        for index, order in enumerate(self.orders):
            mixed = shifted[0] * self.multipliers[0]
            for position in range(1, order):
                mixed = torch.bitwise_xor(mixed, shifted[position] * self.multipliers[position])
            capacities = self.head_capacities[index * self.num_heads:(index + 1) * self.num_heads]
            blocks.append(torch.remainder(mixed.unsqueeze(-1), capacities))
        hashes = torch.cat(blocks, dim=-1)[:, tail.size(1):tail.size(1) + length]
        return hashes, context[:, -self.tail_size:].clone()


class QwenReadout(nn.Module):
    """One normalized key per stream, shared value, reference signed-sqrt gate."""

    def __init__(self, memory_dim, hidden_size, streams):
        super().__init__()
        self.streams = streams
        self.hidden_size = hidden_size
        self.key_proj = nn.Linear(memory_dim, streams * hidden_size, bias=False)
        self.value_proj = nn.Linear(memory_dim, hidden_size, bias=False)
        self.key_norm = QwenRMSNorm(streams * hidden_size, group_size=hidden_size)
        self.query_norm = QwenRMSNorm(streams * hidden_size, group_size=hidden_size)

    def forward(self, memory, streams, token_mask=None):
        key = self.key_norm(self.key_proj(memory)).unflatten(-1, (self.streams, self.hidden_size))
        query = self.query_norm(streams.flatten(-2)).unflatten(-1, (self.streams, self.hidden_size))
        logits = (key * query).sum(-1, keepdim=True) / math.sqrt(self.hidden_size)
        logits = logits.abs().clamp_min(1e-6).sqrt() * logits.sign()
        return torch.sigmoid(logits) * self.value_proj(memory).unsqueeze(-2)


class QwenCausalConv(nn.Module):
    """Gated residual + SiLU(depthwise causal conv(group-norm(gated residual)))."""

    def __init__(self, hidden_size, streams, kernel_size, dilation):
        super().__init__()
        self.tail_size = (kernel_size - 1) * dilation
        width = streams * hidden_size
        self.norm = QwenRMSNorm(width, group_size=hidden_size)
        self.conv = nn.Conv1d(width, width, kernel_size, groups=width, dilation=dilation, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, values, state=None, token_mask=None):
        batch, length, streams, width = values.shape
        normalized = self.norm(values.flatten(-2)).unflatten(-1, (streams, width))
        if token_mask is not None:
            mask = token_mask[:, :, None, None].to(values)
            values = values * mask
            normalized = normalized * mask
        state = values.new_empty(batch, 0, streams, width) if state is None else state.to(values)
        if length == 0:
            return values, state
        source = torch.cat((state, normalized), dim=1)
        padded = F.pad(source.flatten(-2).transpose(1, 2), (self.tail_size, 0))
        convolved = F.silu(self.conv(padded[..., -(self.tail_size + length):]))
        convolved = convolved.transpose(1, 2).unflatten(-1, (streams, width))
        tail = source[:, -self.tail_size:] if self.tail_size else source[:, :0]
        return values + convolved, tail.clone()


class DeepSeekHasher(nn.Module):
    """Compressed token history; DEAD blocks all earlier lookbacks, including across calls."""

    def __init__(self, spec, layer_id, memory_layer_index):
        super().__init__()
        self.layer_id = layer_id
        self.orders = spec.ngram_orders
        self.num_heads = spec.num_heads
        self.tail_size = max(self.orders) - 1
        count = len(self.orders) * self.num_heads
        prime = _find_nth_prime_after(spec.bucket_size - 1, memory_layer_index * count + 1)
        sizes = [prime]
        for _ in range(count - 1):
            prime = _find_nth_prime_after(prime, 1)
            sizes.append(prime)
        self.bucket_sizes = tuple(sizes)
        self.register_buffer("head_capacities", torch.tensor(sizes, dtype=torch.long))
        self.register_buffer("multipliers", torch.zeros(self.tail_size + 1, dtype=torch.long))
        self.register_buffer("pad_id", torch.tensor(-1, dtype=torch.long))

    def prepare_mapping(self, compressed_vocab_size, compressed_pad_id):
        # NumPy's reference RNG is intentional: torch RNG produces different addresses.
        import numpy as np
        bound = max(1, ((2**63 - 1) // compressed_vocab_size) // 2)
        generator = np.random.default_rng(10007 * self.layer_id)
        values = generator.integers(0, bound, size=self.tail_size + 1, dtype=np.int64)
        self.multipliers.copy_(torch.as_tensor(values * 2 + 1, device=self.multipliers.device))
        self.pad_id.fill_(compressed_pad_id)

    def forward(self, token_ids, tail=None, token_mask=None):
        if self.pad_id.item() < 0:
            raise RuntimeError("DeepSeek hash mapping is not ready; prepare the token map with config.pad_token_id set")
        if token_mask is not None:
            token_ids = token_ids.masked_fill(~token_mask.bool(), -1)
        batch, length = token_ids.shape
        tail = token_ids.new_empty(batch, 0) if tail is None else tail.to(token_ids)
        context = torch.cat((tail, token_ids), dim=1)
        positions = torch.arange(tail.size(1), context.size(1), device=token_ids.device).expand(batch, length)
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        rolling = torch.zeros_like(positions)
        hashes = []
        order_index = 0
        for shift in range(self.tail_size + 1):
            source = context.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == -1)
            tokens = torch.where(blocked, self.pad_id, source)
            rolling = torch.bitwise_xor(rolling, tokens * self.multipliers[shift])
            if shift + 1 in self.orders:
                capacities = self.head_capacities[order_index * self.num_heads:(order_index + 1) * self.num_heads]
                hashes.append(torch.remainder(rolling.unsqueeze(-1), capacities))
                order_index += 1
        return torch.cat(hashes, dim=-1), context[:, -self.tail_size:].clone()


class DeepSeekReadout(nn.Module):
    """Reference FP32 gate with separate per-dimension q/k weights."""

    def __init__(self, memory_dim, hidden_size, streams):
        super().__init__()
        self.hidden_size = hidden_size
        self.streams = streams
        self.wkv = nn.Linear(memory_dim, hidden_size * (streams + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(streams, hidden_size))
        self.k_weight = nn.Parameter(torch.ones(streams, hidden_size))
        self.eps = 1e-6

    def forward(self, memory, streams, token_mask=None):
        key, value = self.wkv(memory).split([self.streams * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.streams, self.hidden_size))
        hidden = streams.float()
        weight = self.q_weight.float() * self.k_weight.float()
        rstd = torch.rsqrt(hidden.square().mean(-1) + self.eps)
        rstd = rstd * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hidden * weight * key).sum(-1) * rstd * self.hidden_size ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.bool().unsqueeze(-1), 0)
        return (gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(streams.dtype)


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
        gated = self.readout(memory, streams, token_mask)
        delta, post_state = self.postprocessor(gated, state.post_state, token_mask)
        return delta, EngramState(hash_tail, post_state)

    @torch.no_grad()
    def reset_special_parameters(self):
        """Preserve padding and zero-convolution initialization after HF post_init."""
        self.memory_store.reset_padding()
        if isinstance(self.postprocessor, (LegacyCausalConv, QwenCausalConv)):
            nn.init.zeros_(self.postprocessor.conv.weight)
        for module in self.modules():
            if isinstance(module, QwenRMSNorm):
                nn.init.zeros_(module.weight)
        if isinstance(self.readout, DeepSeekReadout):
            nn.init.ones_(self.readout.q_weight)
            nn.init.ones_(self.readout.k_weight)


@torch.no_grad()
def set_engram_token_map(layers, config, token_map):
    """Install the offline reference lookup and synchronize dependent hash metadata."""
    targets = [layer for layer in layers.values() if isinstance(layer.token_mapper, CompressedTokenMapper)]
    if not targets:
        raise ValueError("No compressed Engram token mapper is enabled")
    lookup, compressed_size, pad_id = prepare_token_map(
        token_map, config.vocab_size, config.pad_token_id,
        [layer.token_mapper for layer in targets],
    )
    for memory_layer_index, layer in enumerate(targets):
        hasher = layer.hasher
        if isinstance(hasher, DeepSeekHasher):
            hasher.prepare_mapping(compressed_size, pad_id)
        elif isinstance(hasher, QwenHasher):
            multipliers = _build_layer_multipliers(
                compressed_size, hasher.tail_size + 1, memory_layer_index,
                config.engram_overrides["hash_seed"],
            )
            hasher.multipliers.copy_(multipliers.to(hasher.multipliers.device))
            eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
            hasher.eos_token_id.copy_(lookup[eos].to(hasher.eos_token_id))
        layer.token_mapper.set_mapping(lookup, compressed_size)


def build_engram_layers(config) -> nn.ModuleDict:
    """Assemble independent memory layers; compressed mappings can be prepared later."""
    if not config.use_engrams:
        return nn.ModuleDict()
    spec = EngramSpec(**config.engram_overrides)
    store_class = {"separate": SeparateMemoryStore, "packed": PackedMemoryStore}[spec.memory_store]
    layers = nn.ModuleDict()
    for memory_layer_index, layer_id in enumerate(config.engram_n_layer_list):
        mapper = (CompressedTokenMapper(config.vocab_size) if spec.token_mapper == "compressed"
                  else IdentityTokenMapper())
        padding_idx = None
        if spec.hasher == "legacy":
            hasher = LegacyHasher(spec, layer_id)
            padding_idx = 0
        elif spec.hasher == "qwen_xor":
            hasher = QwenHasher(spec, config.vocab_size, config.eos_token_id, memory_layer_index)
        else:
            hasher = DeepSeekHasher(spec, layer_id, memory_layer_index)
            if spec.token_mapper == "identity" and config.pad_token_id is not None:
                hasher.prepare_mapping(config.vocab_size, config.pad_token_id)
        memory_dim = len(hasher.bucket_sizes) * spec.head_dim
        if spec.readout == "legacy":
            readout = LegacyReadout(memory_dim, config.hidden_size)
        elif spec.readout == "qwen_signed_sqrt":
            readout = QwenReadout(memory_dim, config.hidden_size, config.residual_channels)
        else:
            readout = DeepSeekReadout(memory_dim, config.hidden_size, config.residual_channels)
        if spec.postprocessor == "legacy_conv":
            postprocessor = LegacyCausalConv(config.hidden_size, spec.conv_kernel_size)
        elif spec.postprocessor == "causal_conv":
            postprocessor = QwenCausalConv(
                config.hidden_size, config.residual_channels, spec.conv_kernel_size, spec.conv_dilation,
            )
        else:
            postprocessor = IdentityPostProcessor()
        layers[str(layer_id)] = EngramLayer(
            spec, mapper, hasher,
            store_class(hasher.bucket_sizes, spec.head_dim, padding_idx=padding_idx),
            readout, postprocessor,
        )
    return layers
