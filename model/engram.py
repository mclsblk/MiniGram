"""Engram configuration and temporal state, independent of model assembly.

    forward(input_ids, streams, state=None, token_mask=None) -> delta, new_state

Both streams and delta use [B, S, R, D]. Memory heads are ordered by ascending
n-gram order, then head index. Hashers produce head-local bucket IDs; stores
alone translate them to packed offsets. Concrete algorithms land in stages 3,
6 and 7; the builder must never substitute one preset for another.
"""

from dataclasses import asdict, dataclass, fields
import math

import torch
from torch import nn


@dataclass(frozen=True)
class EngramSpec:
    ngram_orders: tuple[int, ...]
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
    if not isinstance(variant, str) or variant not in presets:
        raise ValueError(f"Unknown engram_variant: {variant!r}")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("engram_overrides must be a shallow dictionary")
    unknown = set(overrides) - {field.name for field in fields(EngramSpec)}
    if unknown:
        raise ValueError(f"Unknown engram_overrides fields: {sorted(map(str, unknown))}")
    values = {**presets[variant], "bucket_size": 1024, "num_heads": 4, **overrides}
    choices = {
        "token_mapper": ("identity", "compressed"),
        "hasher": ("legacy", "qwen_xor", "deepseek_xor"),
        "memory_store": ("separate", "packed"),
        "readout": ("legacy", "qwen_signed_sqrt", "deepseek_signed_sqrt"),
        "postprocessor": ("identity", "legacy_conv", "causal_conv"),
        "insertion": ("before_attention", "after_attention"),
    }
    for name, supported in choices.items():
        if values[name] not in supported:
            raise ValueError(f"engram_overrides.{name} must be one of {supported}")
    orders = values["ngram_orders"]
    if (not isinstance(orders, (list, tuple)) or not orders
            or any(type(order) is not int or order < 2 for order in orders)
            or list(orders) != sorted(set(orders))):
        raise ValueError("engram_overrides.ngram_orders must be nonempty, strictly increasing integers >= 2")
    values["ngram_orders"] = tuple(orders)
    for name in ("bucket_size", "num_heads"):
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"engram_overrides.{name} must be a positive integer")
    if values["bucket_size"] < 2:
        raise ValueError("engram_overrides.bucket_size must be at least 2")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    values.setdefault("head_dim", math.ceil(hidden_size / (len(orders) * values["num_heads"])))
    processor = values["postprocessor"]
    values.setdefault("conv_kernel_size", {"identity": 1, "legacy_conv": 3, "causal_conv": 4}[processor])
    values.setdefault("conv_dilation", max(orders) if processor == "causal_conv" else 1)
    for name in ("head_dim", "conv_kernel_size", "conv_dilation"):
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"engram_overrides.{name} must be a positive integer")
    if processor == "legacy_conv" and values["conv_dilation"] != 1:
        raise ValueError("legacy_conv requires conv_dilation=1; use causal_conv for dilation")
    hasher = values["hasher"]
    values.setdefault("hash_seed", {"legacy": 17, "qwen_xor": 1234, "deepseek_xor": None}[hasher])
    if hasher == "deepseek_xor":
        if values["hash_seed"] is not None:
            raise ValueError("deepseek_xor derives seeds from layer IDs; hash_seed must be None")
    elif type(values["hash_seed"]) is not int or values["hash_seed"] < 0:
        raise ValueError(f"{hasher} requires a nonnegative integer hash_seed")
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


def build_engram_layers(config) -> nn.ModuleDict:
    """Build layer-index-keyed modules from an already resolved MiniGramConfig.

    Layers own independent parameters and histories. A disabled Engram creates
    no parameters or caches. Enabled presets become available in later stages.
    """
    if not config.use_engrams:
        return nn.ModuleDict()
    raise NotImplementedError(
        f"Engram preset {config.engram_variant!r} is configured but not implemented yet; "
        "legacy, qwen and deepseek arrive in stages 3, 6 and 7 respectively"
    )
