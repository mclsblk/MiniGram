"""Residual channel implementations and depth-local state.

initialize(hidden) -> ChannelState
read(state, branch) -> hidden, BranchContext
write(state, branch_output, context) -> ChannelState
inject(state, delta) -> ChannelState
finalize(state) -> hidden

branch identifies (layer_index, 'attention' or 'ffn'). Each layer/branch owns
its mapping parameters; the final mixer is registered separately. The same
read context must be passed to write. No module stores activation-dependent
coefficients between calls.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .common import RMSNorm


@dataclass(frozen=True)
class ChannelState:
    """Depth-local activations, reinitialized on every model forward.

    streams: [B, S, R, D]; pre_mix: optional FP32 [B, S, R] delayed mHC read
    coefficients. Neither field is a past_key_values entry or token history.
    """

    streams: torch.Tensor
    pre_mix: torch.Tensor | None = None


@dataclass(frozen=True)
class BranchContext:
    """One branch's write coefficients, consumed immediately by write.

    post_mix: [B, S, R]; comb_mix: [B, S, R, R]; next_pre_mix: [B, S, R].
    All are absent for single; GR uses post_mix; mHC uses all three.
    """

    post_mix: torch.Tensor | None = None
    comb_mix: torch.Tensor | None = None
    next_pre_mix: torch.Tensor | None = None


class SingleResidualChannel(nn.Module):
    """One stream, with the existing pre-norm sublayers and final RMSNorm."""

    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.branch_norms = nn.ModuleList([
            nn.ModuleDict({"attention": RMSNorm(hidden_size), "ffn": RMSNorm(hidden_size)})
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_size)

    def initialize(self, hidden):
        return ChannelState(hidden.unsqueeze(2))

    def read(self, state, branch):
        layer_index, kind = branch
        return self.branch_norms[layer_index][kind](state.streams.squeeze(2)), BranchContext()

    def write(self, state, branch_output, context):
        return ChannelState(state.streams + branch_output.unsqueeze(2))

    def inject(self, state, delta):
        if delta.shape != state.streams.shape:
            raise ValueError("Engram delta must have the same [B,S,R,D] shape as streams")
        return ChannelState(state.streams + delta)

    def finalize(self, state):
        return self.final_norm(state.streams.squeeze(2))


def build_residual_channel(config) -> nn.Module:
    """Sole construction entry; never silently fall back to single."""
    if config.residual_variant == "single":
        return SingleResidualChannel(config.hidden_size, config.num_hidden_layers)
    raise NotImplementedError(
        f"Residual channel {config.residual_variant!r} is configured but not implemented yet; "
        "gr4 and mhc4 arrive in stages 4 and 5 respectively"
    )
