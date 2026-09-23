"""Residual channel contracts; concrete implementations begin in stage 2.

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


def build_residual_channel(config) -> nn.Module:
    """Sole construction entry; never silently fall back to single."""
    raise NotImplementedError(
        f"Residual channel {config.residual_variant!r} is configured but not implemented yet; "
        "single, gr4 and mhc4 arrive in stages 2, 4 and 5 respectively"
    )
