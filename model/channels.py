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
import math

import torch
from torch import nn
import torch.nn.functional as F

from .common import RMSNorm, QwenRMSNorm, DeepSeekRMSNorm


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

    def reset_special_parameters(self):
        pass

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


class GRMixer(nn.Module):
    """Qwen grouped norm, per-feature read gates and per-stream write gates."""

    def __init__(self, hidden_size, rank, initializer_range, use_combine=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.norm = QwenRMSNorm(4 * hidden_size, group_size=hidden_size)
        self.read_down = nn.Linear(4 * hidden_size, rank, bias=False)
        self.read_up = nn.Linear(rank, 4 * hidden_size, bias=False)
        self.write_proj = nn.Linear(4 * hidden_size, 4, bias=False) if use_combine else None
        self.reset_special_parameters()

    @torch.no_grad()
    def reset_special_parameters(self):
        self.norm.weight.zero_()
        nn.init.normal_(self.read_down.weight, std=self.initializer_range)
        nn.init.normal_(self.read_up.weight, std=self.initializer_range)
        if self.write_proj is not None:
            nn.init.normal_(self.write_proj.weight, std=self.initializer_range)

    def forward(self, streams):
        normalized = self.norm(streams.flatten(-2))
        read_mix = torch.sigmoid(self.read_up(F.silu(self.read_down(normalized) / 4)))
        hidden = (read_mix * normalized).unflatten(-1, (4, self.hidden_size)).mean(-2)
        post_mix = None if self.write_proj is None else 2 * torch.sigmoid(self.write_proj(normalized) / 4)
        return hidden, post_mix


class GRResidualChannel(nn.Module):
    def __init__(self, hidden_size, num_layers, rank, initializer_range):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.ModuleDict({
                name: GRMixer(hidden_size, rank, initializer_range)
                for name in ("attention", "ffn")
            }) for _ in range(num_layers)
        ])
        self.final_mixer = GRMixer(hidden_size, rank, initializer_range, use_combine=False)

    def reset_special_parameters(self):
        for branches in self.branches:
            for mixer in branches.values():
                mixer.reset_special_parameters()
        self.final_mixer.reset_special_parameters()

    def initialize(self, hidden):
        return ChannelState(hidden.unsqueeze(2).expand(-1, -1, 4, -1))

    def read(self, state, branch):
        layer_index, kind = branch
        hidden, post_mix = self.branches[layer_index][kind](state.streams)
        return hidden, BranchContext(post_mix=post_mix)

    def write(self, state, branch_output, context):
        return ChannelState(state.streams + context.post_mix.unsqueeze(-1) * branch_output.unsqueeze(2))

    def inject(self, state, delta):
        if delta.shape != state.streams.shape:
            raise ValueError("Engram delta must have the same [B,S,R,D] shape as streams")
        return ChannelState(state.streams + delta)

    def finalize(self, state):
        hidden, _ = self.final_mixer(state.streams)
        return hidden


class MHCMixer(nn.Module):
    """FP32 dynamic coefficients; comb axes are [source, destination]."""

    def __init__(self, hidden_size, initializer_range):
        super().__init__()
        self.eps = 1e-6
        self.sinkhorn_iters = 20
        self.initializer_range = initializer_range
        self.projection = nn.Parameter(torch.empty(24, 4 * hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(24, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.norm = DeepSeekRMSNorm(hidden_size)
        self.reset_special_parameters()

    @torch.no_grad()
    def reset_special_parameters(self):
        nn.init.normal_(self.projection, std=self.initializer_range)
        self.scale.fill_(0.01)
        self.base.zero_()
        self.base[:4].fill_(-math.log(3))
        self.base[8:].view(4, 4).diagonal().fill_(8)
        self.norm.weight.fill_(1)

    def forward(self, streams):
        # Explicitly disable autocast: even the projection feeding Sinkhorn is FP32.
        with torch.autocast(device_type=streams.device.type, enabled=False):
            values = streams.flatten(-2).float()
            mixes = F.linear(values, self.projection.float())
            mixes = mixes * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
            base, scale = self.base.float(), self.scale.float()
            pre = torch.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + self.eps
            post = 2 * torch.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
            logits = (mixes[..., 8:] * scale[2] + base[8:]).unflatten(-1, (4, 4))
            # Match the reference: row softmax + eps, column normalization,
            # followed by 19 row/column normalization pairs.
            comb = torch.softmax(logits, dim=-1) + self.eps
            comb = comb / (comb.sum(-2, keepdim=True) + self.eps)
            for _ in range(self.sinkhorn_iters - 1):
                comb = comb / (comb.sum(-1, keepdim=True) + self.eps)
                comb = comb / (comb.sum(-2, keepdim=True) + self.eps)
        return BranchContext(post_mix=post, comb_mix=comb, next_pre_mix=pre)


def _mhc_collapse(streams, pre_mix):
    return (pre_mix.unsqueeze(-1) * streams.float()).sum(-2).to(streams.dtype)


class MHCResidualChannel(nn.Module):
    def __init__(self, hidden_size, num_layers, initializer_range):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.ModuleDict({
                name: MHCMixer(hidden_size, initializer_range)
                for name in ("attention", "ffn")
            }) for _ in range(num_layers)
        ])
        self.final_norm = DeepSeekRMSNorm(hidden_size)

    @torch.no_grad()
    def reset_special_parameters(self):
        for branches in self.branches:
            for mixer in branches.values():
                mixer.reset_special_parameters()
        self.final_norm.weight.fill_(1)

    def initialize(self, hidden):
        streams = hidden.unsqueeze(2).expand(-1, -1, 4, -1)
        pre_mix = hidden.new_full((*hidden.shape[:-1], 4), 0.25, dtype=torch.float32)
        return ChannelState(streams, pre_mix)

    def read(self, state, branch):
        layer_index, kind = branch
        mixer = self.branches[layer_index][kind]
        context = mixer(state.streams)
        # This branch generates the next pre; its own input uses the previous pre.
        hidden = mixer.norm(_mhc_collapse(state.streams, state.pre_mix))
        return hidden, context

    def write(self, state, branch_output, context):
        residual = (context.comb_mix.unsqueeze(-1) * state.streams.float().unsqueeze(-2)).sum(-3)
        injected = context.post_mix.unsqueeze(-1) * branch_output.float().unsqueeze(-2)
        return ChannelState((residual + injected).to(branch_output.dtype), context.next_pre_mix)

    def inject(self, state, delta):
        if delta.shape != state.streams.shape:
            raise ValueError("Engram delta must have the same [B,S,R,D] shape as streams")
        return ChannelState(state.streams + delta, state.pre_mix)

    def finalize(self, state):
        return self.final_norm(_mhc_collapse(state.streams, state.pre_mix))


def build_residual_channel(config) -> nn.Module:
    """Sole construction entry; never silently fall back to single."""
    if config.residual_variant == "single":
        return SingleResidualChannel(config.hidden_size, config.num_hidden_layers)
    if config.residual_variant == "gr4":
        return GRResidualChannel(
            config.hidden_size, config.num_hidden_layers,
            config.residual_low_rank, config.initializer_range,
        )
    if config.residual_variant == "mhc4":
        return MHCResidualChannel(
            config.hidden_size, config.num_hidden_layers, config.initializer_range,
        )
