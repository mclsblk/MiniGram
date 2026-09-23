"""Shared primitives; this module never imports model assembly or plugins."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """MiniGram norm: cast normalized values before multiplying by the weight."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class QwenRMSNorm(nn.Module):
    """Reference zero-centered norm, optionally grouped within the last axis.

    Inputs have last dimension ``dim``. The weight is applied in FP32 before
    casting back; this deliberately differs from MiniGram's RMSNorm.
    """

    def __init__(self, dim, group_size=None, eps=1e-6):
        super().__init__()
        if dim <= 0 or (group_size is not None and (group_size <= 0 or dim % group_size)):
            raise ValueError("QwenRMSNorm requires positive dim divisible by group_size")
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        self.eps = eps

    def forward(self, hidden_states):
        values = hidden_states.float()
        if self.group_size is not None:
            values = values.reshape(*values.shape[:-1], -1, self.group_size)
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            values = values.flatten(-2)
        return (values * (1.0 + self.weight.float())).to(hidden_states.dtype)
