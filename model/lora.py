import math

import torch
from torch import nn


def _unwrap_model(model):
    if hasattr(model, "module"):
        model = model.module
    return getattr(model, "_orig_mod", model)


class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16.0, dropout=0.05):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be greater than 0.")

        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def merged_linear(self):
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base_layer.weight.data.add_(delta.to(device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype))
        return self.base_layer


def apply_lora(model, target_modules, r=8, alpha=16.0, dropout=0.05):
    raw_model = _unwrap_model(model)
    targets = set(target_modules)
    replaced = 0

    def visit(module):
        nonlocal replaced
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
            else:
                visit(child)

    visit(raw_model)
    if replaced == 0:
        raise ValueError(f"No modules matched LoRA targets: {sorted(targets)}")
    return replaced


def mark_only_lora_as_trainable(model):
    raw_model = _unwrap_model(model)
    for parameter in raw_model.parameters():
        parameter.requires_grad = False
    for module in raw_model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True


def lora_state_dict(model, dtype=None):
    raw_model = _unwrap_model(model)
    state = {}
    for name, parameter in raw_model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        tensor = parameter.detach()
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        state[name] = tensor.cpu()
    return state


def load_lora_state_dict(model, state_dict, strict=True):
    raw_model = _unwrap_model(model)
    expected = set(lora_state_dict(raw_model).keys())
    found = set(state_dict.keys())
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    if strict and (missing or unexpected):
        raise RuntimeError(f"LoRA state mismatch. missing={missing}, unexpected={unexpected}")
    raw_model.load_state_dict(state_dict, strict=False)
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def merge_lora_weights(model):
    raw_model = _unwrap_model(model)
    merged = 0

    def visit(module):
        nonlocal merged
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged_linear())
                merged += 1
            else:
                visit(child)

    visit(raw_model)
    return merged
