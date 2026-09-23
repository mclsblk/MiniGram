"""Boundary validation only; no imports from model, Engram or channel modules.

Functions inspect plain configuration values and external cache containers.
They neither resolve defaults nor mutate input or construct components.
"""


def validate_model_config(hidden_size, num_hidden_layers, residual_variant, rank,
                          layers, use_engrams, extra_fields):
    deprecated = {
        "engram_vocab_size": "bucket_size", "engram_n_gram_list": "ngram_orders",
        "engram_num_heads": "num_heads", "engram_conv_size": "conv_kernel_size",
        "engram_hash_seed": "hash_seed",
    }
    supplied = sorted(deprecated.keys() & extra_fields.keys())
    if supplied:
        replacements = ", ".join(f"{name} -> {deprecated[name]}" for name in supplied)
        raise ValueError(f"Removed Engram configuration fields; use engram_overrides instead: {replacements}")
    for name, value in (("hidden_size", hidden_size), ("num_hidden_layers", num_hidden_layers)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    channel_counts = {"single": 1, "gr4": 4, "mhc4": 4}
    if residual_variant not in channel_counts:
        raise ValueError(f"Unknown residual_variant: {residual_variant!r}")
    channels = channel_counts[residual_variant]
    if extra_fields.get("residual_channels", channels) != channels:
        raise ValueError(f"residual_variant={residual_variant!r} requires residual_channels={channels}")
    if residual_variant == "gr4" and rank <= 0:
        raise ValueError("residual_low_rank must be positive for gr4")
    if any(layer < 0 for layer in layers) or len(set(layers)) != len(layers):
        raise ValueError("engram_n_layer_list must contain distinct nonnegative integer layer indices")
    if use_engrams and any(layer >= num_hidden_layers for layer in layers):
        raise ValueError("engram_n_layer_list contains an index outside num_hidden_layers")


def validate_engram_combination(use_engrams, variant, residual_variant, readout, postprocessor):
    if use_engrams and residual_variant != "single":
        if variant == "legacy" or readout == "legacy" or postprocessor == "legacy_conv":
            raise ValueError("legacy Engram, readout and convolution require residual_variant='single'")


def validate_engram_selection(variant, overrides, presets, fields):
    if variant not in presets:
        raise ValueError(f"Unknown engram_variant: {variant!r}")
    unknown = set(overrides) - fields
    if unknown:
        raise ValueError(f"Unknown engram_overrides fields: {sorted(map(str, unknown))}")


def validate_engram_options(values):
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
    if not orders or any(order < 2 for order in orders) or list(orders) != sorted(set(orders)):
        raise ValueError("engram_overrides.ngram_orders must be nonempty, strictly increasing integers >= 2")
    if values["bucket_size"] < 2:
        raise ValueError("engram_overrides.bucket_size must be at least 2")
    if values["num_heads"] <= 0:
        raise ValueError("engram_overrides.num_heads must be positive")


def validate_engram_parameters(values):
    if values["head_dim"] <= 0:
        raise ValueError("engram_overrides.head_dim must be positive")
    processor = values["postprocessor"]
    if processor != "identity":
        for name in ("conv_kernel_size", "conv_dilation"):
            if values[name] <= 0:
                raise ValueError(f"engram_overrides.{name} must be positive for {processor}")
    if processor == "legacy_conv" and values["conv_dilation"] != 1:
        raise ValueError("legacy_conv requires conv_dilation=1; use causal_conv for dilation")
    hasher = values["hasher"]
    if hasher == "deepseek_xor":
        if values["hash_seed"] is not None:
            raise ValueError("deepseek_xor derives seeds from layer IDs; hash_seed must be None")
    elif values["hash_seed"] < 0:
        raise ValueError(f"{hasher} requires a nonnegative hash_seed")


def validate_past_key_values(past_key_values, num_layers, use_cache=True):
    if past_key_values is None:
        return
    if not use_cache:
        raise ValueError("Supplying past_key_values requires use_cache=True")
    if not isinstance(past_key_values, (list, tuple)):
        raise TypeError("past_key_values must be a sequence of layer dictionaries")
    if len(past_key_values) != num_layers:
        raise ValueError("past_key_values must contain one cache per layer")
    for cache in past_key_values:
        if not isinstance(cache, dict) or set(cache) - {"attn", "engram"}:
            raise ValueError("Layer cache accepts only 'attn' and 'engram'; old cache formats are unsupported")
        attn = cache.get("attn")
        if attn is not None and (not isinstance(attn, tuple) or len(attn) != 2):
            raise TypeError("The 'attn' cache entry must be a (key, value) tuple")
