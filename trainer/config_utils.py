"""Strict, stage-specific TOML input and a small, explicit CLI override surface."""
import copy
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# These are input fields, not arbitrary PretrainedConfig kwargs.
MODEL_FIELDS = {
    **dict.fromkeys(("hidden_size", "num_hidden_layers", "num_attention_heads", "num_kv_heads",
                     "intermediate_size", "max_length", "residual_low_rank", "num_experts",
                     "num_expert_per_token"), int),
    **dict.fromkeys(("dropout", "initializer_range", "rope_theta", "aux_loss_coef"), float),
    **dict.fromkeys(("use_moe", "use_engrams", "flash_attention"), bool),
    **dict.fromkeys(("hidden_act", "engram_variant", "residual_variant"), str),
    "engram_n_layer_list": list,
    "engram_overrides": dict,
    "rope_scaling_params": dict,
}
ENGRAM_FIELDS = {
    **dict.fromkeys(("bucket_size", "num_heads", "head_dim", "hash_seed", "conv_kernel_size",
                     "conv_dilation"), int),
    **dict.fromkeys(("token_mapper", "hasher", "memory_store", "readout", "postprocessor",
                     "insertion"), str),
    "ngram_orders": list,
}
ROPE_FIELDS = {"type": str, **dict.fromkeys(("beta_fast", "beta_slow", "factor",
              "attention_factor"), float), "original_max_position_embeddings": int}
SCHEMA = {
    "data": {"data_path": str, "tokenizer_path": str, "tokenizer_name": str,
             "max_length": int, "num_workers": int},
    "model": MODEL_FIELDS,
    "train": {"epochs": int, "batch_size": int, "accumulation_steps": int,
              "learning_rate": float, "min_lr": float, "warmup_ratio": float,
              "weight_decay": float, "grad_clip": float, "dtype": str, "seed": int},
    "runtime": {"device": str, "use_compile": bool},
    "output": {"save_dir": str, "save_name": str, "save_interval": int, "log_interval": int},
}
DEFAULTS = {"model": {}, "runtime": {"device": "auto", "use_compile": False},
            "train": {"weight_decay": 0.01}, "data": {"num_workers": 0}}
CLI_FIELDS = {"device": ("runtime", "device"), "save_dir": ("output", "save_dir"),
              **{key: ("train", key) for key in ("batch_size", "learning_rate", "epochs")}}


def _check_fields(values, schema, prefix):
    if not isinstance(values, dict):
        raise ValueError(f"{prefix}: expected a table")
    for key, value in values.items():
        name = f"{prefix}.{key}"
        if key not in schema:
            raise ValueError(f"{name}: unknown field")
        kind = schema[key]
        valid = type(value) is kind or (kind is float and type(value) is int)
        if not valid:
            raise ValueError(f"{name}: expected {kind.__name__}")
        if kind is float and not math.isfinite(value):
            raise ValueError(f"{name}: must be finite")
        if kind is str and not value.strip():
            raise ValueError(f"{name}: must not be empty")
        if kind is list and any(type(item) is not int for item in value):
            raise ValueError(f"{name}: expected an integer array")


def validate_train_config(config, stage="pretrain"):
    if stage not in ("pretrain", "sft") or config.get("stage") != stage:
        raise ValueError(f"stage: expected supported stage {stage!r}")
    stage_schema = dict(SCHEMA)
    if stage == "sft":
        stage_schema.pop("model")
        stage_schema["data"] = {**SCHEMA["data"], "train_on_prompt": bool}
    unknown = set(config) - {"stage", *stage_schema}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    for group, schema in stage_schema.items():
        if group not in config:
            raise ValueError(f"{group}: missing table")
        _check_fields(config[group], schema, group)
    required = {
        "data": ("data_path", "max_length", "num_workers"),
        "train": tuple(SCHEMA["train"]), "output": tuple(SCHEMA["output"]),
        "runtime": tuple(SCHEMA["runtime"]),
    }
    for group, fields in required.items():
        for field in fields:
            if field not in config[group]:
                raise ValueError(f"{group}.{field}: required")
    data, model, train = config["data"], config.get("model", {}), config["train"]
    if ("tokenizer_path" in data) == ("tokenizer_name" in data):
        raise ValueError("data: specify exactly one of tokenizer_path and tokenizer_name")
    if "engram_overrides" in model:
        _check_fields(model["engram_overrides"], ENGRAM_FIELDS, "model.engram_overrides")
    if "rope_scaling_params" in model:
        rope = model["rope_scaling_params"]
        _check_fields(rope, ROPE_FIELDS, "model.rope_scaling_params")
        if set(rope) != set(ROPE_FIELDS):
            raise ValueError("model.rope_scaling_params: provide all rope fields")
    for group, fields in {
        "data": ("max_length",), "train": ("epochs", "batch_size", "accumulation_steps", "learning_rate"),
        "output": ("save_interval", "log_interval"),
    }.items():
        for field in fields:
            if config[group][field] <= 0:
                raise ValueError(f"{group}.{field}: must be positive")
    if data["max_length"] < 2:
        raise ValueError("data.max_length: must be at least 2 for shifted labels")
    for group, field in (("data", "num_workers"), ("train", "seed"), ("train", "min_lr"),
                         ("train", "weight_decay"), ("train", "grad_clip")):
        if config[group][field] < 0:
            raise ValueError(f"{group}.{field}: must be nonnegative")
    if not 0 <= train["warmup_ratio"] <= 1:
        raise ValueError("train.warmup_ratio: must be between 0 and 1")
    if train["min_lr"] > train["learning_rate"]:
        raise ValueError("train.min_lr: must not exceed train.learning_rate")
    if train["dtype"] not in ("fp32", "bf16", "fp16"):
        raise ValueError("train.dtype: expected fp32, bf16 or fp16")
    if train["seed"] >= 2**32:
        raise ValueError("train.seed: must be less than 2**32")
    name = config["output"]["save_name"]
    if name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("output.save_name: expected a filename, not a path")
    if stage == "sft":
        return config
    for field, kind in MODEL_FIELDS.items():
        if kind is int and field in model and model[field] <= 0:
            raise ValueError(f"model.{field}: must be positive")
    if not 0 <= model.get("dropout", 0.1) < 1:
        raise ValueError("model.dropout: must be in [0, 1)")
    for field in ("initializer_range", "aux_loss_coef"):
        if model.get(field, 0) < 0:
            raise ValueError(f"model.{field}: must be nonnegative")
    if model.get("rope_theta", 100000) <= 0:
        raise ValueError("model.rope_theta: must be positive")
    if data["max_length"] > model.get("max_length", 32768):
        raise ValueError("data.max_length: exceeds model.max_length")
    return config


def load_train_config(path, stage="pretrain"):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    stage_defaults = copy.deepcopy(DEFAULTS)
    if stage == "sft":
        stage_defaults.pop("model")
        stage_defaults["data"]["train_on_prompt"] = False
        stage_defaults["train"]["weight_decay"] = 0.1
    for group, defaults in stage_defaults.items():
        if group not in config:
            config[group] = {}
        if isinstance(config[group], dict):
            config[group] = {**defaults, **config[group]}
    validate_train_config(config, stage)
    for group, field in (("data", "data_path"), ("data", "tokenizer_path"), ("output", "save_dir")):
        if field in config[group]:
            config[group][field] = str((path.parent / Path(config[group][field]).expanduser()).resolve())
    return config


def apply_cli_overrides(config, overrides):
    result = copy.deepcopy(config)
    for key, value in overrides.items():
        if key not in CLI_FIELDS:
            raise ValueError(f"CLI override {key!r} is not supported")
        if value is None:
            continue
        group, field = CLI_FIELDS[key]
        if key == "save_dir":
            value = str(Path(value).expanduser().resolve())
        result[group][field] = value
    return validate_train_config(result, result["stage"])
