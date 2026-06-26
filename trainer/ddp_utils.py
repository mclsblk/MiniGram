import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    device_type: str
    backend: Optional[str] = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}.") from exc


def init_distributed(args: Any = None) -> DistributedState:
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    enabled = world_size > 1
    requested_device = getattr(args, "device", None) if args is not None else None

    use_explicit_cpu = requested_device is not None and torch.device(requested_device).type == "cpu"
    if enabled and torch.cuda.is_available() and not use_explicit_cpu:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif requested_device is not None:
        device = torch.device(requested_device)
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device_type = device.type
    backend = os.environ.get("DDP_BACKEND") or ("nccl" if device_type == "cuda" else "gloo")
    if enabled:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build.")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        backend = None

    return DistributedState(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        device_type=device_type,
        backend=backend,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(state: Optional[DistributedState] = None) -> bool:
    if state is not None:
        return state.is_main
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier(state: Optional[DistributedState] = None) -> None:
    if state is not None and not state.enabled:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def rank0_log(state: DistributedState, msg: str, log_fn: Callable[[str], None]) -> None:
    if state.is_main:
        log_fn(msg)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while True:
        if hasattr(model, "module"):
            model = model.module
            continue
        original = getattr(model, "_orig_mod", None)
        if original is not None:
            model = original
            continue
        return model


def get_model_dtype(model: torch.nn.Module, default: torch.dtype = torch.float32) -> torch.dtype:
    raw_model = unwrap_model(model)
    for parameter in raw_model.parameters():
        return parameter.dtype
    for buffer in raw_model.buffers():
        return buffer.dtype
    return default


def wrap_ddp(
    model: torch.nn.Module,
    state: DistributedState,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
) -> torch.nn.Module:
    if not state.enabled:
        return model

    ddp_kwargs = {
        "find_unused_parameters": find_unused_parameters,
        "broadcast_buffers": broadcast_buffers,
    }
    if state.device_type == "cuda":
        ddp_kwargs["device_ids"] = [state.local_rank]
        ddp_kwargs["output_device"] = state.local_rank
    return DistributedDataParallel(model, **ddp_kwargs)


def maybe_no_sync(model: torch.nn.Module, state: DistributedState, should_sync: bool):
    if state.enabled and not should_sync and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


def build_distributed_sampler(
    dataset,
    state: DistributedState,
    shuffle: bool,
    drop_last: bool,
    seed: int = 0,
):
    if not state.enabled:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=state.world_size,
        rank=state.rank,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )


def set_sampler_epoch(sampler, epoch: int) -> None:
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def reduce_mean(value, state: DistributedState):
    if not state.enabled:
        return value

    if torch.is_tensor(value):
        tensor = value.detach().clone()
        if not tensor.is_floating_point():
            tensor = tensor.float()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor / state.world_size

    tensor = torch.tensor(float(value), dtype=torch.float32, device=state.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / state.world_size).item())


def _metric_value(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def reduce_metrics(metrics: Dict[str, Any], state: DistributedState) -> Dict[str, float]:
    if not metrics:
        return {}
    if not state.enabled:
        return {key: _metric_value(value) for key, value in metrics.items()}

    keys = list(metrics.keys())
    values = torch.tensor(
        [_metric_value(metrics[key]) for key in keys],
        dtype=torch.float32,
        device=state.device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / state.world_size
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def seed_worker(worker_id: int, base_seed: int = 0, rank: int = 0) -> None:
    worker_seed = int(base_seed) + int(rank) * 100_000 + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def build_worker_seed_fn(base_seed: int, rank: int = 0):
    return partial(seed_worker, base_seed=base_seed, rank=rank)
