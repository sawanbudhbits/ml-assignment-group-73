from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    backend: str


def init_distributed(backend: str = "gloo", init_method: Optional[str] = None) -> DistInfo:
    """Initialize torch.distributed.

    - If launched with torchrun, init_method defaults to env://.
    - For local multi-process without networking issues, pass init_method like
      file:///tmp/some_store.
    - For single-process runs (plain `python`), initialization is skipped.

    On macOS, NCCL is generally unavailable; default to gloo.
    """
    has_rank_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    should_init = init_method is not None or has_rank_env

    if should_init and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method or "env://",
            timeout=timedelta(seconds=60),
        )

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank, backend=backend)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def allreduce_mean_scalar(x: float, device: Optional[torch.device] = None) -> float:
    """All-reduce mean for a python float."""
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)

    t = torch.tensor([x], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


class Timer:
    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._start is not None:
            self.elapsed += time.perf_counter() - self._start
