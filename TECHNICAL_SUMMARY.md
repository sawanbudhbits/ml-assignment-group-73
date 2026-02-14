# Technical Summary (Repo Codebook)

This document is a **single, printable Markdown “codebook”** for the repository. It includes:
- A technical overview (objective, architecture, runtime flow)
- A per-file summary
- The **full source/config content** of each key file under a fenced code block

It is designed so you can convert it to PDF (e.g., using Pandoc).

## What’s included / excluded

Included (human-authored / source-of-truth):
- `src/*.py` (implementation)
- `configs/*.json` (example run config)
- `requirements.txt`, `.gitignore`
- `README.md`, `FacingSheet.md`, `Assignment-2.md`

Excluded (generated or very large / binary):
- `runs/**` (logs, CSV/JSON summaries, plots)
- `.venv/**`, `__pycache__/**`
- `Group 73 - ML Assignment (1).pdf` (binary)

---

## 1) Objective (what this repo does)

The repo implements a minimal **synchronous distributed PPO (DD-PPO style)** training loop using **PyTorch** + **Gymnasium**.

Goal:
- Parallelize rollout collection and PPO learning across **W worker processes**.
- Keep on-policy correctness by synchronizing the policy at iteration boundaries.

Core measurable outputs:
- **Throughput**: steps/sec (`fps`)
- **Iteration time**: `t_iter` with breakdown `t_rollout`, `t_learn`, `t_sync`
- PPO diagnostics: `approx_kl`, `clip_frac`, `entropy`, `value_loss`

---

## 2) Architecture (how it works)

### 2.1 Components per worker
Each worker process contains:
- Vectorized environments (`gymnasium.vector.SyncVectorEnv`) for fast rollout
- An Actor-Critic neural network (policy logits + value head)
- PPO optimization loop (epochs × minibatches)
- Optional distributed communication via `torch.distributed` and DDP

### 2.2 One iteration (control flow)
For iteration `k`:
1. Collect rollout of length `T` on `E` environments (local batch `B_local = T × E`).
2. Compute **GAE** advantages and returns.
3. Normalize advantages (currently **local per worker**).
4. Run PPO updates for `epochs × minibatches`.
   - In distributed mode, DDP **all-reduces gradients** across workers.
5. Barrier synchronize to keep iteration boundaries aligned.
6. Rank 0 logs a JSON line with metrics.

### 2.3 Distributed execution on macOS
- Backend: `gloo` (CPU-friendly; NCCL typically unavailable on macOS)
- Local multi-process launcher: uses `file://` rendezvous to avoid TCP/IPv6 issues.

---

## 3) How to run (quick reference)

### 3.1 Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 3.2 Single-process (W=1)
```bash
python -m src.train_ddppo --env-id CartPole-v1 --total-iters 50 --log-jsonl runs/cartpole_w1.jsonl
```

### 3.3 Multi-process local (W>1)
```bash
python -m src.launch_local_ddppo --nproc 4 -- \
  --env-id CartPole-v1 --total-iters 50 --rollout-steps 64 --num-envs 2 \
  --log-jsonl runs/cartpole_w4.jsonl
```

### 3.4 Scaling benchmark (P3)
```bash
python -m src.p3_benchmark --world-sizes 1,2,4 --repeats 1 --output-dir runs/p3
```

---

## 4) File-by-file guide + full contents

### 4.1 `src/train_ddppo.py`
**Role:** Main entrypoint for training. Runs rollouts, computes GAE, performs PPO updates, and logs timing/metrics. When `WORLD_SIZE>1`, wraps the model with `DistributedDataParallel`.

```python
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.dist_utils import DistInfo, Timer, allreduce_mean_scalar, barrier, init_distributed
from src.ppo_core import ActorCritic, PPOHyperParams, compute_gae, ppo_loss


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def rollout(
    envs: gym.vector.VectorEnv,
    model: ActorCritic,
    device: torch.device,
    rollout_steps: int,
) -> Dict[str, np.ndarray]:
    obs, _ = envs.reset()
    obs = obs.astype(np.float32)

    T = rollout_steps
    E = envs.num_envs

    obs_buf = np.zeros((T, E, obs.shape[-1]), dtype=np.float32)
    act_buf = np.zeros((T, E), dtype=np.int64)
    logp_buf = np.zeros((T, E), dtype=np.float32)
    val_buf = np.zeros((T, E), dtype=np.float32)
    rew_buf = np.zeros((T, E), dtype=np.float32)
    done_buf = np.zeros((T, E), dtype=np.float32)

    for t in range(T):
        obs_buf[t] = obs

        obs_t = torch.from_numpy(obs).to(device)
        action_t, logp_t, value_t = model.act(obs_t)

        actions = action_t.cpu().numpy()
        next_obs, rewards, terminated, truncated, _ = envs.step(actions)
        dones = np.logical_or(terminated, truncated).astype(np.float32)

        act_buf[t] = actions
        logp_buf[t] = logp_t.cpu().numpy()
        val_buf[t] = value_t.cpu().numpy()
        rew_buf[t] = rewards.astype(np.float32)
        done_buf[t] = dones

        obs = next_obs.astype(np.float32)

    with torch.no_grad():
        last_obs_t = torch.from_numpy(obs).to(device)
        _, last_value_t = model.forward(last_obs_t)
        last_value = last_value_t.cpu().numpy().astype(np.float32)

    return {
        "obs": obs_buf,
        "actions": act_buf,
        "logp": logp_buf,
        "values": val_buf,
        "rewards": rew_buf,
        "dones": done_buf,
        "last_value": last_value,
    }


def flatten_time_env(x: np.ndarray) -> np.ndarray:
    # (T, E, ...) -> (T*E, ...)
    return x.reshape((-1,) + x.shape[2:]) if x.ndim >= 3 else x.reshape(-1)


def build_minibatches(
    batch_size: int,
    num_minibatches: int,
    seed: int,
) -> np.ndarray:
    if batch_size % num_minibatches != 0:
        raise ValueError(f"batch_size={batch_size} must be divisible by num_minibatches={num_minibatches}")

    rng = np.random.default_rng(seed)
    indices = np.arange(batch_size)
    rng.shuffle(indices)

    mb_size = batch_size // num_minibatches
    return indices.reshape(num_minibatches, mb_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronous distributed PPO (DD-PPO style) using torch.distributed")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--total-iters", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument(
        "--init-method",
        type=str,
        default="",
        help="Explicit init method for torch.distributed (e.g., file:///tmp/store). Empty means env:// under torchrun.",
    )
    parser.add_argument("--log-jsonl", type=str, default="")
    args = parser.parse_args()

    init_method = args.init_method.strip() or None
    dist_info: DistInfo = init_distributed(backend=args.backend, init_method=init_method)

    # Device: CPU by default for macOS. (GPU can be enabled later if available.)
    device = torch.device("cpu")

    # Seed per-rank for rollout diversity but deterministic reproducibility.
    base_seed = args.seed
    rank_seed = base_seed + 1000 * dist_info.rank
    torch.manual_seed(rank_seed)
    np.random.seed(rank_seed)

    # Vectorized envs per worker
    def env_fn(i: int):
        return make_env(args.env_id, seed=rank_seed + i)

    envs = gym.vector.SyncVectorEnv([lambda i=i: env_fn(i) for i in range(args.num_envs)])

    obs_space = envs.single_observation_space
    act_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 1:
        raise ValueError("This reference implementation assumes 1D continuous observations.")
    if not isinstance(act_space, gym.spaces.Discrete):
        raise ValueError("This reference implementation assumes discrete action spaces.")

    obs_dim = int(obs_space.shape[0])
    act_dim = int(act_space.n)

    hparams = PPOHyperParams()

    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim).to(device)

    # Wrap with DDP only in distributed runs.
    ddp_model: nn.Module
    if dist_info.world_size > 1:
        ddp_model = DDP(model)
    else:
        ddp_model = model

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=hparams.lr)

    local_batch = args.rollout_steps * args.num_envs
    if local_batch % args.minibatches != 0:
        raise ValueError("Local batch (rollout_steps*num_envs) must be divisible by minibatches.")

    # Optional JSONL logging (rank0 only)
    log_fp = None
    if dist_info.rank == 0 and args.log_jsonl:
        os.makedirs(os.path.dirname(args.log_jsonl) or ".", exist_ok=True)
        log_fp = open(args.log_jsonl, "a", encoding="utf-8")

    if dist_info.rank == 0:
        print("=== Distributed PPO (initial P2 implementation) ===")
        print(f"env={args.env_id} world_size={dist_info.world_size} local_batch={local_batch} global_batch={local_batch*dist_info.world_size}")
        print(f"hparams={asdict(hparams)}")

    # Main training loop
    for it in range(1, args.total_iters + 1):
        t_iter_start = time.perf_counter()

        timers = {"rollout": Timer(), "learn": Timer(), "sync": Timer()}

        with timers["rollout"]:
            traj = rollout(envs, model, device, rollout_steps=args.rollout_steps)

        # Compute GAE + returns (numpy)
        advantages, returns = compute_gae(
            rewards=traj["rewards"],
            dones=traj["dones"],
            values=traj["values"],
            last_value=traj["last_value"],
            gamma=hparams.gamma,
            gae_lambda=hparams.gae_lambda,
        )

        # Flatten to (B_local, ...)
        obs = flatten_time_env(traj["obs"])
        actions = flatten_time_env(traj["actions"]).astype(np.int64)
        old_logp = flatten_time_env(traj["logp"]).astype(np.float32)
        adv = flatten_time_env(advantages).astype(np.float32)
        ret = flatten_time_env(returns).astype(np.float32)

        # Normalize advantages (local normalization for initial version)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Torch tensors
        obs_t = torch.from_numpy(obs).to(device)
        actions_t = torch.from_numpy(actions).to(device)
        old_logp_t = torch.from_numpy(old_logp).to(device)
        adv_t = torch.from_numpy(adv).to(device)
        ret_t = torch.from_numpy(ret).to(device)

        # Learning
        metrics_accum: Dict[str, float] = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }
        num_updates = 0

        with timers["learn"]:
            for epoch in range(args.epochs):
                minibatches = build_minibatches(local_batch, args.minibatches, seed=rank_seed + 10_000 * it + epoch)

                for mb_idx in range(args.minibatches):
                    idx = minibatches[mb_idx]

                    logits, values = ddp_model(obs_t[idx])
                    loss, mb_metrics = ppo_loss(
                        logits=logits,
                        values=values,
                        actions=actions_t[idx],
                        old_logp=old_logp_t[idx],
                        advantages=adv_t[idx],
                        returns=ret_t[idx],
                        clip_eps=hparams.clip_eps,
                        vf_coef=hparams.vf_coef,
                        ent_coef=hparams.ent_coef,
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=hparams.max_grad_norm)
                    optimizer.step()

                    for k in metrics_accum.keys():
                        metrics_accum[k] += float(mb_metrics[k])
                    num_updates += 1

        # Sync to keep iteration boundaries aligned (esp. for logging/timers)
        with timers["sync"]:
            barrier()

        # Aggregate metrics across ranks (mean)
        metrics_mean = {k: metrics_accum[k] / max(1, num_updates) for k in metrics_accum}
        metrics_mean = {k: allreduce_mean_scalar(v) for k, v in metrics_mean.items()}

        # Throughput estimate: local steps = rollout_steps*num_envs; global = *world_size
        t_iter = time.perf_counter() - t_iter_start
        steps_global = args.rollout_steps * args.num_envs * dist_info.world_size
        fps = steps_global / max(t_iter, 1e-9)

        log_obj = {
            "iter": it,
            "world_size": dist_info.world_size,
            "steps_global": steps_global,
            "fps": fps,
            "t_iter": t_iter,
            "t_rollout": timers["rollout"].elapsed,
            "t_learn": timers["learn"].elapsed,
            "t_sync": timers["sync"].elapsed,
            **metrics_mean,
        }

        if dist_info.rank == 0:
            print(
                f"it={it:04d} fps={fps:8.1f} t_iter={t_iter:6.3f}s "
                f"kl={metrics_mean['approx_kl']:+.4f} clip={metrics_mean['clip_frac']:.3f} "
                f"loss={metrics_mean['loss']:.3f}"
            )
            if log_fp is not None:
                log_fp.write(json.dumps(log_obj) + "\n")
                log_fp.flush()

    if log_fp is not None:
        log_fp.close()

    envs.close()


if __name__ == "__main__":
    main()
```

### 4.2 `src/ppo_core.py`
**Role:** PPO algorithm core (model, GAE, PPO-Clip loss + metrics).

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


@dataclass
class PPOHyperParams:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized GAE for (T, E) arrays.

    rewards: (T, E)
    dones: (T, E) with 1.0 when episode ended at step t
    values: (T, E)
    last_value: (E,) value estimate for observation after last step

    Returns:
      advantages: (T, E)
      returns: (T, E)
    """
    T, E = rewards.shape
    advantages = np.zeros((T, E), dtype=np.float32)

    gae = np.zeros((E,), dtype=np.float32)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def ppo_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> Tuple[torch.Tensor, dict]:
    dist = Categorical(logits=logits)
    logp = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(logp - old_logp)
    adv = advantages

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = 0.5 * (returns - values).pow(2).mean()

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    with torch.no_grad():
        approx_kl = (old_logp - logp).mean()
        clip_frac = (torch.abs(ratio - 1.0) > clip_eps).float().mean()

    metrics = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "approx_kl": approx_kl.item(),
        "clip_frac": clip_frac.item(),
    }
    return loss, metrics
```

### 4.3 `src/dist_utils.py`
**Role:** Distributed initialization, barrier, mean all-reduce for scalars, and a basic wall-clock timer.

```python
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
```

### 4.4 `src/launch_local_ddppo.py`
**Role:** Local launcher that spawns N worker processes on one machine, using a `file://` store rendezvous (macOS-friendly).

```python
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
from typing import List


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local multi-process launcher for src.train_ddppo using file:// init_method (avoids macOS socket/IPv6 issues)."
    )
    parser.add_argument("--nproc", type=int, default=2, help="Number of worker processes to launch")
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use (defaults to current interpreter)",
    )

    # Everything after '--' is forwarded to train_ddppo.
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    nproc = int(args.nproc)
    if nproc < 1:
        raise ValueError("--nproc must be >= 1")

    # Create a shared file store for rendezvous (avoids TCPStore + hostname/IPv6 issues).
    # IMPORTANT: init_method must be of the form file:///abs/path.
    fd, store_path = tempfile.mkstemp(prefix="ddppo_store_", suffix=".tmp")
    os.close(fd)
    init_method_base = f"file://{store_path}"
    debug = os.environ.get("DDPPO_LAUNCHER_DEBUG", "0") == "1"
    if debug:
        print(f"[launcher] init_method_base={init_method_base}")

    # Pick a free local TCP port (not used when file:// init_method is passed, but keep
    # env vars set for compatibility if a user overrides --init-method).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        master_port = s.getsockname()[1]

    procs: List[subprocess.Popen] = []
    try:
        for rank in range(nproc):
            # Start from a sanitized environment to avoid inheriting torchrun/elastic variables
            # that can override rendezvous behavior.
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("TORCHELASTIC_")
                and not k.startswith("PET_")
                and k
                not in {
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                    "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                }
            }
            env["RANK"] = str(rank)
            env["WORLD_SIZE"] = str(nproc)
            env["LOCAL_RANK"] = str(rank)

            # Force local rendezvous on loopback to avoid hostname/IPv6 resolution issues.
            env["MASTER_ADDR"] = "127.0.0.1"
            env["MASTER_PORT"] = str(master_port)
            env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
            env.setdefault("PYTHONUNBUFFERED", "1")

            cmd = [
                args.python,
                "-m",
                "src.train_ddppo",
                "--backend",
                "gloo",
            ]

            # If the user didn't specify an init method, force the file store.
            train_args = args.train_args
            if train_args and train_args[0] == "--":
                train_args = train_args[1:]

            if "--init-method" not in train_args:
                init_method_rank = f"{init_method_base}?rank={rank}&world_size={nproc}"
                cmd += ["--init-method", init_method_rank]

            cmd += train_args

            if debug:
                print(f"[launcher] rank={rank} cmd={' '.join(cmd)}")

            procs.append(subprocess.Popen(cmd, env=env))

        exit_codes = [p.wait() for p in procs]
        if any(code != 0 for code in exit_codes):
            raise SystemExit(max(exit_codes))
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=2)
            except Exception:
                if p.poll() is None:
                    p.kill()

        try:
            os.remove(store_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
```

### 4.5 `src/p3_benchmark.py`
**Role:** P3 evaluation harness. Runs multiple world sizes, summarizes logs, writes CSV/JSON summaries, optionally plots if `matplotlib` exists.

```python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunConfig:
    env_id: str
    total_iters: int
    warmup_iters: int
    rollout_steps: int
    num_envs: int
    epochs: int
    minibatches: int
    seed: int
    backend: str


@dataclass(frozen=True)
class RunResult:
    world_size: int
    repeat: int
    env_id: str
    rollout_steps: int
    num_envs: int
    global_batch: int
    mean_fps: float
    mean_t_iter: float
    mean_t_rollout: float
    mean_t_learn: float
    mean_t_sync: float
    sync_frac: float


def parse_world_sizes(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    ws: List[int] = []
    for p in parts:
        w = int(p)
        if w < 1:
            raise ValueError("world sizes must be >= 1")
        ws.append(w)
    if not ws:
        raise ValueError("no world sizes provided")
    return ws


def load_jsonl(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_log(rows: Sequence[Dict[str, float]], warmup_iters: int) -> Tuple[float, float, float, float, float]:
    if not rows:
        raise ValueError("empty log")
    usable = [r for r in rows if int(r.get("iter", 0)) > warmup_iters]
    if not usable:
        usable = list(rows)

    def avg(key: str) -> float:
        vals = [float(r[key]) for r in usable if key in r]
        if not vals:
            raise ValueError(f"missing key '{key}' in log")
        return float(mean(vals))

    return (
        avg("fps"),
        avg("t_iter"),
        avg("t_rollout"),
        avg("t_learn"),
        avg("t_sync"),
    )


def choose_num_envs_for_world_size(base_rollout_steps: int, base_num_envs: int, world_size: int) -> int:
    """Keep global batch roughly constant vs the W=1 baseline.

    Baseline global batch (W=1): B0 = base_rollout_steps * base_num_envs.
    For W>1 we target per-worker batch ceil(B0/W), keeping rollout_steps fixed,
    and adjust num_envs to the nearest integer >= 1.
    """
    target_global = base_rollout_steps * base_num_envs
    per_worker_target = int(math.ceil(target_global / max(1, world_size)))
    return max(1, int(math.ceil(per_worker_target / max(1, base_rollout_steps))))


def run_command(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(cmd, env=env, check=True)


def run_one(
    python_exe: str,
    world_size: int,
    repeat: int,
    cfg: RunConfig,
    output_dir: Path,
) -> RunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"w{world_size}_r{repeat}.jsonl"

    cmd_train = [
        python_exe,
        "-m",
        "src.train_ddppo",
        "--env-id",
        cfg.env_id,
        "--total-iters",
        str(cfg.total_iters),
        "--rollout-steps",
        str(cfg.rollout_steps),
        "--num-envs",
        str(cfg.num_envs),
        "--epochs",
        str(cfg.epochs),
        "--minibatches",
        str(cfg.minibatches),
        "--seed",
        str(cfg.seed + 10_000 * repeat),
        "--backend",
        cfg.backend,
        "--log-jsonl",
        str(log_path),
    ]

    if world_size == 1:
        run_command(cmd_train)
    else:
        cmd = [
            python_exe,
            "-m",
            "src.launch_local_ddppo",
            "--nproc",
            str(world_size),
            "--",
            *cmd_train[3:],
        ]
        run_command(cmd)

    rows = load_jsonl(log_path)
    mean_fps, mean_t_iter, mean_t_rollout, mean_t_learn, mean_t_sync = summarize_log(rows, cfg.warmup_iters)
    global_batch = world_size * cfg.rollout_steps * cfg.num_envs
    sync_frac = float(mean_t_sync / mean_t_iter) if mean_t_iter > 0 else 0.0

    return RunResult(
        world_size=world_size,
        repeat=repeat,
        env_id=cfg.env_id,
        rollout_steps=cfg.rollout_steps,
        num_envs=cfg.num_envs,
        global_batch=global_batch,
        mean_fps=float(mean_fps),
        mean_t_iter=float(mean_t_iter),
        mean_t_rollout=float(mean_t_rollout),
        mean_t_learn=float(mean_t_learn),
        mean_t_sync=float(mean_t_sync),
        sync_frac=float(sync_frac),
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def group_by_world_size(results: Sequence[RunResult]) -> Dict[int, List[RunResult]]:
    out: Dict[int, List[RunResult]] = {}
    for r in results:
        out.setdefault(r.world_size, []).append(r)
    return out


def aggregate(results: Sequence[RunResult]) -> List[Dict[str, object]]:
    grouped = group_by_world_size(results)
    if 1 not in grouped:
        raise ValueError("benchmark must include world_size=1 for speedup/efficiency")

    base_fps = mean([r.mean_fps for r in grouped[1]])
    rows: List[Dict[str, object]] = []
    for w in sorted(grouped.keys()):
        rs = grouped[w]
        fps_vals = [r.mean_fps for r in rs]
        t_iter_vals = [r.mean_t_iter for r in rs]
        sync_frac_vals = [r.sync_frac for r in rs]

    	fps_mean = float(mean(fps_vals))
    	fps_std = float(pstdev(fps_vals)) if len(fps_vals) > 1 else 0.0
    	t_iter_mean = float(mean(t_iter_vals))
    	t_iter_std = float(pstdev(t_iter_vals)) if len(t_iter_vals) > 1 else 0.0
    	sync_frac_mean = float(mean(sync_frac_vals))

    	speedup = float(fps_mean / base_fps) if base_fps > 0 else 0.0
    	efficiency = float(speedup / w) if w > 0 else 0.0

    	rows.append(
    		{
    			"world_size": w,
    			"repeats": len(rs),
    			"env_id": rs[0].env_id,
    			"rollout_steps": rs[0].rollout_steps,
    			"num_envs_per_worker": rs[0].num_envs,
    			"global_batch": rs[0].global_batch,
    			"fps_mean": fps_mean,
    			"fps_std": fps_std,
    			"t_iter_mean": t_iter_mean,
    			"t_iter_std": t_iter_std,
    			"speedup_vs_w1": speedup,
    			"efficiency": efficiency,
    			"sync_frac_mean": sync_frac_mean,
    		}
    	)
    return rows


def maybe_plot(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    ws = [int(r["world_size"]) for r in summary_rows]
    fps = [float(r["fps_mean"]) for r in summary_rows]
    speedup = [float(r["speedup_vs_w1"]) for r in summary_rows]
    eff = [float(r["efficiency"]) for r in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    axes[0].plot(ws, fps, marker="o")
    axes[0].set_title("Throughput (FPS)")
    axes[0].set_xlabel("W")
    axes[0].set_ylabel("steps/sec")

    axes[1].plot(ws, speedup, marker="o")
    axes[1].plot(ws, ws, linestyle="--", linewidth=1, label="ideal")
    axes[1].set_title("Speedup vs W=1")
    axes[1].set_xlabel("W")
    axes[1].set_ylabel("x")
    axes[1].legend(frameon=False)

    axes[2].plot(ws, eff, marker="o")
    axes[2].set_title("Efficiency")
    axes[2].set_xlabel("W")
    axes[2].set_ylabel("S(W)/W")
    axes[2].set_ylim(0.0, 1.05)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 benchmark: run DD-PPO for multiple world sizes and summarize scaling.")
    parser.add_argument("--world-sizes", type=str, default="1,2,4")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="runs/p3")

    # Training parameters
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--total-iters", type=int, default=30)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=8, help="Baseline num envs for W=1. Larger W auto-adjusts to keep global batch roughly constant.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", type=str, default="gloo")
    args = parser.parse_args()

    ws = parse_world_sizes(args.world_sizes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable

    # Run benchmark
    per_run_results: List[RunResult] = []
    for w in ws:
        num_envs_w = choose_num_envs_for_world_size(args.rollout_steps, args.num_envs, w)
        cfg = RunConfig(
            env_id=args.env_id,
            total_iters=args.total_iters,
            warmup_iters=args.warmup_iters,
            rollout_steps=args.rollout_steps,
            num_envs=num_envs_w,
            epochs=args.epochs,
            minibatches=args.minibatches,
            seed=args.seed,
            backend=args.backend,
        )

        for r in range(args.repeats):
            print(f"[p3] running W={w} repeat={r} num_envs={num_envs_w}")
            res = run_one(python_exe=python_exe, world_size=w, repeat=r, cfg=cfg, output_dir=output_dir / "logs")
            per_run_results.append(res)

    # Write per-run CSV
    per_run_csv_rows: List[Dict[str, object]] = []
    for r in per_run_results:
        per_run_csv_rows.append(asdict(r))
    write_csv(output_dir / "per_run.csv", per_run_csv_rows)
    (output_dir / "per_run.json").write_text(json.dumps(per_run_csv_rows, indent=2), encoding="utf-8")

    # Aggregate
    summary_rows = aggregate(per_run_results)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    plotted = maybe_plot(summary_rows, output_dir / "plots.png")
    print(f"[p3] wrote {output_dir / 'summary.csv'}")
    print(f"[p3] plot={'yes' if plotted else 'no (matplotlib not installed)'}")


if __name__ == "__main__":
    main()
```

### 4.6 `configs/cartpole_local.json`
**Role:** Example configuration (not auto-loaded by the code; equivalent CLI flags are used).

```json
{
  "env_id": "CartPole-v1",
  "total_iters": 200,
  "rollout_steps": 256,
  "num_envs": 8,
  "epochs": 4,
  "minibatches": 4,
  "hidden_dim": 64,
  "seed": 42,
  "backend": "gloo",
  "log_jsonl": "runs/cartpole_ddppo.jsonl"
}
```

### 4.7 `requirements.txt`
**Role:** Python dependencies.

```text
numpy

gymnasium

torch

matplotlib
```

### 4.8 `.gitignore`
**Role:** Ignore virtualenvs, caches, and OS/editor junk.

```text
# Python
__pycache__/
*.py[cod]
*.pyd
*.so
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environments
.venv/
venv/
ENV/

# OS / editor
.DS_Store
.vscode/

# Jupyter
.ipynb_checkpoints/
```

### 4.9 `README.md`
**Role:** Main usage and architecture doc.

````markdown
# ML Assignment 2 (Group 73)

Parallel and Distributed Training of **Proximal Policy Optimization (PPO)** using **synchronous, data-parallel** updates with `torch.distributed` (DD-PPO style).

This repository contains a minimal reference implementation that:
- Collects on-policy rollouts in parallel (multiple processes / workers)
- Runs PPO optimization locally per worker
- Synchronizes gradients across workers (via `DistributedDataParallel`)
- Logs timing/throughput so you can evaluate scaling behavior

---

## Objective

Implement and evaluate **synchronous distributed PPO** to reduce wall-clock training time while maintaining on-policy correctness.

Primary metrics:
- **Throughput** (steps/sec, reported as `fps`)
- **Iteration latency** (`t_iter`) and breakdown (`t_rollout`, `t_learn`, `t_sync`)
- PPO diagnostics: `approx_kl`, `clip_frac`, `entropy`, `value_loss`

---

## Repository structure

- [src/](src/)
  - [src/train_ddppo.py](src/train_ddppo.py): main training entrypoint (single-process or distributed)
  - [src/launch_local_ddppo.py](src/launch_local_ddppo.py): local multi-process launcher for macOS (uses `file://` rendezvous)
  - [src/ppo_core.py](src/ppo_core.py): PPO core components (Actor-Critic, GAE, PPO loss)
  - [src/dist_utils.py](src/dist_utils.py): `torch.distributed` init + simple collectives + timers
  - [src/p3_benchmark.py](src/p3_benchmark.py): runs multiple world sizes and writes scaling summaries to `runs/p3/`

- [configs/](configs/)
  - [configs/cartpole_local.json](configs/cartpole_local.json): example hyperparameters/flags for a local CartPole run

- [runs/](runs/)
  - Example logs and P3 benchmark outputs (CSV/JSON summaries, per-run JSONL logs)

- [Assignment-2.md](Assignment-2.md): assignment write-up / design notes

---

## Architecture (high level)

### Components per worker
Each worker process runs the same code:
- **Vectorized environments** (`gymnasium.vector.SyncVectorEnv`) to collect rollouts
- **Actor-Critic model** (shared MLP trunk with policy/value heads)
- **PPO update loop** over the collected on-policy batch
- **Distributed gradient synchronization** via `torch.nn.parallel.DistributedDataParallel` (DDP)

### One training iteration
For each iteration:
1. **Rollout**: collect `T` steps in each of `E` envs → local batch size `B_local = T * E`
2. **GAE/returns**: compute advantages and returns using `compute_gae()`
3. **Optimization**: run `epochs × minibatches` PPO updates
   - If `world_size > 1`, DDP all-reduces gradients during backprop so all workers apply the same optimizer step
4. **Barrier**: `dist.barrier()` to align iteration boundaries and logging
5. **Logging (rank 0)**: write one JSON object per iteration to a JSONL file

### Notes for macOS
- Default backend is `gloo` and training runs on CPU.
- `src/launch_local_ddppo.py` uses a `file://` store rendezvous and forces loopback (`lo0`) to avoid common macOS hostname/IPv6 issues.

---

## Setup

### 1) Create a virtual environment
From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## How to run

### A) Single-process training (W=1)

```bash
python -m src.train_ddppo \
  --env-id CartPole-v1 \
  --total-iters 200 \
  --rollout-steps 256 \
  --num-envs 8 \
  --epochs 4 \
  --minibatches 4 \
  --seed 42 \
  --backend gloo \
  --log-jsonl runs/cartpole_w1.jsonl
```

What you’ll see on stdout (rank 0):
- `fps`, `t_iter`, and PPO diagnostics per iteration

What gets written:
- `runs/cartpole_w1.jsonl` (one JSON record per iteration)

### B) Multi-process local distributed training (W>1)

Use the provided launcher (recommended on macOS):

```bash
python -m src.launch_local_ddppo --nproc 4 -- \
  --env-id CartPole-v1 \
  --total-iters 200 \
  --rollout-steps 256 \
  --num-envs 8 \
  --epochs 4 \
  --minibatches 4 \
  --seed 42 \
  --backend gloo \
  --log-jsonl runs/cartpole_w4.jsonl
```

Notes:
- Only **rank 0** writes the JSONL log.
- This implementation currently normalizes advantages **locally per worker** (simple baseline).

### C) Run the P3 scaling benchmark (generates CSV/JSON summaries)

This runs a short benchmark for multiple world sizes and writes outputs under `runs/p3/`:

```bash
python -m src.p3_benchmark \
  --world-sizes 1,2,4 \
  --repeats 1 \
  --output-dir runs/p3 \
  --env-id CartPole-v1 \
  --total-iters 30 \
  --warmup-iters 5 \
  --rollout-steps 64 \
  --num-envs 8 \
  --epochs 2 \
  --minibatches 4
```

Outputs:
- `runs/p3/logs/*.jsonl`: raw per-iteration logs per run
- `runs/p3/per_run.csv` + `runs/p3/per_run.json`: metrics per run
- `runs/p3/summary.csv` + `runs/p3/summary.json`: aggregated scaling metrics (speedup/efficiency)
- `runs/p3/plots.png` (only if `matplotlib` is available)

---

## Using the provided JSON config

[configs/cartpole_local.json](configs/cartpole_local.json) is an example set of flags (env, iters, rollout steps, etc.).

The training script does not currently parse JSON configs directly; run it by passing equivalent CLI flags (see examples above).

---

## Key log fields (JSONL)

Each line in a JSONL log is one iteration, with fields such as:
- `world_size`, `steps_global`, `fps`
- `t_iter`, `t_rollout`, `t_learn`, `t_sync`
- PPO metrics: `loss`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_frac`

---

## Troubleshooting

- **Gym / Torch import errors**: confirm you activated the venv and installed `requirements.txt`.
- **Multi-process hangs**: prefer [src/launch_local_ddppo.py](src/launch_local_ddppo.py) on macOS (it avoids common `torchrun` networking issues).
- **No plot generated in P3**: the benchmark prints `plot=no (matplotlib not installed)` if `matplotlib` is missing.

````

### 4.10 `FacingSheet.md`
**Role:** Standalone facing sheet (template-style).

```markdown
# Facing Sheet

## Assignment
- **Title:** Assignment 2 — Parallelization/Distribution of an ML Algorithm
- **Topic:** Parallel and Distributed Training of Proximal Policy Optimization (PPO)
- **Group:** 73
- **Date:** 2026-02-14

## Contributors
> Update the table with full names + student IDs if required by your course.

| # | Name (from git) | Email (from git) | Student ID / Roll No. |
|---|-----------------|------------------|------------------------|
| 1 | **sawan** | sawan.budhbhatti@se.com | _TBD_ |

## GitHub Repository
- https://github.com/sawanbudhbits/ml-assignment-group-73

## Notes
- The contributors list above is derived from `git shortlog` (commit authors). If you have teammates who did not commit from their machine/account, add them to the table manually.
```

### 4.11 `Assignment-2.md`
**Role:** Main assignment report: P0 problem formulation, P1 design, P2 implementation, P3 evaluation.

```markdown
(See `Assignment-2.md` in the repo for the full report content.)
```

### 4.12 `Part1.md`
**Role:** Present but currently empty.

```text

```

### 4.13 `test.md`
**Role:** Additional copy/notes (appears to duplicate early sections of the report).

```markdown
(Truncated in this codebook. See `test.md` in the repo.)
```

---

## 5) Converting this Markdown to PDF

### 5.1 Using Pandoc (macOS)
```bash
brew install pandoc
pandoc TECHNICAL_SUMMARY.md -o TECHNICAL_SUMMARY.pdf
```

If you want syntax highlighting in the PDF, tell me which PDF engine you prefer (default vs LaTeX), and I can provide a tuned command.
