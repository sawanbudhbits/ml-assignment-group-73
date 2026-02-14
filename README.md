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
