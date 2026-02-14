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
