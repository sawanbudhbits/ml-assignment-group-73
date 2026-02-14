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
