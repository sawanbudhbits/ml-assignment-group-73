
# Assignment 2 — Parallelization/Distribution of an ML Algorithm

## Topic
**Parallel and Distributed Training of Proximal Policy Optimization (PPO)**

One-line summary: P0 defines the distributed PPO problem, P1 proposes the system design, P2 implements it in code, and P3 benchmarks scaling results.

## GitHub Repository
- https://github.com/sawanbudhbits/ml-assignment-group-73

# Facing Sheet

## Contributors

| Name                          | Roll Number  | Contribution                          |
|-------------------------------|--------------|---------------------------------------|
| **BUDHBHATTI SAWAN**         | 2024aa05243 | 100% (Led distributed PPO implementation) |
| CHAKKILAM VENKATA PARDHASARADHI | 2024ac05021 | 100% (Wrote problem statement and design) |
| TONY ABRAHAM MAMMEN         | 2024ac05596 | 100% (Ran local benchmark and collected logs) |
| CHAUDHARI PARESH DILIP      | 2023aa05696 | 100% (Analyzed scaling summarizing results) |
| ASHWINI PRAKASH             | 2022ac05536 | 100% (Group discussions) |


## P0 — Problem Formulation

### P0.1 Motivation
Proximal Policy Optimization (PPO) is widely used because it is comparatively stable for policy-gradient reinforcement learning (RL) and is easy to implement with first-order optimization. However, PPO training is often bottlenecked by:

- **Simulation throughput** (environment stepping is expensive)
- **Wall-clock training time** (multiple PPO epochs per iteration)
- **Limited single-machine scaling** (GPU under-utilization if rollouts are slow)

The goal is to parallelize and/or distribute PPO so we can reduce **time-to-target-return** while preserving PPO’s on-policy correctness and stability.

### P0.2 What is being parallelized/distributed?
We focus on **synchronous data-parallel PPO** across $W$ workers.

At a high level, each training iteration consists of:

1. **Rollout collection (acting):** workers collect trajectories using the current policy parameters
2. **Advantage/return computation:** GAE and returns computed from trajectories
3. **Learning (PPO updates):** multiple epochs of minibatch SGD on the collected batch
4. **Synchronization:** workers synchronize parameters/gradients and proceed to the next iteration

Parallelization targets:

- **Rollout collection parallelism**: $W$ workers run environments concurrently to increase frames-per-second (FPS).
- **Learner parallelism (data-parallel SGD)**: gradient computation is replicated; gradients are aggregated and applied synchronously.

We do **not** assume asynchronous off-policy methods (e.g., IMPALA/APPO) for P0; the problem is explicitly on-policy PPO scaling.

### P0.3 System model and assumptions
We consider $W$ identical workers. Each worker has:

- A learner replica (GPU/accelerator optional) that performs PPO updates
- A rollout component that runs **$E$ vectorized environments** per worker

Per iteration $k$:

- All workers start with identical parameters $\theta_k$.
- Each worker collects $T$ steps from each of its $E$ environments (total $E\cdot T$ transitions per worker).
- The **global batch size** per iteration is:

$$
B = W \cdot E \cdot T
$$

We assume:

- Efficient collective communication is available (e.g., NCCL/MPI/Gloo all-reduce).
- Workers can be on one node (multi-GPU) or multiple nodes (cluster).
- Synchronization is **synchronous per iteration** (barriers exist).

### P0.4 Formal objective
We want a distributed PPO training system that minimizes wall-clock time to reach a target performance level while keeping training behavior comparable to a single-worker PPO baseline.

Define:

- $R(t)$ = evaluation return at wall-clock time $t$
- $R_\text{target}$ = predefined target return
- $T_W$ = wall-clock time required with $W$ workers to reach $R_\text{target}$ (median across seeds)

Primary optimization objective:

$$
\min \; T_W \quad \text{s.t.} \quad R_W(\text{end}) \approx R_1(\text{end}) \;\; \text{and on-policy correctness is maintained.}
$$

### P0.5 Expectations and measurable success criteria
We state expectations in terms of throughput, speedup, communication cost, and response time.

#### A) Throughput
Let $\text{FPS}(W)$ be environment frames per second (or steps per second).

- Expectation: $\text{FPS}(W)$ should increase with $W$; near-linear scaling is possible for small/moderate $W$ when simulation dominates.

#### B) Speedup and scaling efficiency
Let $T_1$ be time-to-target with one worker and $T_W$ with $W$ workers.

$$
S(W) = \frac{T_1}{T_W}, \qquad E(W) = \frac{S(W)}{W}
$$

- Expectation: $S(W)$ should be increasing in $W$.
- Expectation: $E(W)$ should remain “high” (close to 1) until communication and synchronization dominate.

#### C) Communication cost
We model iteration time as:

$$
t_\text{iter}(W) = t_\text{rollout}(W) + t_\text{learn}(W) + t_\text{comm}(W) + t_\text{sync}(W)
$$

Where:

- $t_\text{comm}(W)$ includes gradient all-reduce / parameter synchronization
- $t_\text{sync}(W)$ includes barrier/straggler waiting

Report:

- **Communication fraction**: $\rho_\text{comm}(W)=\frac{t_\text{comm}(W)}{t_\text{iter}(W)}$
- **Synchronization fraction**: $\rho_\text{sync}(W)=\frac{t_\text{sync}(W)}{t_\text{iter}(W)}$

Expectation: $\rho_\text{comm}$ and $\rho_\text{sync}$ increase with $W$; they explain sub-linear scaling.

#### D) “Response time” (iteration latency)
For interactive debugging and for stable on-policy learning, we care about iteration latency:

- **Iteration response time** = $t_\text{iter}(W)$

Expectation: even if FPS increases, $t_\text{iter}(W)$ should not blow up excessively with $W$ (otherwise on-policy cadence becomes slow and variance may increase).

### P0.6 On-policy correctness and constraints
PPO is on-policy: trajectories used for updates must be generated by (or very close to) the policy being updated.

Key constraints:

1. **Policy staleness constraint**
	- Data collected with parameters $\theta_k$ should not be used to update a significantly different $\theta$.
	- Synchronous iteration helps enforce this.

2. **Batch size and optimization dynamics**
	- Increasing $B$ changes gradient variance and the PPO clipping behavior.
	- Scaling may require retuning learning rate, number of epochs, minibatch size, entropy/value coefficients.

3. **Stragglers and heterogeneity**
	- Slow environments or uneven hardware can cause barrier waits and reduce efficiency.

4. **Network and bandwidth limits**
	- Beyond a certain $W$, all-reduce costs can dominate.

### P0.7 Evaluation plan (what we will measure)
We will evaluate the distributed PPO system across $W \in \{1,2,4,8,16,32\}$ (or the maximum available).

For each $W$ we will measure:

- **Learning quality:** mean/median episodic return vs environment steps and vs wall-clock time (with multiple seeds)
- **Time-to-target:** $T_W$ for a fixed $R_\text{target}$
- **Throughput:** FPS / steps-per-second
- **Speedup and efficiency:** $S(W)$ and $E(W)$
- **System breakdown:** rollout time, learn time, communication time, sync/idle time
- **PPO stability diagnostics:** approx-KL, clip fraction, entropy, value loss

Success criteria for P0 (to be validated later in P2/P3):

- $S(W)$ increases with $W$ and provides meaningful wall-clock gains
- Final performance is comparable to a single-worker baseline (within an acceptable tolerance across seeds)
- Communication/sync overhead is quantified and explains deviations from ideal scaling

---

## P1 — Design (Initial)

### P1.1 Design goal
Solve P0 by designing a **synchronous, decentralized, data-parallel PPO** training system that:

- Increases rollout throughput via parallel environment execution.
- Preserves **on-policy correctness** by ensuring rollouts are generated from the same parameters being updated.
- Scales learning with **gradient all-reduce** to keep all workers’ parameters identical at iteration boundaries.
- Measures and reports system/learning metrics needed to validate $S(W)$, $E(W)$, and communication overhead.

This is an **initial** design intended to be revised after instructor feedback.

### P1.2 High-level architecture
We adopt a DD-PPO–style architecture (no parameter server).

**Worker (replicated $W$ times):**

- **EnvRunner**: runs $E$ vectorized environments; produces on-policy trajectories.
- **Learner**: holds a replica of the policy/value networks; performs PPO update steps.
- **Communicator**: performs collective operations for gradient aggregation.

**Synchronization model:**

- Synchronous iteration boundary: all workers start iteration $k$ with identical $\theta_k$.
- Learning is **data-parallel**: each worker computes gradients on a shard of the minibatch stream; gradients are aggregated via **all-reduce**; the optimizer step is applied identically on all workers.

### P1.3 Control flow (one training iteration)
Let $k$ denote iteration index.

1. **Parameter agreement**
	- Ensure all workers have identical parameters $\theta_k$ (true by construction after the previous sync).

2. **Rollout collection (on-policy)**
	- Each worker runs $E$ environments for $T$ steps using $\pi_{\theta_k}$.
	- Each worker collects trajectories: $(s_t, a_t, r_t, s_{t+1}, \log \pi_{\theta_k}(a_t|s_t), V_{\theta_k}(s_t), \text{done}_t)$.

3. **Advantage and return computation**
	- Compute GAE($\lambda$) advantages and discounted returns locally.
	- Normalize advantages (global or local; see P1.6).

4. **Batch formation**
	- Conceptually, the global batch size is $B=W\cdot E\cdot T$.
	- Implementation: avoid physically concatenating all data on one node; each worker keeps its local batch and participates in synchronized SGD.

5. **PPO optimization (K epochs, M minibatches)**
	- Shuffle the batch indices.
	- For each minibatch step:
		- Compute PPO losses (policy loss with clipping, value loss, entropy bonus).
		- Compute gradients on the local minibatch.
		- **All-reduce gradients** across workers (sum) and divide by $W$ to obtain the averaged gradient.
		- Apply optimizer step identically on all workers.

6. **Diagnostics + evaluation**
	- Log per-iteration: FPS, time breakdown, approx-KL, clip fraction, entropy, value loss.
	- Periodically run evaluation episodes with deterministic policy (or fixed stochastic seed) to estimate return.

### P1.4 PPO core (algorithmic details)
We implement PPO-Clip with standard components.

Given old policy $\pi_{\theta_\text{old}}$ and new $\pi_\theta$:

- Probability ratio: $r_t(\theta)=\exp(\log \pi_\theta(a_t|s_t)-\log \pi_{\theta_\text{old}}(a_t|s_t))$
- Clipped objective:

$$
L^{\text{CLIP}}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_t\right)\right]
$$

Total loss (to minimize):

$$
\mathcal{L}(\theta) = -L^{\text{CLIP}}(\theta) + c_v\,\mathbb{E}_t\big[(V_\theta(s_t)-\hat{R}_t)^2\big] - c_e\,\mathbb{E}_t\big[\mathcal{H}(\pi_\theta(\cdot|s_t))\big]
$$

We monitor:

- **approx-KL** between old and new policies (stability proxy)
- **clip fraction** (fraction of samples where clipping is active)

### P1.5 Communication and distributed backend
We use **gradient all-reduce** once per minibatch step.

Assumptions/choices:

- GPU: NCCL all-reduce (preferred)
- CPU fallback: Gloo or MPI

Communication optimizations (optional in initial version, but in-scope):

- Bucket gradients by tensor size to reduce latency overhead.
- Overlap communication with backprop when supported.

We avoid a parameter server because it introduces a central bottleneck and can create staleness.

### P1.6 Data and batch sizing strategy
Key parameters:

- $W$: number of workers
- $E$: vectorized environments per worker
- $T$: rollout horizon per env
- $B=W\cdot E\cdot T$: global batch per iteration
- $K$: PPO epochs per iteration
- $M$: number of minibatches (or minibatch size)

Initial scaling strategy (to preserve learning dynamics):

1. **Keep global batch $B$ approximately constant** while increasing $W$ by decreasing $E$ and/or $T$ per worker (for scaling experiments focused on system speedup).
2. After verifying stability, explore **increasing $B$ with $W$** to maximize throughput and examine the effect on sample efficiency.

Advantage normalization choice:

- Initial: normalize advantages **per worker** (simpler).
- If needed for stability at scale: compute global mean/std via all-reduce of sufficient statistics.

### P1.7 Default hyperparameters (initial)
These are starting points; they may be tuned based on stability diagnostics.

- Discount: $\gamma=0.99$
- GAE: $\lambda=0.95$
- PPO clip: $\varepsilon=0.2$
- Value coefficient: $c_v=0.5$
- Entropy coefficient: $c_e \in [0.01, 0.02]$
- Optimizer: Adam
- PPO epochs: $K=4$ (typical)
- Minibatches: $M=4$ to $8$

Environment-dependent rollout horizon (examples):

- Atari-like: $T\approx 128$ (short horizon, high throughput)
- Continuous control (MuJoCo-like): $T\approx 1024$ to $2048$

### P1.8 Instrumentation and metrics mapping to P0
We will record, per iteration and per worker:

- **System metrics:** FPS, $t_\text{rollout}$, $t_\text{learn}$, $t_\text{comm}$, $t_\text{sync}$
- **Scaling metrics:** $S(W)$, $E(W)$, $\rho_\text{comm}(W)$, $\rho_\text{sync}(W)$
- **Learning metrics:** episodic return (eval), value loss, policy loss, entropy
- **Stability metrics:** approx-KL, clip fraction

These directly support the P0 expectations and explain deviations from ideal scaling.

### P1.9 Risks and mitigations
- **Communication bottleneck at high $W$**
	- Mitigation: bucketed all-reduce; reduce minibatch steps; prefer fewer, larger minibatches.
- **Stragglers due to environment variability**
	- Mitigation: increase vectorization $E$; balance env assignment; cap per-iteration rollout time.
- **Large batch changes optimization dynamics**
	- Mitigation: keep $B$ constant first; tune LR/K/M; use KL-based early stopping if needed.
- **On-policy drift**
	- Mitigation: strict synchronous iterations; avoid asynchronous gradient updates in initial version.

### P1.10 Alternatives considered (and why not chosen for P1)
- **Parameter server PPO**: simpler conceptual model but becomes a bottleneck and may introduce staleness.
- **Asynchronous/APPO-style PPO**: higher utilization but policy lag complicates on-policy assumptions; not chosen for initial submission.
- **IMPALA-style off-policy actor-learner**: excellent scaling but changes the algorithmic objective (off-policy corrections); out of scope for on-policy PPO formulation.

---

## P2 — Implementation (Code + Platform Choices)

### P2.1 What was implemented
We implemented a **minimal synchronous distributed PPO (DD-PPO style)** prototype using **PyTorch** and **Gymnasium**, with a focus on correctness, debuggability, and being runnable on a laptop. The training update uses **synchronous data-parallel SGD**: each worker computes gradients on its local minibatch shard and gradients are **averaged across workers** before applying the optimizer step.

At a high level the implementation matches P1’s loop:

1. Each worker collects an on-policy rollout (vectorized envs per worker)
2. Each worker computes GAE and returns locally
3. Each worker runs PPO updates (multiple epochs / minibatches)
4. In distributed runs, gradients are synchronized via **all-reduce** (PyTorch `DistributedDataParallel`)
5. Workers synchronize at iteration boundaries (barrier) so the policy remains on-policy across iterations

### P2.2 Code organization (key files)
The implementation is intentionally small and split into a few focused modules:

- `src/train_ddppo.py`
	- Main training entrypoint.
	- Builds a vectorized Gymnasium environment per worker.
	- Collects rollouts, computes GAE, performs PPO update steps, and logs per-iteration metrics.
	- Uses `DistributedDataParallel` only when `world_size > 1` (so W=1 remains a plain single-process PPO baseline).

- `src/ppo_core.py`
	- `ActorCritic` network (MLP policy logits + value head).
	- `compute_gae()` for GAE(λ) and returns.
	- `ppo_loss()` implementing PPO-Clip + diagnostics (approx-KL, clip fraction, entropy).

- `src/dist_utils.py`
	- `init_distributed()` helper to initialize `torch.distributed` (or skip it for single-process runs).
	- `barrier()` and `allreduce_mean_scalar()` utilities.
	- Adds a bounded init timeout so rendezvous failures do not hang indefinitely.

- `src/launch_local_ddppo.py`
	- A local multi-process launcher to spawn N Python processes without relying on `torchrun`.
	- Uses a `file://` rendezvous store so macOS hostname/IPv6 issues do not block local experiments.

### P2.3 Platform and backend choices (macOS)
This project was executed on **macOS** (CPU). That drove three practical choices:

1. **Backend: Gloo**
	- On macOS, NCCL GPU collectives are typically unavailable.
	- We use `--backend gloo` for portability and CPU-based collectives.

2. **Rendezvous: `file://` store for local multi-process**
	- `torchrun` / TCP rendezvous can be sensitive to macOS hostname and IPv6 resolution.
	- We use a local filesystem-based rendezvous (`file://…`) in `src/launch_local_ddppo.py`, which avoids TCPStore networking altogether.

3. **Execution model: one process per worker**
	- Each worker is a separate OS process; within that process we run vectorized environments (`SyncVectorEnv`) and PPO updates.
	- This matches the “W workers” model in P0/P1 and is easy to scale from W=1 to W>1.

### P2.4 Where synchronization happens in our code
We make the “distributed” part explicit in terms of *when* communication happens:

- **Gradient averaging (all-reduce):** in `src/train_ddppo.py`, when `world_size > 1` the model is wrapped by PyTorch `DistributedDataParallel (DDP)`. During each minibatch update, calling `loss.backward()` triggers DDP’s gradient bucket all-reduces, producing averaged gradients across workers.
- **Synchronous update:** each worker then runs `optimizer.step()` locally. Because gradients were averaged and each worker processes the same sequence of minibatch steps, parameters remain aligned.
- **Iteration boundary sync:** after the PPO update loop, workers call a `barrier()` (via `src/dist_utils.py`) so iteration timers/logging stay comparable and the next rollout starts from a consistent iteration boundary.

### P2.5 How to run (single-process and multi-process)

**Single-process (W=1):**

```bash
./.venv/bin/python -m src.train_ddppo --env-id CartPole-v1 --total-iters 50
```

**Multi-process local (W>1) using the custom launcher:**

```bash
./.venv/bin/python -m src.launch_local_ddppo --nproc 4 -- \
	--env-id CartPole-v1 --total-iters 50 --rollout-steps 64 --num-envs 2
```

Notes:

- In distributed runs, rank 0 prints logs; metrics are averaged across ranks where appropriate.
- For deterministic-but-diverse rollouts, the base seed is offset by rank.

### P2.5 Current limitations (intentional for an initial P2)
- No separate evaluation loop / episodic return curve yet; P3 focuses on systems metrics and scaling.
- Advantage normalization is local (per-worker) in this initial version.
- The implementation targets discrete-action environments like CartPole for simplicity.

## P3 — Evaluation (Scaling Results)

### P3.1 Goal
Validate the P0 scaling metrics on a local machine by measuring:

- Throughput: **FPS(W)**
- Speedup: **S(W) = FPS(W) / FPS(1)** (throughput-based speedup)
- Efficiency: **E(W) = S(W) / W**
- System breakdown: **t_iter**, **t_rollout**, **t_learn**, **t_sync**

This P3 focuses on **systems scaling** (iteration throughput/latency) rather than “time-to-target return” because the minimal CartPole baseline does not include periodic evaluation episodes yet.

### P3.2 Method
We run the same implementation from P2 with **world sizes W ∈ {1,2,4}** and keep the **global batch approximately constant** across W:

- Baseline (W=1): rollout_steps=64, num_envs=8 ⇒ global_batch = 1·64·8 = 512
- W=2: num_envs is auto-adjusted to 4 ⇒ global_batch = 2·64·4 = 512
- W=4: num_envs is auto-adjusted to 2 ⇒ global_batch = 4·64·2 = 512

We exclude early transient iterations using a warmup window.

**Command used (generates logs + CSV + plot):**

```bash
./.venv/bin/python -m src.p3_benchmark \
	--world-sizes 1,2,4 \
	--repeats 1 \
	--output-dir runs/p3 \
	--env-id CartPole-v1 \
	--total-iters 25 \
	--warmup-iters 5 \
	--rollout-steps 64 \
	--num-envs 8 \
	--epochs 2 \
	--minibatches 4
```

Artifacts produced:

- `runs/p3/logs/` (per-run JSONL training logs, rank0)
- `runs/p3/per_run.csv` and `runs/p3/summary.csv` (aggregated metrics)
- `runs/p3/plots.png` (FPS/speedup/efficiency plot)

### P3.3 Results (local, macOS, Gloo backend)
The table below is copied from `runs/p3/summary.csv`.

| W | rollout_steps | num_envs/worker | global_batch | FPS mean | speedup vs W=1 | efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 8 | 512 | 34985.84 | 1.000 | 1.000 |
| 2 | 64 | 4 | 512 | 27694.95 | 0.792 | 0.396 |
| 4 | 64 | 2 | 512 | 14262.60 | 0.408 | 0.102 |

Additional system metric (sync fraction):

- W=1: 0.0002
- W=2: 0.0067
- W=4: 0.0076

### P3.4 Discussion
Observed scaling is **sub-linear** and even **negative** for this CPU-only CartPole setup when keeping the global batch fixed. This is expected because:

- Each worker’s per-iteration compute becomes smaller as W increases (since we reduced num_envs/worker), so **fixed distributed overheads** (process scheduling + inter-process synchronization + gradient all-reduce) become a larger fraction of the iteration.
- The environment and model are lightweight; there is limited work to amortize synchronization.

For larger environments/models (or for runs where per-worker workload is held constant so global batch grows with W), we expect the scaling curve to improve until communication dominates.


