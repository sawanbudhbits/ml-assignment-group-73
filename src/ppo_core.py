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
