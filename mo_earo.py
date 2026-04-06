"""
MO-EARO — Multi-Objective Edge-AI Route Optimizer
Layer 2: PPO Policy Agent + Pareto Cost Evaluator

State space:  s_t = [h_origin, h_dest, TSV_current_edge, step/max_steps]
Action space: discrete — index of next candidate edge/node
Reward:       R = -F(route)  where F = alpha*T + beta*E + gamma*C + delta*L

Pareto weights (from paper):
  alpha=0.35  (travel time)
  beta =0.25  (energy)
  gamma=0.25  (congestion)
  delta=0.15  (latency)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Objective Cost Function  F = αT + βE + γC + δL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouteMetrics:
    """Computed metrics for a candidate route."""
    route: List[str]          # ordered list of edge/node IDs
    T: float                  # total travel time (seconds)
    E: float                  # energy consumption (kWh or normalized)
    C: float                  # congestion index [0,1]
    L: float                  # cumulative latency (ms)


class MOCostEvaluator:
    """
    Evaluates F = alpha*T + beta*E + gamma*C + delta*L for any route.

    Weights are normalized so alpha+beta+gamma+delta = 1.
    All metrics are normalized to [0,1] before weighting via
    min-max normalization over the candidate set.
    """

    ALPHA = 0.35
    BETA  = 0.25
    GAMMA = 0.25
    DELTA = 0.15

    def __init__(self, alpha=None, beta=None, gamma=None, delta=None):
        self.alpha = alpha or self.ALPHA
        self.beta  = beta  or self.BETA
        self.gamma = gamma or self.GAMMA
        self.delta = delta or self.DELTA
        # Normalize weights
        s = self.alpha + self.beta + self.gamma + self.delta
        self.alpha /= s; self.beta /= s; self.gamma /= s; self.delta /= s

    def scalar_cost(self, m: RouteMetrics,
                    T_range=(0,1), E_range=(0,1),
                    C_range=(0,1), L_range=(0,1)) -> float:
        """
        F = alpha * T_norm + beta * E_norm + gamma * C_norm + delta * L_norm
        Ranges are (min, max) from the candidate set for normalization.
        """
        def norm(v, lo, hi):
            if hi == lo: return 0.0
            return (v - lo) / (hi - lo)

        T_n = norm(m.T, *T_range)
        E_n = norm(m.E, *E_range)
        C_n = norm(m.C, *C_range)
        L_n = norm(m.L, *L_range)

        return self.alpha*T_n + self.beta*E_n + self.gamma*C_n + self.delta*L_n

    def best_route(self, candidates: List[RouteMetrics]) -> Tuple[RouteMetrics, float]:
        """
        Select Pareto-optimal route from a set of candidates.
        Returns (best_route, cost_score).
        """
        if not candidates:
            raise ValueError("No route candidates provided")

        T_vals = [m.T for m in candidates]
        E_vals = [m.E for m in candidates]
        C_vals = [m.C for m in candidates]
        L_vals = [m.L for m in candidates]

        ranges = dict(
            T_range=(min(T_vals), max(T_vals)),
            E_range=(min(E_vals), max(E_vals)),
            C_range=(min(C_vals), max(C_vals)),
            L_range=(min(L_vals), max(L_vals)),
        )

        costs = [self.scalar_cost(m, **ranges) for m in candidates]
        best_idx = int(np.argmin(costs))
        return candidates[best_idx], costs[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# PPO Actor-Critic Network
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """
        Shared-backbone Actor-Critic for PPO.

        State:   s_t = [h_origin(d), h_dest(d), tsv(4), step_frac(1)]
                       dimension = 2*emb_dim + 5
        Action:  discrete over max_actions candidate next edges

        Actor:   pi(a|s) — policy distribution
        Critic:  V(s)    — state value estimate
        """

        def __init__(self, state_dim: int, max_actions: int,
                     hidden_dim: int = 128):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.actor  = nn.Linear(hidden_dim, max_actions)
            self.critic = nn.Linear(hidden_dim, 1)

        def forward(self, s: torch.Tensor):
            feat = self.backbone(s)
            logits = self.actor(feat)
            value  = self.critic(feat).squeeze(-1)
            return logits, value

        def act(self, s: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None):
            """
            Sample action from policy.
            valid_mask: boolean tensor (max_actions,) — True = valid next hop.
            Returns (action_idx, log_prob, value_estimate).
            """
            logits, value = self.forward(s)
            if valid_mask is not None:
                logits = logits.masked_fill(~valid_mask, -1e9)
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action), value


# ─────────────────────────────────────────────────────────────────────────────
# PPO Training Buffer + Update
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    @dataclass
    class PPOBuffer:
        """Rollout buffer for one PPO epoch."""
        states:     List[torch.Tensor] = None
        actions:    List[int]          = None
        log_probs:  List[torch.Tensor] = None
        rewards:    List[float]        = None
        values:     List[torch.Tensor] = None
        dones:      List[bool]         = None

        def __post_init__(self):
            self.states    = []; self.actions = []
            self.log_probs = []; self.rewards = []
            self.values    = []; self.dones   = []

        def store(self, s, a, lp, r, v, done):
            self.states.append(s); self.actions.append(a)
            self.log_probs.append(lp); self.rewards.append(r)
            self.values.append(v); self.dones.append(done)

        def compute_gae(self, gamma=0.99, lam=0.95) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Generalized Advantage Estimation (GAE):
              delta_t = r_t + gamma*V_{t+1} - V_t
              A_t     = sum_{l=0}^{T} (gamma*lam)^l * delta_t+l
            """
            T   = len(self.rewards)
            adv = torch.zeros(T)
            ret = torch.zeros(T)
            vals = torch.stack(self.values).detach()
            gae = 0.0
            for t in reversed(range(T)):
                next_val = vals[t+1] if t < T-1 and not self.dones[t] else 0.0
                delta = self.rewards[t] + gamma * next_val - vals[t]
                gae   = delta + gamma * lam * (0 if self.dones[t] else gae)
                adv[t] = gae
                ret[t] = gae + vals[t]
            # Normalize advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            return adv, ret


    def ppo_update(model: ActorCritic,
                   optimizer,
                   buffer: PPOBuffer,
                   clip_eps: float = 0.2,
                   vf_coef:  float = 0.5,
                   ent_coef: float = 0.01,
                   epochs:   int   = 4) -> Dict[str, float]:
        """
        PPO clipped surrogate update.

        L_CLIP = E[ min(r_t*A_t, clip(r_t, 1-eps, 1+eps)*A_t) ]
        r_t    = pi_theta(a|s) / pi_theta_old(a|s)
        """
        adv, ret = buffer.compute_gae()
        states    = torch.stack(buffer.states)
        actions   = torch.tensor(buffer.actions, dtype=torch.long)
        old_lps   = torch.stack(buffer.log_probs).detach()

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(epochs):
            logits, values = model(states)
            dist    = Categorical(logits=logits)
            new_lps = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_lps - old_lps).exp()
            clip  = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
            p_loss = -torch.min(ratio * adv, clip * adv).mean()
            v_loss = F.mse_loss(values, ret)

            loss = p_loss + vf_coef * v_loss - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            stats["policy_loss"] += p_loss.item()
            stats["value_loss"]  += v_loss.item()
            stats["entropy"]     += entropy.item()

        for k in stats: stats[k] /= epochs
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Numpy fallback: greedy cost-minimizing route selection
# ─────────────────────────────────────────────────────────────────────────────

class GreedyRouter:
    """Greedy route selector using MO cost. No RL needed."""
    def __init__(self): self.evaluator = MOCostEvaluator()

    def select(self, candidates: List[RouteMetrics]) -> RouteMetrics:
        best, _ = self.evaluator.best_route(candidates)
        return best


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test cost evaluator
    routes = [
        RouteMetrics(["A","B","C"], T=420, E=1.2, C=0.7, L=85),
        RouteMetrics(["A","D","C"], T=510, E=0.9, C=0.3, L=60),
        RouteMetrics(["A","E","C"], T=380, E=1.5, C=0.9, L=110),
    ]
    ev = MOCostEvaluator()
    best, score = ev.best_route(routes)
    print(f"Best route: {best.route}  (F={score:.4f})")
    print(f"  T={best.T}s  E={best.E}kWh  C={best.C}  L={best.L}ms")

    if TORCH_AVAILABLE:
        state_dim = 2*64 + 5   # 2x emb_dim + tsv(4) + step(1)
        model = ActorCritic(state_dim=state_dim, max_actions=8)
        s = torch.randn(state_dim)
        a, lp, v = model.act(s)
        print(f"\nPPO sample: action={a}  log_prob={lp.item():.4f}  value={v.item():.4f}")
