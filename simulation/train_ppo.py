r"""
train_ppo.py — PPO Routing Policy Training
==========================================
Trains the PPO agent inside the SUMO environment and saves
policy weights to simulation/data/ppo_weights.pt

Usage:
    cd C:\Users\ghoda\Downloads\leacer
    python simulation/train_ppo.py [--episodes 500] [--gui]

Output:
    simulation/data/ppo_weights.pt
    simulation/data/ppo_training_rewards.png
"""

import os, sys, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
SIM_DIR  = ROOT / "simulation"
DATA_DIR = SIM_DIR / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIM_DIR))

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int,   default=500)
parser.add_argument("--steps",    type=int,   default=300,
                    help="Steps per episode (shorter = faster training)")
parser.add_argument("--gui",      action="store_true")
parser.add_argument("--lr",       type=float, default=3e-4)
parser.add_argument("--gamma",    type=float, default=0.99)
parser.add_argument("--lam",      type=float, default=0.95)
parser.add_argument("--clip",     type=float, default=0.2)
parser.add_argument("--vf_coef",  type=float, default=0.5)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--ppo_epochs", type=int, default=4)
args = parser.parse_args()

SEED = 42
np.random.seed(SEED)

print("=" * 65)
print("LEACER — PPO Routing Policy Training")
print("=" * 65)
print(f"  Episodes  : {args.episodes}")
print(f"  Steps/ep  : {args.steps}")
print(f"  LR        : {args.lr}")

# ── GRID NODES and adjacency (matches leacer_network) ────────────────────────
GRID_NODES = ["N00","N01","N02","N10","N11","N12","N20","N21","N22"]
ENTRY_NODES = ["IN_W","IN_N","IN_E","IN_S"]
ALL_NODES   = GRID_NODES + ENTRY_NODES

ADJACENCY = {
    "N00": ["N01","N10"],  "N01": ["N00","N02","N11"],
    "N02": ["N01","N12"],  "N10": ["N00","N11","N20"],
    "N11": ["N01","N10","N12","N21"], "N12": ["N02","N11","N22"],
    "N20": ["N10","N21"],  "N21": ["N11","N20","N22"],
    "N22": ["N12","N21"],
    "IN_W": ["N00","N10","N20"], "IN_N": ["N00","N01","N02"],
    "IN_E": ["N02","N12","N22"], "IN_S": ["N20","N21","N22"],
}
NODE_IDX = {n: i for i, n in enumerate(ALL_NODES)}
MAX_ACTIONS = max(len(v) for v in ADJACENCY.values())   # 4

# ── State dimension: [h_origin(8), h_dest(8), TSV(4), step_frac(1)] ──────────
EMB_DIM    = 8
STATE_DIM  = 2 * EMB_DIM + 4 + 1   # = 21

# ── Load GRU normaliser (if available) for TSV normalisation ─────────────────
NORM_PATH = DATA_DIR / "gru_normalizer.npz"
if NORM_PATH.exists():
    nrm = np.load(NORM_PATH)
    TSV_MU  = nrm["mu"].astype(np.float32)
    TSV_SIG = nrm["sig"].astype(np.float32)
else:
    TSV_MU  = np.array([47.0, 6.0, 0.5, 40.0], dtype=np.float32)
    TSV_SIG = np.array([5.0,  4.0, 0.5, 20.0], dtype=np.float32)

# ── Multi-objective cost function ─────────────────────────────────────────────
ALPHA, BETA, GAMMA, DELTA = 0.35, 0.25, 0.25, 0.15   # T, E, C, L

def mo_cost(speed_kmh, queue, latency_ms, density):
    """Scalar MO cost F in [0,1] — lower is better."""
    T = max(0.0, (60.0 - speed_kmh) / 60.0)   # normalised travel-time proxy
    E = min(1.0, density / 120.0)              # energy proxy via density
    C = min(1.0, queue  / 20.0)               # congestion
    L = min(1.0, latency_ms / 300_000.0)       # latency (travel_time in ms)
    return ALPHA*T + BETA*E + GAMMA*C + DELTA*L

# ── Routing environment wrapper ────────────────────────────────────────────────
class RoutingEnv:
    """
    Wraps SUMOEnv into a step-by-step routing MDP.
    State:  [origin_emb, dest_emb, current_TSV_norm, step_fraction]
    Action: index into sorted neighbour list of current node
    Reward: -F(current_edge_TSV) + bonus if destination reached
    """

    def __init__(self, use_gui=False):
        self.use_gui = use_gui
        self._sumo   = None
        self._step   = 0
        self._max_route_steps = 12   # max hops before episode terminates

        # Simple node embeddings (random fixed, shared across training)
        rng = np.random.RandomState(0)
        self._node_emb = {n: rng.randn(EMB_DIM).astype(np.float32)
                          for n in ALL_NODES}

    # ── SUMO lifecycle ────────────────────────────────────────────────────
    def _start_sumo(self):
        try:
            from sumo_env import SUMOEnv
            cfg = SIM_DIR / "sumo_cfg" / "leacer.sumocfg"
            self._sumo = SUMOEnv(cfg_path=str(cfg), use_gui=self.use_gui,
                                 max_steps=args.steps)
            self._sumo.start()
        except Exception as e:
            print(f"[WARN] SUMO unavailable ({e}) — using mock TSV")
            self._sumo = None

    def _get_tsv(self, node_id):
        """Get TSV for the edge leading into node_id, or mock if SUMO unavailable."""
        if self._sumo is not None:
            try:
                import traci
                # Pull speed + queue for edges connected to this node
                edge_candidates = [e for e in self._sumo.edge_ids
                                   if node_id.replace("N","") in e or
                                      node_id in e]
                if edge_candidates:
                    states = self._sumo.get_edge_states()
                    for s in states:
                        if s.edge_id in edge_candidates:
                            return np.array([s.mean_speed * 3.6,
                                             s.mean_density,
                                             float(s.queue_length),
                                             s.travel_time], dtype=np.float32)
            except Exception:
                pass
        # Mock: random traffic state
        return np.array([
            np.random.uniform(35, 55),   # speed_kmh
            np.random.uniform(2,  30),   # density
            float(np.random.randint(0, 8)),
            np.random.uniform(20, 120),  # travel_time
        ], dtype=np.float32)

    # ── Episode control ───────────────────────────────────────────────────
    def reset(self, episode: int):
        """Start new episode, sample random origin/destination."""
        # Pick origin != destination, at least 3 hops apart
        nodes = list(ADJACENCY.keys())
        self._origin = np.random.choice(nodes)
        candidates   = [n for n in nodes if n != self._origin]
        self._dest   = np.random.choice(candidates)
        self._current = self._origin
        self._route   = [self._origin]
        self._step    = 0

        # Reset SUMO environment per episode start
        if self._sumo is not None:
            try:
                self._sumo.reset()
            except Exception:
                self._sumo = None

        return self._make_state()

    def step(self, action_idx: int):
        neighbours = sorted(ADJACENCY.get(self._current, []))
        if not neighbours:
            return self._make_state(), -1.0, True

        action_idx  = min(action_idx, len(neighbours) - 1)
        next_node   = neighbours[action_idx]
        tsv         = self._get_tsv(next_node)

        # Reward
        cost   = mo_cost(tsv[0], tsv[2], tsv[3]*1000, tsv[1])
        reward = -cost

        self._current = next_node
        self._route.append(next_node)
        self._step   += 1

        done = (next_node == self._dest) or (self._step >= self._max_route_steps)
        if next_node == self._dest:
            reward += 2.0   # destination bonus

        return self._make_state(), reward, done

    def _make_state(self) -> np.ndarray:
        tsv = self._get_tsv(self._current)
        tsv_norm = (tsv - TSV_MU) / TSV_SIG
        step_frac = np.array([self._step / self._max_route_steps], dtype=np.float32)
        return np.concatenate([
            self._node_emb[self._origin],
            self._node_emb[self._dest],
            tsv_norm,
            step_frac,
        ])  # (21,)

    def valid_mask(self):
        """Boolean mask of valid actions for current node."""
        neighbours = sorted(ADJACENCY.get(self._current, []))
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        mask[:len(neighbours)] = True
        return mask

    def close(self):
        if self._sumo is not None:
            try:
                self._sumo.stop()
            except Exception:
                pass


# ── PPO Actor-Critic ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH = True
except ImportError:
    TORCH = False
    print("[ERROR] PyTorch required for PPO training.")
    print("  pip install torch")
    sys.exit(1)

torch.manual_seed(SEED)

class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=MAX_ACTIONS, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, s):
        f  = self.backbone(s)
        return self.actor(f), self.critic(f).squeeze(-1)

    def act(self, s: np.ndarray, mask: np.ndarray):
        st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        logits, value = self.forward(st)
        logits = logits.squeeze(0)
        logits[~torch.tensor(mask)] = -1e9
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist    = Categorical(logits=logits)
        lp      = dist.log_prob(actions)
        entropy = dist.entropy()
        return lp, values, entropy


def ppo_update(model, optimizer, rollout, gamma, lam, clip, vf_coef, ent_coef, ppo_epochs):
    states  = torch.tensor(np.array(rollout["states"]),  dtype=torch.float32)
    actions = torch.tensor(rollout["actions"],            dtype=torch.long)
    old_lp  = torch.tensor(rollout["log_probs"],          dtype=torch.float32)
    rewards = rollout["rewards"]
    dones   = rollout["dones"]
    values  = torch.tensor(rollout["values"],             dtype=torch.float32)

    # GAE
    T   = len(rewards)
    adv = torch.zeros(T)
    ret = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        nv  = values[t+1] if t < T-1 and not dones[t] else 0.0
        d   = rewards[t] + gamma * nv - values[t]
        gae = d + gamma * lam * (0.0 if dones[t] else gae)
        adv[t] = gae
        ret[t] = gae + values[t]
    
    if T > 1:
        std = adv.std()
        if std > 1e-8:
            adv = (adv - adv.mean()) / (std + 1e-8)
        else:
            adv = adv - adv.mean()
    else:
        adv = adv - adv.mean() # For T=1, adv becomes 0

    total_pl = total_vl = total_ent = 0.0
    for _ in range(ppo_epochs):
        lp, vals, ent = model.evaluate(states, actions)
        ratio = (lp - old_lp.detach()).exp()
        p_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1-clip, 1+clip) * adv).mean()
        v_loss = F.mse_loss(vals, ret)
        loss   = p_loss + vf_coef * v_loss - ent_coef * ent.mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_pl += p_loss.item(); total_vl += v_loss.item(); total_ent += ent.mean().item()

    return total_pl/ppo_epochs, total_vl/ppo_epochs, total_ent/ppo_epochs


# ── Main training loop ────────────────────────────────────────────────────────
model     = ActorCritic()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
env       = RoutingEnv(use_gui=args.gui)
env._start_sumo()

ep_rewards  = []
ep_lengths  = []
avg_rewards = []
window      = deque(maxlen=50)

print(f"\n{'Ep':>6}  {'Reward':>9}  {'Avg50':>9}  {'Steps':>7}  {'P-Loss':>8}  {'V-Loss':>8}")
print("-" * 58)

best_avg = -float("inf")
best_state_dict = None

for episode in range(1, args.episodes + 1):
    state = env.reset(episode)
    rollout = {"states":[], "actions":[], "log_probs":[], "rewards":[], "dones":[], "values":[]}
    ep_reward = 0.0
    done      = False

    while not done:
        mask  = env.valid_mask()
        with torch.no_grad():
            a, lp, val = model.act(state, mask)
        next_state, reward, done = env.step(a)
        rollout["states"].append(state)
        rollout["actions"].append(a)
        rollout["log_probs"].append(lp.item())
        rollout["rewards"].append(reward)
        rollout["dones"].append(done)
        rollout["values"].append(val.item())
        state      = next_state
        ep_reward += reward

    pl, vl, ent = ppo_update(model, optimizer, rollout,
                              args.gamma, args.lam, args.clip,
                              args.vf_coef, args.ent_coef, args.ppo_epochs)

    ep_rewards.append(ep_reward)
    ep_lengths.append(len(rollout["actions"]))
    window.append(ep_reward)
    avg50 = np.mean(window)
    avg_rewards.append(avg50)

    if avg50 > best_avg:
        best_avg = avg50
        best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    if episode % 25 == 0 or episode == 1:
        print(f"{episode:>6}  {ep_reward:>9.3f}  {avg50:>9.3f}  "
              f"{ep_lengths[-1]:>7}  {pl:>8.4f}  {vl:>8.4f}")

env.close()

# ── Save weights ──────────────────────────────────────────────────────────────
model.load_state_dict(best_state_dict)
save_path = DATA_DIR / "ppo_weights.pt"
torch.save({
    "model_state": best_state_dict,
    "model_config": dict(state_dim=STATE_DIM, n_actions=MAX_ACTIONS, hidden=128),
    "best_avg_reward": best_avg,
    "episodes_trained": args.episodes,
    "node_embeddings": {k: v.tolist() for k, v in
                        list(zip(ALL_NODES, [env._node_emb[n] for n in ALL_NODES]))},
    "hyperparams": vars(args),
}, save_path)
print(f"\n[Saved] {save_path}")
print(f"[Best]  avg50_reward = {best_avg:.4f}")

# ── Training curve ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(ep_rewards,  lw=0.8, alpha=0.5, color="#1f77b4", label="Episode reward")
ax1.plot(avg_rewards, lw=1.6, color="#d62728", label="Avg(50)")
ax1.set_xlabel("Episode"); ax1.set_ylabel("Cumulative Reward")
ax1.set_title("PPO Routing Agent — Reward Convergence")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(ep_lengths, lw=0.8, color="#2ca02c")
ax2.set_xlabel("Episode"); ax2.set_ylabel("Route Length (hops)")
ax2.set_title("Route Length per Episode")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / "ppo_training_rewards.png", dpi=120)
plt.close()
print(f"[Plot]  {DATA_DIR / 'ppo_training_rewards.png'}")

print("\n" + "=" * 65)
print("PPO training complete.")
print("=" * 65)
