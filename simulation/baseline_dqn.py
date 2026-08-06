"""
baseline_dqn.py — DQN Routing Baseline ("Cloud-RL" archetype)
================================================================
A centralized DQN routing agent (Mnih et al., Nature 2015) evaluated
inside the SAME SUMO environment as LEACER, with a genuine CLOUD
LATENCY PENALTY implemented as two real mechanisms (not a cosmetic
wall-clock sleep, which would NOT affect simulated outcomes):

  1. Decisions are computed on a traffic-state snapshot that is
     STALE by LATENCY_STALE_STEPS (emulates round-trip delay to a
     centralized inference server — the model reacts to old data).
  2. The reroute cycle is 3x LONGER than edge-native baselines
     (CLOUD_REROUTE_INTERVAL=90 vs 30), reflecting that round-trip
     communication overhead caps how often centralized control can
     refresh decisions.

Honest framing for the paper: this is a standard DQN applied to the
routing MDP, NOT a reproduction of any specific 2025 paper. Cite as
Mnih et al. 2015 (already in your bibliography as ref_dqn).

Usage:
    python simulation/baseline_dqn.py --mode train --episodes 400
    python simulation/baseline_dqn.py --mode eval  --steps 3600
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque

SIM_DIR = Path(__file__).parent
sys.path.insert(0, str(SIM_DIR))

from sumo_env import SUMOEnv
from routing_utils import RoadGraph, TelemetryRecorder, commit_route

# ── Shared topology / state definition (mirrors train_ppo.py) ────────────────
ALL_NODES = ["N00","N01","N02","N10","N11","N12","N20","N21","N22",
             "IN_W","IN_N","IN_E","IN_S"]
ADJACENCY = {
    "N00": ["N01","N10"],  "N01": ["N00","N02","N11"],
    "N02": ["N01","N12"],  "N10": ["N00","N11","N20"],
    "N11": ["N01","N10","N12","N21"], "N12": ["N02","N11","N22"],
    "N20": ["N10","N21"],  "N21": ["N11","N20","N22"],
    "N22": ["N12","N21"],
    "IN_W": ["N00","N10","N20"], "IN_N": ["N00","N01","N02"],
    "IN_E": ["N02","N12","N22"], "IN_S": ["N20","N21","N22"],
}
EMB_DIM, STATE_DIM, MAX_ACTIONS = 8, 2*8 + 4 + 1, 4
ALPHA, BETA, GAMMA, DELTA = 0.35, 0.25, 0.25, 0.15

def mo_cost(speed_kmh, queue, latency_ms, density):
    T = max(0.0, (60.0 - speed_kmh) / 60.0)
    E = min(1.0, density / 120.0)
    C = min(1.0, queue / 20.0)
    L = min(1.0, latency_ms / 300_000.0)
    return ALPHA*T + BETA*E + GAMMA*C + DELTA*L

LATENCY_STALE_STEPS     = 8    # decisions use an 8-step-old TSV snapshot
CLOUD_REROUTE_INTERVAL  = 90   # vs 30 for edge-native baselines
DATA_DIR = SIM_DIR / "data"


class DQNNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=MAX_ACTIONS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, n_actions))
    def forward(self, x): return self.net(x)


class StaleRoutingEnv:
    """Mock routing MDP for offline DQN training (same state def as PPO)."""
    def __init__(self):
        self._max_hops = 12
        rng = np.random.RandomState(1)
        self._emb = {n: rng.randn(EMB_DIM).astype(np.float32) for n in ALL_NODES}
        self._mu, self._sig = np.array([47.,6.,.5,40.]), np.array([5.,4.,.5,20.])

    def reset(self):
        nodes = list(ADJACENCY.keys())
        self.origin = np.random.choice(nodes)
        self.dest   = np.random.choice([n for n in nodes if n != self.origin])
        self.cur, self.step_i = self.origin, 0
        return self._state()

    def _tsv(self):
        return np.array([np.random.uniform(35,55), np.random.uniform(2,30),
                         float(np.random.randint(0,8)), np.random.uniform(20,120)], dtype=np.float32)

    def _state(self):
        tsv = (self._tsv() - self._mu) / self._sig
        frac = np.array([self.step_i / self._max_hops], dtype=np.float32)
        return np.concatenate([self._emb[self.origin], self._emb[self.dest], tsv, frac])

    def valid_mask(self):
        nb = sorted(ADJACENCY.get(self.cur, []))
        m = np.zeros(MAX_ACTIONS, dtype=bool); m[:len(nb)] = True
        return m

    def step(self, action):
        nb = sorted(ADJACENCY.get(self.cur, []))
        if not nb: return self._state(), -1.0, True
        action = min(action, len(nb)-1); nxt = nb[action]
        tsv = self._tsv()
        r = -mo_cost(tsv[0], tsv[2], tsv[3]*1000, tsv[1])
        self.cur, self.step_i = nxt, self.step_i + 1
        done = (nxt == self.dest) or (self.step_i >= self._max_hops)
        if nxt == self.dest: r += 2.0
        return self._state(), r, done


def train(episodes=400, gamma=0.99, lr=1e-3, batch=64,
          eps_start=1.0, eps_end=0.05, eps_decay=300):
    print("="*60); print("Baseline DQN Training (Cloud-RL archetype)"); print("="*60)

    env = StaleRoutingEnv()
    policy, target = DQNNet(), DQNNet()
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    buf = deque(maxlen=50_000)
    eps = eps_start
    ep_rewards = []

    for ep in range(1, episodes + 1):
        s, done, ep_r = env.reset(), False, 0.0
        while not done:
            mask = env.valid_mask()
            if np.random.rand() < eps:
                a = np.random.choice(np.where(mask)[0])
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                    q[~torch.tensor(mask)] = -1e9
                    a = int(q.argmax())
            s2, r, done = env.step(a)
            buf.append((s, a, r, s2, done))
            s, ep_r = s2, ep_r + r

            if len(buf) >= batch:
                idx = np.random.choice(len(buf), batch, replace=False)
                bs, ba, br, bs2, bd = map(np.array, zip(*[buf[i] for i in idx]))
                bs=torch.tensor(bs,dtype=torch.float32); ba=torch.tensor(ba,dtype=torch.long)
                br=torch.tensor(br,dtype=torch.float32); bs2=torch.tensor(bs2,dtype=torch.float32)
                bd=torch.tensor(bd,dtype=torch.float32)
                qsa = policy(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    qt = br + gamma * target(bs2).max(1)[0] * (1-bd)
                loss = F.mse_loss(qsa, qt)
                opt.zero_grad(); loss.backward(); opt.step()

        eps = max(eps_end, eps_start * np.exp(-ep / eps_decay))
        ep_rewards.append(ep_r)
        if ep % 20 == 0:
            target.load_state_dict(policy.state_dict())
            print(f"  ep {ep:4d}  reward={ep_r:7.3f}  avg20={np.mean(ep_rewards[-20:]):7.3f}  eps={eps:.3f}")

    DATA_DIR.mkdir(exist_ok=True)
    torch.save({"model_state": policy.state_dict(), "episodes": episodes},
               DATA_DIR / "dqn_weights.pt")
    print(f"\nSaved -> {DATA_DIR/'dqn_weights.pt'}")


def evaluate(steps=3600, use_gui=False):
    print("="*60); print(f"Baseline DQN Evaluation — {steps} steps (SUMO)"); print("="*60)

    policy = DQNNet()
    wpath = DATA_DIR / "dqn_weights.pt"
    if wpath.exists():
        ckpt = torch.load(wpath, map_location="cpu")
        policy.load_state_dict(ckpt["model_state"])
        print(f"Loaded trained DQN weights ({ckpt['episodes']} episodes)")
    else:
        print("[WARN] No trained weights — run --mode train first for a fair comparison.")
    policy.eval()

    cfg  = str(SIM_DIR / "sumo_cfg" / "leacer.sumocfg")
    env  = SUMOEnv(cfg_path=cfg, use_gui=use_gui, max_steps=steps)
    road = RoadGraph()
    rec  = TelemetryRecorder(algorithm="DQN_CLOUDRL")

    tsv_history = deque(maxlen=LATENCY_STALE_STEPS + 1)
    rng = np.random.RandomState(1)
    node_emb = {n: rng.randn(EMB_DIM).astype(np.float32) for n in ALL_NODES}
    mu, sig = np.array([47.,6.,.5,40.]), np.array([5.,4.,.5,20.])

    env.start()
    step = 0
    while not env.is_done:
        edge_states = env.step()
        if edge_states is None: break
        tsv_history.append(edge_states)

        reroute_triggered = False
        if step % CLOUD_REROUTE_INTERVAL == 0 and len(tsv_history) > LATENCY_STALE_STEPS:
            stale_states = tsv_history[0]   # oldest snapshot = the "stale" cloud view
            reroute_triggered = _dqn_reroute_cycle(policy, road, stale_states, node_emb, mu, sig)

        rec.record_step(step, env.sim_time, edge_states, env=env,
                        reroute_triggered=reroute_triggered)
        step += 1
        if step % 200 == 0:
            print(f"  step {step:5d} | reroutes={rec._reroutes}")

    env.stop()
    rec.save(SIM_DIR / "results" / "baseline_dqn_telemetry.csv")


def _dqn_reroute_cycle(policy, road, stale_states, node_emb, mu, sig):
    try:
        import traci
    except ImportError:
        return False

    veh_ids = traci.vehicle.getIDList()
    if not veh_ids: return False
    stale_by_edge = {s.edge_id: s for s in stale_states}
    any_rerouted = False

    for vid in veh_ids[:20]:
        try:
            route = traci.vehicle.getRoute(vid)
            if not route: continue
            cur_edge, dest_edge = traci.vehicle.getRoadID(vid), route[-1]

            s = stale_by_edge.get(cur_edge)
            tsv = (np.array([s.mean_speed*3.6, s.mean_density,
                             float(s.queue_length), s.travel_time*1000])
                   if s is not None else np.array([40.,10.,2.,80.]))
            tsv_norm = (tsv - mu) / sig

            cur_uv, dest_uv = road.uv_of(cur_edge), road.uv_of(dest_edge)
            if cur_uv is None or dest_uv is None: continue
            cur_node = cur_uv[1]

            state = np.concatenate([node_emb.get(cur_node, np.zeros(EMB_DIM)),
                                    node_emb.get("N22", np.zeros(EMB_DIM)),
                                    tsv_norm, [0.5]]).astype(np.float32)

            nb = sorted(ADJACENCY.get(cur_node, []))
            if not nb: continue
            mask = np.zeros(MAX_ACTIONS, dtype=bool); mask[:len(nb)] = True

            with torch.no_grad():
                q = policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                q[~torch.tensor(mask)] = -1e9
                a = min(int(q.argmax()), len(nb)-1)
            next_node = nb[a]

            rest = road.dijkstra(next_node, dest_uv[1])
            first_edge = road.edge_of(cur_node, next_node)
            if rest is not None and first_edge:
                if commit_route(vid, [cur_edge, first_edge] + rest):
                    any_rerouted = True
        except Exception:
            continue

    return any_rerouted


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--steps", type=int, default=3600)
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()
    train(episodes=args.episodes) if args.mode == "train" else evaluate(steps=args.steps, use_gui=args.gui)
