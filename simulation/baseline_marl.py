"""
baseline_marl.py — Independent Multi-Agent DQN ("DRL-ITS" archetype)
=======================================================================
Four independent DQN agents, one per RSU coverage region (reuses the
RSU_LAYOUT grid partition already defined in multi_rsu.py). Each agent
trains and decides ENTIRELY independently — no shared weights, no
cross-agent messages. This is the classical Independent Learners
multi-agent formulation (Tan, ICML 1993 — already ref_marl_its in your
bibliography), contrasted with LEACER's cooperative RSU mesh (Eq. 38).

Usage:
    python simulation/baseline_marl.py --mode train --episodes 300
    python simulation/baseline_marl.py --mode eval  --steps 3600
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
from multi_rsu import RSU_LAYOUT   # reuse existing 4-region grid partition

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
DATA_DIR = SIM_DIR / "data"
REGIONS = list(RSU_LAYOUT.keys())   # ["RSU_NW","RSU_NE","RSU_SW","RSU_SE"]

def mo_cost(speed_kmh, queue, latency_ms, density):
    T = max(0., (60.-speed_kmh)/60.); E = min(1., density/120.)
    C = min(1., queue/20.); L = min(1., latency_ms/300_000.)
    return ALPHA*T + BETA*E + GAMMA*C + DELTA*L

def node_region(node_id):
    for rid, cfg in RSU_LAYOUT.items():
        if node_id in cfg["nodes"]:
            return rid
    return REGIONS[0]


class DQNNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=MAX_ACTIONS, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, n_actions))
    def forward(self, x): return self.net(x)


class MockEnv:
    """Per-region training MDP — confines agents to nodes in ONE RSU region,
    enforcing genuine decentralisation during training."""
    def __init__(self, region):
        self.nodes = RSU_LAYOUT[region]["nodes"]
        rng = np.random.RandomState(hash(region) % 2**31)
        self._emb = {n: rng.randn(EMB_DIM).astype(np.float32) for n in ALL_NODES}
        self._mu, self._sig = np.array([47.,6.,.5,40.]), np.array([5.,4.,.5,20.])
        self._max_hops = 6

    def reset(self):
        self.origin = np.random.choice(self.nodes)
        cand = [n for n in self.nodes if n != self.origin]
        self.dest = np.random.choice(cand) if cand else self.origin
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
        nb = sorted([x for x in ADJACENCY.get(self.cur, []) if x in self.nodes]
                    or ADJACENCY.get(self.cur, []))
        m = np.zeros(MAX_ACTIONS, dtype=bool); m[:len(nb)] = True
        return m, nb

    def step(self, action):
        mask, nb = self.valid_mask()
        if not nb: return self._state(), -1., True
        action = min(action, len(nb)-1); nxt = nb[action]
        tsv = self._tsv()
        r = -mo_cost(tsv[0], tsv[2], tsv[3]*1000, tsv[1])
        self.cur, self.step_i = nxt, self.step_i + 1
        done = (nxt == self.dest) or (self.step_i >= self._max_hops)
        if nxt == self.dest: r += 2.0
        return self._state(), r, done


def train_one_agent(region, episodes, lr=1e-3, gamma=0.99, batch=64):
    env = MockEnv(region)
    policy, target = DQNNet(), DQNNet()
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    buf = deque(maxlen=20_000)
    eps = 1.0

    for ep in range(1, episodes + 1):
        s, done = env.reset(), False
        while not done:
            mask, nb = env.valid_mask()
            if np.random.rand() < eps or not nb:
                a = np.random.randint(0, max(len(nb), 1))
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                    q[~torch.tensor(mask)] = -1e9
                    a = int(q.argmax())
            s2, r, done = env.step(a)
            buf.append((s, a, r, s2, done)); s = s2

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

        eps = max(0.05, eps * 0.98)
        if ep % 20 == 0:
            target.load_state_dict(policy.state_dict())

    return policy


def train(episodes=300):
    print("="*60); print("Independent Multi-Agent DQN Training (DRL-ITS archetype)"); print("="*60)
    DATA_DIR.mkdir(exist_ok=True)
    agents = {}
    for region in REGIONS:
        print(f"\n-- Training agent for {region} --")
        agents[region] = train_one_agent(region, episodes)
    torch.save({r: a.state_dict() for r, a in agents.items()}, DATA_DIR / "marl_weights.pt")
    print(f"\nSaved 4 independent agents -> {DATA_DIR/'marl_weights.pt'}")


def evaluate(steps=3600, use_gui=False):
    print("="*60); print(f"MARL Evaluation — {steps} steps (SUMO)"); print("="*60)

    agents = {r: DQNNet() for r in REGIONS}
    wpath = DATA_DIR / "marl_weights.pt"
    if wpath.exists():
        sd = torch.load(wpath, map_location="cpu")
        for r in REGIONS: agents[r].load_state_dict(sd[r])
        print("Loaded 4 trained independent agents")
    else:
        print("[WARN] No trained MARL weights — run --mode train first for a fair comparison.")
    for a in agents.values(): a.eval()

    road = RoadGraph()
    cfg  = str(SIM_DIR / "sumo_cfg" / "leacer.sumocfg")
    env  = SUMOEnv(cfg_path=cfg, use_gui=use_gui, max_steps=steps)
    rec  = TelemetryRecorder(algorithm="MARL_DRLITS")

    rng = np.random.RandomState(2)
    node_emb = {n: rng.randn(EMB_DIM).astype(np.float32) for n in ALL_NODES}
    mu, sig  = np.array([47.,6.,.5,40.]), np.array([5.,4.,.5,20.])

    env.start()
    step = 0
    while not env.is_done:
        edge_states = env.step()
        if edge_states is None: break

        reroute_triggered = False
        if step % 30 == 0:   # edge-native cadence — no cloud latency penalty
            reroute_triggered = _marl_reroute_cycle(agents, road, edge_states, node_emb, mu, sig)

        rec.record_step(step, env.sim_time, edge_states, env=env,
                        reroute_triggered=reroute_triggered)
        step += 1
        if step % 200 == 0:
            print(f"  step {step:5d} | reroutes={rec._reroutes}")

    env.stop()
    rec.save(SIM_DIR / "results" / "baseline_marl_telemetry.csv")


def _marl_reroute_cycle(agents, road, edge_states, node_emb, mu, sig):
    try:
        import traci
    except ImportError:
        return False

    state_by_edge = {s.edge_id: s for s in edge_states}
    veh_ids = traci.vehicle.getIDList()
    if not veh_ids: return False
    any_rerouted = False

    for vid in veh_ids[:25]:
        try:
            route = traci.vehicle.getRoute(vid)
            if not route: continue
            cur_edge, dest_edge = traci.vehicle.getRoadID(vid), route[-1]
            cur_uv, dest_uv = road.uv_of(cur_edge), road.uv_of(dest_edge)
            if cur_uv is None or dest_uv is None: continue
            cur_node = cur_uv[1]
            agent = agents[node_region(cur_node)]

            s = state_by_edge.get(cur_edge)
            tsv = (np.array([s.mean_speed*3.6, s.mean_density,
                             float(s.queue_length), s.travel_time*1000])
                   if s else np.array([40.,10.,2.,80.]))
            tsv_norm = (tsv - mu) / sig

            state = np.concatenate([node_emb.get(cur_node, np.zeros(EMB_DIM)),
                                    node_emb.get("N22", np.zeros(EMB_DIM)),
                                    tsv_norm, [0.5]]).astype(np.float32)

            nb = sorted(ADJACENCY.get(cur_node, []))
            if not nb: continue
            mask = np.zeros(MAX_ACTIONS, dtype=bool); mask[:len(nb)] = True

            with torch.no_grad():
                q = agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
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
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=3600)
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()
    train(episodes=args.episodes) if args.mode == "train" else evaluate(steps=args.steps, use_gui=args.gui)
