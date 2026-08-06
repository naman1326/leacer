"""
baseline_gcn.py — GCN Routing Baseline ("GCN-Route" archetype)
=================================================================
A 2-layer Graph Convolutional Network (Kipf & Welling, ICLR 2017)
scores road-network junctions from live traffic state; the network
is re-routed periodically using GCN-informed Dijkstra weights.

Implemented in raw PyTorch (no torch_geometric dependency, avoids
Windows wheel-matching issues) via normalised adjacency propagation:
    H' = D^-1/2 (A+I) D^-1/2 H W

Trained with a genuine self-supervised objective: predict each
junction's next-step average speed from its current traffic state,
using a short real SUMO rollout as training data (conceptually the
same forecasting objective as T-GCN, already in your Related Work).

Usage:
    python simulation/baseline_gcn.py --mode train --rollout-steps 400 --epochs 60
    python simulation/baseline_gcn.py --mode eval  --steps 3600 --update-interval 50
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

SIM_DIR = Path(__file__).parent
sys.path.insert(0, str(SIM_DIR))

from sumo_env import SUMOEnv
from routing_utils import RoadGraph, TelemetryRecorder, commit_route

DATA_DIR = SIM_DIR / "data"


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, X, A_norm):
        return torch.relu(self.lin(A_norm @ X))


class GCNRouter(nn.Module):
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
    def forward(self, X, A_norm):
        h = self.gcn1(X, A_norm)
        h = self.gcn2(h, A_norm)
        return self.head(h).squeeze(-1)   # (n_nodes,) predicted next-step speed (normalised)


def normalise_adjacency(G, node_order):
    n = len(node_order)
    idx = {node: i for i, node in enumerate(node_order)}
    A = np.eye(n)
    for u, v in G.edges():
        if u in idx and v in idx:
            A[idx[u], idx[v]] = 1.0
            A[idx[v], idx[u]] = 1.0
    deg = A.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return torch.tensor(D_inv_sqrt @ A @ D_inv_sqrt, dtype=torch.float32)


def _node_features(road, node_order, node_idx, edge_states):
    n = len(node_order)
    feat, counts = np.zeros((n, 4), dtype=np.float32), np.zeros(n)
    state_by_edge = {s.edge_id: s for s in edge_states}
    for u, v, data in road.G.edges(data=True):
        s = state_by_edge.get(data["id"])
        if s is None: continue
        vec = np.array([s.mean_speed*3.6, s.mean_density,
                        float(s.queue_length), s.travel_time*1000])
        for node in (u, v):
            i = node_idx[node]; feat[i] += vec; counts[i] += 1
    counts[counts == 0] = 1
    return feat / counts[:, None]


def _collect_rollout(road, node_order, node_idx, steps, use_gui):
    cfg = str(SIM_DIR / "sumo_cfg" / "leacer.sumocfg")
    env = SUMOEnv(cfg_path=cfg, use_gui=use_gui, max_steps=steps)
    env.start()
    feats = []
    while not env.is_done:
        edge_states = env.step()
        if edge_states is None: break
        feats.append(_node_features(road, node_order, node_idx, edge_states))
    env.stop()
    return np.array(feats)   # (T, n_nodes, 4)


def train(rollout_steps=400, epochs=60, use_gui=False):
    print("="*60); print("GCN-Route Self-Supervised Pretraining"); print("="*60)
    road = RoadGraph()
    node_order = road.nodes
    node_idx = {n: i for i, n in enumerate(node_order)}
    A_norm = normalise_adjacency(road.G, node_order)

    print(f"Collecting {rollout_steps}-step SUMO rollout for training data...")
    feats = _collect_rollout(road, node_order, node_idx, rollout_steps, use_gui)
    print(f"Collected {feats.shape[0]} timesteps x {feats.shape[1]} nodes")

    mu  = feats.reshape(-1, 4).mean(0)
    sig = feats.reshape(-1, 4).std(0) + 1e-6
    feats_norm = (feats - mu) / sig

    model = GCNRouter(in_dim=4, hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.tensor(feats_norm[:-1], dtype=torch.float32)
    Y = torch.tensor(feats_norm[1:, :, 0], dtype=torch.float32)   # next-step speed

    for ep in range(1, epochs + 1):
        total = 0.0
        for t in range(X.shape[0]):
            pred = model(X[t], A_norm)
            loss = F.mse_loss(pred, Y[t])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 10 == 0:
            print(f"  epoch {ep:3d}  avg_MSE={total/X.shape[0]:.4f}")

    DATA_DIR.mkdir(exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "feat_mu": mu.tolist(), "feat_sig": sig.tolist()},
               DATA_DIR / "gcn_weights.pt")
    print(f"\nSaved -> {DATA_DIR/'gcn_weights.pt'}")


def evaluate(steps=3600, update_interval=50, use_gui=False):
    print("="*60); print(f"GCN-Route Evaluation — {steps} steps, update every {update_interval}"); print("="*60)

    road = RoadGraph()
    node_order = road.nodes
    node_idx = {n: i for i, n in enumerate(node_order)}
    A_norm = normalise_adjacency(road.G, node_order)

    model = GCNRouter(in_dim=4, hidden=32)
    wpath = DATA_DIR / "gcn_weights.pt"
    if wpath.exists():
        ckpt = torch.load(wpath, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        mu, sig = np.array(ckpt["feat_mu"]), np.array(ckpt["feat_sig"])
        print("Loaded pretrained GCN weights")
    else:
        mu, sig = np.array([47.,6.,.5,40.]), np.array([5.,4.,.5,20.])
        print("[WARN] No pretrained GCN weights — run --mode train first for a fair comparison.")
    model.eval()

    cfg = str(SIM_DIR / "sumo_cfg" / "leacer.sumocfg")
    env = SUMOEnv(cfg_path=cfg, use_gui=use_gui, max_steps=steps)
    rec = TelemetryRecorder(algorithm="GCN_ROUTE")

    env.start()
    step = 0
    while not env.is_done:
        edge_states = env.step()
        if edge_states is None: break

        reroute_triggered = False
        if step % update_interval == 0:
            reroute_triggered = _gcn_reroute_cycle(
                model, road, node_order, node_idx, A_norm, edge_states, mu, sig)

        rec.record_step(step, env.sim_time, edge_states, env=env,
                        reroute_triggered=reroute_triggered)
        step += 1
        if step % 200 == 0:
            print(f"  step {step:5d} | reroutes={rec._reroutes}")

    env.stop()
    rec.save(SIM_DIR / "results" / "baseline_gcn_telemetry.csv")


def _gcn_reroute_cycle(model, road, node_order, node_idx, A_norm, edge_states, mu, sig):
    try:
        import traci
    except ImportError:
        return False

    feat = _node_features(road, node_order, node_idx, edge_states)
    feat_norm = (feat - mu) / sig

    with torch.no_grad():
        pred_speed = model(torch.tensor(feat_norm, dtype=torch.float32), A_norm).numpy()

    # Higher predicted speed => more desirable => lower routing weight
    for u, v, data in road.G.edges(data=True):
        desirability = (pred_speed[node_idx[u]] + pred_speed[node_idx[v]]) / 2.0
        base_weight = data["length"] / max(data["speed"], 0.1)
        data["weight"] = base_weight * float(np.exp(-desirability))

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

            new_path = road.dijkstra(cur_uv[1], dest_uv[1])
            if new_path:
                if commit_route(vid, [cur_edge] + new_path):
                    any_rerouted = True
        except Exception:
            continue

    return any_rerouted


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--rollout-steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps", type=int, default=3600)
    p.add_argument("--update-interval", type=int, default=50)
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()
    if args.mode == "train":
        train(rollout_steps=args.rollout_steps, epochs=args.epochs, use_gui=args.gui)
    else:
        evaluate(steps=args.steps, update_interval=args.update_interval, use_gui=args.gui)
