"""
leacer_runner.py -- LEACER Closed-Loop Simulation Runner
==========================================================
Runs the full LEACER pipeline using trained GRU + PPO weights, and
ACTUALLY commits rerouted paths to SUMO via TraCI (this was previously
a no-op stub: it computed a PPO action but never called setRoute,
which is why LEACER's SUMO stats were identical to Static's in earlier
runs). Falls back to heuristic routing if weights are not trained.

Two correctness fixes vs. the earlier version:
  1. Topology now comes from topology_cache.py (auto-generated from
     whatever network is currently loaded, e.g. MoST) instead of the
     old hardcoded 13-node grid dict, which would silently mismatch
     any network you've since swapped to.
  2. Node embeddings used at inference are the SAME embeddings PPO
     was actually trained with (loaded from ppo_weights.pt's saved
     "node_embeddings"), not freshly regenerated with a different
     random seed. Using mismatched embeddings makes a trained policy's
     action selection effectively meaningless, since the network
     never saw these particular vectors during training.

Called by: run_simulation.py --mode leacer
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

ROOT    = Path(__file__).parent.parent
SIM_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIM_DIR))

from sumo_env          import SUMOEnv, SimulationMetrics, EdgeState
from sumo_data_adapter  import SumoDataAdapter
from routing_utils      import RoadGraph, commit_route, compute_co2_step
from topology_cache     import ADJACENCY, ALL_NODES
from scenario_config    import SUMOCFG_PATH, NET_FILE_PATH

EMB_DIM      = 8
MAX_ACTIONS  = 4

# -- CEFAR KPI thresholds ------------------------------------------------------
TAU_C    = 0.75
TAU_L_MS = 150.0
TAU_Q    = 20
COOLDOWN = 30

MAX_VEHICLES_PER_CYCLE = 20   # matches the cap used by every baseline runner


# -- Model loader ----------------------------------------------------------------
class ModelLoader:
    """Loads trained GRU + PPO weights if available."""

    def __init__(self, data_dir: Path):
        self.data_dir           = data_dir
        self.gru_model          = None
        self.ppo_model          = None
        self.ppo_node_embeddings: Dict[str, np.ndarray] = {}
        self.norm_mu    = np.array([47.0, 6.0, 0.5, 40.0], dtype=np.float32)
        self.norm_sig   = np.array([5.0,  4.0, 0.5, 20.0], dtype=np.float32)
        self._load()

    def _load(self):
        try:
            import torch
            import torch.nn as nn

            # -- GRU -----------------------------------------------------
            gru_path = self.data_dir / "gru_weights.pt"
            if gru_path.exists():
                ckpt = torch.load(gru_path, map_location="cpu", weights_only=False)
                cfg  = ckpt["model_config"]

                class GRUModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.gru  = nn.GRU(cfg["feat"], cfg["hidden"], cfg["layers"],
                                           batch_first=True)
                        self.head = nn.Sequential(
                            nn.Linear(cfg["hidden"], cfg["hidden"]), nn.ReLU(),
                            nn.Linear(cfg["hidden"], cfg["horizon"] * cfg["feat"]),
                        )
                        self.horizon = cfg["horizon"]
                        self.feat    = cfg["feat"]

                    def forward(self, x):
                        _, h = self.gru(x)
                        return self.head(h[-1]).view(x.size(0), self.horizon, self.feat)

                m = GRUModel()
                m.load_state_dict(ckpt["model_state"])
                m.eval()
                self.gru_model  = m
                self.norm_mu    = np.array(ckpt["norm_mu"],  dtype=np.float32)
                self.norm_sig   = np.array(ckpt["norm_sig"], dtype=np.float32)
                print(f"[LEACER] GRU weights loaded  (val_MSE={ckpt['best_val_mse']:.5f})")
            else:
                print("[LEACER] GRU weights not found, using heuristic predictor")

            # -- PPO -----------------------------------------------------
            ppo_path = self.data_dir / "ppo_weights.pt"
            if ppo_path.exists():
                ckpt = torch.load(ppo_path, map_location="cpu", weights_only=False)
                cfg  = ckpt["model_config"]

                class AC(nn.Module):
                    def __init__(self):
                        super().__init__()
                        h = cfg["hidden"]
                        self.bb = nn.Sequential(
                            nn.Linear(cfg["state_dim"], h), nn.Tanh(),
                            nn.Linear(h, h), nn.Tanh())
                        self.actor  = nn.Linear(h, cfg["n_actions"])
                        self.critic = nn.Linear(h, 1)

                    def forward(self, s):
                        f = self.bb(s)
                        return self.actor(f), self.critic(f).squeeze(-1)

                m2 = AC()
                m2.load_state_dict(ckpt["model_state"])
                m2.eval()
                self.ppo_model = m2

                # Load the SAME node embeddings PPO was trained with, not
                # freshly regenerated ones, or its learned actions are
                # meaningless at inference.
                saved_emb = ckpt.get("node_embeddings", {})
                self.ppo_node_embeddings = {
                    k: np.array(v, dtype=np.float32) for k, v in saved_emb.items()
                }
                print(f"[LEACER] PPO weights loaded  (best_avg={ckpt['best_avg_reward']:.4f}, "
                      f"{len(self.ppo_node_embeddings)} node embeddings restored)")

                missing = [n for n in ALL_NODES if n not in self.ppo_node_embeddings]
                if missing:
                    print(f"[LEACER][WARN] {len(missing)} nodes in the current network have "
                          f"no saved embedding (network changed since training?). "
                          f"Falling back to zero-vectors for those nodes.")
            else:
                print("[LEACER] PPO weights not found, using greedy MO routing")

        except ImportError:
            print("[LEACER] PyTorch not available, using heuristics only")
        except Exception as e:
            print(f"[LEACER] Model load warning: {e}")

    def node_embedding(self, node_id: str) -> np.ndarray:
        return self.ppo_node_embeddings.get(node_id, np.zeros(EMB_DIM, dtype=np.float32))

    def predict_tsv(self, window: np.ndarray) -> np.ndarray:
        """Predict next-step TSV. window: (tau, 4). Returns (4,)."""
        if self.gru_model is not None:
            try:
                import torch
                x = torch.tensor(
                    ((window - self.norm_mu) / self.norm_sig)[None],
                    dtype=torch.float32)
                with torch.no_grad():
                    pred = self.gru_model(x)[0, 0].numpy()
                return pred * self.norm_sig + self.norm_mu
            except Exception:
                pass
        alpha = 0.3
        lvl = window[0].copy()
        for row in window:
            lvl = alpha * row + (1 - alpha) * lvl
        return lvl

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        if self.ppo_model is not None:
            try:
                import torch
                from torch.distributions import Categorical
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = self.ppo_model(s)
                logits = logits.squeeze(0)
                logits[~torch.tensor(mask)] = -1e9
                return Categorical(logits=logits).sample().item()
            except Exception:
                pass
        valid = np.where(mask)[0]
        return int(valid[0]) if len(valid) else 0


# -- CEFAR Layer-3 monitor -------------------------------------------------------
class CEFARMonitor:
    def __init__(self):
        self._last = {}

    def check(self, step: int, states: List[EdgeState]) -> bool:
        for s in states:
            cong = s.occupancy / 100.0
            if cong > TAU_C and (step - self._last.get("C", -9999)) > COOLDOWN:
                self._last["C"] = step; return True
            if s.travel_time * 1000 > TAU_L_MS and (step - self._last.get("L", -9999)) > COOLDOWN:
                self._last["L"] = step; return True
            if s.queue_length > TAU_Q and (step - self._last.get("Q", -9999)) > COOLDOWN:
                self._last["Q"] = step; return True
        return False


# -- Main runner -------------------------------------------------------------
class LEACERRunner:
    RESULTS_DIR = SIM_DIR / "results"

    def __init__(self, cfg_path: str = None, use_gui: bool = False, max_steps: int = 3600):
        self.max_steps = max_steps

        cfg = cfg_path or str(SUMOCFG_PATH)
        self.env     = SUMOEnv(cfg_path=cfg, use_gui=use_gui, max_steps=max_steps)
        self.adapter = SumoDataAdapter()
        self.cefar   = CEFARMonitor()
        self.models  = ModelLoader(SIM_DIR / "data")
        self.road    = RoadGraph(net_file=str(NET_FILE_PATH))   # same network the env above is actually simulating

        self.RESULTS_DIR.mkdir(exist_ok=True)

        self._records: List[dict] = []
        self._co2_cum  = 0.0
        self._reroutes = 0   # counts CYCLES with >=1 successfully committed reroute,
                              # matching the convention every baseline runner uses

        from collections import deque
        self._deque = deque
        self._tsv_window: dict = {}

    def run(self) -> SimulationMetrics:
        print("=" * 70)
        print("  LEACER Closed-Loop Simulation Starting")
        print("=" * 70)

        self.env.start()
        t0 = time.perf_counter()
        step = 0

        while not self.env.is_done:
            edge_states = self.env.step()

            if edge_states is None:
                print("\n[LEACER] Simulation stopped early.")
                break
            if not edge_states:
                step += 1
                continue

            tsv_map = self.adapter.adapt(edge_states)

            for eid, tsv in tsv_map.items():
                if eid not in self._tsv_window:
                    self._tsv_window[eid] = self._deque(maxlen=20)
                self._tsv_window[eid].append([tsv.S, tsv.D, float(tsv.Q), tsv.L])

            for s in edge_states:
                length_km = self.env.get_edge_length(s.edge_id) / 1000.0
                self._co2_cum += compute_co2_step(s.mean_speed * 3.6, s.mean_density, length_km)

            reroute_committed = False
            if self.cefar.check(step, edge_states):
                reroute_committed = self._do_reroute(tsv_map)
                if reroute_committed:
                    self._reroutes += 1

            avg_speed = np.mean([s.mean_speed * 3.6 for s in edge_states]) if edge_states else 0.0
            avg_queue = np.mean([s.queue_length for s in edge_states]) if edge_states else 0.0
            avg_lat   = np.mean([s.travel_time for s in edge_states]) if edge_states else 0.0

            self._records.append({
                "step":            step,
                "sim_time":        self.env.sim_time,
                "avg_speed_kmh":   round(avg_speed, 3),
                "avg_queue":       round(avg_queue, 4),
                "avg_latency_ms":  round(avg_lat * 1000, 2),
                "co2_total":       round(self._co2_cum, 6),
                "reroute_events":  self._reroutes,
            })

            step += 1
            if step % 200 == 0:
                print(f"  Step {step:5d} | t={self.env.sim_time:7.1f}s "
                      f"| Speed={avg_speed:5.1f}km/h | Queue={avg_queue:5.1f} "
                      f"| Reroute cycles={self._reroutes}")

        wall = time.perf_counter() - t0
        metrics = self.env.stop()
        metrics.reroute_events = self._reroutes
        metrics.total_co2_kg   = self._co2_cum

        self._print_summary(metrics, wall, step)
        self._save(metrics)
        return metrics

    # -- Rerouting via PPO -- now actually commits via TraCI -----------------
    def _do_reroute(self, tsv_map) -> bool:
        """
        Selects a next-hop for a sample of active vehicles using the trained
        PPO policy, completes the rest of the path via Dijkstra, and commits
        it with traci.vehicle.setRoute. Returns True if at least one vehicle
        was actually rerouted this cycle (same convention as every baseline).
        """
        try:
            import traci
        except ImportError:
            return False

        veh_ids = traci.vehicle.getIDList()
        if not veh_ids:
            return False

        any_rerouted = False

        for vid in veh_ids[:MAX_VEHICLES_PER_CYCLE]:
            try:
                route = traci.vehicle.getRoute(vid)
                if not route:
                    continue
                cur_edge, dest_edge = traci.vehicle.getRoadID(vid), route[-1]
                if cur_edge == dest_edge:
                    continue

                cur_uv, dest_uv = self.road.uv_of(cur_edge), self.road.uv_of(dest_edge)
                if cur_uv is None or dest_uv is None:
                    continue
                cur_node = cur_uv[1]

                outgoing_edges = self.road.get_outgoing_edges(cur_edge)
                if not outgoing_edges:
                    continue
                candidate_nodes = {}
                for out_eid in outgoing_edges:
                    out_uv = self.road.uv_of(out_eid)
                    if out_uv and out_uv[1] != cur_uv[0]:
                        candidate_nodes[out_uv[1]] = out_eid
                if not candidate_nodes:
                    continue

                neighbours = sorted(ADJACENCY.get(cur_node, []))
                if not neighbours:
                    continue
                mask = np.zeros(MAX_ACTIONS, dtype=bool)
                for i, nb in enumerate(neighbours[:MAX_ACTIONS]):
                    if nb in candidate_nodes:
                        mask[i] = True
                if not mask.any():
                    continue

                tsv = tsv_map.get(cur_edge)
                tsv_arr = (np.array([tsv.S, tsv.D, float(tsv.Q), tsv.L], dtype=np.float32)
                          if tsv is not None else self.models.norm_mu.copy())
                tsv_norm = (tsv_arr - self.models.norm_mu) / self.models.norm_sig

                state = np.concatenate([
                    self.models.node_embedding(cur_node),
                    self.models.node_embedding(dest_uv[1]),
                    tsv_norm,
                    [0.5],
                ]).astype(np.float32)

                action = self.models.select_action(state, mask)
                next_node = neighbours[action]
                first_edge = candidate_nodes[next_node]

                rest = self.road.dijkstra_edges(first_edge, dest_edge)
                if rest:
                    candidate_route = [cur_edge] + rest
                    if commit_route(vid, candidate_route, road=self.road):
                        any_rerouted = True
            except Exception:
                continue

        return any_rerouted

    # -- Summary + save --------------------------------------------------
    def _print_summary(self, m: SimulationMetrics, wall: float, steps: int):
        print("=" * 70)
        print(f"  Simulation complete in {wall:.1f}s wall time")
        print(f"  Steps            : {steps}")
        print(f"  Avg Speed        : {m.avg_speed_kmh:.2f} km/h")
        print(f"  Avg Queue        : {m.avg_queue_length:.2f} veh")
        print(f"  Mean Latency     : {m.mean_latency_ms:.1f} ms")
        print(f"  CO2 Total        : {m.total_co2_kg:.4f} kg")
        print(f"  Reroute cycles   : {m.reroute_events}")
        print("=" * 70)

    def _save(self, metrics: SimulationMetrics):
        df   = pd.DataFrame(self._records)
        path = self.RESULTS_DIR / "leacer_run_telemetry.csv"
        df.to_csv(path, index=False)
        print(f"  Telemetry saved -> {path}")

        summary = {
            "avg_speed_kmh":    metrics.avg_speed_kmh,
            "avg_queue_length": metrics.avg_queue_length,
            "mean_latency_ms":  metrics.mean_latency_ms,
            "total_co2_kg":     metrics.total_co2_kg,
            "reroute_events":   metrics.reroute_events,
            "total_steps":      metrics.total_steps,
            "completed_trips":  metrics.completed_trips,
        }
        spath = self.RESULTS_DIR / "leacer_run_summary.json"
        with open(spath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved   -> {spath}")
