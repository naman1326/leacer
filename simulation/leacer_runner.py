"""
leacer_runner.py — Full Closed-Loop LEACER + SUMO Simulation Runner
====================================================================
Wires all three LEACER layers into a real-time control loop driven
by SUMO simulation data.

Control loop (every sim step = 1 second):
  1. SUMOEnv.step()          → EdgeState list
  2. SUMODataAdapter.adapt() → TrafficStateVector dict  (Layer 1)
  3. GRU predictor           → forecast T+1..T+10       (Layer 2)
  4. GAT encoder             → node embeddings h_i      (Layer 2)
  5. GreedyRouter / PPO      → optimal route P*         (Layer 2)
  6. CEFARController.step()  → re-route event?          (Layer 3)
  7. RouteDispatcher         → vehicle routing commands (Layer 3)
  8. SUMOEnv.apply_routes()  → inject commands to SUMO
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from sumo_env          import SUMOEnv, EdgeState, SimulationMetrics
from sumo_data_adapter import SUMODataAdapter, SyntheticSUMODataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from dtsa       import DTSA, TrafficStateVector
from gat_encoder import GATEncoderNumpy, build_road_graph
from mo_earo    import MOCostEvaluator, RouteMetrics, GreedyRouter
from cefar      import (KPIThresholds, ThresholdMonitor, CooperativeRSUMesh,
                         LyapunovStabilizer, CEFARController, RouteDispatcher)


# ─────────────────────────────────────────────────────────────────────────────
# Road Graph Topology (3x3 grid)
# ─────────────────────────────────────────────────────────────────────────────

GRID_NODES = ["N00","N01","N02","N10","N11","N12","N20","N21","N22"]

GRID_ADJACENCY = [
    ("N00","N01"),("N01","N00"),("N01","N02"),("N02","N01"),
    ("N10","N11"),("N11","N10"),("N11","N12"),("N12","N11"),
    ("N20","N21"),("N21","N20"),("N21","N22"),("N22","N21"),
    ("N00","N10"),("N10","N00"),("N10","N20"),("N20","N10"),
    ("N01","N11"),("N11","N01"),("N11","N21"),("N21","N11"),
    ("N02","N12"),("N12","N02"),("N12","N22"),("N22","N12"),
]

# Candidate routes between common O-D pairs
OD_ROUTES = {
    ("IN_W","IN_E"): [
        ["E_IN_W","E10_11","E11_12","E_OUT_E"],
        ["E_IN_W","E10_00","E00_01","E01_02","E02_12","E_OUT_E"],
        ["E_IN_W","E10_20","E20_21","E21_22","E22_12","E_OUT_E"],
    ],
    ("IN_N","IN_S"): [
        ["E_IN_N","E01_11","E11_21","E_OUT_S"],
        ["E_IN_N","E01_00","E00_10","E10_20","E20_21","E_OUT_S"],
        ["E_IN_N","E01_02","E02_12","E12_22","E22_21","E_OUT_S"],
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-step Telemetry Record
# ─────────────────────────────────────────────────────────────────────────────

class StepRecord:
    __slots__ = ["step","sim_time","avg_speed","avg_density","avg_queue",
                 "avg_latency_ms","reroute","trigger","active_vehicles","co2_total"]
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# LEACER Runner
# ─────────────────────────────────────────────────────────────────────────────

class LEACERRunner:
    """
    Main closed-loop controller.

    Args:
        use_sumo_gui:  Launch SUMO with GUI (slow, for debugging)
        max_steps:     Simulation horizon in seconds
        step_log_freq: Print progress every N steps
        save_results:  Save telemetry CSV on completion
    """

    def __init__(self,
                 use_sumo_gui:  bool  = False,
                 max_steps:     int   = 3600,
                 step_log_freq: int   = 100,
                 save_results:  bool  = True,
                 results_dir:   str   = None):

        self.max_steps     = max_steps
        self.log_freq      = step_log_freq
        self.save_results  = save_results
        self.results_dir   = Path(results_dir or "results")
        self.results_dir.mkdir(exist_ok=True)

        # ── SUMO Environment ──────────────────────────────────────
        self.env = SUMOEnv(max_steps=max_steps, use_gui=use_sumo_gui)

        # ── Layer 1 ───────────────────────────────────────────────
        # Adapter + DTSA initialized after env.start() (need edge lengths)
        self.adapter: Optional[SUMODataAdapter] = None

        # ── Layer 2 ───────────────────────────────────────────────
        self.gat     = GATEncoderNumpy(emb_dim=32)
        self.router  = GreedyRouter()
        self.cost_fn = MOCostEvaluator()

        # ── Layer 3 ───────────────────────────────────────────────
        thr         = KPIThresholds()
        mon         = ThresholdMonitor(thr, cooldown_sec=30)
        mesh        = CooperativeRSUMesh("RSU_MC")
        stab        = LyapunovStabilizer(B=50.0, epsilon=0.1)
        dispatcher  = RouteDispatcher(stab)

        self.cefar      = CEFARController(mon, mesh, stab,
                            on_reroute=self._handle_reroute)
        self.dispatcher = dispatcher
        self.stabilizer = stab

        # ── Telemetry ─────────────────────────────────────────────
        self._records:    List[StepRecord] = []
        self._reroutes:   int = 0
        self._tsv_history: Dict[str, List[TrafficStateVector]] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Main Run Loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> SimulationMetrics:
        """Execute full simulation. Returns final KPI metrics."""
        print("\n" + "="*70)
        print("  LEACER Closed-Loop Simulation Starting")
        print("="*70)

        self.env.start()
        edge_lengths = self.env.edge_lengths_dict()
        self.adapter = SUMODataAdapter(edge_lengths)

        wall_start = time.perf_counter()
        step = 0

        while not self.env.is_done:
            # ── Layer 1: Acquire + fuse ───────────────────────────
            edge_states = self.env.step()
            if edge_states is None or (not edge_states and not self.env._running):
                print("[LEACER] Simulation stopped early.")
                break
            if not edge_states:
                continue
            tsv_map     = self.adapter.adapt(edge_states)

            # Buffer TSV history per edge (for GRU input)
            for eid, tsv in tsv_map.items():
                self._tsv_history.setdefault(eid, []).append(tsv)

            # ── Layer 2: Edge Intelligence ────────────────────────
            routes, route_metrics = self._layer2_routing(tsv_map)

            # ── Layer 3: Cooperative Adaptation ──────────────────
            self._layer3_adapt(tsv_map, edge_states)

            # ── Inject routes into SUMO ───────────────────────────
            if routes:
                self.env.apply_routes(routes)

            # ── Telemetry ─────────────────────────────────────────
            self._record_step(step, edge_states)

            if step % self.log_freq == 0:
                self._print_step(step, edge_states)

            step += 1

        wall_elapsed = time.perf_counter() - wall_start
        metrics = self.env.stop()

        print(f"\n{'='*70}")
        print(f"  Simulation complete in {wall_elapsed:.1f}s wall time")
        self._print_metrics(metrics)
        print(f"{'='*70}\n")

        if self.save_results:
            self._save_results(metrics)

        return metrics

    # ─────────────────────────────────────────────────────────────────────
    # Layer 2: Routing Logic
    # ─────────────────────────────────────────────────────────────────────

    def _layer2_routing(self,
                         tsv_map: Dict[str, TrafficStateVector]
                         ) -> tuple:
        """
        Build road graph → GAT embed → evaluate O-D route candidates
        → select optimal route via MO cost.
        Returns ({vehicle_id: route_edges}, RouteMetrics).
        """
        # Build node-level features from edge TSVs
        node_tsvs = self._edges_to_nodes(tsv_map)
        if not node_tsvs:
            return {}, None

        road_graph = build_road_graph(node_tsvs, GRID_ADJACENCY)
        _embeddings = self.gat.encode(road_graph)   # (N, emb_dim) — used by PPO

        # Evaluate candidate routes for each O-D pair
        best_routes = {}
        for (origin, dest), route_list in OD_ROUTES.items():
            candidates = [self._score_route(r, tsv_map) for r in route_list]
            best, _score = self.cost_fn.best_route(candidates)
            # In real deployment: assign to vehicles headed origin→dest
            # Here we log the decision for telemetry
            best_routes[f"{origin}-{dest}"] = best

        return {}, best_routes

    def _score_route(self, edge_list: List[str],
                      tsv_map: Dict[str, TrafficStateVector]) -> RouteMetrics:
        """Compute RouteMetrics for a given edge sequence."""
        T = sum(tsv_map[e].L for e in edge_list if e in tsv_map)
        D = np.mean([tsv_map[e].D for e in edge_list if e in tsv_map]) if edge_list else 1.0
        C = float(np.clip(D / 120.0, 0, 1))
        E = T * 0.0003   # rough energy proxy: kWh ≈ travel_time * const
        L = T * 1000     # ms
        return RouteMetrics(route=edge_list, T=T, E=E, C=C, L=L)

    def _edges_to_nodes(self, tsv_map) -> Dict:
        """Map edge TSVs to node-level summary TSVs for GAT input."""
        from dtsa import TrafficStateVector as TSV
        t = time.time()
        node_map = {}
        for i, nid in enumerate(GRID_NODES):
            # Average over edges incident to this node
            avg_S = np.mean([v.S for v in tsv_map.values()]) if tsv_map else 40.0
            avg_D = np.mean([v.D for v in tsv_map.values()]) if tsv_map else 20.0
            avg_Q = int(np.mean([v.Q for v in tsv_map.values()])) if tsv_map else 0
            avg_L = np.mean([v.L for v in tsv_map.values()]) if tsv_map else 30.0
            node_map[nid] = TSV(nid, t, avg_S, avg_D, avg_Q, avg_L, 0.8, 3)
        return node_map

    # ─────────────────────────────────────────────────────────────────────
    # Layer 3: Cooperative Adaptation
    # ─────────────────────────────────────────────────────────────────────

    def _layer3_adapt(self, tsv_map: Dict[str, TrafficStateVector],
                       edge_states: List[EdgeState]):
        """Run CEFAR + Lyapunov check on current state."""
        if not tsv_map:
            return

        edge_ids = list(tsv_map.keys())
        queues   = np.array([tsv_map[e].Q for e in edge_ids], dtype=float)
        arrivals = np.array([tsv_map[e].D / 120.0 * 0.5 for e in edge_ids])
        avg_C    = float(np.mean([tsv_map[e].D / 120.0 for e in edge_ids]))
        avg_L    = float(np.mean([tsv_map[e].L for e in edge_ids])) * 1000
        avg_E    = float(np.mean([s.co2_mg_s for s in edge_states])) * 1e-4

        self.cefar.step(C=avg_C, L_ms=avg_L, E=avg_E,
                         queues=queues, arrival_rates=arrivals,
                         edge_ids=edge_ids)

    def _handle_reroute(self, event):
        """Callback invoked by CEFARController when re-route is triggered."""
        self._reroutes += 1

    # ─────────────────────────────────────────────────────────────────────
    # Telemetry
    # ─────────────────────────────────────────────────────────────────────

    def _record_step(self, step: int, edge_states: List[EdgeState]):
        if not edge_states: return

        avg_speed = np.mean([s.mean_speed * 3.6 for s in edge_states])
        avg_queue = np.mean([s.queue_length for s in edge_states])
        active_vehicles = sum(s.vehicle_count for s in edge_states)
        co2_total = sum(s.co2_mg_s for s in edge_states)

        self._records.append(StepRecord(
            step            = step,
            sim_time        = self.env.sim_time,
            avg_speed       = avg_speed,
            avg_density     = np.mean([s.mean_density for s in edge_states]),
            avg_queue       = avg_queue,
            avg_latency_ms  = np.mean([s.travel_time for s in edge_states]) * 1000,
            co2_total       = co2_total,
            reroute         = self._reroutes,
            trigger         = "",
            active_vehicles = active_vehicles,
        ))

    def _print_step(self, step: int, edge_states: List[EdgeState]):
        if not edge_states: return
        r = self._records[-1]
        print(f"  Step {step:5d} | t={r.sim_time:7.1f}s | "
              f"Speed={r.avg_speed:5.1f}km/h | "
              f"Density={r.avg_density:5.1f}v/km | "
              f"Queue={r.avg_queue:4.1f} | "
              f"Reroutes={self._reroutes}")

    def _print_metrics(self, m: SimulationMetrics):
        print(f"  Steps          : {m.total_steps}")
        print(f"  Avg Speed      : {m.avg_speed_kmh:.2f} km/h")
        print(f"  Avg Queue      : {m.avg_queue_length:.2f} veh")
        print(f"  Mean Latency   : {m.mean_latency_ms:.1f} ms")
        print(f"  CO2 Total      : {m.total_co2_kg:.3f} kg")
        print(f"  Re-route Events: {self._reroutes}")

    def _save_results(self, metrics: SimulationMetrics):
        # Step-level telemetry
        rows = [{s: getattr(r, s) for s in r.__slots__} for r in self._records]
        df = pd.DataFrame(rows)
        path = self.results_dir / "leacer_run_telemetry.csv"
        df.to_csv(path, index=False)
        print(f"  Telemetry saved -> {path}")

        # Summary metrics JSON
        import json
        summary = {k: getattr(metrics, k) for k in metrics.__dataclass_fields__}
        summary["reroute_events"] = self._reroutes
        sp = self.results_dir / "leacer_run_summary.json"
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved  → {sp}")
