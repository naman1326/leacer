"""
routing_utils.py — Shared SUMO Routing Utilities
==================================================
Provides REAL graph-based routing (Dijkstra, A*) over the actual
LEACER SUMO network via sumolib, plus TraCI route-commit helpers
and a shared telemetry recorder used by EVERY algorithm (LEACER,
classical baselines, DQN, MARL, GCN).

This ensures every algorithm is scored under IDENTICAL conditions:
same network graph, same CO2 model, same telemetry schema — so
differences in results reflect genuine differences in routing
decisions, not fabricated presets.
"""

import os, sys, math
import numpy as np
import networkx as nx
from pathlib import Path

SIM_DIR = Path(__file__).parent

# ── SUMO_HOME / sumolib setup ─────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

try:
    import sumolib
    HAVE_SUMOLIB = True
except ImportError:
    HAVE_SUMOLIB = False
    print("[routing_utils][WARN] sumolib not importable — check SUMO_HOME is set. "
          "Falling back to a synthetic grid graph whose edge IDs will NOT match "
          "your real network, so route commits will fail. Fix SUMO_HOME before "
          "running any baseline.")

# ── Fallback grid (only used if sumolib truly unavailable) ────────────────────
FALLBACK_ADJACENCY = {
    "N00": ["N01","N10"],  "N01": ["N00","N02","N11"],
    "N02": ["N01","N12"],  "N10": ["N00","N11","N20"],
    "N11": ["N01","N10","N12","N21"], "N12": ["N02","N11","N22"],
    "N20": ["N10","N21"],  "N21": ["N11","N20","N22"],
    "N22": ["N12","N21"],
    "IN_W": ["N00","N10","N20"], "IN_N": ["N00","N01","N02"],
    "IN_E": ["N02","N12","N22"], "IN_S": ["N20","N21","N22"],
}


class RoadGraph:
    """
    Wraps the SUMO network as a networkx DiGraph for routing.
    Nodes = SUMO junctions.  Edges = SUMO road edges (id, length, speed).
    """

    def __init__(self, net_file: str = None):
        self.net_file = net_file or str(SIM_DIR / "sumo_cfg" / "leacer_network.net.xml")
        self.G = nx.DiGraph()
        self._edge_lookup = {}     # (u,v) -> sumo edge id
        self._id_to_uv    = {}     # sumo edge id -> (u,v)
        self._net = None

        if HAVE_SUMOLIB and os.path.exists(self.net_file):
            self._load_from_sumolib()
        else:
            self._load_fallback()

    def _load_from_sumolib(self):
        self._net = sumolib.net.readNet(self.net_file)
        for edge in self._net.getEdges():
            if edge.isSpecial():
                continue
            u, v = edge.getFromNode().getID(), edge.getToNode().getID()
            eid, length, speed = edge.getID(), edge.getLength(), edge.getSpeed()
            self.G.add_edge(u, v, id=eid, length=length,
                             speed=speed, weight=length / max(speed, 0.1))
            self._edge_lookup[(u, v)] = eid
            self._id_to_uv[eid] = (u, v)
        print(f"[RoadGraph] Loaded {self.G.number_of_nodes()} junctions, "
              f"{self.G.number_of_edges()} edges from {os.path.basename(self.net_file)}")

    def _load_fallback(self):
        for u, neighbours in FALLBACK_ADJACENCY.items():
            for v in neighbours:
                eid = f"E{u}_{v}"
                self.G.add_edge(u, v, id=eid, length=100.0, speed=13.9, weight=100.0/13.9)
                self._edge_lookup[(u, v)] = eid
                self._id_to_uv[eid] = (u, v)
        print(f"[RoadGraph] FALLBACK grid loaded ({self.G.number_of_nodes()} nodes) — "
              f"route commits will likely fail until sumolib works.")

    def update_weights_from_traci(self):
        """Pull live travel times from TraCI and refresh graph edge weights."""
        try:
            import traci
            for _, _, data in self.G.edges(data=True):
                try:
                    tt = traci.edge.getTraveltime(data["id"])
                    if tt > 0:
                        data["weight"] = tt
                except Exception:
                    pass
        except Exception:
            pass

    def node_path_to_edges(self, node_path):
        edges = []
        for u, v in zip(node_path[:-1], node_path[1:]):
            eid = self._edge_lookup.get((u, v))
            if eid is None:
                return None
            edges.append(eid)
        return edges

    def dijkstra(self, src_node, dst_node):
        try:
            path = nx.dijkstra_path(self.G, src_node, dst_node, weight="weight")
            return self.node_path_to_edges(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def astar(self, src_node, dst_node):
        def heuristic(a, b):
            try:
                ax, ay = self._net.getNode(a).getCoord()
                bx, by = self._net.getNode(b).getCoord()
                return math.hypot(ax - bx, ay - by) / 15.0
            except Exception:
                return 0.0
        try:
            path = nx.astar_path(self.G, src_node, dst_node, heuristic=heuristic, weight="weight")
            return self.node_path_to_edges(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def edge_of(self, u, v):
        return self._edge_lookup.get((u, v))

    def uv_of(self, edge_id):
        return self._id_to_uv.get(edge_id)

    @property
    def nodes(self):
        return list(self.G.nodes)


# ── CO2 model (HBEFA 3.1-style — SAME for every algorithm, for fairness) ─────
KAPPA0, KAPPA1, KAPPA2 = 0.12, 0.08, 0.01

def compute_co2_step(speed_kmh: float, density: float, length_km: float) -> float:
    speed_safe = max(speed_kmh, 1.0)
    energy_kwh = (KAPPA0 * length_km
                  + KAPPA1 * (length_km / speed_safe) ** 2
                  + KAPPA2 * density * length_km)
    return energy_kwh * 0.233   # kg CO2 per kWh, avg EU electricity mix


# ── Route commit ───────────────────────────────────────────────────────────────
def commit_route(veh_id: str, edge_path: list) -> bool:
    if not edge_path:
        return False
    try:
        import traci
        traci.vehicle.setRoute(veh_id, edge_path)
        return True
    except Exception:
        return False


# ── Shared telemetry recorder ─────────────────────────────────────────────────
class TelemetryRecorder:
    """
    Accumulates per-step network telemetry in the SAME schema across every
    algorithm, so results_analyzer.py works unmodified for all of them.
    """
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.records = []
        self._co2_cum = 0.0
        self._reroutes = 0

    def record_step(self, step, sim_time, edge_states, env=None, reroute_triggered=False):
        if not edge_states:
            return
        avg_speed = float(np.mean([s.mean_speed * 3.6 for s in edge_states]))
        avg_queue = float(np.mean([s.queue_length for s in edge_states]))
        avg_lat   = float(np.mean([s.travel_time for s in edge_states]))

        for s in edge_states:
            length_km = 0.15
            if env is not None:
                try:
                    length_km = env.get_edge_length(s.edge_id) / 1000.0
                except Exception:
                    pass
            self._co2_cum += compute_co2_step(s.mean_speed * 3.6, s.mean_density, length_km)

        if reroute_triggered:
            self._reroutes += 1

        self.records.append({
            "step": step, "sim_time": sim_time,
            "avg_speed_kmh":  round(avg_speed, 3),
            "avg_queue":      round(avg_queue, 4),
            "avg_latency_ms": round(avg_lat * 1000, 2),
            "co2_total":      round(self._co2_cum, 6),
            "reroute_events": self._reroutes,
            "algorithm":      self.algorithm,
        })

    def save(self, path):
        import pandas as pd
        df = pd.DataFrame(self.records)
        df.to_csv(path, index=False)
        print(f"[{self.algorithm}] Telemetry saved -> {path}  "
              f"(avg_speed={df.avg_speed_kmh.mean():.2f} km/h, "
              f"avg_queue={df.avg_queue.mean():.3f}, "
              f"co2_final={df.co2_total.iloc[-1]:.4f} kg, "
              f"reroutes={self._reroutes})")
        return df
