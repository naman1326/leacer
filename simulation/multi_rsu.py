"""
multi_rsu.py — Cooperative Multi-RSU Layer (Step 6)
=====================================================
Implements the full cooperative RSU mesh from Section 4.4.2 of the paper:

    T_coop_rn(t) = (1-ω)·T_rn(t) + ω · Σ m_rm(t) / |N(rn)|

Each RSU covers a geographic sub-region of the 3x3 grid.
RSUs broadcast compact state summaries to neighbours and fuse
them using inverse-distance weighting.

Usage (standalone test):
    python multi_rsu.py

Integrated usage (in leacer_runner.py):
    from multi_rsu import RSUMesh, RSUConfig, build_mesh_from_sumo
"""

import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


# ── Grid layout — 4 RSUs cover 9 intersections ────────────────────────────────
#
#  RSU_NW  |  RSU_NE
#  covers  |  covers
#  N00,N01 | N01,N02
#  N10,N11 | N11,N12
#  --------+---------
#  RSU_SW  |  RSU_SE
#  N10,N20 | N12,N22
#  N11,N21 | N21,N22
#
# Edges are mapped to the RSU whose centroid is nearest.

RSU_LAYOUT = {
    "RSU_NW": dict(
        x=0.0, y=1.0,
        nodes=["N00","N01","N10","N11"],
        edges=["E00_01","E01_00","E00_10","E10_00",
               "E01_11","E11_01","E10_11","E11_10"],
    ),
    "RSU_NE": dict(
        x=1.0, y=1.0,
        nodes=["N01","N02","N11","N12"],
        edges=["E01_02","E02_01","E11_12","E12_11",
               "E01_11","E11_01","E02_12","E12_02"],
    ),
    "RSU_SW": dict(
        x=0.0, y=0.0,
        nodes=["N10","N11","N20","N21"],
        edges=["E10_20","E20_10","E11_21","E21_11",
               "E10_11","E11_10","E20_21","E21_20"],
    ),
    "RSU_SE": dict(
        x=1.0, y=0.0,
        nodes=["N11","N12","N21","N22"],
        edges=["E11_12","E12_11","E21_22","E22_21",
               "E12_22","E22_12","E11_21","E21_11"],
    ),
}

# Neighbour pairs (bidirectional mesh links)
RSU_NEIGHBORS = {
    "RSU_NW": ["RSU_NE", "RSU_SW"],
    "RSU_NE": ["RSU_NW", "RSU_SE"],
    "RSU_SW": ["RSU_NW", "RSU_SE"],
    "RSU_SE": ["RSU_NE", "RSU_SW"],
}


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class EdgeTSV:
    """Traffic State Vector for a single edge at a given time."""
    edge_id:     str
    timestamp:   float
    speed_kmh:   float   # S
    density:     float   # D (veh/km)
    queue:       float   # Q (vehicles)
    latency_ms:  float   # L (ms)


@dataclass
class RSUStateBroadcast:
    """
    Compact state summary broadcast by one RSU to its neighbours.
    Eq. (37) from paper:  m_rm(t) = [D̃, Q̃, ρ, T̂(t+1)]
    """
    rsu_id:          str
    timestamp:       float
    avg_density:     float    # D̃  (normalised mean density)
    avg_queue:       float    # Q̃  (normalised mean queue)
    server_util:     float    # ρ   (server utilisation [0,1])
    pred_speed_kmh:  float    # T̂(t+1) — 1-step speed forecast
    pred_latency_ms: float    # T̂(t+1) — 1-step latency forecast
    distance_m:      float = 500.0   # set by receiver


@dataclass
class CoopState:
    """Fused cooperative state for one RSU after receiving peer broadcasts."""
    rsu_id:     str
    timestamp:  float
    speed_kmh:  float    # ω-weighted average of peer + local speed
    density:    float
    queue:      float
    latency_ms: float
    n_peers:    int


# ── Scalar Kalman Filter (per-feature per-edge) ────────────────────────────────
class KF:
    def __init__(self, Q=1.0, R=5.0):
        self.Q = Q; self.R = R; self.P = 10.0; self.x = None

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z; return z
        P_ = self.P + self.Q
        K = P_ / (P_ + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_
        return self.x


# ── Single RSU ────────────────────────────────────────────────────────────────
class RSU:
    """
    One Roadside Unit covering a sub-region of the grid.

    Responsibilities:
      1. Maintain per-edge TSVs using Kalman fusion
      2. Compute 1-step GRU forecast (or EMA fallback)
      3. Build compact broadcast message for mesh peers
      4. Receive peer messages and compute cooperative fused state
    """

    COOP_WEIGHT = 0.3   # ω in paper Eq. (38)
    MAX_PEER_AGE = 30.0  # discard broadcasts older than this (seconds)

    def __init__(self, rsu_id: str, config: dict,
                 gru_model=None, norm_mu=None, norm_sig=None):
        self.id      = rsu_id
        self.edges   = config["edges"]
        self.nodes   = config["nodes"]
        self.x, self.y = config["x"], config["y"]

        # GRU predictor (optional)
        self._gru   = gru_model
        self._mu    = norm_mu  if norm_mu  is not None else np.array([47.,6.,.5,40.])
        self._sig   = norm_sig if norm_sig is not None else np.array([5., 4.,.5,20.])

        # Per-edge state: Kalman filters for each of {speed, density, queue, latency}
        self._kf: Dict[str, List[KF]] = {
            e: [KF() for _ in range(4)] for e in self.edges}

        # TSV history window for GRU (tau=20 steps, 4 features per edge)
        self._history: Dict[str, deque] = {
            e: deque(maxlen=20) for e in self.edges}

        # Current fused TSVs
        self._tsvs: Dict[str, EdgeTSV] = {}

        # Peer message inbox
        self._inbox: Dict[str, RSUStateBroadcast] = {}

        # Server utilisation (simulated as queue pressure)
        self._util = 0.0

    # ── Ingest raw edge state (from TraCI / SUMOEnv) ─────────────────────
    def ingest(self, edge_id: str, speed_ms: float, density: float,
               queue: int, travel_time_s: float):
        """Feed one step of raw edge telemetry into Kalman filters."""
        if edge_id not in self._kf:
            return

        raw = [speed_ms * 3.6, density, float(queue), travel_time_s * 1000]
        fused = [self._kf[edge_id][i].update(raw[i]) for i in range(4)]

        tsv = EdgeTSV(
            edge_id    = edge_id,
            timestamp  = time.time(),
            speed_kmh  = fused[0],
            density    = fused[1],
            queue      = fused[2],
            latency_ms = fused[3],
        )
        self._tsvs[edge_id] = tsv
        self._history[edge_id].append(fused)

        # Update server utilisation proxy
        self._util = min(1.0, np.mean([t.queue for t in self._tsvs.values()]) / 20.0)

    # ── 1-step forecast via GRU or EMA ───────────────────────────────────
    def _forecast(self, edge_id: str) -> Tuple[float, float]:
        """Returns (pred_speed_kmh, pred_latency_ms) for next step."""
        hist = list(self._history.get(edge_id, []))
        if len(hist) < 2:
            tsv = self._tsvs.get(edge_id)
            if tsv:
                return tsv.speed_kmh, tsv.latency_ms
            return 47.0, 40.0

        window = np.array(hist, dtype=np.float32)  # (T, 4)

        if self._gru is not None and len(hist) >= 5:
            try:
                import torch
                T = min(len(hist), 20)
                w = window[-T:]
                w_norm = (w - self._mu) / self._sig
                x = torch.tensor(w_norm[None], dtype=torch.float32)
                with torch.no_grad():
                    pred = self._gru(x)[0, 0].numpy()
                pred_denorm = pred * self._sig + self._mu
                return float(pred_denorm[0]), float(pred_denorm[3])
            except Exception:
                pass

        # EMA fallback
        alpha = 0.3
        lvl = window[0]
        for row in window:
            lvl = alpha * row + (1 - alpha) * lvl
        return float(lvl[0]), float(lvl[3])

    # ── Build broadcast message ───────────────────────────────────────────
    def build_broadcast(self) -> RSUStateBroadcast:
        """Eq. (37) — compact state summary for mesh peers."""
        if not self._tsvs:
            return RSUStateBroadcast(
                self.id, time.time(), 0.0, 0.0, self._util, 47.0, 40.0)

        tsvs = list(self._tsvs.values())
        avg_d = float(np.mean([t.density for t in tsvs]))
        avg_q = float(np.mean([t.queue   for t in tsvs]))

        # Use most-congested edge for forecast
        worst = max(tsvs, key=lambda t: t.queue)
        pred_spd, pred_lat = self._forecast(worst.edge_id)

        # Normalise D̃ and Q̃
        d_norm = min(avg_d / 120.0, 1.0)
        q_norm = min(avg_q / 20.0,  1.0)

        return RSUStateBroadcast(
            rsu_id         = self.id,
            timestamp      = time.time(),
            avg_density    = d_norm,
            avg_queue      = q_norm,
            server_util    = self._util,
            pred_speed_kmh = pred_spd,
            pred_latency_ms= pred_lat,
        )

    # ── Receive peer broadcast ────────────────────────────────────────────
    def receive(self, msg: RSUStateBroadcast, distance_m: float):
        msg.distance_m = distance_m
        self._inbox[msg.rsu_id] = msg

    # ── Compute cooperative fused state Eq. (38) ──────────────────────────
    def fused_state(self) -> CoopState:
        """
        T_coop_rn(t) = (1-ω)·T_local + ω · Σ ω_i·m_i / Σ ω_i
        """
        if not self._tsvs:
            return CoopState(self.id, time.time(), 47.0, 6.0, 0.5, 40.0, 0)

        # Local averages
        tsvs   = list(self._tsvs.values())
        loc_spd = np.mean([t.speed_kmh  for t in tsvs])
        loc_den = np.mean([t.density    for t in tsvs])
        loc_que = np.mean([t.queue      for t in tsvs])
        loc_lat = np.mean([t.latency_ms for t in tsvs])

        # Filter fresh peer messages
        now      = time.time()
        peers    = [m for m in self._inbox.values()
                    if (now - m.timestamp) < self.MAX_PEER_AGE]

        if not peers:
            return CoopState(self.id, now, loc_spd, loc_den, loc_que, loc_lat, 0)

        # Inverse-distance weights  ω_i = (1/d_i) / Σ (1/d_j)
        dists   = np.array([m.distance_m for m in peers], dtype=np.float64)
        inv_d   = 1.0 / np.maximum(dists, 1.0)
        omegas  = inv_d / inv_d.sum()

        # Peer contributions (use pred_speed and normalised queue)
        peer_spd = sum(w * m.pred_speed_kmh  for w, m in zip(omegas, peers))
        peer_den = sum(w * m.avg_density * 120.0 for w, m in zip(omegas, peers))
        peer_que = sum(w * m.avg_queue   * 20.0  for w, m in zip(omegas, peers))
        peer_lat = sum(w * m.pred_latency_ms      for w, m in zip(omegas, peers))

        ω = self.COOP_WEIGHT
        return CoopState(
            rsu_id    = self.id,
            timestamp = now,
            speed_kmh = (1 - ω) * loc_spd + ω * peer_spd,
            density   = (1 - ω) * loc_den + ω * peer_den,
            queue     = (1 - ω) * loc_que + ω * peer_que,
            latency_ms= (1 - ω) * loc_lat + ω * peer_lat,
            n_peers   = len(peers),
        )

    # ── Lyapunov stability check ──────────────────────────────────────────
    def lyapunov_drift(self, prev_queues: np.ndarray,
                       curr_queues: np.ndarray) -> float:
        """ΔV(t) = V(Q(t+1)) - V(Q(t)) = 0.5*(||Q_next||² - ||Q||²)"""
        return 0.5 * (float(np.dot(curr_queues, curr_queues)) -
                      float(np.dot(prev_queues, prev_queues)))

    def queue_array(self) -> np.ndarray:
        return np.array([self._tsvs[e].queue for e in self.edges
                         if e in self._tsvs], dtype=np.float32)

    # ── CEFAR KPI check ───────────────────────────────────────────────────
    def kpi_violation(self, tau_C=0.75, tau_L=150.0, tau_Q=20) -> Optional[str]:
        """Returns violation type string or None."""
        for tsv in self._tsvs.values():
            cong = tsv.density / 120.0
            if cong > tau_C:    return "CONGESTION"
            if tsv.latency_ms > tau_L: return "HIGH_LATENCY"
            if tsv.queue > tau_Q:      return "QUEUE_OVERFLOW"
        return None

    @property
    def mean_speed_kmh(self) -> float:
        if not self._tsvs: return 47.0
        return float(np.mean([t.speed_kmh for t in self._tsvs.values()]))

    @property
    def mean_queue(self) -> float:
        if not self._tsvs: return 0.0
        return float(np.mean([t.queue for t in self._tsvs.values()]))


# ── RSU Mesh ──────────────────────────────────────────────────────────────────
class RSUMesh:
    """
    Manages all 4 RSUs and coordinates broadcast/receive cycles.

    Integration with simulation:
        mesh = RSUMesh(gru_model, norm_mu, norm_sig)
        for each SUMO step:
            mesh.ingest_from_sumo(edge_states)
            mesh.broadcast_cycle()
            coop = mesh.cooperative_states()
    """

    def __init__(self, gru_model=None, norm_mu=None, norm_sig=None):
        self.rsus: Dict[str, RSU] = {
            rid: RSU(rid, cfg, gru_model, norm_mu, norm_sig)
            for rid, cfg in RSU_LAYOUT.items()
        }
        # Pre-compute inter-RSU distances
        self._distances: Dict[Tuple[str,str], float] = {}
        for r1 in self.rsus:
            for r2 in self.rsus:
                if r1 != r2:
                    dx = RSU_LAYOUT[r1]["x"] - RSU_LAYOUT[r2]["x"]
                    dy = RSU_LAYOUT[r1]["y"] - RSU_LAYOUT[r2]["y"]
                    self._distances[(r1,r2)] = math.sqrt(dx**2 + dy**2) * 1000.0  # m

        # Edge → RSU mapping (each edge assigned to nearest RSU centroid)
        self._edge_to_rsu: Dict[str, str] = {}
        for rid, cfg in RSU_LAYOUT.items():
            for eid in cfg["edges"]:
                self._edge_to_rsu[eid] = rid

    # ── Ingest from SUMO EdgeState objects ────────────────────────────────
    def ingest_from_sumo(self, edge_states):
        """Feed all EdgeState objects from SUMOEnv into the correct RSU."""
        for s in edge_states:
            rsu_id = self._edge_to_rsu.get(s.edge_id)
            if rsu_id and rsu_id in self.rsus:
                self.rsus[rsu_id].ingest(
                    s.edge_id, s.mean_speed, s.mean_density,
                    s.queue_length, s.travel_time)

    # ── Broadcast cycle ────────────────────────────────────────────────────
    def broadcast_cycle(self):
        """Each RSU builds its message and sends to all neighbours."""
        messages = {rid: rsu.build_broadcast()
                    for rid, rsu in self.rsus.items()}
        for sender_id, msg in messages.items():
            for receiver_id in RSU_NEIGHBORS.get(sender_id, []):
                d = self._distances.get((receiver_id, sender_id), 500.0)
                self.rsus[receiver_id].receive(msg, d)

    # ── Get cooperative states ─────────────────────────────────────────────
    def cooperative_states(self) -> Dict[str, CoopState]:
        return {rid: rsu.fused_state() for rid, rsu in self.rsus.items()}

    # ── Global Lyapunov check ──────────────────────────────────────────────
    def global_lyapunov(self) -> float:
        """V(Q) = 0.5 * Σ Q_i² across all RSUs."""
        queues = []
        for rsu in self.rsus.values():
            queues.extend(rsu.queue_array().tolist())
        q = np.array(queues)
        return float(0.5 * np.dot(q, q))

    # ── Summary ───────────────────────────────────────────────────────────
    def summary(self) -> dict:
        states = self.cooperative_states()
        return {
            "global_lyapunov":  self.global_lyapunov(),
            "rsus": {rid: {
                "speed_kmh":  round(s.speed_kmh, 2),
                "queue":      round(s.queue, 2),
                "latency_ms": round(s.latency_ms, 2),
                "n_peers":    s.n_peers,
            } for rid, s in states.items()}
        }


# ── Factory: build mesh with loaded GRU model ─────────────────────────────────
def build_mesh_from_sumo(data_dir) -> RSUMesh:
    """Load GRU weights and build RSUMesh ready to ingest from SUMO."""
    gru_model = None
    mu = sig = None

    try:
        import torch, torch.nn as nn
        gru_path = Path(data_dir) / "gru_weights.pt"
        if gru_path.exists():
            ckpt = torch.load(gru_path, map_location="cpu")
            cfg  = ckpt["model_config"]

            class GRU(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gru  = nn.GRU(cfg["feat"], cfg["hidden"], cfg["layers"],
                                        batch_first=True)
                    self.head = nn.Sequential(
                        nn.Linear(cfg["hidden"], cfg["hidden"]), nn.ReLU(),
                        nn.Linear(cfg["hidden"], cfg["horizon"] * cfg["feat"]))
                    self.h, self.f = cfg["horizon"], cfg["feat"]

                def forward(self, x):
                    _, h = self.gru(x)
                    return self.head(h[-1]).view(x.size(0), self.h, self.f)

            m = GRU()
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            gru_model = m
            mu  = np.array(ckpt["norm_mu"],  dtype=np.float32)
            sig = np.array(ckpt["norm_sig"], dtype=np.float32)
            print("[MultiRSU] GRU model loaded for cooperative prediction")
    except Exception as e:
        print(f"[MultiRSU] Using EMA fallback ({e})")

    return RSUMesh(gru_model, mu, sig)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-RSU Cooperative Layer — Smoke Test")
    print("=" * 60)

    mesh = RSUMesh()   # no GRU, uses EMA

    # Simulate 10 steps of synthetic edge data
    edges_all = [e for cfg in RSU_LAYOUT.values() for e in cfg["edges"]]
    for step in range(10):
        for eid in edges_all:
            # Synthetic raw values (speed_ms, density, queue, travel_time_s)
            mesh.rsus.get(
                list(mesh._edge_to_rsu.get(eid, "RSU_NW") and
                     [mesh._edge_to_rsu[eid]])[0],
                list(mesh.rsus.values())[0]
            ).ingest(eid,
                     speed_ms=np.random.uniform(8, 14),
                     density=np.random.uniform(5, 40),
                     queue=np.random.randint(0, 10),
                     travel_time_s=np.random.uniform(20, 80))

        mesh.broadcast_cycle()

    states = mesh.cooperative_states()
    print(f"\n{'RSU':12s}  {'Speed':>10}  {'Queue':>8}  {'Latency':>10}  {'Peers':>6}")
    print("-" * 52)
    for rid, s in states.items():
        print(f"{rid:12s}  {s.speed_kmh:>10.2f}  {s.queue:>8.2f}  "
              f"{s.latency_ms:>10.2f}  {s.n_peers:>6}")

    print(f"\nGlobal Lyapunov V(Q) = {mesh.global_lyapunov():.4f}")
    print("\n✅ Multi-RSU cooperative layer OK")
