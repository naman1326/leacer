"""
sumo_data_adapter.py — Converts SUMO EdgeState output into DTSA input formats
==============================================================================
Bridges the gap between raw SUMO telemetry and the LEACER Layer 1 data
structures (VehicleOBU, IoTSensorReading, SignalPhaseData, RSUAggregator).

Also provides a SyntheticSUMODataset class for replay-based offline training
when SUMO is not available.
"""

import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass

from sumo_env import EdgeState, SimulationMetrics

# Import Layer 1 data structures
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from dtsa import (RSUAggregator, VehicleOBU, IoTSensorReading,
                   SignalPhaseData, TrafficStateVector, DTSA)


# ─────────────────────────────────────────────────────────────────────────────
# RSU Coverage Map
# ─────────────────────────────────────────────────────────────────────────────

# Maps each RSU to the set of edges it covers (~500m radius)
RSU_COVERAGE: Dict[str, List[str]] = {
    "RSU_NW": ["E00_01", "E01_00", "E00_10", "E10_00"],
    "RSU_NC": ["E01_02", "E02_01", "E01_11", "E11_01", "E01_00", "E00_01"],
    "RSU_NE": ["E02_12", "E12_02", "E02_01", "E01_02"],
    "RSU_MW": ["E10_11", "E11_10", "E10_20", "E20_10", "E10_00", "E00_10"],
    "RSU_MC": ["E11_12", "E12_11", "E11_21", "E21_11", "E11_10", "E10_11",
               "E11_01", "E01_11"],
    "RSU_ME": ["E12_22", "E22_12", "E12_11", "E11_12", "E12_02", "E02_12"],
    "RSU_SW": ["E20_21", "E21_20", "E20_10", "E10_20"],
    "RSU_SC": ["E21_22", "E22_21", "E21_11", "E11_21", "E21_20", "E20_21"],
    "RSU_SE": ["E22_12", "E12_22", "E22_21", "E21_22"],
}


# ─────────────────────────────────────────────────────────────────────────────
# SUMO → DTSA Adapter
# ─────────────────────────────────────────────────────────────────────────────

class SUMODataAdapter:
    """
    Converts SUMO EdgeState list into RSUAggregator ingestions.

    Each call to adapt() takes the latest step's EdgeState list and
    feeds synthetic OBU/IoT/signal records into the relevant RSU buffers,
    then triggers DTSA to produce TrafficStateVectors.

    OBU synthesis: We approximate OBU packets by sampling 'vehicle_count'
    virtual vehicles uniformly distributed along the edge, each with
    Gaussian speed noise around edge mean_speed.
    """

    def __init__(self,
                 edge_lengths: Dict[str, float],
                 obu_sample_rate: float = 0.6):
        """
        obu_sample_rate: fraction of vehicles that are V2X-equipped [0-1].
        """
        self.edge_lengths    = edge_lengths
        self.obu_sample_rate = obu_sample_rate

        # One RSUAggregator per RSU site
        self.rsus: Dict[str, RSUAggregator] = {
            rsu_id: RSUAggregator(rsu_id) for rsu_id in RSU_COVERAGE
        }

        # DTSA instances per RSU
        self.dtsas: Dict[str, DTSA] = {
            rsu_id: DTSA(edge_lengths, rsu) for rsu_id, rsu in self.rsus.items()
        }

        self._veh_counter = 0

    def adapt(self, edge_states: List[EdgeState]) -> Dict[str, TrafficStateVector]:
        """
        Main entry point. Ingests one simulation step and returns
        {edge_id: TrafficStateVector} for all edges.
        """
        state_map = {s.edge_id: s for s in edge_states}
        t_now = time.time()

        for edge_id, es in state_map.items():
            rsu_ids = self._rsus_for_edge(edge_id)
            for rsu_id in rsu_ids:
                rsu = self.rsus[rsu_id]
                self._inject_iot(rsu, es, t_now)
                self._inject_obu(rsu, es, t_now)
                self._inject_signal(rsu, es, t_now)

        # Compute TSVs from each DTSA, merge (last-write wins per edge)
        tsv_map: Dict[str, TrafficStateVector] = {}
        for rsu_id, dtsa in self.dtsas.items():
            for eid in RSU_COVERAGE[rsu_id]:
                if eid in state_map:
                    tsv = dtsa.compute_tsv(eid)
                    tsv_map[eid] = tsv
        return tsv_map

    def _rsus_for_edge(self, edge_id: str) -> List[str]:
        return [rid for rid, edges in RSU_COVERAGE.items() if edge_id in edges]

    def _inject_iot(self, rsu: RSUAggregator, es: EdgeState, t: float):
        """Synthesize IoT loop detector reading from SUMO EdgeState."""
        rsu.ingest_iot(IoTSensorReading(
            edge_id       = es.edge_id,
            timestamp     = t,
            vehicle_count = es.vehicle_count,
            occupancy_pct = es.occupancy / 100.0,
            avg_speed_kmh = es.mean_speed * 3.6,
            camera_density= es.mean_density,
        ))

    def _inject_obu(self, rsu: RSUAggregator, es: EdgeState, t: float):
        """Synthesize V2X OBU packets for V2X-equipped vehicles."""
        n_equipped = max(1, int(es.vehicle_count * self.obu_sample_rate))
        length_m   = self.edge_lengths.get(es.edge_id, 500.0)
        for i in range(n_equipped):
            self._veh_counter += 1
            noisy_speed = max(0.0, es.mean_speed * 3.6 + np.random.randn() * 2.0)
            rsu.ingest_obu(VehicleOBU(
                vehicle_id  = f"V_{es.edge_id}_{self._veh_counter}",
                timestamp   = t - np.random.uniform(0, 2.0),
                latitude    = 1.3521 + np.random.randn() * 0.001,
                longitude   = 103.82 + np.random.randn() * 0.001,
                speed_kmh   = noisy_speed,
                heading_deg = np.random.uniform(0, 360),
                edge_id     = es.edge_id,
                battery_soc = np.random.uniform(0.3, 1.0),
                v2x_rssi    = np.random.uniform(-80, -55),
            ))

    def _inject_signal(self, rsu: RSUAggregator, es: EdgeState, t: float):
        """Synthesize signal phase data from queue & speed info."""
        congested = es.mean_speed < 5.0 or es.queue_length > 5
        phase = "red" if congested else "green"
        rsu.ingest_signal(SignalPhaseData(
            node_id         = es.edge_id,
            timestamp       = t,
            phase           = phase,
            remaining_sec   = np.random.uniform(5, 45),
            cycle_length_sec= 90.0,
            queue_length_veh= es.queue_length,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic SUMO Dataset — for offline replay / training
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SUMOEpisode:
    """One recorded simulation episode."""
    episode_id:  int
    steps:       int
    edge_ids:    List[str]
    data:        pd.DataFrame    # columns: step, edge_id, + EdgeState fields
    metrics:     SimulationMetrics


class SyntheticSUMODataset:
    """
    Generates or loads a dataset of SUMO simulation episodes for
    offline training of GRU predictor and PPO agent.

    When generated (no real SUMO data available):
      - Uses stochastic macroscopic traffic model
      - Reproduces realistic rush-hour demand patterns
      - Injects random incidents (lane closures, accidents)

    When loading from CSV (real SUMO output):
      - Reads telemetry CSV saved by SUMOEnv.save_telemetry()
    """

    EDGE_IDS = [
        "E00_01","E01_02","E10_11","E11_12","E20_21","E21_22",
        "E00_10","E10_20","E01_11","E11_21","E02_12","E12_22",
    ]
    EDGE_LENGTHS = {e: np.random.uniform(350, 650) for e in EDGE_IDS}

    def __init__(self, n_episodes: int = 50, steps_per_ep: int = 3600,
                 seed: int = 42):
        self.n_episodes   = n_episodes
        self.steps_per_ep = steps_per_ep
        self.rng          = np.random.RandomState(seed)

    # ─────────────────────────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────────────────────────

    def generate(self, save_path: str = None) -> List[SUMOEpisode]:
        """Generate n_episodes synthetic episodes."""
        print(f"[Dataset] Generating {self.n_episodes} episodes "
              f"x {self.steps_per_ep} steps...")
        episodes = []
        for ep in range(self.n_episodes):
            ep_data = self._generate_episode(ep)
            episodes.append(ep_data)
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1}/{self.n_episodes} done")

        if save_path:
            self._save(episodes, save_path)
        return episodes

    def _generate_episode(self, ep_id: int) -> SUMOEpisode:
        """Generate one episode using stochastic traffic model."""
        records = []
        # Randomly pick a day type: weekday (high demand) or weekend
        day_factor = 1.0 if self.rng.rand() > 0.3 else 0.65
        # Random incident: 30% chance, affects 1-2 edges for 10-30 min
        incident_edges, incident_start, incident_end = self._random_incident()

        for t in range(self.steps_per_ep):
            tod = self._time_of_day_factor(t, day_factor)
            for eid in self.EDGE_IDS:
                incident_factor = 0.25 if (
                    eid in incident_edges and incident_start <= t <= incident_end
                ) else 1.0

                k_j, v_f = 120.0, 60.0
                base_k = 20.0 * tod * day_factor * incident_factor
                k = float(np.clip(base_k + self.rng.randn() * 3, 1, k_j))
                v = v_f * max(0, 1 - k / k_j) + self.rng.randn() * 2
                v = float(np.clip(v, 2.0, v_f))

                lkm  = self.EDGE_LENGTHS[eid] / 1000.0
                cnt  = int(k * lkm)
                occ  = min(k / k_j * 100, 100)
                tt   = self.EDGE_LENGTHS[eid] / max(v / 3.6, 0.5)
                q    = max(0, int((k - 40) * lkm)) if k > 40 else 0
                co2  = cnt * 130 * (1 + (v_f - v) / v_f)

                records.append({
                    "episode": ep_id, "step": t, "sim_time": float(t),
                    "edge_id": eid, "vehicle_count": cnt,
                    "mean_speed": v / 3.6, "occupancy": occ,
                    "mean_density": k, "queue_length": q,
                    "travel_time": tt, "co2_mg_s": co2,
                    "throughput": max(0, int(cnt * 0.08 + self.rng.randint(0, 2))),
                    "incident": int(eid in incident_edges and incident_start <= t <= incident_end),
                })

        df = pd.DataFrame(records)
        metrics = SimulationMetrics(
            total_steps      = self.steps_per_ep,
            avg_speed_kmh    = float(df.mean_speed.mean() * 3.6),
            avg_queue_length = float(df.queue_length.mean()),
            mean_latency_ms  = float(df.travel_time.mean() * 1000),
        )
        return SUMOEpisode(ep_id, self.steps_per_ep, self.EDGE_IDS, df, metrics)

    def _random_incident(self) -> Tuple[List[str], int, int]:
        if self.rng.rand() > 0.3:
            return [], -1, -1
        edges = list(self.rng.choice(self.EDGE_IDS, size=self.rng.randint(1, 3), replace=False))
        start = self.rng.randint(300, 2700)
        dur   = self.rng.randint(600, 1800)
        return edges, start, start + dur

    @staticmethod
    def _time_of_day_factor(t_step: int, day_factor: float = 1.0) -> float:
        t_h = (t_step / 3600.0 + 6.0) % 24.0
        am  = np.exp(-0.5 * ((t_h - 8.0) / 1.0) ** 2) * 1.5
        pm  = np.exp(-0.5 * ((t_h - 18.0) / 1.0) ** 2) * 1.3
        return day_factor * (0.4 + am + pm)

    # ─────────────────────────────────────────────────────────────────
    # Sequence extraction for GRU training
    # ─────────────────────────────────────────────────────────────────

    def to_gru_sequences(self, episodes: List[SUMOEpisode],
                          tau: int = 20, horizon: int = 10
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert episodes to (X, Y) training pairs for GRU predictor.

        For each edge, slides a window of length tau over the time series
        to produce:
          X: (N_samples, tau, 4)      — look-back windows [S,D,Q,L]
          Y: (N_samples, horizon, 4)  — forecast targets
        """
        X_list, Y_list = [], []
        features = ["mean_speed", "mean_density", "queue_length", "travel_time"]

        for ep in episodes:
            for eid in self.EDGE_IDS:
                df_e = ep.data[ep.data.edge_id == eid].sort_values("step")
                arr  = df_e[features].values.astype(np.float32)
                T    = len(arr)
                for i in range(T - tau - horizon):
                    X_list.append(arr[i: i + tau])
                    Y_list.append(arr[i + tau: i + tau + horizon])

        X = np.array(X_list)
        Y = np.array(Y_list)
        print(f"[Dataset] GRU sequences: X={X.shape}  Y={Y.shape}")
        return X, Y

    # ─────────────────────────────────────────────────────────────────
    # Load/Save
    # ─────────────────────────────────────────────────────────────────

    def _save(self, episodes: List[SUMOEpisode], path: str):
        all_dfs = [ep.data for ep in episodes]
        df = pd.concat(all_dfs, ignore_index=True)
        df.to_csv(path, index=False)
        print(f"[Dataset] Saved {len(df)} rows → {path}")

    def load_from_csv(self, path: str) -> List[SUMOEpisode]:
        """Load a previously saved dataset CSV."""
        df = pd.read_csv(path)
        episodes = []
        for ep_id in df.episode.unique():
            ep_df = df[df.episode == ep_id].reset_index(drop=True)
            metrics = SimulationMetrics(
                total_steps      = int(ep_df.step.max()),
                avg_speed_kmh    = float(ep_df.mean_speed.mean() * 3.6),
                avg_queue_length = float(ep_df.queue_length.mean()),
            )
            episodes.append(SUMOEpisode(ep_id, int(ep_df.step.max()),
                                         self.EDGE_IDS, ep_df, metrics))
        print(f"[Dataset] Loaded {len(episodes)} episodes from {path}")
        return episodes
