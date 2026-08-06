"""
sumo_env.py — SUMO Simulation Environment for LEACER Framework
==============================================================
Wraps the TraCI API to:
  - Launch / step / reset a SUMO simulation
  - Extract per-edge traffic state → feeds DTSA Layer 1
  - Inject LEACER routing decisions back into SUMO vehicles
  - Record per-step telemetry for results analysis

Dependencies:
  pip install traci sumolib numpy pandas

SUMO must be installed: https://sumo.dlr.de/docs/Installing/
Set SUMO_HOME environment variable before running.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ── TraCI import (graceful fallback for environments without SUMO) ──────────
try:
    import traci
    import traci.constants as tc
    import sumolib
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("[WARN] traci not found. SUMOEnv will run in MOCK mode.")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SUMO_CFG_DIR  = Path(__file__).parent / "sumo_cfg"
RESULTS_DIR   = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_CFG   = SUMO_CFG_DIR / "leacer.sumocfg"
SUMO_BINARY   = os.path.join(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo"), "bin", "sumo.exe")
SUMO_GUI      = os.path.join(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo"), "bin", "sumo-gui.exe")

# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeState:
    """
    Per-edge traffic state extracted from SUMO via TraCI each time step.
    Directly maps to TrafficStateVector fields used by DTSA.
    """
    edge_id:        str
    step:           int
    sim_time:       float
    vehicle_count:  int           # vehicles currently on edge
    mean_speed:     float         # m/s
    occupancy:      float         # % road space occupied [0-100]
    mean_density:   float         # veh/km (computed)
    queue_length:   int           # vehicles waiting (speed < 0.1 m/s)
    travel_time:    float         # seconds (SUMO edge attribute)
    co2_mg_s:       float         # CO2 emissions mg/s (proxy for energy)
    noise_db:       float         # noise pollution dB
    throughput:     int           # vehicles that left edge this step


@dataclass
class VehicleState:
    """Per-vehicle state extracted from SUMO."""
    vehicle_id:   str
    step:         int
    edge_id:      str
    lane_id:      str
    position:     float           # metres along edge
    speed:        float           # m/s
    acceleration: float           # m/s^2
    co2_mg_s:     float
    fuel_ml_s:    float
    waiting_time: float           # cumulative seconds at speed~0
    route_id:     str
    origin:       str
    dest:         str


@dataclass
class SimulationMetrics:
    """Aggregated KPIs for the full simulation run."""
    total_steps:        int   = 0
    total_vehicles:     int   = 0
    completed_trips:    int   = 0
    avg_travel_time:    float = 0.0
    avg_speed_kmh:      float = 0.0
    total_co2_kg:       float = 0.0
    total_fuel_L:       float = 0.0
    avg_queue_length:   float = 0.0
    reroute_events:     int   = 0
    mean_latency_ms:    float = 0.0
    throughput_veh_h:   float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SUMO Environment
# ─────────────────────────────────────────────────────────────────────────────

class SUMOEnv:
    """
    OpenAI-Gym-style SUMO environment wrapper for LEACER.

    Usage:
        env = SUMOEnv(cfg_path="sumo_cfg/leacer.sumocfg", use_gui=False)
        env.start()
        for step in range(3600):
            edge_states = env.step()
            # feed edge_states → DTSA → Layer 2 → Layer 3
            env.apply_routes(vehicle_routes)
        metrics = env.stop()
    """

    # TraCI subscriptions — what we pull every step per edge
    EDGE_SUBS = [
        tc.LAST_STEP_VEHICLE_NUMBER,
        tc.LAST_STEP_MEAN_SPEED,
        tc.LAST_STEP_OCCUPANCY,
        tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
        tc.VAR_CURRENT_TRAVELTIME,
    ]

    def __init__(self,
                 cfg_path:    str   = None,
                 use_gui:     bool  = False,
                 step_length: float = 1.0,
                 seed:        int   = 42,
                 max_steps:   int   = 3600):

        self.cfg_path    = cfg_path or str(DEFAULT_CFG)
        self.use_gui     = use_gui
        self.step_length = step_length
        self.seed        = seed
        self.max_steps   = max_steps

        self._step       = 0
        self._running    = False
        self._edge_ids:  List[str] = []
        self._veh_ids:   List[str] = []

        # Telemetry buffers
        self._edge_history: List[List[EdgeState]] = []
        self._veh_history:  List[VehicleState]    = []
        self._step_times:   List[float]            = []

        # Completed trip tracking
        self._departed:  Dict[str, float] = {}   # veh_id → departure sim_time
        self._arrived:   List[Tuple[str, float, float]] = []  # (id, dep, arr)

        print(f"[SUMOEnv] Initialized  cfg={self.cfg_path}  gui={use_gui}")

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def start(self):
        """Launch SUMO (with or without GUI) and initialise subscriptions."""
        if not TRACI_AVAILABLE:
            self._running = True
            self._init_mock_network()
            return

        binary = SUMO_GUI if self.use_gui else SUMO_BINARY

        cmd = [
            binary,
            "-c", str(self.cfg_path),
            "--no-step-log",
            "--waiting-time-memory", "100",
        ]

        if self.use_gui:
            cmd.extend(["--start", "true", "--quit-on-end", "true"])

        traci.start(cmd)

        self._running  = True
        # Filter out internal junction edges (start with ":")
        self._edge_ids = [e for e in traci.edge.getIDList()
                          if not e.startswith(":")]

        # Subscribe to per-edge data so every step is O(1) per edge
        for eid in self._edge_ids:
            traci.edge.subscribe(eid, self.EDGE_SUBS)

        print(f"[SUMOEnv] Started. Edges: {len(self._edge_ids)}")

    def _init_mock_network(self):
        """Initialize synthetic 3x3 grid for mock mode (no SUMO installed)."""
        self._edge_ids = [
            "E00_01","E01_02","E10_11","E11_12","E20_21","E21_22",
            "E00_10","E10_20","E01_11","E11_21","E02_12","E12_22",
            "E01_00","E02_01","E11_10","E12_11","E21_20","E22_21",
            "E10_00","E20_10","E11_01","E21_11","E12_02","E22_12",
        ]
        self._edge_lengths = {e: np.random.uniform(400, 600) for e in self._edge_ids}

    def stop(self) -> SimulationMetrics:
        """Close TraCI connection and compute final metrics."""
        if TRACI_AVAILABLE and self._running:
            try:
                traci.close()
            except Exception:
                pass
        self._running = False
        return self._compute_metrics()

    def reset(self) -> List[EdgeState]:
        """Reset simulation to t=0."""
        if self._running:
            self.stop()
        self._step = 0
        self._edge_history.clear()
        self._veh_history.clear()
        self._step_times.clear()
        self._departed.clear()
        self._arrived.clear()
        self.start()
        return self.get_edge_states()

    # ─────────────────────────────────────────────────────────────────────
    # Stepping
    # ─────────────────────────────────────────────────────────────────────

    def step(self) -> List[EdgeState]:
        if not self._running:
            raise RuntimeError("Call env.start() first")

        t0 = time.perf_counter()

        if TRACI_AVAILABLE:
            try:
                traci.simulationStep()
                self._track_vehicles()
            except Exception as e:
                # If TraCI connection is lost (simulation ended), stop and break.
                # However, if we're below max_steps, we might want to continue in mock mode, 
                # but TraCI being gone usually means the binary closed.
                # Let's signal and stop.
                print(f"\n[SUMOEnv] TraCI stopped (Step {self._step}): {e}")
                self._running = False
                return None
        else:
            self._mock_step()

        self._step += 1
        self._step_times.append(time.perf_counter() - t0)

        states = self.get_edge_states()
        self._edge_history.append(states)
        return states

    def _track_vehicles(self):
        """Track vehicle departures and arrivals for trip time computation."""
        sim_time = traci.simulation.getTime()
        for vid in traci.simulation.getDepartedIDList():
            self._departed[vid] = sim_time
        for vid in traci.simulation.getArrivedIDList():
            dep = self._departed.pop(vid, sim_time)
            self._arrived.append((vid, dep, sim_time))

    # ─────────────────────────────────────────────────────────────────────
    # State Extraction
    # ─────────────────────────────────────────────────────────────────────

    def get_edge_states(self) -> List[EdgeState]:
        """Extract EdgeState for every edge in the network."""
        if TRACI_AVAILABLE:
            return self._get_edge_states_traci()
        else:
            return self._get_edge_states_mock()

    def _get_edge_states_traci(self) -> List[EdgeState]:
        """Pull subscribed values from TraCI."""
        sim_time = traci.simulation.getTime()
        states = []
        for eid in self._edge_ids:
            sub = traci.edge.getSubscriptionResults(eid)
            if sub is None:
                continue

            count    = sub.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
            speed    = sub.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
            occ      = sub.get(tc.LAST_STEP_OCCUPANCY, 0.0)
            halting  = sub.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            tt       = sub.get(tc.VAR_CURRENT_TRAVELTIME, 0.0)
            co2      = traci.edge.getCO2Emission(eid)
            noise    = traci.edge.getNoiseEmission(eid)

            try:
                length_km = traci.lane.getLength(eid + "_0") / 1000.0
            except Exception:
                length_km = 0.5
            density    = count / max(length_km, 0.01)
            # throughput = traci.edge.getLastStepVehicleIDs(eid) # This can be very large output, better use count if needed, but let's keep it if small
            
            # Using count as proxy if getLastStepVehicleIDs is not subscribed
            tp_ids = traci.edge.getLastStepVehicleIDs(eid)
            tp_count = len(tp_ids)

            states.append(EdgeState(
                edge_id=eid, step=self._step, sim_time=sim_time,
                vehicle_count=count, mean_speed=speed,
                occupancy=occ, mean_density=density,
                queue_length=halting, travel_time=tt,
                co2_mg_s=co2, noise_db=noise,
                throughput=tp_count
            ))
        return states

    def _mock_step(self):
        """Advance mock simulation using stochastic traffic model."""
        pass  # state generated in _get_edge_states_mock

    def _get_edge_states_mock(self) -> List[EdgeState]:
        """
        Generate synthetic EdgeState using a stochastic macroscopic model.
        Greenshields speed-density:  v(k) = v_f * (1 - k/k_j)
        """
        sim_time = float(self._step * self.step_length)
        tod_factor = self._time_of_day_factor(sim_time)
        states = []

        for eid in self._edge_ids:
            base_density = 15.0 * tod_factor + np.random.randn() * 2.0
            density = float(np.clip(base_density, 0.5, 110.0))

            v_f, k_j = 60.0, 120.0
            speed_kmh = v_f * (1.0 - density / k_j)
            speed_kmh = float(np.clip(speed_kmh + np.random.randn() * 2, 5.0, v_f))
            speed_ms  = speed_kmh / 3.6

            length_m   = self._edge_lengths.get(eid, 500.0)
            length_km  = length_m / 1000.0
            count      = int(density * length_km)
            occupancy  = min(density / k_j * 100, 100.0)
            travel_time = length_m / max(speed_ms, 0.5)
            queue      = max(0, int((density - 40) * length_km)) if density > 40 else 0
            co2        = count * 130.0 * (1.0 + (v_f - speed_kmh) / v_f)
            throughput = max(0, int(count * 0.1 + np.random.randint(0, 3)))

            states.append(EdgeState(
                edge_id=eid, step=self._step, sim_time=sim_time,
                vehicle_count=count, mean_speed=speed_ms,
                occupancy=occupancy, mean_density=density,
                queue_length=queue, travel_time=travel_time,
                co2_mg_s=co2, noise_db=65.0 + density * 0.1,
                throughput=throughput
            ))
        return states

    @staticmethod
    def _time_of_day_factor(sim_time_s: float) -> float:
        """
        Returns a scaling factor [0.3, 4.0] based on time of day.
        Simulates morning (8AM) and evening (6PM) peaks with added randomness.
        """
        t_h = (sim_time_s / 3600.0 + 6.0) % 24.0
        # Aggressive peaks
        am   = np.exp(-0.5 * ((t_h - 8.0)  / 1.2) ** 2) * 3.0
        pm   = np.exp(-0.5 * ((t_h - 18.0) / 1.2) ** 2) * 2.5
        
        # Add a low-frequency oscillation for more natural curve variance
        osc = 0.2 * np.sin(2 * np.pi * t_h / 24.0)
        
        # Add some noise
        noise = np.random.normal(0, 0.05)
        
        return max(0.2, 0.3 + am + pm + osc + noise)

    # ─────────────────────────────────────────────────────────────────────
    # Vehicle Control
    # ─────────────────────────────────────────────────────────────────────

    def get_vehicle_states(self) -> List[VehicleState]:
        """Extract state of all active vehicles (TraCI only)."""
        if not TRACI_AVAILABLE:
            return []
        states = []
        sim_time = traci.simulation.getTime()
        for vid in traci.vehicle.getIDList():
            try:
                states.append(VehicleState(
                    vehicle_id   = vid,
                    step         = self._step,
                    edge_id      = traci.vehicle.getRoadID(vid),
                    lane_id      = traci.vehicle.getLaneID(vid),
                    position     = traci.vehicle.getLanePosition(vid),
                    speed        = traci.vehicle.getSpeed(vid),
                    acceleration = traci.vehicle.getAcceleration(vid),
                    co2_mg_s     = traci.vehicle.getCO2Emission(vid),
                    fuel_ml_s    = traci.vehicle.getFuelConsumption(vid),
                    waiting_time = traci.vehicle.getAccumulatedWaitingTime(vid),
                    route_id     = traci.vehicle.getRouteID(vid),
                    origin       = traci.vehicle.getRoute(vid)[0],
                    dest         = traci.vehicle.getRoute(vid)[-1],
                ))
            except traci.TraCIException:
                continue
        return states

    def apply_routes(self, vehicle_routes: Dict[str, List[str]]):
        """
        Inject LEACER routing decisions into SUMO vehicles.
        vehicle_routes: {vehicle_id: [edge_id, edge_id, ...]}
        """
        if not TRACI_AVAILABLE:
            return
        for vid, route in vehicle_routes.items():
            try:
                traci.vehicle.setRoute(vid, route)
            except traci.TraCIException as e:
                print(f"[WARN] Could not set route for {vid}: {e}")

    def set_traffic_light(self, tl_id: str, phase_index: int):
        """Override a traffic light phase (for signal control experiments)."""
        if TRACI_AVAILABLE:
            traci.trafficlight.setPhase(tl_id, phase_index)

    # ─────────────────────────────────────────────────────────────────────
    # Properties & Utilities
    # ─────────────────────────────────────────────────────────────────────

    @property
    def sim_time(self) -> float:
        if TRACI_AVAILABLE and self._running:
            return traci.simulation.getTime()
        return float(self._step * self.step_length)

    @property
    def edge_ids(self) -> List[str]:
        return list(self._edge_ids)

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def is_done(self) -> bool:
        return self._step >= self.max_steps

    def get_edge_length(self, edge_id: str) -> float:
        """Return edge length in metres."""
        if TRACI_AVAILABLE:
            try:
                return traci.lane.getLength(edge_id + "_0")
            except Exception:
                return 500.0
        return self._edge_lengths.get(edge_id, 500.0)

    def edge_lengths_dict(self) -> Dict[str, float]:
        return {eid: self.get_edge_length(eid) for eid in self._edge_ids}

    # ─────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────

    def _compute_metrics(self) -> SimulationMetrics:
        if not self._edge_history:
            return SimulationMetrics()

        all_states = [s for step in self._edge_history for s in step]
        trip_times = [(arr - dep) for _, dep, arr in self._arrived]

        metrics = SimulationMetrics(
            total_steps      = self._step,
            total_vehicles   = len(self._departed) + len(self._arrived),
            completed_trips  = len(self._arrived),
            avg_travel_time  = float(np.mean(trip_times)) if trip_times else 0.0,
            avg_speed_kmh    = float(np.mean([s.mean_speed * 3.6 for s in all_states])),
            total_co2_kg     = float(sum(s.co2_mg_s for s in all_states)) * 1e-6,
            avg_queue_length = float(np.mean([s.queue_length for s in all_states])),
            mean_latency_ms  = float(np.mean([s.travel_time for s in all_states])) * 1000,
            throughput_veh_h = float(sum(s.throughput for s in all_states)) / max(self._step / 3600, 1e-6),
        )
        return metrics

    def save_telemetry(self, path: str = None) -> str:
        """Save edge telemetry history to CSV."""
        path = path or str(RESULTS_DIR / f"telemetry_step{self._step}.csv")
        records = []
        for step_states in self._edge_history:
            for s in step_states:
                records.append({
                    "step": s.step, "sim_time": s.sim_time, "edge_id": s.edge_id,
                    "vehicles": s.vehicle_count, "speed_ms": s.mean_speed,
                    "density_veh_km": s.mean_density, "queue": s.queue_length,
                    "travel_time_s": s.travel_time, "co2_mg_s": s.co2_mg_s,
                    "occupancy_pct": s.occupancy,
                })
        df = pd.DataFrame(records)
        df.to_csv(path, index=False)
        print(f"[SUMOEnv] Telemetry saved → {path}  ({len(df)} rows)")
        return path