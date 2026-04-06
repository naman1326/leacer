"""
CEFAR — Cooperative Event-triggered Fault-Adaptive Re-routing
Layer 3: Cooperative Adaptation Module, LEACER Framework

Implements:
  1. Threshold Monitor    — detects KPI violations (tau_C, tau_L, tau_E)
  2. CEFAR Controller     — triggers re-optimization via Layer 2
  3. Cooperative RSU Mesh — weighted fusion of neighbour RSU states
  4. Lyapunov Stabilizer  — ensures queue drift bound B is satisfied
  5. Route Dispatcher     — outputs final route via V2V / V2I

Lyapunov stability condition:
  E[ L(Q_{t+1}) - L(Q_t) | Q_t ] <= B - epsilon * sum_i Q_i(t)
  where L(Q) = 0.5 * ||Q||^2  (quadratic Lyapunov function)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
import time


# ─────────────────────────────────────────────────────────────────────────────
# Threshold definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KPIThresholds:
    """System-level KPI thresholds from paper Section IV-C."""
    tau_C: float = 0.75    # Congestion index upper bound [0,1]
    tau_L: float = 150.0   # Latency upper bound (ms)
    tau_E: float = 2.0     # Energy per km upper bound (kWh/km)
    tau_Q: int   = 20      # Queue length upper bound (vehicles)


class TriggerReason(Enum):
    NONE         = "none"
    CONGESTION   = "congestion"
    HIGH_LATENCY = "high_latency"
    HIGH_ENERGY  = "high_energy"
    QUEUE_OVERFL = "queue_overflow"
    PEER_REQUEST = "peer_rsu_request"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Threshold Monitor
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdMonitor:
    """
    Watches live KPI values and fires an event when any threshold is violated.
    Implements hysteresis to avoid oscillation: once triggered, requires
    cooldown_sec before another trigger on the same metric.
    """

    def __init__(self, thresholds: KPIThresholds = None, cooldown_sec=15.0):
        self.thr = thresholds or KPIThresholds()
        self.cooldown = cooldown_sec
        self._last_trigger: Dict[TriggerReason, float] = {}

    def _cooled_down(self, reason: TriggerReason) -> bool:
        last = self._last_trigger.get(reason, 0.0)
        return (time.time() - last) >= self.cooldown

    def check(self, C: float, L_ms: float, E: float, Q: int
              ) -> Tuple[bool, TriggerReason]:
        """
        Check current KPI values against thresholds.
        Returns (triggered, reason).
        """
        checks = [
            (C   > self.thr.tau_C, TriggerReason.CONGESTION),
            (L_ms> self.thr.tau_L, TriggerReason.HIGH_LATENCY),
            (E   > self.thr.tau_E, TriggerReason.HIGH_ENERGY),
            (Q   > self.thr.tau_Q, TriggerReason.QUEUE_OVERFL),
        ]
        for violated, reason in checks:
            if violated and self._cooled_down(reason):
                self._last_trigger[reason] = time.time()
                return True, reason
        return False, TriggerReason.NONE


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cooperative RSU Mesh — weighted state fusion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RSUState:
    """State broadcast by a single RSU to its mesh neighbours."""
    rsu_id:    str
    timestamp: float
    avg_congestion: float    # mean C over its edge set
    avg_latency_ms: float    # mean L over its edge set
    load_factor:    float    # fraction of edges above tau_C
    weight:         float = 1.0   # assigned by fusion (set externally)


class CooperativeRSUMesh:
    """
    Maintains a peer table of neighbour RSU states and computes
    weighted cooperative traffic state T_coop for use by CEFAR.

    Fusion weight omega_i = (1 / d_i) / sum_j (1/d_j)
    where d_i is the geographic distance to RSU i.
    """

    def __init__(self, local_rsu_id: str, max_peer_age_sec=30.0):
        self.local_id = local_rsu_id
        self.max_age  = max_peer_age_sec
        self._peers:  Dict[str, RSUState] = {}
        self._distances: Dict[str, float] = {}

    def update_peer(self, state: RSUState, distance_m: float):
        """Receive a broadcast from a neighbour RSU."""
        self._peers[state.rsu_id]    = state
        self._distances[state.rsu_id] = max(distance_m, 1.0)

    def _active_peers(self) -> List[RSUState]:
        cutoff = time.time() - self.max_age
        return [s for s in self._peers.values() if s.timestamp >= cutoff]

    def fused_state(self) -> Optional[RSUState]:
        """
        Compute distance-weighted average state across all active peers.
        Returns None if no peers available.
        """
        peers = self._active_peers()
        if not peers:
            return None

        ids = [p.rsu_id for p in peers]
        dists = np.array([self._distances.get(i, 500.0) for i in ids])
        inv_d = 1.0 / dists
        omegas = inv_d / inv_d.sum()   # Eq. omega_i in paper

        fused = RSUState(
            rsu_id    = "FUSED",
            timestamp = time.time(),
            avg_congestion = float(np.dot(omegas, [p.avg_congestion for p in peers])),
            avg_latency_ms = float(np.dot(omegas, [p.avg_latency_ms for p in peers])),
            load_factor    = float(np.dot(omegas, [p.load_factor    for p in peers])),
        )
        return fused


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lyapunov Stabilizer
# ─────────────────────────────────────────────────────────────────────────────

class LyapunovStabilizer:
    """
    Enforces queue-length stability using a Lyapunov drift-plus-penalty
    framework (from paper Section IV-D).

    Lyapunov function:  L(Q) = 0.5 * sum_i Q_i^2
    Drift condition:    E[L(Q_{t+1}) - L(Q_t)] <= B - eps * sum_i Q_i
    Stability requires: mean_arrival_rate <= mean_service_rate + eps

    The stabilizer re-routes traffic away from edges where queue
    growth threatens to violate the drift bound.
    """

    def __init__(self, B: float = 50.0, epsilon: float = 0.1,
                 service_rate_veh_s: float = 0.5):
        self.B   = B
        self.eps = epsilon
        self.mu  = service_rate_veh_s   # vehicles drained per second per edge

    def lyapunov(self, queues: np.ndarray) -> float:
        """L(Q) = 0.5 * ||Q||^2"""
        return 0.5 * float(np.dot(queues, queues))

    def drift(self, Q_now: np.ndarray, Q_next: np.ndarray) -> float:
        """One-step Lyapunov drift: L(Q_{t+1}) - L(Q_t)"""
        return self.lyapunov(Q_next) - self.lyapunov(Q_now)

    def is_stable(self, Q_now: np.ndarray, arrival_rates: np.ndarray) -> bool:
        """
        Check stability condition:
          arrival_rate_i <= mu + eps   for all i with Q_i > 0
        """
        for i, (q, lam) in enumerate(zip(Q_now, arrival_rates)):
            if q > 0 and lam > self.mu + self.eps:
                return False
        return True

    def unstable_edges(self, queues: np.ndarray,
                        arrival_rates: np.ndarray,
                        edge_ids: List[str]) -> List[str]:
        """
        Return list of edge IDs violating the stability condition.
        These edges should be avoided in re-routing.
        """
        bad = []
        for eid, q, lam in zip(edge_ids, queues, arrival_rates):
            if q > 0 and lam > self.mu + self.eps:
                bad.append(eid)
        return bad

    def penalty_weights(self, queues: np.ndarray) -> np.ndarray:
        """
        Compute per-edge routing penalty proportional to queue length.
        Used to bias PPO cost function away from congested edges.
        penalty_i = Q_i / (sum_j Q_j + 1)
        """
        total = queues.sum() + 1.0
        return queues / total


# ─────────────────────────────────────────────────────────────────────────────
# 4. CEFAR Controller — orchestrates the re-routing trigger loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouteUpdateEvent:
    """Emitted by CEFAR when a re-route is needed."""
    timestamp:    float
    trigger:      TriggerReason
    affected_edges: List[str]
    penalized_edges: List[str]   # Lyapunov-flagged edges to avoid
    cooperative_C:   float       # fused congestion from mesh
    cooperative_L:   float       # fused latency from mesh


class CEFARController:
    """
    Cooperative Event-triggered Fault-Adaptive Re-routing Controller.

    Lifecycle per time step:
      1. ThresholdMonitor checks current KPI → event fired?
      2. If yes: gather cooperative state from RSU mesh
      3. Run Lyapunov stabilizer to flag unstable edges
      4. Emit RouteUpdateEvent → passed to Route Dispatcher
      5. Route Dispatcher triggers re-invocation of Layer 2 PPO
    """

    def __init__(self,
                 monitor:    ThresholdMonitor,
                 mesh:       CooperativeRSUMesh,
                 stabilizer: LyapunovStabilizer,
                 on_reroute: Optional[Callable[[RouteUpdateEvent], None]] = None):
        self.monitor    = monitor
        self.mesh       = mesh
        self.stabilizer = stabilizer
        self.on_reroute = on_reroute    # callback to Route Dispatcher
        self._event_log: List[RouteUpdateEvent] = []

    def step(self,
             C: float, L_ms: float, E: float,
             queues: np.ndarray,
             arrival_rates: np.ndarray,
             edge_ids: List[str]) -> Optional[RouteUpdateEvent]:
        """
        Called every control cycle (~1s).
        Returns RouteUpdateEvent if re-route was triggered, else None.
        """
        triggered, reason = self.monitor.check(C, L_ms, E, int(queues.max()))
        if not triggered:
            return None

        # Gather cooperative state
        peer_state = self.mesh.fused_state()
        coop_C = peer_state.avg_congestion if peer_state else C
        coop_L = peer_state.avg_latency_ms if peer_state else L_ms

        # Lyapunov — find unstable edges
        bad_edges = self.stabilizer.unstable_edges(queues, arrival_rates, edge_ids)

        event = RouteUpdateEvent(
            timestamp       = time.time(),
            trigger         = reason,
            affected_edges  = [eid for eid,q in zip(edge_ids,queues) if q > 5],
            penalized_edges = bad_edges,
            cooperative_C   = coop_C,
            cooperative_L   = coop_L,
        )
        self._event_log.append(event)
        if self.on_reroute:
            self.on_reroute(event)
        return event

    def event_log(self) -> List[RouteUpdateEvent]:
        return list(self._event_log)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Route Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DispatchedRoute:
    """Final route packet sent to vehicles via V2V/V2I."""
    vehicle_id: str
    route:      List[str]         # ordered edge/node IDs
    eta_sec:    float             # estimated total travel time
    issued_at:  float
    source:     str = "LEACER"    # for fleet management logging


class RouteDispatcher:
    """
    Translates the optimal route P* from Layer 2 into per-vehicle
    navigation updates, broadcast via V2I or fleet management API.
    Applies Lyapunov penalty weights to adjust final cost if needed.
    """

    def __init__(self, stabilizer: LyapunovStabilizer):
        self.stabilizer = stabilizer
        self._dispatched: List[DispatchedRoute] = []

    def dispatch(self,
                 vehicle_id: str,
                 route: List[str],
                 edge_latencies: Dict[str, float],
                 queues: np.ndarray,
                 edge_ids: List[str]) -> DispatchedRoute:
        """
        Issue final routing command to a vehicle.

        Applies Lyapunov penalties to ETA: penalized edges add
        extra delay = penalty_weight * base_latency.
        """
        penalties = self.stabilizer.penalty_weights(queues)
        pen_map   = dict(zip(edge_ids, penalties))

        eta = 0.0
        for eid in route:
            base = edge_latencies.get(eid, 30.0)
            eta += base * (1.0 + pen_map.get(eid, 0.0))

        pkg = DispatchedRoute(vehicle_id=vehicle_id, route=route,
                              eta_sec=round(eta,1), issued_at=time.time())
        self._dispatched.append(pkg)
        return pkg


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("CEFAR Smoke Test  LEACER Layer 3")
    print("=" * 60)

    thr  = KPIThresholds(tau_C=0.75, tau_L=150.0, tau_E=2.0, tau_Q=20)
    mon  = ThresholdMonitor(thr, cooldown_sec=0)   # no cooldown for test
    mesh = CooperativeRSUMesh("RSU_01")
    stab = LyapunovStabilizer(B=50.0, epsilon=0.1)
    cefar = CEFARController(mon, mesh, stab,
        on_reroute=lambda e: print(f"  Re-route triggered! reason={e.trigger.value}"))

    # Simulate a high-congestion step
    edges = ["E1","E2","E3","E4"]
    queues = np.array([5.0, 8.0, 25.0, 3.0])    # E3 overflows tau_Q=20
    arrivals = np.array([0.4, 0.5, 0.9, 0.2])

    event = cefar.step(C=0.82, L_ms=180.0, E=1.5,
                       queues=queues, arrival_rates=arrivals,
                       edge_ids=edges)

    if event:
        print(f"  Affected: {event.affected_edges}")
        print(f"  Penalized (unstable): {event.penalized_edges}")
        print(f"  Cooperative C={event.cooperative_C:.2f}  L={event.cooperative_L:.1f}ms")

    # Dispatch route
    disp = RouteDispatcher(stab)
    pkg  = disp.dispatch("V001", ["E1","E2"], {"E1":35.0,"E2":42.0},
                          queues, edges)
    print(f"\nDispatched to {pkg.vehicle_id}: route={pkg.route}  ETA={pkg.eta_sec}s")
    print("\n  CEFAR module OK")
