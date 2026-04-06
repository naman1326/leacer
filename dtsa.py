"""
DTSA — Dynamic Traffic State Aggregator
Layer 1: Data Acquisition Module for LEACER Framework

Fuses V2X OBU + IoT sensor + signal phase data into
Traffic State Vector T = {S, D, Q, L} per road edge
using a scalar Kalman Filter.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import time


@dataclass
class VehicleOBU:
    vehicle_id: str;  timestamp: float
    latitude: float;  longitude: float
    speed_kmh: float; heading_deg: float
    edge_id: str;     battery_soc: float;  v2x_rssi: float

@dataclass
class IoTSensorReading:
    edge_id: str;     timestamp: float
    vehicle_count: int; occupancy_pct: float
    avg_speed_kmh: float; camera_density: float

@dataclass
class SignalPhaseData:
    node_id: str;     timestamp: float
    phase: str;       remaining_sec: float
    cycle_length_sec: float; queue_length_veh: int

@dataclass
class TrafficStateVector:
    edge_id: str;  timestamp: float
    S: float       # speed km/h
    D: float       # density veh/km
    Q: int         # queue length vehicles
    L: float       # travel time seconds
    confidence: float
    source_mask: int   # bit0=OBU, bit1=IoT, bit2=Signal


class ScalarKalmanFilter:
    """1-D Kalman Filter: x_k = x_{k-1} + w,  z_k = x_k + v"""
    def __init__(self, Q=1.0, R=5.0):
        self.Q = Q; self.R = R; self.P = 10.0; self.x = None

    def update(self, z: float) -> Tuple[float, float]:
        if self.x is None:
            self.x = z; return z, 0.5
        P_ = self.P + self.Q
        K   = P_ / (P_ + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_
        return self.x, min(1.0 / (1.0 + self.P), 1.0)


class RSUAggregator:
    """Buffers V2X / IoT / signal packets from a ~500m RSU coverage area."""
    def __init__(self, rsu_id: str, window_sec=10.0):
        self.rsu_id = rsu_id; self.window = window_sec
        self._obu:  Dict[str, deque] = {}
        self._iot:  Dict[str, deque] = {}
        self._sig:  Dict[str, deque] = {}

    def ingest_obu(self, p: VehicleOBU):
        self._obu.setdefault(p.edge_id, deque(maxlen=200)).append(p)
    def ingest_iot(self, r: IoTSensorReading):
        self._iot.setdefault(r.edge_id, deque(maxlen=50)).append(r)
    def ingest_signal(self, s: SignalPhaseData):
        self._sig.setdefault(s.node_id, deque(maxlen=20)).append(s)

    def recent_obu(self, eid, w=None):
        cutoff = time.time() - (w or self.window)
        return [p for p in self._obu.get(eid, []) if p.timestamp >= cutoff]
    def latest_iot(self, eid) -> Optional[IoTSensorReading]:
        b = self._iot.get(eid, deque()); return b[-1] if b else None
    def latest_sig(self, nid) -> Optional[SignalPhaseData]:
        b = self._sig.get(nid, deque()); return b[-1] if b else None
    def all_edges(self): return list(set(self._obu)|set(self._iot))


class DTSA:
    """
    Fuses all sensor streams into TSV per edge.
    Latency:  L_e = len_e / (S_e / 3.6)   [seconds]
    """
    def __init__(self, edge_lengths: Dict[str,float], rsu: RSUAggregator,
                 v_f=60.0, k_j=120.0):
        self.lens = edge_lengths; self.rsu = rsu
        self.v_f = v_f; self.k_j = k_j
        self._kfs: Dict[str, ScalarKalmanFilter] = {}
        self._kfd: Dict[str, ScalarKalmanFilter] = {}

    def _kf(self, store, eid):
        return store.setdefault(eid, ScalarKalmanFilter())

    def compute_tsv(self, eid: str, nid: str = None) -> TrafficStateVector:
        mask = 0
        obus = self.rsu.recent_obu(eid)
        iot  = self.rsu.latest_iot(eid)
        sig  = self.rsu.latest_sig(nid or eid)

        # Speed
        spds = []
        if obus: spds.append(np.mean([p.speed_kmh for p in obus])); mask |= 1
        if iot:  spds.append(iot.avg_speed_kmh);                    mask |= 2
        raw_s = float(np.mean(spds)) if spds else self.v_f
        S, cs = self._kf(self._kfs, eid).update(raw_s)

        # Density
        lkm = self.lens.get(eid, 500.0) / 1000.0
        dens = [len(self.rsu.recent_obu(eid, 30)) / max(lkm,.01)]
        if iot: dens.append(iot.vehicle_count / max(lkm,.01))
        D, cd = self._kf(self._kfd, eid).update(float(np.mean(dens)))

        # Queue
        Q = sig.queue_length_veh if sig else 0
        if Q: mask |= 4

        # Latency
        L = self.lens.get(eid, 500.0) / max(S/3.6, 0.5)

        return TrafficStateVector(eid, time.time(),
            round(S,2), round(D,2), Q, round(L,3),
            round(float(np.mean([cs,cd])),4), mask)

    def compute_all(self, node_map=None):
        return [self.compute_tsv(e,(node_map or {}).get(e)) for e in self.rsu.all_edges()]


if __name__ == "__main__":
    rsu = RSUAggregator("RSU_01"); t = time.time()
    for i in range(8):
        rsu.ingest_obu(VehicleOBU(f"V{i}",t-i,1.35,103.82,
            38+np.random.randn()*3, 90, "E1", .75, -65))
    rsu.ingest_iot(IoTSensorReading("E1",t,12,.45,35.5,24.0))
    rsu.ingest_signal(SignalPhaseData("E1",t,"red",18,90,7))
    tsv = DTSA({"E1":600}, rsu).compute_tsv("E1","E1")
    print(f"S={tsv.S} km/h  D={tsv.D} veh/km  Q={tsv.Q}  L={tsv.L}s  conf={tsv.confidence}")
