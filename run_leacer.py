"""
run_leacer.py — Full LEACER pipeline integration entry point.
Wires Layer 1 → Layer 2 → Layer 3 together for a single routing cycle.
"""

import time
import numpy as np
from dtsa    import DTSA, RSUAggregator, VehicleOBU, IoTSensorReading, SignalPhaseData
from gat_encoder import GATEncoderNumpy, build_road_graph
from mo_earo import MOCostEvaluator, RouteMetrics, GreedyRouter
from cefar   import (KPIThresholds, ThresholdMonitor, CooperativeRSUMesh,
                      LyapunovStabilizer, CEFARController, RouteDispatcher)


def run_single_cycle(vehicle_id="V001", origin="N0", dest="N3"):
    print(f"\n{'='*65}")
    print(f"  LEACER  single routing cycle   {vehicle_id}: {origin} -> {dest}")
    print(f"{'='*65}\n")

    # ── Layer 1: Data Acquisition ─────────────────────────────────
    rsu = RSUAggregator("RSU_01"); t = time.time()
    edges = ["E01","E12","E23","E03","E13"]
    lens  = {"E01":400,"E12":550,"E23":300,"E03":700,"E13":480}

    for eid in edges:
        for i in range(5):
            rsu.ingest_obu(VehicleOBU(f"V{i}",t-i,1.35,103.82,
                35+np.random.randn()*5,90,eid,.8,-65))
        rsu.ingest_iot(IoTSensorReading(eid,t,
            np.random.randint(5,20),.4,32+np.random.randn()*4,20.0))
        rsu.ingest_signal(SignalPhaseData(eid,t,"red",20,90,
            np.random.randint(2,15)))

    dtsa = DTSA(lens, rsu)
    tsvs = {tsv.edge_id: tsv for tsv in dtsa.compute_all()}
    print("Layer 1 — TSVs:")
    for eid, tsv in tsvs.items():
        print(f"  {eid}: S={tsv.S:5.1f}km/h  D={tsv.D:5.1f}veh/km  "
              f"Q={tsv.Q:2d}  L={tsv.L:6.1f}s  conf={tsv.confidence:.2f}")

    # ── Layer 2: Edge Intelligence ────────────────────────────────
    adj  = [("N0","N1"),("N1","N2"),("N2","N3"),("N0","N3"),("N1","N3")]
    nodes = ["N0","N1","N2","N3"]
    # Build node-level TSVs from edge TSVs (simplified: map edge→node)
    from dtsa import TrafficStateVector as TSV
    node_tsvs = {n: TSV(n,t,40-i*3,15+i*2,i*2,30+i*10,0.8,3)
                 for i,n in enumerate(nodes)}

    graph = build_road_graph(node_tsvs, adj)
    encoder = GATEncoderNumpy(emb_dim=16)
    embs  = encoder.encode(graph)
    print(f"\nLayer 2 — GAT node embeddings: {embs.shape}")

    # Candidate routes and MO cost
    candidates = [
        RouteMetrics(["N0","N1","N2","N3"], T=420, E=1.2, C=0.65, L=85),
        RouteMetrics(["N0","N3"],           T=510, E=0.9, C=0.30, L=60),
        RouteMetrics(["N0","N1","N3"],      T=360, E=1.4, C=0.80, L=100),
    ]
    router = GreedyRouter()
    best   = router.select(candidates)
    print(f"  Best route: {best.route}")
    print(f"  F = alpha*{best.T} + beta*{best.E} + gamma*{best.C} + delta*{best.L}")

    # ── Layer 3: Cooperative Adaptation ──────────────────────────
    thr   = KPIThresholds()
    mon   = ThresholdMonitor(thr, cooldown_sec=0)
    mesh  = CooperativeRSUMesh("RSU_01")
    stab  = LyapunovStabilizer()
    cefar = CEFARController(mon, mesh, stab)

    all_edges = list(lens.keys())
    queues    = np.array([tsvs[e].Q for e in all_edges], dtype=float)
    arrivals  = np.random.rand(len(all_edges)) * 0.6
    avg_C     = float(np.mean([tsvs[e].D / 120.0 for e in all_edges]))
    avg_L_ms  = float(np.mean([tsvs[e].L * 1000 for e in all_edges]))

    event = cefar.step(C=avg_C, L_ms=avg_L_ms, E=1.1,
                       queues=queues, arrival_rates=arrivals,
                       edge_ids=all_edges)

    if event:
        print(f"\nLayer 3 — Re-route event: {event.trigger.value}")
        print(f"  Unstable edges: {event.penalized_edges}")
    else:
        print(f"\nLayer 3 — No re-route triggered (KPIs within bounds)")

    # Dispatch
    disp = RouteDispatcher(stab)
    pkg  = disp.dispatch(vehicle_id, best.route,
                          {e: tsvs.get(e, type('',(),{'L':30.0})()).L
                           for e in best.route},
                          queues, all_edges)
    print(f"\n  Dispatched: {pkg.vehicle_id} -> {pkg.route}  ETA={pkg.eta_sec}s")
    print(f"\n{'='*65}")
    print("  LEACER cycle complete")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_single_cycle()
