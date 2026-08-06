"""
baseline_classical.py — Static / Dijkstra / A* Baselines (REAL SUMO)
========================================================================
Replaces the old generate_baselines.py synthetic-preset approach.
Every algorithm here genuinely reroutes vehicles inside SUMO via
TraCI — results are emergent, not fabricated.

  STATIC   : vehicles keep their original demand-file route
  DIJKSTRA : periodic shortest-path rerouting on live travel-time weights
  ASTAR    : periodic A* rerouting (straight-line heuristic)

Usage:
    python simulation/baseline_classical.py --algo static   --steps 3600
    python simulation/baseline_classical.py --algo dijkstra --steps 3600
    python simulation/baseline_classical.py --algo astar    --steps 3600
"""

import os, sys, argparse, time
from pathlib import Path

SIM_DIR = Path(__file__).parent
sys.path.insert(0, str(SIM_DIR))

from sumo_env import SUMOEnv
from routing_utils import RoadGraph, TelemetryRecorder, commit_route

REROUTE_INTERVAL = 30          # steps between reroute cycles
MAX_VEHICLES_PER_CYCLE = 25    # cap per-cycle cost


def run(algo: str, steps: int, use_gui: bool):
    assert algo in ("static", "dijkstra", "astar")
    cfg = str(SIM_DIR / "sumo_cfg" / "leacer.sumocfg")

    env  = SUMOEnv(cfg_path=cfg, use_gui=use_gui, max_steps=steps)
    road = RoadGraph() if algo != "static" else None
    rec  = TelemetryRecorder(algorithm=algo.upper())

    print(f"\n{'='*60}\n  Baseline: {algo.upper()}  ({steps} steps)\n{'='*60}")
    env.start()
    t0 = time.perf_counter()
    step = 0

    while not env.is_done:
        edge_states = env.step()
        if edge_states is None:
            break

        reroute_triggered = False
        if algo != "static" and step % REROUTE_INTERVAL == 0:
            reroute_triggered = _reroute_cycle(algo, road)

        rec.record_step(step, env.sim_time, edge_states, env=env,
                        reroute_triggered=reroute_triggered)

        step += 1
        if step % 200 == 0:
            print(f"  step {step:5d} | t={env.sim_time:7.1f}s | reroutes={rec._reroutes}")

    wall = time.perf_counter() - t0
    env.stop()
    print(f"  Done in {wall:.1f}s wall time | {step} steps")

    out = SIM_DIR / "results" / f"baseline_{algo}_telemetry.csv"
    rec.save(out)
    return rec


def _reroute_cycle(algo: str, road: RoadGraph) -> bool:
    try:
        import traci
    except ImportError:
        return False

    road.update_weights_from_traci()
    veh_ids = traci.vehicle.getIDList()
    if not veh_ids:
        return False

    any_rerouted = False
    for vid in veh_ids[:MAX_VEHICLES_PER_CYCLE]:
        try:
            route = traci.vehicle.getRoute(vid)
            if not route:
                continue
            cur_edge_id, dest_edge_id = traci.vehicle.getRoadID(vid), route[-1]
            cur_uv, dest_uv = road.uv_of(cur_edge_id), road.uv_of(dest_edge_id)
            if cur_uv is None or dest_uv is None:
                continue

            src_node, dst_node = cur_uv[1], dest_uv[1]
            new_path = road.dijkstra(src_node, dst_node) if algo == "dijkstra" \
                       else road.astar(src_node, dst_node)

            if new_path:
                if commit_route(vid, [cur_edge_id] + new_path):
                    any_rerouted = True
        except Exception:
            continue

    return any_rerouted


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["static", "dijkstra", "astar"], required=True)
    p.add_argument("--steps", type=int, default=3600)
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()
    run(args.algo, args.steps, args.gui)
