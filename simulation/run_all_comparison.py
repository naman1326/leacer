"""
run_all_comparison.py — Master Comparison Runner
====================================================
Runs LEACER + all baselines against the SAME SUMO scenario (same
network, same demand file), producing directly comparable telemetry
CSVs, then regenerates all 12 comparison graphs.

"Simultaneously" in the sense of one coordinated pipeline: SUMO/TraCI
only supports one active connection per process, so runs are executed
back-to-back under identical conditions — this is standard practice
for SUMO comparison studies (nobody parallelises SUMO processes for
this; sequential runs under a fixed seed are what makes results
directly comparable in the first place).

Usage:
    python simulation/run_all_comparison.py --steps 3600
    python simulation/run_all_comparison.py --steps 3600 --skip-training
"""

import subprocess, sys, argparse, time
from pathlib import Path

SIM_DIR = Path(__file__).parent


def run_cmd(cmd, label):
    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(SIM_DIR))
    ok = result.returncode == 0
    print(f"  [{label}] finished in {time.perf_counter()-t0:.1f}s "
          f"({'OK' if ok else 'FAILED, exit '+str(result.returncode)})")
    return ok


def main(steps, skip_training, gui, dqn_episodes, marl_episodes, gcn_rollout, gcn_epochs):
    py = sys.executable
    gui_flag = ["--gui"] if gui else []

    if not skip_training:
        run_cmd([py, "baseline_dqn.py",  "--mode","train","--episodes",str(dqn_episodes)],
                 "Training DQN (Cloud-RL archetype)")
        run_cmd([py, "baseline_marl.py", "--mode","train","--episodes",str(marl_episodes)],
                 "Training Independent MARL (DRL-ITS archetype)")
        run_cmd([py, "baseline_gcn.py",  "--mode","train",
                 "--rollout-steps",str(gcn_rollout),"--epochs",str(gcn_epochs)],
                 "Pretraining GCN (GCN-Route archetype)")
    else:
        print("Skipping training — using existing weights in simulation/data/ if present.")

    run_cmd([py, "baseline_classical.py","--algo","static",   "--steps",str(steps)]+gui_flag,
             "Evaluating STATIC routing")
    run_cmd([py, "baseline_classical.py","--algo","dijkstra", "--steps",str(steps)]+gui_flag,
             "Evaluating DIJKSTRA routing")
    run_cmd([py, "baseline_classical.py","--algo","astar",    "--steps",str(steps)]+gui_flag,
             "Evaluating A* routing")
    run_cmd([py, "baseline_dqn.py", "--mode","eval","--steps",str(steps)]+gui_flag,
             "Evaluating DQN (Cloud-RL archetype)")
    run_cmd([py, "baseline_marl.py","--mode","eval","--steps",str(steps)]+gui_flag,
             "Evaluating MARL (DRL-ITS archetype)")
    run_cmd([py, "baseline_gcn.py", "--mode","eval","--steps",str(steps)]+gui_flag,
             "Evaluating GCN-Route archetype")
    run_cmd([py, "run_simulation.py","--mode","leacer","--steps",str(steps)]+gui_flag,
             "Evaluating LEACER")

    run_cmd([py, "results_analyzer.py"], "Regenerating all 12 comparison graphs")

    print("\n" + "="*70)
    print("  Done. Check simulation/results/plots/ for updated graphs,")
    print("  and simulation/results/*_telemetry.csv for raw data.")
    print("="*70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3600)
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--dqn-episodes", type=int, default=400)
    p.add_argument("--marl-episodes", type=int, default=300)
    p.add_argument("--gcn-rollout", type=int, default=400)
    p.add_argument("--gcn-epochs", type=int, default=60)
    args = p.parse_args()
    main(args.steps, args.skip_training, args.gui, args.dqn_episodes,
         args.marl_episodes, args.gcn_rollout, args.gcn_epochs)
