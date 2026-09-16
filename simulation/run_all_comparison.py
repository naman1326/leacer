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

    # Native SUMO visualization integration
    run_official_sumo_plots()

    print("\n" + "="*70)
    print("  Done. Check simulation/results/plots/ for updated graphs,")
    print("  and simulation/results/*_telemetry.csv for raw data.")
    print("="*70)


def run_official_sumo_plots():
    """Generates official SUMO visualization plots if SUMO_HOME and output XMLs are found."""
    try:
        from sumo_env import find_sumo_home
        sumo_home = find_sumo_home()
    except Exception:
        sumo_home = None

    if not sumo_home:
        print("\n[SKIP] SUMO_HOME not detected — skipping native SUMO plot scripts.")
        return

    vis_dir = Path(sumo_home) / "tools" / "visualization"
    plot_summary_py = vis_dir / "plot_summary.py"
    plot_xml_py = vis_dir / "plotXMLAttributes.py"
    results_dir = SIM_DIR / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # 1. Multi-algorithm network speed / running vehicle comparison
    algo_order = [
        ("leacer_summary.xml", "LEACER"),
        ("dqn_summary.xml", "DQN"),
        ("marl_summary.xml", "MARL"),
        ("gcn_summary.xml", "GCN"),
    ]
    present_files = []
    present_labels = []
    for fname, label in algo_order:
        fpath = results_dir / fname
        if fpath.exists():
            present_files.append(str(fpath))
            present_labels.append(label)

    if len(present_files) >= 2 and plot_summary_py.exists():
        cmd = [
            py, str(plot_summary_py),
            "-i", ",".join(present_files),
            "-l", ",".join(present_labels),
            "-m", "running",
            "-o", str(plots_dir / "sumo_speed_comparison.png"),
            "-b"
        ]
        run_cmd(cmd, "Generating SUMO native plot_summary.py (running vehicles)")

    # 2. Per-vehicle CO2 plot from emissions.xml
    emission_candidates = [
        results_dir / "leacer_emissions.xml",
        results_dir / "emissions.xml",
        results_dir / "sumo_emissions.xml",
    ]
    target_emission = next((f for f in emission_candidates if f.exists()), None)
    if target_emission and plot_xml_py.exists():
        cmd = [
            py, str(plot_xml_py),
            "-x", "time", "-y", "CO2",
            "-o", str(plots_dir / "co2_check.png"),
            str(target_emission),
            "-i", "id", "--xelem", "timestep", "--yelem", "vehicle",
            "--legend", "-b"
        ]
        run_cmd(cmd, f"Generating per-vehicle CO2 plot with plotXMLAttributes.py ({target_emission.name})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3600)
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--gui", action="store_true", default=False,
                   help="Launch SUMO with GUI")
    p.add_argument("--dqn-episodes", type=int, default=400)
    p.add_argument("--marl-episodes", type=int, default=300)
    p.add_argument("--gcn-rollout", type=int, default=400)
    p.add_argument("--gcn-epochs", type=int, default=60)
    args = p.parse_args()
    main(args.steps, args.skip_training, args.gui, args.dqn_episodes,
         args.marl_episodes, args.gcn_rollout, args.gcn_epochs)
