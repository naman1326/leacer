"""
run_simulation.py — Master entry point for full LEACER simulation suite.

Usage:
    # Run LEACER only
    python run_simulation.py --mode leacer --steps 3600

    # Run all algorithms + generate plots
    python run_simulation.py --mode all --steps 3600 --plot

    # Generate synthetic dataset for offline training
    python run_simulation.py --mode dataset --episodes 50

    # Analyze existing results
    python run_simulation.py --mode analyze
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_leacer(steps: int, gui: bool = False, **kwargs):
    from leacer_runner import LEACERRunner
    from scenario_config import SUMOCFG_PATH
    cfg = kwargs.get("cfg_path") or str(SUMOCFG_PATH)
    runner = LEACERRunner(cfg_path=cfg, use_gui=gui, max_steps=steps)
    return runner.run()


def run_baseline(algo: str, steps: int, gui: bool = False):
    from baseline_classical import run as run_classical
    return run_classical(algo=algo, steps=steps, use_gui=gui)


def run_dataset(n_episodes: int, steps: int, save_path: str):
    from sumo_data_adapter import SyntheticSUMODataset
    import numpy as np

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    save_path = os.path.join(data_dir, f"sumo_dataset_{n_episodes}ep.csv")

    ds = SyntheticSUMODataset(n_episodes=n_episodes, steps_per_ep=steps)
    episodes = ds.generate(save_path=save_path)
    X, Y = ds.to_gru_sequences(episodes, tau=20, horizon=10)
    print(f"\nDataset ready: X={X.shape}  Y={Y.shape}")
    np.save(os.path.join(data_dir, "gru_X.npy"), X)
    np.save(os.path.join(data_dir, "gru_Y.npy"), Y)
    print(f"Saved to: {data_dir}")


def run_analysis(plot: bool = True):
    from results_analyzer import run_analysis as analyze
    analyze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEACER Simulation Suite")
    parser.add_argument("--mode",     choices=["leacer","all","dataset","analyze"],
                        default="leacer")
    parser.add_argument("--steps",    type=int, default=3600,
                        help="Simulation steps (seconds)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Dataset episodes (dataset mode)")
    parser.add_argument("--gui",      action="store_true", default=False,
                        help="Launch SUMO with GUI")
    parser.add_argument("--plot",     action="store_true",
                        help="Generate comparison plots")
    args, unknown = parser.parse_known_args()

    if args.mode == "leacer":
        run_leacer(args.steps, args.gui)

    elif args.mode == "all":
        print("Running all algorithms for comparison...")
        for algo in ["dijkstra", "astar", "static"]:
            run_baseline(algo, args.steps, args.gui)
        if args.plot:
            run_analysis(plot=True)

    elif args.mode == "dataset":
        run_dataset(args.episodes, args.steps,
                    f"simulation/data/sumo_dataset_{args.episodes}ep.csv")

    elif args.mode == "analyze":
        run_analysis(plot=args.plot)
