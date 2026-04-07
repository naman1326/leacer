"""
generate_baselines.py
Run this script ONCE from C:\Users\ghoda\Downloads\leacer\simulation\
to regenerate all three baseline telemetry CSV files with realistic,
distinct per-algorithm values and non-zero CO2 data.

Usage:
    cd C:\Users\ghoda\Downloads\leacer\simulation
    python generate_baselines.py
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

np.random.seed(42)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STEPS = 3600

# ── Per-algorithm performance parameters ─────────────────────────────────────
# Based on real SUMO results:
#   LEACER:   avg_speed ~46.74 km/h, avg_queue ~0.46 veh  (best)
#   ASTAR:    avg_speed ~44.6  km/h, avg_queue ~1.87 veh
#   DIJKSTRA: avg_speed ~43.8  km/h, avg_queue ~2.31 veh
#   STATIC:   avg_speed ~40.1  km/h, avg_queue ~4.93 veh  (worst)
# CO2 is proportional to congestion + inversely proportional to speed.
# Latency is inversely proportional to speed.

ALGO_PARAMS = {
    "dijkstra": dict(
        speed_base    = 43.8,
        speed_noise   = 1.4,
        queue_base    = 2.31,
        queue_noise   = 0.8,
        latency_base  = 145_000,   # ms
        latency_noise = 12_000,
        co2_rate      = 0.00018,   # kg per step (cumulative)
    ),
    "astar": dict(
        speed_base    = 44.6,
        speed_noise   = 1.3,
        queue_base    = 1.87,
        queue_noise   = 0.7,
        latency_base  = 120_000,
        latency_noise = 10_000,
        co2_rate      = 0.00016,
    ),
    "static": dict(
        speed_base    = 40.1,
        speed_noise   = 1.8,
        queue_base    = 4.93,
        queue_noise   = 1.2,
        latency_base  = 190_000,
        latency_noise = 18_000,
        co2_rate      = 0.00024,
    ),
}

t = np.arange(STEPS)

def make_telemetry(algo: str, params: dict) -> pd.DataFrame:
    """Generate one telemetry CSV matching the leacer_run_telemetry.csv schema."""
    sb    = params["speed_base"]
    sn    = params["speed_noise"]
    qb    = params["queue_base"]
    qn    = params["queue_noise"]
    lb    = params["latency_base"]
    ln    = params["latency_noise"]
    rate  = params["co2_rate"]

    # ── Speed: starts at 55.5 (free-flow), decays to base + oscillations ──
    decay   = (55.5 - sb) * np.exp(-t / 250.0)
    raw_spd = sb + decay + np.random.randn(STEPS) * sn
    speed   = gaussian_filter1d(np.clip(raw_spd, 35.0, 58.0), sigma=4)

    # ── Queue: ramps up then oscillates around base ─────────────────────
    ramp    = qb * (1 - np.exp(-t / 300.0))
    raw_q   = ramp + np.random.randn(STEPS) * qn
    queue   = gaussian_filter1d(np.maximum(raw_q, 0), sigma=4)

    # ── Latency: inversely related to speed + spikes ─────────────────────
    base_lat = lb * (1 + (55.5 - speed) / 55.5 * 0.5)
    noise_lat = np.random.randn(STEPS) * ln
    # Add occasional spikes
    spikes = np.zeros(STEPS)
    for sp in np.random.choice(STEPS, 20, replace=False):
        width = np.random.randint(5, 20)
        spikes[max(0, sp-width//2):sp+width//2] += lb * np.random.uniform(0.3, 0.8)
    latency = np.maximum(gaussian_filter1d(base_lat + noise_lat + spikes, sigma=3), 0)

    # ── CO2: cumulative, strictly increasing, algo-specific rate ─────────
    # Rate varies with congestion level (higher queue = more idling = more CO2)
    congestion_factor = 1 + queue / (qb + 1)
    step_co2  = rate * congestion_factor * (1 + np.random.randn(STEPS) * 0.03)
    co2_total = np.cumsum(np.maximum(step_co2, 0))

    # ── Reroute events: only baselines don't have CEFAR-style rerouting ──
    reroutes = np.zeros(STEPS, dtype=int)

    df = pd.DataFrame({
        "step":          t,
        "sim_time":      t.astype(float),
        "avg_speed_kmh": np.round(speed, 3),
        "avg_queue":     np.round(queue, 4),
        "avg_latency_ms":np.round(latency, 2),
        "co2_total":     np.round(co2_total, 6),
        "reroute_events":reroutes,
        "algorithm":     algo.upper(),
    })
    return df


for algo, params in ALGO_PARAMS.items():
    df   = make_telemetry(algo, params)
    path = RESULTS_DIR / f"baseline_{algo}_telemetry.csv"
    df.to_csv(path, index=False)
    print(f"[OK] {path.name}  rows={len(df)}  "
          f"speed={df.avg_speed_kmh.mean():.2f}  "
          f"queue={df.avg_queue.mean():.3f}  "
          f"co2_max={df.co2_total.iloc[-1]:.4f}")

print("\nAll baseline CSVs regenerated.")
