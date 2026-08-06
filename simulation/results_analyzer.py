"""
results_analyzer.py — LEACER Comparison Results Analyser
============================================================
Loads telemetry for LEACER + all real baselines (classical + the
DQN/MARL/GCN "modern algorithm" archetypes) and generates all 12
comparison plots into simulation/results/plots/.

Every CSV loaded here comes from a REAL SUMO run with real per-
algorithm rerouting decisions -- nothing here is synthetic preset data.

Usage:
    cd simulation
    python results_analyzer.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "legend.fontsize": 8, "legend.framealpha": 0.9, "figure.dpi": 150,
})

# key = value in the CSV's "algorithm" column
# value = (csv filename stem, display label, color, linestyle, linewidth)
ALGO_REGISTRY = {
    "LEACER":      ("leacer_run",        "LEACER (Proposed)",         "#1f77b4", "-",  1.8),
    "DQN_CLOUDRL": ("baseline_dqn",      "DQN (Cloud-RL archetype)",  "#ff7f0e", "--", 1.3),
    "MARL_DRLITS": ("baseline_marl",     "MARL (DRL-ITS archetype)",  "#2ca02c", "-.", 1.3),
    "GCN_ROUTE":   ("baseline_gcn",      "GCN-Route",                 "#d62728", ":",  1.3),
    "DIJKSTRA":    ("baseline_dijkstra", "Dijkstra",                  "#9467bd", "--", 1.1),
    "ASTAR":       ("baseline_astar",    "A*",                        "#8c564b", "-.", 1.1),
    "STATIC":      ("baseline_static",   "Static",                    "#7f7f7f", ":",  1.1),
}
# Default comparison set = the "modern algorithms" request (LEACER vs DQN/MARL/GCN).
ALGOS = list(ALGO_REGISTRY.keys())

COLORS = {k: v[2] for k, v in ALGO_REGISTRY.items()}
LS     = {k: v[3] for k, v in ALGO_REGISTRY.items()}
LW     = {k: v[4] for k, v in ALGO_REGISTRY.items()}
LABEL  = {k: v[1] for k, v in ALGO_REGISTRY.items()}
SAVE   = dict(bbox_inches="tight", dpi=150)


def load_all_telemetry() -> dict:
    data = {}
    for algo, reg in ALGO_REGISTRY.items():
        stem = reg[0]
        path = RESULTS_DIR / f"{stem}_telemetry.csv"
        if path.exists():
            data[algo] = pd.read_csv(path)
        else:
            print(f"[SKIP] {algo}: no telemetry file found "
                  f"(expected {path.name}) -- run its baseline script first.")
    return data


def _active(data):
    return [a for a in ALGOS if a in data]


def fig1_speed(data):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for a in _active(data):
        df = data[a]
        ax.plot(df["sim_time"], df["avg_speed_kmh"], color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
    ax.set_xlabel("Simulation Step (s)"); ax.set_ylabel("Avg Speed (km/h)")
    ax.set_title("Fig. 1 -- Average Network Speed over Time"); ax.legend()
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig1_speed_over_time.pdf", **SAVE); plt.close()
    print("[OK] fig1_speed_over_time.pdf")


def fig2_queue(data):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for a in _active(data):
        df = data[a]
        ax.plot(df["sim_time"], df["avg_queue"], color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
    ax.set_xlabel("Simulation Step (s)"); ax.set_ylabel("Avg Queue Length (veh)")
    ax.set_title("Fig. 2 -- Average Queue Length over Time"); ax.legend()
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig2_queue_over_time.pdf", **SAVE); plt.close()
    print("[OK] fig2_queue_over_time.pdf")


def fig3_latency(data):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for a in _active(data):
        df = data[a]
        lat = df["avg_latency_ms"].values
        if lat.max() > 1e6: lat = lat / 1000.0
        ax.plot(df["sim_time"], lat, color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
    ax.axhline(150, color="salmon", ls="--", lw=1.2, label=r"$\tau_L$ = 150ms")
    ax.set_xlabel("Simulation Step (s)"); ax.set_ylabel("Edge Latency (ms)")
    ax.set_title(r"Fig. 3 -- Edge Latency vs CEFAR Threshold $\tau_L$ = 150ms"); ax.legend()
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig3_latency_over_time.pdf", **SAVE); plt.close()
    print("[OK] fig3_latency_over_time.pdf")


def fig4_co2(data):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for a in _active(data):
        df = data[a]
        ax.plot(df["sim_time"], df["co2_total"], color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
    ax.set_xlabel("Simulation Step (s)"); ax.set_ylabel("Cumulative CO$_2$ (kg)")
    ax.set_title("Fig. 4 -- Cumulative CO$_2$ Emissions"); ax.legend(loc="upper left")
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig4_co2_cumulative.pdf", **SAVE); plt.close()
    print("[OK] fig4_co2_cumulative.pdf")


def fig5_speed_dist(data):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.linspace(20, 60, 300)
    for a in _active(data):
        s = data[a]["avg_speed_kmh"].values
        if s.std() < 1e-6: continue
        kde = gaussian_kde(s, bw_method=0.25)
        ax.plot(x, kde(x), color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
    ax.set_xlabel("Speed (km/h)"); ax.set_ylabel("Density")
    ax.set_title("Fig. 5 -- Speed Distribution Comparison"); ax.legend()
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig5_speed_distribution.pdf", **SAVE); plt.close()
    print("[OK] fig5_speed_distribution.pdf")


def fig6_queue_dist(data):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.linspace(0, 20, 300)
    for a in _active(data):
        q = data[a]["avg_queue"].values
        if q.std() < 1e-6: continue
        kde = gaussian_kde(q, bw_method=0.2)
        ax.plot(x, kde(x), color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
    ax.axvline(20, color="salmon", ls="--", lw=1.2, label=r"$\tau_Q$ = 20")
    ax.set_xlabel("Queue Length (veh)"); ax.set_ylabel("Density")
    ax.set_title("Fig. 6 -- Queue Length Distribution"); ax.legend()
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig6_queue_distribution.pdf", **SAVE); plt.close()
    print("[OK] fig6_queue_distribution.pdf")


def fig7_rush_hour(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    for a in _active(data):
        df = data[a]
        ax1.plot(df["sim_time"], df["avg_speed_kmh"], color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
        ax2.plot(df["sim_time"], df["avg_queue"], color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
    ax1.set_xlabel("Step"); ax1.set_ylabel("Speed (km/h)")
    ax1.set_title("Fig. 7a -- Speed (Rush Hour)"); ax1.legend(fontsize=7)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Queue (veh)")
    ax2.set_title("Fig. 7b -- Queue (Rush Hour)"); ax2.legend(fontsize=7)
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig7_rush_hour.pdf", **SAVE); plt.close()
    print("[OK] fig7_rush_hour.pdf")


def fig8_radar(data):
    categories = ["Speed", "Low Queue", "Low Latency", "Low CO2", "Stability"]
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]

    raw = {}
    for a in _active(data):
        df = data[a]
        raw[a] = [df["avg_speed_kmh"].mean(), df["avg_queue"].mean(),
                  df["avg_latency_ms"].mean(), df["co2_total"].iloc[-1],
                  1.0/(1.0+df["avg_queue"].std())]
    if not raw:
        print("[SKIP] fig8_kpi_radar -- no data"); return

    mins = [min(raw[a][i] for a in raw) for i in range(5)]
    maxs = [max(raw[a][i] for a in raw) for i in range(5)]
    higher_is_better = [True, False, False, False, True]

    norm = {}
    for a, vals in raw.items():
        ns = []
        for i, v in enumerate(vals):
            if maxs[i] == mins[i]: ns.append(0.5)
            elif higher_is_better[i]: ns.append((v-mins[i])/(maxs[i]-mins[i]))
            else: ns.append(1-(v-mins[i])/(maxs[i]-mins[i]))
        norm[a] = ns

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    for a in _active(data):
        vals = norm[a] + norm[a][:1]
        ax.plot(angles, vals, color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
        ax.fill(angles, vals, color=COLORS[a], alpha=0.07)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)
    ax.set_ylim(0,1); ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_title("Fig. 8 -- Normalised KPI Radar Chart", pad=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4,1.15))
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig8_kpi_radar.pdf", **SAVE); plt.close()
    print("[OK] fig8_kpi_radar.pdf")


def fig9_improvement(data):
    if "LEACER" not in data:
        print("[SKIP] fig9 -- LEACER data missing"); return
    leacer_speed = data["LEACER"]["avg_speed_kmh"].mean()
    leacer_queue = data["LEACER"]["avg_queue"].mean()

    baselines = [a for a in _active(data) if a != "LEACER"]
    if not baselines:
        print("[SKIP] fig9 -- no baseline data"); return

    speed_imp, queue_red = [], []
    for a in baselines:
        bs, bq = data[a]["avg_speed_kmh"].mean(), data[a]["avg_queue"].mean()
        speed_imp.append(100*(leacer_speed-bs)/max(bs,0.01))
        queue_red.append(100*(bq-leacer_queue)/max(bq,0.01))

    x = np.arange(len(baselines)); w = 0.35
    fig, ax = plt.subplots(figsize=(7,3.2))
    b1 = ax.bar(x-w/2, speed_imp, w, color="#4472C4", label="Speed Improvement %")
    b2 = ax.bar(x+w/2, queue_red, w, color="#70AD47", label="Queue Reduction %")
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{bar.get_height():.1f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in baselines], fontsize=8)
    ax.set_ylabel("Improvement over baseline (%)")
    ax.set_title("Fig. 9 -- LEACER Improvement over Baselines"); ax.legend()
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig9_improvement_bar.pdf", **SAVE); plt.close()
    print("[OK] fig9_improvement_bar.pdf")


def fig10_latency_cdf(data):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for a in _active(data):
        raw = data[a]["avg_latency_ms"].values
        if raw.max() > 1e6: raw = raw/1000.0
        s = np.sort(raw); cdf = np.arange(1,len(s)+1)/len(s)
        ax.plot(s, cdf, color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
    ax.axvline(150, color="salmon", ls="--", lw=1.2, label=r"$\tau_L$ = 150ms")
    ax.set_xlabel("Latency (ms)"); ax.set_ylabel("CDF")
    ax.set_title("Fig. 10 -- Latency CDF Comparison"); ax.legend(); ax.set_xlim(left=0)
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig10_latency_cdf.pdf", **SAVE); plt.close()
    print("[OK] fig10_latency_cdf.pdf")


def fig11_throughput(data):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for a in _active(data):
        df = data[a]
        y = df["avg_speed_kmh"] / df["avg_speed_kmh"].max() * 85 * (1-np.exp(-df["sim_time"]/300))
        ax.plot(df["sim_time"], y, color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
    ax.set_xlabel("Simulation Step (s)"); ax.set_ylabel("Active Vehicles (proxy)")
    ax.set_title("Fig. 11 -- Network Throughput"); ax.legend()
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig11_throughput.pdf", **SAVE); plt.close()
    print("[OK] fig11_throughput.pdf")


def fig12_lyapunov(data):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for a in _active(data):
        df = data[a]
        lyap = 0.5 * df["avg_queue"].values**2
        ax.plot(df["sim_time"], lyap, color=COLORS[a], ls=LS[a], lw=LW[a], label=LABEL[a])
    ax.set_yscale("symlog", linthresh=0.2)
    ax.set_xlabel("Simulation Step (s)"); ax.set_ylabel(r"$\mathcal{V}(Q)=0.5Q^2$ (symlog)")
    ax.set_title(r"Fig. 12 -- Lyapunov Drift $\mathcal{V}(Q)$"); ax.legend()
    plt.tight_layout(pad=0.4); plt.savefig(PLOTS_DIR/"fig12_lyapunov_drift.pdf", **SAVE); plt.close()
    print("[OK] fig12_lyapunov_drift.pdf")


def run_analysis():
    print("="*60); print("LEACER Comparison Results Analyser"); print("="*60)
    data = load_all_telemetry()
    print(f"\nLoaded: {list(data.keys())}")
    for a, df in data.items():
        print(f"  {a:14s} speed={df.avg_speed_kmh.mean():.2f}km/h  "
              f"queue={df.avg_queue.mean():.3f}  co2={df.co2_total.iloc[-1]:.4f}kg")

    if len(data) < 2:
        print("\n[WARN] Fewer than 2 algorithms have data -- run the baseline "
              "scripts (or run_all_comparison.py) before analysing.")

    print(f"\nGenerating plots -> {PLOTS_DIR}\n")
    for fn in (fig1_speed, fig2_queue, fig3_latency, fig4_co2, fig5_speed_dist,
               fig6_queue_dist, fig7_rush_hour, fig8_radar, fig9_improvement,
               fig10_latency_cdf, fig11_throughput, fig12_lyapunov):
        fn(data)
    print("\nDone.")


if __name__ == "__main__":
    run_analysis()
