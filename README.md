# LEACER: Lightweight Edge-AI Cooperative Event-triggered Re-routing

**LEACER** is a multi-layered Edge-AI traffic management and vehicle routing framework designed for the **SUMO (Simulation of Urban MObility)** environment. It optimizes urban traffic flow, reduces energy consumption (CO₂ emissions), and ensures system stability using a combination of Graph Attention Networks (GAT), Deep Reinforcement Learning (PPO), and Lyapunov Stability theory.

## 🚀 Key Features

*   **Layer 1: Dynamic Traffic State Analyzer (DTSA)** — Real-time data acquisition and fusion from RSUs, vehicle OBUs, and IoT sensors to generate high-fidelity Traffic State Vectors (TSVs).
*   **Layer 2: Multi-Objective Edge-AI Route Optimizer (MO-EARO)** — Uses GAT for road graph embeddings and PPO to compute Pareto-optimal routes balancing travel time, energy, congestion, and latency.
*   **Layer 3: Cooperative Event-triggered Fault-Adaptive Re-routing (CEFAR)** — A robust adaptation layer that monitors KPIs and triggers re-routing during congestion or failures while maintaining queue stability via Lyapunov drift-plus-penalty constraints.
*   **Comprehensive Benchmark Suite** — Full simulation pipeline comparing LEACER against 6 classical and AI baselines (**Dijkstra, A*, Static, Cloud-DQN, GCN-Route, and MARL**) with detailed telemetry tracking (CO₂, Speed, Queue Length, Latency).
*   **Predictive Analytics** — Includes a GRU-based predictor for short-term traffic forecasting.

## 📊 Performance Visualization

The framework generates 12 detailed IEEE-standard comparison plots, including:
*   Cumulative CO₂ Emissions (Fig. 4)
*   Average Network Speed (Fig. 1)
*   Queue Length Distribution (Fig. 6)
*   KPI Radar Charts (Fig. 8)
*   Latency CDF & Throughput (Fig. 10 & 11)
*   Lyapunov Drift & Stability Analysis (Fig. 12)

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8+
*   [SUMO (Simulation of Urban MObility)](https://sumo.dlr.de/docs/Installing/index.html)
*   `traci`, `sumolib`, `numpy`, `pandas`, `matplotlib`, `torch`

### Quick Start

1.  **Train the Intelligence Layer & AI Baselines (Optional):**
    ```bash
    # Train Traffic Predictor & PPO Policy
    python simulation/train_gru.py
    python simulation/train_ppo.py --episodes 100

    # Train Baseline Models (DQN, MARL, GCN)
    python simulation/baseline_dqn.py --mode train --episodes 400
    python simulation/baseline_marl.py --mode train --episodes 300
    python simulation/baseline_gcn.py --mode train --epochs 60
    ```

2.  **Run Master Comparison Pipeline (Runs LEACER + All 6 Baselines):**
    ```bash
    # Run complete end-to-end evaluation & regenerate all 12 IEEE plots
    python simulation/run_all_comparison.py --steps 3600
    ```

3.  **Run Single Simulation with GUI Visualization:**
    ```bash
    python simulation/run_simulation.py --mode leacer --steps 3600 --gui
    ```

### Results & Output Directory
*   **KPI Summaries:** `simulation/results/leacer_run_summary.json`, `simulation/results/kpi_summary.csv`
*   **Telemetry Data:** `simulation/results/*_telemetry.csv` (Git-ignored)
*   **Generated Plots:** `simulation/results/plots/*.pdf` & `.png` (Git-ignored)
*   **Trained Models:** `simulation/data/*.pt`
*   **Architecture Diagram:** `LEACER.drawio`

## 📂 Data & Results Access

All generated simulation telemetry, CSV logs, XML outputs, and plot PDFs are generated locally in `simulation/results/` and `simulation/results/plots/`. 

To maintain a lightweight repository, generated results, raw telemetry CSVs, and output graphs are excluded from Git tracking. You can regenerate all telemetry and figures at any time using:

```bash
# Evaluate and plot stored telemetry data
python simulation/results_analyzer.py
```

## 📖 Research Paper
The technical details, mathematical formulations, and experimental results of the LEACER framework are documented in the following research paper:

*   **View on Overleaf:** [LEACER: Lightweight Edge-AI Cooperative Event-triggered Re-routing](https://www.overleaf.com/read/mstksgzdtkzm#c00324)

## 📐 Three-Layer Architecture

<img src="leacer_ss.png" width="100%" alt="LEACER Framework Architecture">

```mermaid
graph TD
    subgraph "Layer 1 — DTSA (Data Acquisition)"
        A["RSU Aggregator"] --> B["Kalman Filter Fusion"]
        C["Vehicle OBU Data"] --> A
        D["IoT Sensor Data"] --> A
        E["Signal Phase Data"] --> A
        B --> F["Traffic State Vectors (TSV)"]
    end

    subgraph "Layer 2 — MO-EARO (Edge Intelligence)"
        F --> G["GAT Encoder (Graph Embeddings)"]
        G --> H["PPO Policy Agent"]
        F --> GRU["GRU Traffic Predictor"]
        GRU --> H
        H --> I["Multi-Objective Cost: F = αT + βE + γC + δL"]
        I --> J["Pareto-Optimal Route"]
    end

    subgraph "Layer 3 — CEFAR (Cooperative Adaptation)"
        J --> K["Threshold Monitor"]
        K --> L["Cooperative RSU Mesh"]
        L --> M["Lyapunov Stabilizer"]
        M --> N["Route Dispatcher (V2I / V2V)"]
    end
```

| Layer | Module | Purpose | Source |
|-------|--------|---------|--------|
| **1** | **DTSA** — Dynamic Traffic State Analyzer | Fuses V2X OBU, IoT sensor, and signal-phase data via Kalman filtering into per-edge Traffic State Vectors `{S, D, Q, L}` | `dtsa.py` |
| **2** | **GAT Encoder** | Encodes road-network graph `G=(V,E)` into spatial node embeddings using multi-head Graph Attention | `gat_encoder.py` |
| **2** | **GRU Predictor** | Short-term traffic forecasting over a look-back window of TSVs | `gru_predictor.py` |
| **2** | **MO-EARO** — Multi-Objective Route Optimizer | PPO actor-critic selects Pareto-optimal routes minimizing `F = 0.35·T + 0.25·E + 0.25·C + 0.15·L` | `mo_earo.py` |
| **3** | **CEFAR** — Cooperative Event-triggered Fault-Adaptive Re-routing | Monitors KPIs against thresholds, fuses cooperative RSU mesh states, enforces queue stability via Lyapunov drift-plus-penalty, and dispatches re-routing events | `cefar.py` |

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch (with NumPy fallbacks for all neural modules) |
| Traffic Simulator | SUMO via `traci` / `sumolib` |
| Core Algorithms | GAT, PPO (Actor-Critic), GRU (Seq2Seq), Kalman Filter, Lyapunov Stability |
| Visualization | Matplotlib (12 IEEE-standard comparison plots) |
| Data Processing | NumPy, Pandas |

## 📁 File Structure

```
leacer/
├── config.py                  # All hyperparameters & thresholds
├── dtsa.py                    # Layer 1: Traffic State Aggregator (Kalman fusion)
├── gat_encoder.py             # Layer 2: GAT graph encoder (PyTorch + NumPy fallback)
├── gru_predictor.py           # Layer 2: GRU traffic forecaster (PyTorch + NumPy fallback)
├── mo_earo.py                 # Layer 2: PPO actor-critic + MO cost evaluator
├── cefar.py                   # Layer 3: CEFAR controller + Lyapunov stabilizer
├── run_leacer.py              # Standalone single-cycle pipeline demo
├── LEACER.drawio              # Architecture diagram (draw.io)
│
└── simulation/
    ├── sumo_env.py              # SUMO environment wrapper (traci)
    ├── sumo_data_adapter.py     # Bridges SUMO ↔ LEACER data structures
    ├── multi_rsu.py             # Multi-RSU simulation with cooperative mesh
    ├── routing_utils.py         # Graph routing utilities (Dijkstra, A*, etc.)
    ├── leacer_runner.py         # Full LEACER simulation runner
    ├── run_simulation.py        # CLI entry point: single-mode simulation (--gui)
    ├── run_all_comparison.py    # CLI entry point: LEACER + 6 baselines end-to-end
    ├── results_analyzer.py      # Post-hoc analysis & 12-plot generation
    │
    ├── train_gru.py             # GRU training script
    ├── train_ppo.py             # PPO training script
    ├── baseline_classical.py    # Dijkstra / A* / Static baselines
    ├── baseline_dqn.py          # Cloud-DQN baseline
    ├── baseline_gcn.py          # GCN-Route baseline
    ├── baseline_marl.py         # MARL baseline
    ├── baseline_runner.py       # Unified baseline execution harness
    │
    ├── sumo_cfg/                # SUMO network, routes, detectors, config files
    ├── data/                    # Pre-trained model weights (.pt) + training data
    └── results/                 # Telemetry CSVs, SUMO XMLs, KPI summaries, plots/
```

## 🏁 Benchmarked Baselines (6 Total)

LEACER is evaluated against 3 classical and 3 AI-based routing baselines:

| # | Baseline | Type | Description | Source |
|---|----------|------|-------------|--------|
| 1 | **Dijkstra** | Classical | Shortest-path by edge weight | `simulation/baseline_classical.py` |
| 2 | **A*** | Classical | Heuristic-guided shortest-path | `simulation/baseline_classical.py` |
| 3 | **Static** | Classical | Fixed pre-computed routes (no adaptation) | `simulation/baseline_classical.py` |
| 4 | **Cloud-DQN** | AI (Centralized) | Deep Q-Network with cloud-based inference | `simulation/baseline_dqn.py` |
| 5 | **GCN-Route** | AI (Graph) | Graph Convolutional Network routing | `simulation/baseline_gcn.py` |
| 6 | **MARL** | AI (Multi-Agent) | Multi-Agent Reinforcement Learning | `simulation/baseline_marl.py` |

---
*Developed for research in Intelligent Transportation Systems (ITS) and Edge-AI.*

