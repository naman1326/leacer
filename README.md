# LEACER: Lightweight Edge-AI Cooperative Event-triggered Re-routing

**LEACER** is a multi-layered Edge-AI traffic management and vehicle routing framework designed for the **SUMO (Simulation of Urban MObility)** environment. It optimizes urban traffic flow, reduces energy consumption (CO₂ emissions), and ensures system stability using a combination of Graph Attention Networks (GAT), Deep Reinforcement Learning (PPO), and Lyapunov Stability theory.

## 🚀 Key Features

*   **Layer 1: Dynamic Traffic State Analyzer (DTSA)** — Real-time data acquisition and fusion from RSUs, vehicle OBUs, and IoT sensors to generate high-fidelity Traffic State Vectors (TSVs).
*   **Layer 2: Multi-Objective Edge-AI Route Optimizer (MO-EARO)** — Uses GAT for road graph embeddings and PPO to compute Pareto-optimal routes balancing travel time, energy, congestion, and latency.
*   **Layer 3: Cooperative Event-triggered Fault-Adaptive Re-routing (CEFAR)** — A robust adaptation layer that monitors KPIs and triggers re-routing during congestion or failures while maintaining queue stability via Lyapunov drift-plus-penalty constraints.
*   **Integrated Simulation Suite** — Full pipeline to run LEACER against industry-standard baselines (Dijkstra, A*, Static) with detailed telemetry tracking (CO₂, Speed, Queue Length, Latency).
*   **Predictive Analytics** — Includes a GRU-based predictor for short-term traffic forecasting.

## 📊 Performance Visualization

The framework generates 12 detailed comparison plots, including:
*   Cumulative CO₂ Emissions (Fig. 4)
*   Average Network Speed (Fig. 1)
*   Queue Length Distribution (Fig. 6)
*   KPI Radar Charts (Fig. 8)
*   Lyapunov Drift & Stability Analysis (Fig. 12)

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8+
*   [SUMO (Simulation of Urban MObility)](https://sumo.dlr.de/docs/Installing/index.html)
*   `traci`, `sumolib`, `numpy`, `pandas`, `matplotlib`, `torch`

### Quick Start

1.  **Train the Intelligence Layer (Optional):**
    ```bash
    # Train Traffic Predictor
    python simulation/train_gru.py
    # Train Routing Policy
    python simulation/train_ppo.py --episodes 100
    ```

2.  **Run with GUI Visualization:**
    ```bash
    python simulation/run_simulation.py --mode leacer --steps 3600 --gui
    ```

3.  **Run Full Comparative Analysis (Generates Plots):**
    ```bash
    python simulation/run_simulation.py --mode all --steps 3600 --plot
    ```

### Results & Data
*   **KPI Summary:** `simulation/results/leacer_run_summary.json`
*   **Telemetry CSVs:** `simulation/results/*.csv`
*   **IEEE-Standard Plots:** `simulation/results/plots/*.pdf` (and .png)

## 📖 Framework Architecture

1.  **Data Layer:** RSU Aggregator + IoT Fusion
2.  **Intelligence Layer:** GAT Encoder + PPO Policy
3.  **Control Layer:** CEFAR Trigger + Lyapunov Stabilizer

---
*Developed for research in Intelligent Transportation Systems (ITS) and Edge-AI.*
