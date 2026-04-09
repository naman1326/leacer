r"""
train_gru.py — GRU Traffic Predictor Training
==============================================
Trains the GRU on SUMO-generated sequences from simulation/data/
and saves weights to simulation/data/gru_weights.pt

Usage:
    cd C:\Users\ghoda\Downloads\leacer
    python simulation/train_gru.py

Output:
    simulation/data/gru_weights.pt
    simulation/data/gru_normalizer.npz
    simulation/data/gru_training_loss.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Add leacer root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "simulation" / "data"

# ── Hyper-parameters ─────────────────────────────────────────────────────────
TAU          = 20     # look-back window (steps)
HORIZON      = 10     # prediction horizon (steps)
HIDDEN_DIM   = 64
NUM_LAYERS   = 2
DROPOUT      = 0.1
EPOCHS       = 5
BATCH_SIZE   = 64
LR           = 1e-3
VAL_SPLIT    = 0.15
SEED         = 42

np.random.seed(SEED)

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("=" * 60)
print("LEACER — GRU Training")
print("=" * 60)

X_path = DATA_DIR / "gru_X.npy"
Y_path = DATA_DIR / "gru_Y.npy"

if not X_path.exists() or not Y_path.exists():
    print(f"[ERROR] Dataset not found at {DATA_DIR}")
    print("  Run:  python simulation/run_simulation.py --mode dataset --episodes 50")
    sys.exit(1)

X = np.load(X_path)[:10000]   # (N, TAU, 4)
Y = np.load(Y_path)[:10000]   # (N, HORIZON, 4)

print(f"\n[Data]  X={X.shape}  Y={Y.shape}")
print(f"        Features: [speed_kmh, density_veh_km, queue_veh, travel_time_s]")

# ── 2. Normalise (zero-mean, unit-variance per feature) ───────────────────────
N, tau_actual, n_feat = X.shape

X_flat = X.reshape(-1, n_feat)
mu  = X_flat.mean(axis=0)
sig = X_flat.std(axis=0) + 1e-8

X_norm = ((X - mu) / sig).astype(np.float32)
Y_norm = ((Y - mu) / sig).astype(np.float32)

np.savez(DATA_DIR / "gru_normalizer.npz", mu=mu, sig=sig)
print(f"\n[Norm]  mu ={mu.round(3)}  sig={sig.round(3)}")

# ── 3. Train/val split ────────────────────────────────────────────────────────
idx     = np.random.permutation(N)
val_n   = int(N * VAL_SPLIT)
val_idx = idx[:val_n]
trn_idx = idx[val_n:]

X_trn, Y_trn = X_norm[trn_idx], Y_norm[trn_idx]
X_val, Y_val = X_norm[val_idx], Y_norm[val_idx]
print(f"\n[Split] train={len(X_trn)}  val={len(X_val)}")

# ── 4. Try PyTorch; fall back to numpy ───────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH = True
    print(f"\n[Torch] {torch.__version__}  device=cpu")
except ImportError:
    TORCH = False
    print("\n[WARN] PyTorch not found — saving numpy weights only")

if TORCH:
    torch.manual_seed(SEED)

    class GRUPredictor(nn.Module):
        def __init__(self, feat=4, hidden=HIDDEN_DIM, layers=NUM_LAYERS,
                     horizon=HORIZON, drop=DROPOUT):
            super().__init__()
            self.horizon = horizon
            self.feat    = feat
            self.gru = nn.GRU(feat, hidden, layers,
                               batch_first=True,
                               dropout=drop if layers > 1 else 0.0)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, horizon * feat),
            )

        def forward(self, x):
            # x: (B, tau, feat)
            _, h = self.gru(x)          # h: (layers, B, hidden)
            out  = self.head(h[-1])     # (B, horizon*feat)
            return out.view(x.size(0), self.horizon, self.feat)

    model     = GRUPredictor()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=8, factor=0.5)
    criterion = nn.MSELoss()

    Xt = torch.tensor(X_trn); Yt = torch.tensor(Y_trn)
    Xv = torch.tensor(X_val); Yv = torch.tensor(Y_val)
    loader = DataLoader(TensorDataset(Xt, Yt),
                        batch_size=BATCH_SIZE, shuffle=True)

    trn_losses, val_losses = [], []
    best_val, best_state   = float("inf"), None

    print(f"\n{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}")
    print("-" * 42)

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), Yv).item()

        scheduler.step(val_loss)
        trn_losses.append(ep_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"{ep:>6}  {ep_loss:>10.5f}  {val_loss:>10.5f}  {lr_now:>10.2e}")

    # ── Save best weights ──────────────────────────────────────────────────
    model.load_state_dict(best_state)
    save_path = DATA_DIR / "gru_weights.pt"
    torch.save({
        "model_state": best_state,
        "model_config": dict(feat=4, hidden=HIDDEN_DIM, layers=NUM_LAYERS,
                              horizon=HORIZON, drop=DROPOUT),
        "best_val_mse": best_val,
        "epochs_trained": EPOCHS,
        "norm_mu": mu.tolist(),
        "norm_sig": sig.tolist(),
    }, save_path)
    print(f"\n[Saved] {save_path}")
    print(f"[Best]  val_MSE = {best_val:.5f}")

    # ── Plot training curve ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(trn_losses, label="Train MSE", lw=1.5)
    ax.plot(val_losses, label="Val MSE",   lw=1.5, ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("GRU Traffic Predictor — Training Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "gru_training_loss.png", dpi=120)
    plt.close()
    print(f"[Plot]  {DATA_DIR / 'gru_training_loss.png'}")

    # ── Quick inference test ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        sample = torch.tensor(X_val[:1])
        pred   = model(sample).numpy()[0]        # (HORIZON, 4)
        truth  = Y_val[:1][0]                    # (HORIZON, 4)
        pred_denorm  = pred  * sig + mu
        truth_denorm = truth * sig + mu

    print(f"\n[Test]  1-step ahead prediction vs truth (de-normalised):")
    feat_names = ["speed_kmh", "density", "queue", "travel_time"]
    for i, fn in enumerate(feat_names):
        print(f"  {fn:14s}  pred={pred_denorm[0,i]:7.2f}  truth={truth_denorm[0,i]:7.2f}")

else:
    # ── Numpy fallback: save exponential-smoothing params ──────────────────
    alpha = 0.3
    save_path = DATA_DIR / "gru_weights.pt"
    np.savez(str(save_path).replace(".pt", "_numpy.npz"),
             alpha=np.array([alpha]), mu=mu, sig=sig,
             model_type=np.array(["exp_smooth"]))
    print(f"\n[Saved] numpy fallback weights → gru_weights_numpy.npz")
    print("  Install PyTorch for full GRU training:  pip install torch")

print("\n" + "=" * 60)
print("GRU training complete.")
print("=" * 60)
