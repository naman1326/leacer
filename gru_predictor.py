"""
GRU Traffic Predictor — Layer 2, LEACER Framework

Predicts future traffic state T+h given a look-back window of TSVs.
Input:  (batch, tau, 4)  — tau steps of [S, D, Q, L] per edge
Output: (batch, H, 4)    — H-step prediction horizon
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Model (used when torch is available)
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class GRUTrafficPredictor(nn.Module):
        """
        Multi-step GRU predictor.

        Architecture:
          Input → GRU (hidden_dim, num_layers) → Linear → Output

        Training loss:
          L = (1/H) * sum_h ||T_hat_{t+h} - T_{t+h}||^2
        """

        def __init__(self, input_dim=4, hidden_dim=64,
                     num_layers=2, horizon=5, tau=10, dropout=0.1):
            super().__init__()
            self.horizon    = horizon
            self.hidden_dim = hidden_dim

            self.gru = nn.GRU(
                input_size  = input_dim,
                hidden_size = hidden_dim,
                num_layers  = num_layers,
                batch_first = True,
                dropout     = dropout if num_layers > 1 else 0.0
            )

            # Project to horizon * features in one shot
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, horizon * input_dim)
            )

        def forward(self, x):
            """
            x: (B, tau, input_dim)
            returns: (B, horizon, input_dim)
            """
            _, h_n = self.gru(x)           # h_n: (num_layers, B, H)
            last_h = h_n[-1]               # (B, hidden_dim)
            out = self.head(last_h)        # (B, horizon * input_dim)
            return out.view(x.size(0), self.horizon, -1)

        def predict(self, tsv_window: np.ndarray) -> np.ndarray:
            """
            Convenience wrapper. tsv_window: (tau, 4) numpy array.
            Returns predicted states: (horizon, 4) numpy array.
            """
            self.eval()
            with torch.no_grad():
                x = torch.tensor(tsv_window, dtype=torch.float32).unsqueeze(0)
                return self.forward(x).squeeze(0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Numpy fallback (exponential smoothing) when torch not available
# ─────────────────────────────────────────────────────────────────────────────

class GRUPredictorNumpy:
    """
    Lightweight fallback: double exponential smoothing over look-back window.
    Used when PyTorch is not installed.
    alpha controls smoothing [0,1].  Returns (horizon, 4) forecast.
    """
    def __init__(self, horizon=5, alpha=0.3):
        self.horizon = horizon; self.alpha = alpha

    def predict(self, tsv_window: np.ndarray) -> np.ndarray:
        """tsv_window: (tau, 4)  →  forecast: (horizon, 4)"""
        tau, d = tsv_window.shape
        level = tsv_window[0].copy()
        trend = np.zeros(d)
        for t in range(tau):
            new_level = self.alpha * tsv_window[t] + (1-self.alpha) * (level + trend)
            trend = self.alpha * (new_level - level) + (1-self.alpha) * trend
            level = new_level
        return np.stack([level + (h+1)*trend for h in range(self.horizon)])


# ─────────────────────────────────────────────────────────────────────────────
# TSV Normalizer — zero-mean, unit-variance per feature
# ─────────────────────────────────────────────────────────────────────────────

class TSVNormalizer:
    """Fits mean/std on training data, transforms TSV arrays to [S,D,Q,L]."""
    def __init__(self):
        self.mean_ = None; self.std_ = None

    def fit(self, data: np.ndarray):
        """data: (N, 4)"""
        self.mean_ = data.mean(axis=0)
        self.std_  = data.std(axis=0) + 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_


# ─────────────────────────────────────────────────────────────────────────────
# Training Utility (PyTorch only)
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    def train_gru(model: GRUTrafficPredictor,
                  X: np.ndarray, Y: np.ndarray,
                  epochs=50, lr=1e-3, batch_size=32) -> List[float]:
        """
        X: (N, tau, 4)   Y: (N, horizon, 4)
        Returns per-epoch MSE loss list.
        """
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        N = len(X_t); losses = []
        model.train()
        for ep in range(epochs):
            idx = torch.randperm(N)
            epoch_loss = 0.0; steps = 0
            for i in range(0, N, batch_size):
                b = idx[i:i+batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_t[b]), Y_t[b])
                loss.backward(); optimizer.step()
                epoch_loss += loss.item(); steps += 1
            avg = epoch_loss / max(steps,1)
            losses.append(avg)
            if (ep+1) % 10 == 0:
                print(f"  Epoch {ep+1:3d}/{epochs}  MSE={avg:.4f}")
        return losses


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tau, horizon, feat = 10, 5, 4
    window = np.random.rand(tau, feat) * np.array([60, 50, 10, 120])

    if TORCH_AVAILABLE:
        model = GRUTrafficPredictor(horizon=horizon, tau=tau)
        pred  = model.predict(window)
        print(f"[PyTorch GRU] Forecast shape: {pred.shape}")
    else:
        model = GRUPredictorNumpy(horizon=horizon)
        pred  = model.predict(window)
        print(f"[Numpy fallback] Forecast shape: {pred.shape}")

    print("Predicted T+1:", pred[0].round(2))
    print("Predicted T+5:", pred[-1].round(2))
