"""
LEACER Framework — Configuration
=================================
All hyperparameters, thresholds, and system constants.
"""

DTSA_CONFIG = {
    "kalman_process_noise": 1e-4,
    "kalman_obs_noise":     1e-2,
    "state_vector_dim":     6,
    "fusion_window_sec":    5,
    "rsu_coverage_m":       500,
    "v2x_freq_hz":          10,
}

GRU_CONFIG = {
    "input_dim":        6,
    "hidden_dim":       128,
    "num_layers":       2,
    "dropout":          0.2,
    "lookback_steps":   20,
    "forecast_horizon": 10,
    "learning_rate":    1e-3,
    "batch_size":       64,
    "epochs":           100,
}

GAT_CONFIG = {
    "node_feature_dim": 8,
    "edge_feature_dim": 4,
    "hidden_dim":       64,
    "num_heads":        4,
    "num_layers":       3,
    "dropout":          0.1,
    "output_dim":       32,
    "alpha_leaky_relu": 0.2,
}

PPO_CONFIG = {
    "state_dim":        64,
    "action_dim":       10,
    "lr_actor":         3e-4,
    "lr_critic":        1e-3,
    "gamma":            0.99,
    "gae_lambda":       0.95,
    "clip_epsilon":     0.2,
    "entropy_coeff":    0.01,
    "value_loss_coeff": 0.5,
    "max_grad_norm":    0.5,
    "update_epochs":    10,
    "mini_batch_size":  32,
    "rollout_steps":    512,
}

# F = αT + βE + γC + δL
COST_WEIGHTS = {
    "alpha": 0.35,
    "beta":  0.25,
    "gamma": 0.25,
    "delta": 0.15,
}

EDGE_SERVER_CONFIG = {
    "device":               "cpu",
    "max_inference_ms":     10,
    "model_format":         "onnx",
    "max_candidate_routes": 10,
}

CEFAR_CONFIG = {
    "tau_congestion":       0.75,
    "tau_latency_ms":       150.0,
    "tau_energy_kwh":       0.85,
    "lyapunov_epsilon":     0.01,
    "drift_bound_B":        5.0,
    "omega_weights":        [0.5, 0.3, 0.2],
    "mesh_broadcast_hz":    2,
    "max_reroute_attempts": 3,
    "fallback_to_static":   True,
}
