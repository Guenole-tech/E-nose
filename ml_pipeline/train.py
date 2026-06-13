#!/usr/bin/env python3
"""
ml_pipeline/train.py
====================
Industrial e-Nose — Full ML Training Pipeline

Stages
------
1. Synthetic dataset generation with realistic physics-based drift and noise
2. Signal feature engineering from raw R_ratio time-series
3. Gas classification (Random Forest) + concentration regression (SVR)
4. Cross-validation and performance evaluation
5. Model export to C source code (portable, no runtime dependencies)

Supported gases (class labels):
    0 = Clean Air
    1 = Ethanol (C2H5OH)
    2 = Ammonia (NH3)
    3 = CO (Carbon Monoxide)
    4 = Acetone (C3H6O)

The exported C code implements inference directly using:
    - Random Forest: hard-coded decision tree traversal
    - Regression: SVR dual coefficients baked as float arrays

Usage
-----
    python train.py [--n-samples 5000] [--output-dir ../firmware/src/ml]

Dependencies
------------
    pip install numpy scikit-learn joblib
"""

import argparse
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED     = 42
N_SENSORS       = 4
SEQUENCE_LEN    = 60   # Number of time steps per measurement window (60 × 200 ms = 12 s)
N_GAS_CLASSES   = 5
GAS_NAMES       = ["clean_air", "ethanol", "ammonia", "co", "acetone"]

# Concentration range per gas (ppm)
CONCENTRATION_RANGES = {
    0: (0.0,   0.0),     # clean_air
    1: (5.0,  500.0),    # ethanol
    2: (1.0,  300.0),    # ammonia
    3: (1.0,  200.0),    # co
    4: (5.0,  1000.0),   # acetone
}

# Sensor sensitivity matrix: baseline R_air values (Ω) per sensor
R_AIR_BASELINE = np.array([45000.0, 38000.0, 52000.0, 41000.0])  # Ω

# Gas response matrix: R_ratio = R_gas/R_air = A * exp(-B * C_ppm)
# Each row is a gas, each column is a sensor channel.
# Values calibrated to approximate MQ-series datasheet curves.
RESPONSE_A = np.array([
    [1.00, 1.00, 1.00, 1.00],  # clean_air  — no response
    [0.45, 0.60, 0.80, 0.55],  # ethanol    — strong on ch0, moderate ch1–3
    [0.70, 0.40, 0.65, 0.90],  # ammonia    — strong on ch1, ch3
    [0.85, 0.75, 0.50, 0.80],  # co
    [0.50, 0.85, 0.70, 0.45],  # acetone
])

RESPONSE_B = np.array([
    [0.000, 0.000, 0.000, 0.000],  # clean_air
    [0.012, 0.008, 0.004, 0.010],  # ethanol
    [0.006, 0.015, 0.007, 0.005],  # ammonia
    [0.004, 0.006, 0.010, 0.005],  # co
    [0.010, 0.003, 0.006, 0.012],  # acetone
])

# Drift model: sensors accumulate a slow multiplicative drift over time
DRIFT_RATE_PER_HOUR = 0.001   # 0.1% per hour
MEASUREMENT_NOISE_STD = 0.02  # Fractional Gaussian noise on R_ratio

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SensorWindow:
    """Raw time-series window from one measurement event."""
    r_ratio_sequence: np.ndarray   # shape (SEQUENCE_LEN, N_SENSORS)
    gas_label: int
    concentration_ppm: float
    drift_factor: float            # Simulated aging factor [1.0, 1.5]
    snr_db: float                  # Added noise level


@dataclass
class FeatureVector:
    """Extracted features from one SensorWindow."""
    features: np.ndarray           # shape (N_FEATURES,)
    gas_label: int
    concentration_ppm: float

    # Feature names (class-level, matches extraction order)
    FEATURE_NAMES: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Synthetic Data Generation
# ---------------------------------------------------------------------------

def generate_r_ratio_sequence(
    gas_label: int,
    concentration_ppm: float,
    drift_factor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a realistic R_ratio time-series for one gas exposure event.

    Physical model:
        R_ratio(t, C) = A * exp(-B * C) * (1 - exp(-t/τ_rise)) + 1 * exp(-t/τ_rise)

    This captures:
        - Exponential approach to steady-state (τ_rise ≈ 20–30 s)
        - Sensor cross-selectivity via the A/B matrix
        - Multiplicative drift (aging): R_ratio_drifted = R_ratio * drift_factor
        - Additive Gaussian noise (thermal + electronic)

    Returns
    -------
    np.ndarray, shape (SEQUENCE_LEN, N_SENSORS)
    """
    t = np.arange(SEQUENCE_LEN, dtype=float)

    # Rise time varies slightly per sensor (τ = 20–35 steps × 200 ms = 4–7 s)
    tau_rise = rng.uniform(20.0, 35.0, size=N_SENSORS)

    # Steady-state R_ratio for this gas at this concentration
    if concentration_ppm > 0.0 and gas_label > 0:
        A   = RESPONSE_A[gas_label]   # (N_SENSORS,)
        B   = RESPONSE_B[gas_label]   # (N_SENSORS,)
        r_ss = A * np.exp(-B * concentration_ppm)  # (N_SENSORS,)
    else:
        r_ss = np.ones(N_SENSORS)

    # Temporal envelope: exponential approach to steady state
    # r(t) = r_ss + (1 - r_ss) * exp(-t / tau)
    envelope = np.exp(-t[:, None] / tau_rise[None, :])   # (T, N_SENSORS)
    r_clean  = r_ss[None, :] + (1.0 - r_ss[None, :]) * envelope  # (T, N_SENSORS)

    # Apply drift (multiplicative aging artifact)
    r_drifted = r_clean * drift_factor

    # Add Gaussian noise
    noise_std = MEASUREMENT_NOISE_STD * r_drifted
    noise     = rng.normal(0.0, noise_std)
    r_noisy   = r_drifted + noise

    # Clip to physically plausible range [0.01, 5.0]
    r_noisy = np.clip(r_noisy, 0.01, 5.0)

    return r_noisy.astype(np.float32)


def generate_dataset(n_samples: int, rng: np.random.Generator) -> list[SensorWindow]:
    """
    Generate n_samples measurement windows with balanced gas class distribution
    and random concentration / drift levels.

    Returns list of SensorWindow objects.
    """
    windows: list[SensorWindow] = []
    samples_per_class = n_samples // N_GAS_CLASSES

    for gas_label in range(N_GAS_CLASSES):
        c_min, c_max = CONCENTRATION_RANGES[gas_label]

        for _ in range(samples_per_class):
            concentration = rng.uniform(c_min, c_max) if c_max > 0.0 else 0.0
            drift_factor  = rng.uniform(1.0, 1.3)  # 0–30% aging
            snr_db        = rng.uniform(20.0, 40.0)

            sequence = generate_r_ratio_sequence(
                gas_label, concentration, drift_factor, rng
            )

            windows.append(SensorWindow(
                r_ratio_sequence   = sequence,
                gas_label          = gas_label,
                concentration_ppm  = concentration,
                drift_factor       = drift_factor,
                snr_db             = snr_db,
            ))

    # Shuffle
    indices = rng.permutation(len(windows))
    return [windows[i] for i in indices]


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

def extract_features(window: SensorWindow) -> np.ndarray:
    """
    Extract a discriminative feature vector from a raw R_ratio time-series.

    Features per sensor channel (N_SENSORS × 7 = 28 features):
        1. max_slope          — maximum positive derivative (max responsiveness)
        2. min_slope          — minimum derivative (recovery rate)
        3. steady_state_mean  — mean over last 10 timesteps (equilibrium value)
        4. steady_state_std   — std over last 10 timesteps (stability)
        5. auc_normalized     — area under curve / SEQUENCE_LEN (response strength)
        6. time_to_half_ss    — timestep at which R crosses (1+R_ss)/2 (kinetics)
        7. peak_delta         — max(|R - 1|) (peak sensitivity)

    Global features (4 features):
        8.  ratio_ch0_ch1     — cross-ratio (selectivity discriminant)
        9.  ratio_ch0_ch2
        10. ratio_ch1_ch3
        11. ratio_ch2_ch3

    Total: 28 + 4 = 32 features
    """
    seq = window.r_ratio_sequence   # (T, N_SENSORS)
    T   = seq.shape[0]

    per_channel_features = []

    for ch in range(N_SENSORS):
        s = seq[:, ch]  # (T,)

        # 1. Max slope (max positive first derivative)
        diffs     = np.diff(s)
        max_slope = float(np.max(diffs))

        # 2. Min slope
        min_slope = float(np.min(diffs))

        # 3–4. Steady-state statistics (last 10 samples ≈ last 2 s)
        ss_window = s[-10:]
        ss_mean   = float(np.mean(ss_window))
        ss_std    = float(np.std(ss_window))

        # 5. Normalized AUC using trapezoidal rule
        auc_norm  = float(np.trapz(s) / T)

        # 6. Time to half steady-state
        half_ss   = (1.0 + ss_mean) / 2.0
        crossings = np.where(np.diff(np.sign(s - half_ss)))[0]
        t_half    = float(crossings[0]) / T if len(crossings) > 0 else 1.0

        # 7. Peak delta
        peak_delta = float(np.max(np.abs(s - 1.0)))

        per_channel_features.extend([
            max_slope, min_slope, ss_mean, ss_std, auc_norm, t_half, peak_delta
        ])

    # Global cross-channel ratios (steady-state values)
    ss_values = seq[-10:, :].mean(axis=0)  # (N_SENSORS,)
    eps       = 1e-6
    cross_features = [
        ss_values[0] / (ss_values[1] + eps),
        ss_values[0] / (ss_values[2] + eps),
        ss_values[1] / (ss_values[3] + eps),
        ss_values[2] / (ss_values[3] + eps),
    ]

    return np.array(per_channel_features + cross_features, dtype=np.float32)


FEATURE_NAMES = [
    f"{feat}_ch{ch}"
    for ch in range(N_SENSORS)
    for feat in ["max_slope", "min_slope", "ss_mean", "ss_std", "auc_norm", "t_half", "peak_delta"]
] + ["ratio_ch0_ch1", "ratio_ch0_ch2", "ratio_ch1_ch3", "ratio_ch2_ch3"]

N_FEATURES = len(FEATURE_NAMES)  # 32


def build_feature_matrix(windows: list[SensorWindow]):
    """
    Convert list of SensorWindow → (X, y_class, y_conc) numpy arrays.
    """
    X      = np.zeros((len(windows), N_FEATURES), dtype=np.float32)
    y_cls  = np.zeros(len(windows), dtype=np.int32)
    y_conc = np.zeros(len(windows), dtype=np.float32)

    for i, w in enumerate(windows):
        X[i]      = extract_features(w)
        y_cls[i]  = w.gas_label
        y_conc[i] = w.concentration_ppm

    return X, y_cls, y_conc


# ---------------------------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------------------------

def train_classifier(X: np.ndarray, y: np.ndarray) -> tuple[RandomForestClassifier, np.ndarray]:
    """
    Train Random Forest gas classifier with stratified 5-fold CV evaluation.

    Returns
    -------
    (fitted classifier, OOB-style accuracy per fold)
    """
    print("\n[Classifier] Training Random Forest (100 trees, max_depth=15)...")

    clf = RandomForestClassifier(
        n_estimators      = 100,
        max_depth         = 15,
        min_samples_split = 4,
        min_samples_leaf  = 2,
        max_features      = "sqrt",
        class_weight      = "balanced",
        n_jobs            = -1,
        random_state      = RANDOM_SEED,
        oob_score         = True,
    )

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores  = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"  CV F1-macro: {scores.mean():.4f} ± {scores.std():.4f}")

    clf.fit(X, y)
    print(f"  OOB accuracy: {clf.oob_score_:.4f}")
    print(f"  Feature importances (top 10):")
    top_idx = np.argsort(clf.feature_importances_)[::-1][:10]
    for rank, idx in enumerate(top_idx):
        print(f"    {rank+1:2d}. {FEATURE_NAMES[idx]:30s} {clf.feature_importances_[idx]:.4f}")

    # Final classification report on full dataset
    y_pred = clf.predict(X)
    print("\n  In-sample classification report:")
    print(classification_report(y, y_pred, target_names=GAS_NAMES, digits=4))

    return clf, scores


def train_regressor(X: np.ndarray, y_conc: np.ndarray, y_cls: np.ndarray):
    """
    Train SVR concentration regressor — one per gas class (excluding clean air).

    For each non-zero gas class, train a separate SVR on the subset of samples
    belonging to that class. This avoids conflating the concentration scales of
    different gases.

    Returns dict: {gas_label: sklearn Pipeline(scaler + SVR)}
    """
    print("\n[Regressor] Training SVR per gas class...")

    regressors = {}

    for gas_label in range(1, N_GAS_CLASSES):  # Skip clean_air (label 0)
        mask    = (y_cls == gas_label)
        X_sub   = X[mask]
        y_sub   = y_conc[mask]
        n_sub   = mask.sum()

        if n_sub < 20:
            print(f"  [WARN] Class {gas_label} ({GAS_NAMES[gas_label]}): "
                  f"only {n_sub} samples — skipping regression")
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=100.0, epsilon=0.5, gamma="scale")),
        ])
        pipe.fit(X_sub, y_sub)

        y_pred_reg  = pipe.predict(X_sub)
        mae         = mean_absolute_error(y_sub, y_pred_reg)
        r2          = r2_score(y_sub, y_pred_reg)
        c_min, c_max = CONCENTRATION_RANGES[gas_label]

        print(f"  {GAS_NAMES[gas_label]:12s}: n={n_sub:5d}  "
              f"MAE={mae:8.2f} ppm  R²={r2:.4f}  "
              f"range=[{c_min:.0f}, {c_max:.0f}] ppm")

        regressors[gas_label] = pipe

    return regressors


# ---------------------------------------------------------------------------
# 4. C Code Export
# ---------------------------------------------------------------------------

def _format_c_array(name: str, values, dtype: str = "float", cols: int = 8) -> str:
    """Format a Python list/array as a C static array declaration."""
    lines  = [f"static const {dtype} {name}[] = {{"]
    row    = []
    for v in values:
        if dtype == "float":
            row.append(f"{v:.8f}f")
        elif dtype == "int":
            row.append(f"{int(v)}")
        elif dtype == "double":
            row.append(f"{v:.10}")
        if len(row) == cols:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("};")
    return "\n".join(lines)


def export_random_forest_to_c(clf: RandomForestClassifier, output_path: str) -> None:
    """
    Export a trained Random Forest as a self-contained C source file.

    Each decision tree is encoded as three parallel arrays:
        left_child[node]    — left child index (-1 = leaf)
        right_child[node]   — right child index (-1 = leaf)
        feature[node]       — feature index used at split (-1 = leaf)
        threshold[node]     — split threshold
        class_value[node]   — predicted class at leaf (−1 = internal node)

    Inference function:
        uint8_t rf_predict(const float* features, float* confidence_out)
    """
    trees_code = []
    tree_struct_names = []

    for tree_idx, estimator in enumerate(clf.estimators_):
        tree   = estimator.tree_
        n_node = tree.node_count

        left     = tree.children_left.tolist()
        right    = tree.children_right.tolist()
        feature  = tree.feature.tolist()
        threshold = tree.threshold.tolist()

        # Class label at each leaf node (argmax of value at leaf)
        value     = tree.value  # (n_nodes, n_outputs, n_classes)
        leaf_cls  = []
        for node in range(n_node):
            if left[node] == -1:  # Leaf
                leaf_cls.append(int(np.argmax(value[node, 0, :])))
            else:
                leaf_cls.append(-1)

        prefix = f"tree{tree_idx}"
        tree_struct_names.append(prefix)

        block = [
            f"/* ----- Tree {tree_idx} ({n_node} nodes) ----- */",
            _format_c_array(f"{prefix}_left",      left,      "int"),
            _format_c_array(f"{prefix}_right",     right,     "int"),
            _format_c_array(f"{prefix}_feature",   feature,   "int"),
            _format_c_array(f"{prefix}_threshold", threshold, "float"),
            _format_c_array(f"{prefix}_leaf_cls",  leaf_cls,  "int"),
            "",
            f"static int {prefix}_infer(const float* X) {{",
            f"    int node = 0;",
            f"    while ({prefix}_left[node] != -1) {{",
            f"        if (X[{prefix}_feature[node]] <= {prefix}_threshold[node]) {{",
            f"            node = {prefix}_left[node];",
            f"        }} else {{",
            f"            node = {prefix}_right[node];",
            f"        }}",
            f"    }}",
            f"    return {prefix}_leaf_cls[node];",
            f"}}",
            "",
        ]
        trees_code.append("\n".join(block))

    # Build aggregate voting function
    n_trees = len(clf.estimators_)
    vote_calls = "\n".join(
        f"    votes[{prefix}_infer(X)]++;"
        for prefix in tree_struct_names
    )

    aggregate_fn = textwrap.dedent(f"""\
    uint8_t rf_predict(const float* X, float* confidence_out) {{
        int votes[{N_GAS_CLASSES}] = {{0}};
    {vote_calls}
        int best_cls  = 0;
        int best_votes = votes[0];
        for (int i = 1; i < {N_GAS_CLASSES}; ++i) {{
            if (votes[i] > best_votes) {{
                best_votes = votes[i];
                best_cls   = i;
            }}
        }}
        if (confidence_out) {{
            *confidence_out = (float)best_votes / {n_trees}.0f;
        }}
        return (uint8_t)best_cls;
    }}
    """)

    header_comment = textwrap.dedent(f"""\
    /**
     * @file rf_model.h
     * @brief Auto-generated Random Forest inference (e-Nose gas classifier)
     *        Generated by ml_pipeline/train.py — DO NOT EDIT MANUALLY
     *
     * Input:  float features[{N_FEATURES}]  (see feature extraction order)
     * Output: uint8_t gas_label  (0={', '.join(f'{i}={n}' for i, n in enumerate(GAS_NAMES))})
     *         float confidence   [0.0, 1.0] fraction of trees agreeing
     *
     * Classes: {N_GAS_CLASSES}
     * Trees:   {n_trees}
     * Features: {N_FEATURES}
     */
    #pragma once
    #include <stdint.h>
    """)

    with open(output_path, "w") as f:
        f.write(header_comment + "\n")
        for block in trees_code:
            f.write(block + "\n")
        f.write(aggregate_fn)

    print(f"\n[Export] Random Forest → {output_path}  ({n_trees} trees)")


def export_svr_to_c(regressors: dict, scaler_map: dict, output_path: str) -> None:
    """
    Export SVR regression models as C source.

    For each gas class, the SVR dual form prediction is:
        y = Σ_i (α_i * K(x, x_i)) + b
    with RBF kernel K(x, x_i) = exp(-γ * ||x - x_i||²)

    The support vectors, dual coefficients, γ, and b are baked as float arrays.
    The StandardScaler mean/std are also exported.
    """
    lines = [textwrap.dedent(f"""\
    /**
     * @file svr_model.h
     * @brief Auto-generated SVR concentration regression (e-Nose)
     *        Generated by ml_pipeline/train.py — DO NOT EDIT MANUALLY
     *
     * Input:  float features[{N_FEATURES}] (same order as rf_model.h)
     * Output: float concentration_ppm
     *
     * Usage:
     *   float ppm = svr_predict(gas_label, features);
     */
    #pragma once
    #include <stdint.h>
    #include <math.h>
    """)]

    for gas_label, pipe in regressors.items():
        scaler = pipe.named_steps["scaler"]
        svr    = pipe.named_steps["svr"]
        prefix = f"svr_{GAS_NAMES[gas_label]}"

        sv      = svr.support_vectors_            # (n_sv, N_FEATURES)
        alpha   = (svr.dual_coef_[0]).tolist()    # (n_sv,)
        intercept = float(svr.intercept_[0])
        gamma   = float(svr._gamma)               # Fitted γ

        scaler_mean = scaler.mean_.tolist()
        scaler_std  = scaler.scale_.tolist()
        n_sv        = sv.shape[0]

        lines.append(f"/* ===== SVR for {GAS_NAMES[gas_label]} ({n_sv} support vectors) ===== */")
        lines.append(f"#define {prefix.upper()}_N_SV      {n_sv}")
        lines.append(f"#define {prefix.upper()}_N_FEAT    {N_FEATURES}")
        lines.append(f"static const float {prefix}_gamma   = {gamma:.8f}f;")
        lines.append(f"static const float {prefix}_intercept = {intercept:.8f}f;")
        lines.append(_format_c_array(f"{prefix}_scaler_mean", scaler_mean, "float"))
        lines.append(_format_c_array(f"{prefix}_scaler_std",  scaler_std,  "float"))
        lines.append(_format_c_array(f"{prefix}_alpha",       alpha,       "float"))

        # Flatten support vectors row-major
        sv_flat = sv.flatten().tolist()
        lines.append(_format_c_array(f"{prefix}_sv", sv_flat, "float"))

        # Inference function for this gas
        lines.append(textwrap.dedent(f"""\
        static float {prefix}_predict(const float* x) {{
            /* 1. Standardize input */
            float xs[{N_FEATURES}];
            for (int i = 0; i < {N_FEATURES}; ++i) {{
                xs[i] = (x[i] - {prefix}_scaler_mean[i]) / ({prefix}_scaler_std[i] + 1e-9f);
            }}
            /* 2. Compute RBF kernel dot product */
            float result = {prefix}_intercept;
            for (int sv_i = 0; sv_i < {n_sv}; ++sv_i) {{
                float dist_sq = 0.0f;
                const float* sv_row = &{prefix}_sv[sv_i * {N_FEATURES}];
                for (int f = 0; f < {N_FEATURES}; ++f) {{
                    float d = xs[f] - sv_row[f];
                    dist_sq += d * d;
                }}
                result += {prefix}_alpha[sv_i] * expf(-{prefix}_gamma * dist_sq);
            }}
            return result;
        }}
        """))

    # Dispatcher
    dispatch_cases = "\n".join(
        f"        case {gas_label}: return svr_{GAS_NAMES[gas_label]}_predict(features);"
        for gas_label in regressors
    )
    lines.append(textwrap.dedent(f"""\
    float svr_predict(uint8_t gas_label, const float* features) {{
        switch (gas_label) {{
    {dispatch_cases}
        default: return 0.0f;
        }}
    }}
    """))

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[Export] SVR regressors → {output_path}  ({len(regressors)} models)")


def export_feature_extractor_to_c(output_path: str) -> None:
    """
    Export the feature extraction function to C.
    The microcontroller receives a raw R_ratio buffer and computes features in-place.
    """
    code = textwrap.dedent(f"""\
    /**
     * @file feature_extract.h
     * @brief Auto-generated feature extraction for e-Nose inference
     *        Generated by ml_pipeline/train.py — DO NOT EDIT MANUALLY
     *
     * Input:
     *   r_ratio[{N_SENSORS}][{SEQUENCE_LEN}]  — sensor ratio sequences (row-major, ch × time)
     * Output:
     *   features[{N_FEATURES}]              — feature vector for RF/SVR inference
     */
    #pragma once
    #include <math.h>
    #include <string.h>
    #include <stdint.h>

    #define ENOSE_N_SENSORS   {N_SENSORS}
    #define ENOSE_SEQ_LEN     {SEQUENCE_LEN}
    #define ENOSE_N_FEATURES  {N_FEATURES}

    static void enose_extract_features(
        const float r_ratio[ENOSE_N_SENSORS][ENOSE_SEQ_LEN],
        float features[ENOSE_N_FEATURES])
    {{
        int feat_idx = 0;
        float ss_means[ENOSE_N_SENSORS] = {{0}};

        for (int ch = 0; ch < ENOSE_N_SENSORS; ++ch) {{
            const float* s = r_ratio[ch];

            /* --- Max / min slope --- */
            float max_slope = -1e30f, min_slope = 1e30f;
            for (int t = 0; t < ENOSE_SEQ_LEN - 1; ++t) {{
                float d = s[t+1] - s[t];
                if (d > max_slope) max_slope = d;
                if (d < min_slope) min_slope = d;
            }}

            /* --- Steady-state statistics (last 10 samples) --- */
            float ss_sum = 0.0f, ss_sum2 = 0.0f;
            const int SS_WIN = 10;
            for (int t = ENOSE_SEQ_LEN - SS_WIN; t < ENOSE_SEQ_LEN; ++t) {{
                ss_sum  += s[t];
                ss_sum2 += s[t] * s[t];
            }}
            float ss_mean = ss_sum / SS_WIN;
            float ss_var  = ss_sum2 / SS_WIN - ss_mean * ss_mean;
            float ss_std  = (ss_var > 0.0f) ? sqrtf(ss_var) : 0.0f;
            ss_means[ch]  = ss_mean;

            /* --- Normalized AUC (trapezoidal) --- */
            float auc = 0.0f;
            for (int t = 0; t < ENOSE_SEQ_LEN - 1; ++t) {{
                auc += 0.5f * (s[t] + s[t+1]);
            }}
            float auc_norm = auc / ENOSE_SEQ_LEN;

            /* --- Time to half steady-state --- */
            float half_ss = (1.0f + ss_mean) * 0.5f;
            float t_half  = 1.0f;  /* default: end of window */
            for (int t = 0; t < ENOSE_SEQ_LEN - 1; ++t) {{
                if ((s[t] - half_ss) * (s[t+1] - half_ss) <= 0.0f) {{
                    t_half = (float)t / ENOSE_SEQ_LEN;
                    break;
                }}
            }}

            /* --- Peak delta --- */
            float peak_delta = 0.0f;
            for (int t = 0; t < ENOSE_SEQ_LEN; ++t) {{
                float d = fabsf(s[t] - 1.0f);
                if (d > peak_delta) peak_delta = d;
            }}

            features[feat_idx++] = max_slope;
            features[feat_idx++] = min_slope;
            features[feat_idx++] = ss_mean;
            features[feat_idx++] = ss_std;
            features[feat_idx++] = auc_norm;
            features[feat_idx++] = t_half;
            features[feat_idx++] = peak_delta;
        }}

        /* --- Cross-channel ratios --- */
        const float eps = 1e-6f;
        features[feat_idx++] = ss_means[0] / (ss_means[1] + eps);
        features[feat_idx++] = ss_means[0] / (ss_means[2] + eps);
        features[feat_idx++] = ss_means[1] / (ss_means[3] + eps);
        features[feat_idx++] = ss_means[2] / (ss_means[3] + eps);
    }}
    """)

    with open(output_path, "w") as f:
        f.write(code)

    print(f"[Export] Feature extractor → {output_path}")


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="e-Nose ML training pipeline"
    )
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Total number of synthetic measurement windows (default: 5000)")
    parser.add_argument("--output-dir", type=str, default="../firmware/src/ml",
                        help="Directory for exported C header files")
    parser.add_argument("--save-models", type=str, default="./saved_models",
                        help="Directory to save joblib serialized models")
    return parser.parse_args()


def main():
    args   = parse_args()
    rng    = np.random.default_rng(RANDOM_SEED)
    t_start = time.time()

    os.makedirs(args.output_dir,  exist_ok=True)
    os.makedirs(args.save_models, exist_ok=True)

    print("=" * 60)
    print("  e-Nose Industrial — ML Training Pipeline")
    print("=" * 60)
    print(f"  Sensors    : {N_SENSORS}")
    print(f"  Gas classes: {N_GAS_CLASSES} ({', '.join(GAS_NAMES)})")
    print(f"  Features   : {N_FEATURES}")
    print(f"  Samples    : {args.n_samples}")

    # 1. Generate dataset
    print(f"\n[1/5] Generating {args.n_samples} synthetic windows...")
    windows = generate_dataset(args.n_samples, rng)
    print(f"      Done — {len(windows)} windows generated.")

    # 2. Feature extraction
    print("\n[2/5] Extracting features...")
    X, y_cls, y_conc = build_feature_matrix(windows)
    print(f"      X shape: {X.shape}, y_cls: {y_cls.shape}, y_conc: {y_conc.shape}")
    print(f"      Class distribution: { {n: int((y_cls == i).sum()) for i, n in enumerate(GAS_NAMES)} }")

    # 3. Train classifier
    print("\n[3/5] Training gas classifier...")
    clf, cv_scores = train_classifier(X, y_cls)
    joblib.dump(clf, os.path.join(args.save_models, "rf_classifier.joblib"))

    # 4. Train regressors
    print("\n[4/5] Training concentration regressors...")
    regressors = train_regressor(X, y_conc, y_cls)
    for gas_label, pipe in regressors.items():
        joblib.dump(pipe, os.path.join(
            args.save_models, f"svr_{GAS_NAMES[gas_label]}.joblib"
        ))

    # 5. Export to C
    print("\n[5/5] Exporting models to C...")
    export_feature_extractor_to_c(
        os.path.join(args.output_dir, "feature_extract.h")
    )
    export_random_forest_to_c(
        clf,
        os.path.join(args.output_dir, "rf_model.h")
    )
    export_svr_to_c(
        regressors, {},
        os.path.join(args.output_dir, "svr_model.h")
    )

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f} s")
    print(f"  Models saved to : {args.save_models}")
    print(f"  C headers in    : {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
