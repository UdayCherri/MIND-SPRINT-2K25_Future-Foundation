# explainability.py
"""
BAYMAX Explainability & Insights Engine
Provides feature attribution, trigger-response analysis, and human-readable insights.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from tensorflow.keras.models import load_model
import joblib
import os
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")

# Configuration
MODEL_PATH = "./models/fusion_baseline.h5"
SCALER_PATH = "./models/scaler.joblib"
AUDIO_DIM = 256
TEXT_DIM = 771
BEH_DIM = 6

# Load model & scaler
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, compile=False, safe_mode=False)
    print(f"✓ Loaded model from {MODEL_PATH}")
else:
    model = None
    print(f"⚠️ Model not found: {MODEL_PATH}")

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print(f"✓ Loaded scaler from {SCALER_PATH}")
else:
    scaler = None
    print(f"⚠️ Scaler not found: {SCALER_PATH}")

# Feature mapping with detailed behavior names
def build_feature_names(audio_dim=AUDIO_DIM, text_dim=TEXT_DIM, beh_dim=BEH_DIM):
    """
    Build feature names for all modalities.
    Returns: list of feature names, dict of feature groups
    """
    names = []
    feature_groups = {
        'audio': [],
        'text': [],
        'behavior': []
    }
    
    # Audio features (256 dims)
    for i in range(audio_dim):
        fname = f"audio_feature_{i}"
        names.append(fname)
        feature_groups['audio'].append(len(names) - 1)
    
    # Text embeddings + extras (771 dims)
    # First 768 are DistilBERT embeddings, last 3 are lexical features
    for i in range(text_dim - 3):
        fname = f"text_emb_{i}"
        names.append(fname)
        feature_groups['text'].append(len(names) - 1)
    
    # Last 3 text features are lexical stats
    text_extras = ["lexical_diversity", "avg_sentence_length", "sentence_count"]
    for fname in text_extras:
        names.append(fname)
        feature_groups['text'].append(len(names) - 1)
    
    # Behavior features (6 dims) with meaningful names
    beh_names = [
        "avg_sleep_hours",
        "sleep_variance",
        "screen_time_hours",
        "session_count",
        "break_frequency",
        "activity_minutes"
    ]
    for fname in beh_names[:beh_dim]:
        names.append(fname)
        feature_groups['behavior'].append(len(names) - 1)
    
    return names, feature_groups

FEATURE_NAMES, FEATURE_GROUPS = build_feature_names()

# Behavior feature friendly names for insights
BEHAVIOR_FRIENDLY_NAMES = {
    "avg_sleep_hours": "Sleep Duration",
    "sleep_variance": "Sleep Consistency",
    "screen_time_hours": "Screen Time",
    "session_count": "App Session Count",
    "break_frequency": "Break Frequency",
    "activity_minutes": "Physical Activity"
}

# -------------------------
# 1) Local attributions using SHAP (KernelExplainer)
# -------------------------
# Prepare a small background dataset: sample 100 examples from your validation set (scaled fused inputs)
# background = X_val_scaled[np.random.choice(len(X_val_scaled), 100, replace=False)]
# For demo, you can use zeros or small random noise
background = np.zeros((50, len(FEATURE_NAMES))).astype(np.float32)

explainer = shap.KernelExplainer(lambda x: model.predict([x[:, :256], x[:, 256:256+771], x[:, -6:]])[0], background)

def explain_instance(fused_input):
    """
    fused_input: 1d numpy array scaled (same scale as training)
    Returns: list of (feature, shap_value) sorted desc by abs value
    """
    shap_vals = explainer.shap_values(fused_input.reshape(1,-1), nsamples=200)  # returns list per output; choose pattern head idx or regression
    # shap_vals is for first output head; if multi-output the shape differs. We'll compute for burnout_score (output 1) as example
    # If KernelExplainer returns a list, pick index of interest. Here assume binary: shap_vals[0] is an array
    # Use averaged absolute shap across outputs: flatten
    if isinstance(shap_vals, list):
        # For multi-head, sum absolute across outputs
        arr = np.sum([np.abs(s) for s in shap_vals], axis=0).flatten()
    else:
        arr = np.abs(shap_vals).flatten()
    feat_importances = [(FEATURE_NAMES[i], float(arr[i])) for i in range(len(arr))]
    feat_importances.sort(key=lambda x: x[1], reverse=True)
    return feat_importances

# -------------------------
# 2) Trigger-response correlations (lagged)
# -------------------------
def compute_lagged_correlations(df, trigger_col, target_col, max_lag_days=7):
    """
    df: pandas DataFrame indexed by date, columns include trigger_col and target_col
    returns: DataFrame of lag, correlation, pvalue
    """
    results = []
    for lag in range(0, max_lag_days+1):
        # shift trigger forward by lag so we test trigger at day t vs target at day t+lag
        shifted = df[trigger_col].shift(lag)
        valid = df[[shifted.name, target_col]].dropna()
        if len(valid) < 10:
            continue
        corr, p = stats.spearmanr(valid[shifted.name], valid[target_col])
        results.append({"lag": lag, "corr": corr, "pval": p, "n": len(valid)})
    return pd.DataFrame(results).sort_values("corr", ascending=False)

# -------------------------
# 3) Produce human-readable insight
# -------------------------
def generate_insight_from_attributions(feature_importances, top_k=3, threshold=0.01):
    """
    feature_importances: list of (feature, score) sorted desc by score
    returns: string insight
    """
    top = [f for f in feature_importances if f[1] > threshold][:top_k]
    # Group by modality
    groups = {}
    for fname, score in top:
        group = fname.split("_")[0]
        groups.setdefault(group, []).append((fname, score))
    # Build sentences
    parts = []
    for g, items in groups.items():
        names = ", ".join([i[0] for i in items])
        avg_score = np.mean([i[1] for i in items])
        parts.append(f"{names} (in {g}) contributed strongly (score {avg_score:.3f})")
    return "Top contributors: " + "; ".join(parts)

# Example usage:
# fused_scaled = scaler.transform(fused_input.reshape(1,-1))[0]
# feat_imp = explain_instance(fused_scaled)
# print(generate_insight_from_attributions(feat_imp[:10]))
