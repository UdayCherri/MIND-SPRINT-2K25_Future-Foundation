# train_fusion_baseline.py
import numpy as np
from fusion_model import build_fusion_model, save_scaler
from sklearn.preprocessing import StandardScaler
import os
import tensorflow as tf

# Hyperparams
N_SAMPLES = 2000
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "./models/fusion_baseline.h5"

# Synthetic data generator (replace with real dataset later)
def generate_synthetic_data(n_samples=N_SAMPLES, audio_dim=256, text_dim=771, beh_dim=6):
    """Generate synthetic multi-modal data with realistic correlations"""
    np.random.seed(42)  # Reproducibility
    
    # Generate features with some structure
    X_audio = np.random.normal(0, 1, (n_samples, audio_dim)).astype(np.float32)
    X_text  = np.random.normal(0, 1, (n_samples, text_dim)).astype(np.float32)
    X_beh   = np.random.normal(0, 1, (n_samples, beh_dim)).astype(np.float32)
    
    # Create correlated signal from all modalities
    signal = (X_audio.mean(axis=1) * 0.3 + 
              X_text[:, :10].mean(axis=1) * 0.4 + 
              X_beh.mean(axis=1) * 0.3)
    
    # Synthetic labels: 4 binary patterns with different correlations
    patterns = np.stack([
        (signal + np.random.normal(0, 0.5, n_samples) > 0).astype(float),  # Social Withdrawal
        (signal * 0.8 + np.random.normal(0, 0.6, n_samples) > 0.2).astype(float),  # Sleep Disruption
        (signal * 0.5 + np.random.normal(0, 1, n_samples) > 0).astype(float),  # Emotional Volatility
        (np.random.rand(n_samples) > 0.5).astype(float)  # Random baseline (Cognitive Decline)
    ], axis=1)
    
    # Regression target: burnout_score correlated with patterns
    burnout = np.clip(
        0.3 + 0.15 * patterns.sum(axis=1) / 4 + 0.1 * signal + np.random.normal(0, 0.1, n_samples),
        0.0, 1.0
    ).astype(np.float32)
    
    print(f"Generated {n_samples} synthetic samples:")
    print(f"  Audio: {X_audio.shape}")
    print(f"  Text: {X_text.shape}")
    print(f"  Behavior: {X_beh.shape}")
    print(f"  Patterns: {patterns.shape} (prevalence: {patterns.mean(axis=0)})")
    print(f"  Burnout: mean={burnout.mean():.3f}, std={burnout.std():.3f}")
    
    return X_audio, X_text, X_beh, patterns, burnout

# generate
X_audio, X_text, X_beh, Yp, Yb = generate_synthetic_data()

# scaler fit on concatenated features
X_concat = np.concatenate([X_audio, X_text, X_beh], axis=1)
scaler = StandardScaler()
scaler.fit(X_concat)
save_scaler(scaler)  # save for inference

# apply scaling to FULL concatenated vector, then split
X_concat_scaled = scaler.transform(X_concat)
X_audio_scaled = X_concat_scaled[:, :X_audio.shape[1]]
X_text_scaled  = X_concat_scaled[:, X_audio.shape[1]:X_audio.shape[1]+X_text.shape[1]]
X_beh_scaled   = X_concat_scaled[:, -X_beh.shape[1]:]

# build and train model
model = build_fusion_model(adapter=True)
print("\nModel architecture:")
model.summary()

print(f"\nTraining on {len(X_audio_scaled)} samples for {EPOCHS} epochs...")
history = model.fit([X_audio_scaled, X_text_scaled, X_beh_scaled], 
                    {"pattern_probs": Yp, "burnout_score": Yb},
                    validation_split=0.1, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE,
                    verbose=1)

# Print final metrics
print("\nTraining complete!")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
if 'pattern_probs_auc' in history.history:
    print(f"Final pattern AUC: {history.history['pattern_probs_auc'][-1]:.4f}")
if 'val_burnout_score_loss' in history.history:
    print(f"Final burnout MSE: {history.history['val_burnout_score_loss'][-1]:.4f}")


# save weights
model.save(MODEL_PATH)
print("Saved baseline fusion model to", MODEL_PATH)
