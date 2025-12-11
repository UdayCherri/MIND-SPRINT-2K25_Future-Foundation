# **BAYMAX: ModeM - Phase 2 Submission**

## **Model Architecture + Feature Pipeline**

---

# **1. Overview**

MindRhythm AI is built as a **multi-modal cognitive pattern engine** that processes **audio**, **text**, and **behavioral time-series data** to detect early mental wellness drift.

The architecture integrates:

* Multi-modal encoders
* Temporal pattern detection
* Cross-modal reasoning
* Multi-task prediction heads
* Personalized per-user adapters
* Explainability + trigger-response correlation engine

This ensures the system can generalize across users while adapting precisely to each individual's mental wellness rhythm.

---

# **2. Architecture Diagram**

```
[Weekly User Bundle]
    ├─ Audio samples  ----> Audio Preproc ---> Audio Encoder (wav2vec2-small or CNN+Transformer)
    ├─ Text journals  ----> Text Preproc  ---> Text Encoder (DistilBERT / MiniLM)
    └─ Behavior logs  ---> Behavior Preproc ---> Behavior Encoder (TCN / 1D-CNN)

Per-day Embeddings (audio_day, text_day, behavior_day)
    ↓ (aligned by day / timestamp)
Temporal Aggregator (Multi-head Transformer over N days)
    ↓
Cross-Modal Fusion (1–2 layer Transformer or Attention-MLP)
    ↓
Shared Representation
    ├─ Pattern Classification Heads (stress_loop, routine_fracture, etc.)
    ├─ Regression Heads (burnout_score, stability_index)
    └─ Explainability Module (SHAP-like attribution + trigger correlations)

Personalization Layer (User-Specific Adapter)
    - Bottleneck adapter inserted into fusion/transformer layers
    - Adapter updated weekly using user’s personal data

Outputs → Pattern Radar UI
    • Detected cognitive/behavioral patterns
    • Confidence scores
    • Top contributing features
    • Timeline + actionable insights
```

---

# **3. Feature Pipeline**

## **A. Audio Feature Pipeline**

* Preprocessing: 16kHz resampling, silence trimming, normalization
* Features:

  * Log-mel spectrogram (128 bins)
  * MFCC + delta coefficients
  * Prosody features (pitch mean/variance, energy)
  * Voice rate & jitter/shimmer
* Output: day-level emotional-prosodic embedding

---

## **B. Text Feature Pipeline**

* Preprocessing: cleaning, tokenization
* Features:

  * DistilBERT embeddings
  * Sentiment & emotion distributions
  * Lexical diversity
  * Readability & coherence scores
  * Repetition index
  * Semantic drift across days
* Output: daily cognitive-linguistic embedding

---

## **C. Behavioral Feature Pipeline**

* Inputs: sleep logs, screen-time, productivity sessions, activity patterns
* Features:

  * avg_sleep_hours, sleep_variance
  * screen_time_after_11PM
  * session_count, session_length
  * break frequency
  * circadian rhythm consistency
  * routine-fracturing metric
* Output: daily behavioral embedding

---

## **D. Derived Multi-modal Features**

* Volatility index
* Emotional inertia
* Linguistic fatigue score
* Burnout-risk composite
* Trigger → effect correlations (with lag analysis)

---

# **4. Model Architecture Components**

## **A. Per-Modality Encoders**

* **Audio Encoder**: wav2vec2-small OR CNN+Transformer hybrid
* **Text Encoder**: DistilBERT/MiniLM (frozen or lightly fine-tuned)
* **Behavior Encoder**: 1D-CNN or TCN over daily metrics

Each produces **day-level embeddings**.

---

## **B. Temporal Aggregator**

A **multi-head Transformer** processes 7–28 days of embeddings to model:

* weekly emotional periodicity
* cognitive load buildup
* stress accumulation
* burnout waves
* routine fracturing

This becomes the temporal signature of the user.

---

## **C. Cross-Modal Fusion Layer**

* Lightweight cross-modal Transformer OR attention-MLP
* Learns interactions such as:

  * “text emotional drift + poor sleep = higher volatility”
  * “prosody instability + high screen-time = stress-loop pattern”

---

## **D. Multi-Task Prediction Heads**

1. **Pattern Classification Heads**
   (multi-label sigmoid/BCE)

   * stress_loop
   * burnout_trajectory
   * sleep-emotion mismatch
   * cognitive overload
   * routine fracturing
   * social withdrawal
   * MBTI-modulated responses

2. **Regression Heads**

   * burnout_score
   * stability_index
   * volatility_index

---

## **E. Personalization Layer (User Adapters)**

* Small bottleneck modules:
  **Linear → ReLU → Linear → residual**
* Updated weekly using user’s own data
* Advantages:

  * private, small, fast
  * reduces model drift
  * massively improves accuracy
  * supports federated learning later

---

## **F. Explainability & Pattern Reasoning**

Includes:

* Feature importance (SHAP-like ranking)
* Temporal attention weighting
* Trigger-response correlations with time lag
* Example insight:

  > *“Screen-time after 11 PM increased volatility by +0.22 this week.”*

---

# **5. Learning Algorithms Used**

## **A. Supervised Pretraining**

* RAVDESS for audio emotional pretraining
* GoEmotions & DailyDialog for text emotion understanding
* **Loss:** Cross-Entropy

## **B. Multi-Modal Finetuning (Global Model)**

* Joint training on all modalities
* **Loss:**

```
L = Σ BCE(pattern_predictions)
    + λ1 * MSE(burnout_score)
    + λ2 * MSE(stability_index)
    + λ3 * temporal_consistency_loss
```

## **C. Personalization Algorithm**

* Few-shot adapter finetuning per user
* Backbone frozen → only adapter weights updated
* **Optimizer:** AdamW (lr = 1e-3 for adapters)
* Optional privacy upgrade: **Federated Averaging (FedAvg)**

---

# **6. Hyperparameters (Recommended)**

* Temporal window: 14 days
* Transformer: 2 layers, d_model=512, heads=8
* Adapter bottleneck: 64
* Batch size: 16
* Global learning rate: 5e-5
* Adapter learning rate: 1e-3

---

# **7. Evaluation Strategy (Phase-2 Requirements)**

### **Pattern Metrics**

* Precision, Recall, F1
* AUC-ROC (for multi-label patterns)

### **Regression Metrics**

* MAE, RMSE for burnout & stability scores

### **Personalization Evaluation**

* ΔF1 (before vs after adapter update)
* ΔMAE (risk score improvement)

### **Ablation Studies**

* audio-only
* text-only
* behavior-only
* full multi-modal fusion

---

# **8. Phase-2 Deliverables Completed**

* ✔ Multi-modal feature pipeline defined
* ✔ Model architecture diagram
* ✔ Per-modality encoders specified
* ✔ Temporal Transformer aggregator
* ✔ Cross-modal fusion layer
* ✔ Multi-task prediction heads
* ✔ Personalization adapter strategy
* ✔ Explainability + correlation module design
* ✔ Loss functions & training algorithms
* ✔ Hyperparameters & evaluation metrics
* ✔ Ablation and personalization testing plan

---

# **Summary**

BAYMAX: ModeM’s Phase-2 architecture forms a **scalable, multi-modal cognitive pattern intelligence system** capable of detecting subtle emotional, cognitive, and behavioral micro-patterns. The design supports personalization, privacy, and scientific interpretability, forming a robust foundation for full implementation in Phase-3.

