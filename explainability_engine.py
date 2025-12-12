# explainability_engine.py
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
from dataclasses import dataclass
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

@dataclass
class InsightCard:
    """Structured insight card for UI display."""
    title: str
    subtitle: str
    description: str
    contributors: List[Tuple[str, float]]
    severity: str  # 'low', 'moderate', 'high'
    recommendation: str
    caveat: str = "Non-clinical insight based on user data; consult professionals if concerned."

@dataclass
class TriggerResponse:
    """Trigger-response correlation result."""
    trigger: str
    target: str
    lag_days: int
    correlation: float
    pvalue: float
    confidence: str  # 'low', 'moderate', 'high'
    interpretation: str

class ExplainabilityEngine:
    """Main engine for explainability and insights generation."""
    
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """Initialize the explainability engine."""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._explainer = None
        
        # Load model and scaler
        self._load_resources()
        
        # Build feature names and groups
        self.feature_names, self.feature_groups = self._build_feature_names()
        
        # Behavior feature friendly names
        self.behavior_friendly_names = {
            "avg_sleep_hours": "Sleep Duration",
            "sleep_variance": "Sleep Consistency",
            "screen_time_hours": "Screen Time",
            "session_count": "App Session Count",
            "break_frequency": "Break Frequency",
            "activity_minutes": "Physical Activity"
        }
    
    def _load_resources(self):
        """Load model and scaler."""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path, compile=False, safe_mode=False)
            print(f"✓ Loaded model from {self.model_path}")
        else:
            print(f"⚠️ Model not found: {self.model_path}")
        
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print(f"✓ Loaded scaler from {self.scaler_path}")
        else:
            print(f"⚠️ Scaler not found: {self.scaler_path}")
    
    def _build_feature_names(self):
        """Build feature names for all modalities."""
        names = []
        feature_groups = {
            'audio': [],
            'text': [],
            'behavior': []
        }
        
        # Audio features (256 dims)
        for i in range(AUDIO_DIM):
            fname = f"audio_feature_{i}"
            names.append(fname)
            feature_groups['audio'].append(len(names) - 1)
        
        # Text embeddings (768 DistilBERT)
        for i in range(768):
            fname = f"text_emb_{i}"
            names.append(fname)
            feature_groups['text'].append(len(names) - 1)
        
        # Text lexical features (3)
        text_extras = ["lexical_diversity", "avg_sentence_length", "sentence_count"]
        for fname in text_extras:
            names.append(fname)
            feature_groups['text'].append(len(names) - 1)
        
        # Behavior features (6 dims)
        beh_names = [
            "avg_sleep_hours",
            "sleep_variance",
            "screen_time_hours",
            "session_count",
            "break_frequency",
            "activity_minutes"
        ]
        for fname in beh_names[:BEH_DIM]:
            names.append(fname)
            feature_groups['behavior'].append(len(names) - 1)
        
        return names, feature_groups
    
    def _create_explainer(self, background_samples=50):
        """Create SHAP explainer with background data."""
        if not SHAP_AVAILABLE or self.model is None:
            return None
        
        # Create background dataset (zeros as baseline)
        background = np.zeros((background_samples, len(self.feature_names))).astype(np.float32)
        
        # Wrapper function for SHAP
        def model_predict(fused_inputs):
            audio = fused_inputs[:, :AUDIO_DIM]
            text = fused_inputs[:, AUDIO_DIM:AUDIO_DIM+TEXT_DIM]
            behavior = fused_inputs[:, AUDIO_DIM+TEXT_DIM:]
            return self.model.predict([audio, text, behavior], verbose=0)[0]
        
        explainer = shap.KernelExplainer(model_predict, background)
        return explainer
    
    def get_explainer(self):
        """Get or create the SHAP explainer."""
        if self._explainer is None:
            self._explainer = self._create_explainer()
        return self._explainer
    
    def explain_instance(self, fused_input, nsamples=100):
        """
        Compute local feature attributions for a single instance.
        
        Args:
            fused_input: 1D numpy array (1033 dims, scaled)
            nsamples: Number of samples for SHAP
        
        Returns:
            list of (feature_name, attribution_score) sorted by importance
        """
        explainer = self.get_explainer()
        if explainer is None:
            return self._gradient_attribution(fused_input)
        
        try:
            shap_vals = explainer.shap_values(fused_input.reshape(1, -1), nsamples=nsamples)
            
            # Handle multi-output
            if isinstance(shap_vals, list):
                arr = np.mean([np.abs(s) for s in shap_vals], axis=0).flatten()
            else:
                arr = np.abs(shap_vals).flatten()
            
            feat_importances = [(self.feature_names[i], float(arr[i])) for i in range(len(arr))]
            feat_importances.sort(key=lambda x: x[1], reverse=True)
            
            return feat_importances
        except Exception as e:
            print(f"⚠️ SHAP computation failed: {e}")
            return self._gradient_attribution(fused_input)
    
    def _gradient_attribution(self, fused_input):
        """Fallback gradient-based attribution."""
        import tensorflow as tf
        
        if self.model is None:
            return [(self.feature_names[i], 0.0) for i in range(len(self.feature_names))]
        
        # Split input
        audio = fused_input[:AUDIO_DIM].reshape(1, -1)
        text = fused_input[AUDIO_DIM:AUDIO_DIM+TEXT_DIM].reshape(1, -1)
        behavior = fused_input[AUDIO_DIM+TEXT_DIM:].reshape(1, -1)
        
        # Convert to tensors
        audio_t = tf.constant(audio, dtype=tf.float32)
        text_t = tf.constant(text, dtype=tf.float32)
        behavior_t = tf.constant(behavior, dtype=tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch([audio_t, text_t, behavior_t])
            preds = self.model([audio_t, text_t, behavior_t])
            output = tf.reduce_mean(preds[0])
        
        grads = tape.gradient(output, [audio_t, text_t, behavior_t])
        grad_concat = np.concatenate([g.numpy().flatten() for g in grads])
        
        # Integrated gradients approximation
        attributions = np.abs(grad_concat * fused_input)
        
        feat_importances = [(self.feature_names[i], float(attributions[i])) for i in range(len(attributions))]
        feat_importances.sort(key=lambda x: x[1], reverse=True)
        
        return feat_importances
    
    def compute_global_importance(self, X_samples, top_k=20):
        """
        Compute global feature importance across multiple samples.
        
        Args:
            X_samples: Array of scaled fused inputs (N, 1033)
            top_k: Number of top features to return
        
        Returns:
            list of (feature_name, mean_importance, std_importance)
        """
        all_importances = []
        
        for i, sample in enumerate(X_samples):
            if i % 10 == 0:
                print(f"Computing importance {i}/{len(X_samples)}...")
            importances = self.explain_instance(sample, nsamples=50)
            all_importances.append(dict(importances))
        
        # Aggregate
        importance_df = pd.DataFrame(all_importances)
        mean_importance = importance_df.mean().sort_values(ascending=False)
        std_importance = importance_df.std()
        
        results = [
            (feat, mean_importance[feat], std_importance[feat]) 
            for feat in mean_importance.index[:top_k]
        ]
        
        return results
    
    def compute_lagged_correlations(self, df, trigger_col, target_col, max_lag_days=7):
        """
        Compute lagged correlations between trigger and target.
        
        Args:
            df: pandas DataFrame indexed by date
            trigger_col: Column name for trigger variable
            target_col: Column name for target variable
            max_lag_days: Maximum lag to test
        
        Returns:
            DataFrame of lag, correlation, pvalue
        """
        results = []
        for lag in range(0, max_lag_days + 1):
            shifted = df[trigger_col].shift(lag)
            valid = pd.DataFrame({
                'trigger': shifted,
                'target': df[target_col]
            }).dropna()
            
            if len(valid) < 10:
                continue
            
            corr, p = stats.spearmanr(valid['trigger'], valid['target'])
            results.append({
                "lag": lag,
                "corr": corr,
                "pval": p,
                "n": len(valid)
            })
        
        if results:
            return pd.DataFrame(results).sort_values("corr", key=abs, ascending=False)
        return pd.DataFrame()
    
    def find_trigger_responses(self, df, triggers, targets, threshold_corr=0.25, threshold_pval=0.05):
        """
        Find significant trigger-response pairs across historical data.
        
        Args:
            df: DataFrame with date index and feature columns
            triggers: List of trigger column names
            targets: List of target column names
            threshold_corr: Minimum correlation strength
            threshold_pval: Maximum p-value
        
        Returns:
            List of TriggerResponse objects
        """
        results = []
        
        for trig in triggers:
            if trig not in df.columns:
                continue
            
            for targ in targets:
                if targ not in df.columns:
                    continue
                
                res = self.compute_lagged_correlations(df, trig, targ, max_lag_days=7)
                
                if not res.empty:
                    best = res.iloc[0]
                    if abs(best["corr"]) > threshold_corr and best["pval"] < threshold_pval:
                        # Determine confidence
                        if abs(best["corr"]) > 0.5 and best["pval"] < 0.01:
                            confidence = "high"
                        elif abs(best["corr"]) > 0.35:
                            confidence = "moderate"
                        else:
                            confidence = "low"
                        
                        # Create interpretation
                        direction = "increases" if best["corr"] > 0 else "decreases"
                        interpretation = f"{trig} {direction} {targ}"
                        if best["lag"] > 0:
                            interpretation += f" after {best['lag']} day(s)"
                        
                        results.append(TriggerResponse(
                            trigger=trig,
                            target=targ,
                            lag_days=int(best["lag"]),
                            correlation=float(best["corr"]),
                            pvalue=float(best["pval"]),
                            confidence=confidence,
                            interpretation=interpretation
                        ))
        
        return results
    
    def generate_insight_card(self, feature_importances, predictions, trigger_responses=None, top_k=3):
        """
        Generate a human-readable insight card for UI display.
        
        Args:
            feature_importances: List of (feature_name, score) from explain_instance
            predictions: Dict with 'patterns' and 'burnout' predictions
            trigger_responses: Optional list of TriggerResponse objects
            top_k: Number of top contributors to include
        
        Returns:
            InsightCard object
        """
        # Get top contributors
        top_features = feature_importances[:top_k]
        
        # Group contributors by modality
        contributors_by_group = {'audio': [], 'text': [], 'behavior': []}
        for feat_name, score in top_features:
            if feat_name.startswith('audio'):
                contributors_by_group['audio'].append((feat_name, score))
            elif feat_name.startswith('text') or feat_name in ['lexical_diversity', 'avg_sentence_length', 'sentence_count']:
                contributors_by_group['text'].append((feat_name, score))
            else:
                contributors_by_group['behavior'].append((feat_name, score))
        
        # Determine severity
        burnout = predictions.get('burnout', 0)
        if burnout > 0.7:
            severity = 'high'
            title = "⚠️ High Stress Alert"
        elif burnout > 0.5:
            severity = 'moderate'
            title = "⚡ Moderate Stress Detected"
        else:
            severity = 'low'
            title = "✅ Wellness Profile Stable"
        
        # Build subtitle with top contributors
        top_beh = [f for f in top_features if f[0] in self.behavior_friendly_names][:2]
        if top_beh:
            subtitle_parts = []
            for feat, score in top_beh:
                friendly = self.behavior_friendly_names.get(feat, feat)
                subtitle_parts.append(f"{friendly} ({score:.2f})")
            subtitle = f"Top contributors: {', '.join(subtitle_parts)}"
        else:
            subtitle = "Analysis based on multi-modal data"
        
        # Build description
        description_parts = []
        
        # Add behavior insights
        beh_contributors = contributors_by_group['behavior']
        if beh_contributors:
            beh_text = []
            for feat, score in beh_contributors[:2]:
                friendly = self.behavior_friendly_names.get(feat, feat)
                beh_text.append(friendly)
            description_parts.append(f"Key factors: {', '.join(beh_text)}")
        
        # Add trigger-response if available
        if trigger_responses:
            tr = trigger_responses[0]  # Use strongest correlation
            description_parts.append(f"{tr.interpretation} (r={tr.correlation:.2f}, p={tr.pvalue:.3f})")
        
        description = ". ".join(description_parts) + "."
        
        # Generate recommendation
        recommendation = self._generate_recommendation(predictions, top_features, trigger_responses)
        
        return InsightCard(
            title=title,
            subtitle=subtitle,
            description=description,
            contributors=top_features,
            severity=severity,
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, predictions, top_features, trigger_responses):
        """Generate specific, actionable recommendations."""
        burnout = predictions.get('burnout', 0)
        patterns = predictions.get('patterns', {})
        
        recommendations = []
        
        # Check behavior features in top contributors
        behavior_features = {feat: score for feat, score in top_features 
                           if feat in self.behavior_friendly_names}
        
        if 'screen_time_hours' in behavior_features:
            recommendations.append("Reduce screen time by 30-60 minutes daily")
        
        if 'avg_sleep_hours' in behavior_features:
            if patterns.get('sleep_disruption', 0) > 0.5:
                recommendations.append("Aim for 7-8 hours of consistent sleep")
        
        if 'break_frequency' in behavior_features:
            recommendations.append("Take regular 5-minute breaks every hour")
        
        if 'activity_minutes' in behavior_features:
            recommendations.append("Increase physical activity to 30 minutes daily")
        
        # Add trigger-based recommendations
        if trigger_responses:
            tr = trigger_responses[0]
            if 'screen_time' in tr.trigger.lower() and tr.correlation > 0:
                recommendations.append(f"Late-night screen time shows correlation with stress - consider reducing before bed")
        
        # Burnout-specific recommendations
        if burnout > 0.7:
            recommendations.append("⚠️ Consider consulting a mental health professional")
        elif burnout > 0.5:
            recommendations.append("Practice stress-reduction techniques like meditation or deep breathing")
        
        if not recommendations:
            recommendations.append("Maintain current healthy habits and continue monitoring")
        
        return "; ".join(recommendations[:3])  # Limit to top 3

# Convenience function for quick analysis
def analyze_prediction(fused_input, predictions, user_history_df=None):
    """
    Quick analysis function that returns an insight card.
    
    Args:
        fused_input: Scaled fused input (1033 dims)
        predictions: Dict with 'patterns' and 'burnout'
        user_history_df: Optional DataFrame for trigger-response analysis
    
    Returns:
        InsightCard object
    """
    engine = ExplainabilityEngine()
    
    # Get feature importance
    importances = engine.explain_instance(fused_input, nsamples=50)
    
    # Find trigger-responses if history provided
    trigger_responses = None
    if user_history_df is not None and len(user_history_df) > 14:
        triggers = ['screen_time_hours', 'avg_sleep_hours', 'session_count']
        targets = ['burnout_score', 'emotional_volatility']
        # Filter to existing columns
        triggers = [t for t in triggers if t in user_history_df.columns]
        targets = [t for t in targets if t in user_history_df.columns]
        
        if triggers and targets:
            trigger_responses = engine.find_trigger_responses(
                user_history_df, triggers, targets
            )
    
    # Generate insight card
    card = engine.generate_insight_card(
        importances,
        predictions,
        trigger_responses
    )
    
    return card

if __name__ == "__main__":
    print("BAYMAX Explainability Engine initialized")
    print("=" * 60)
    
    # Example usage
    engine = ExplainabilityEngine()
    print(f"\nFeature groups: {list(engine.feature_groups.keys())}")
    print(f"Total features: {len(engine.feature_names)}")
    print(f"  - Audio: {len(engine.feature_groups['audio'])}")
    print(f"  - Text: {len(engine.feature_groups['text'])}")
    print(f"  - Behavior: {len(engine.feature_groups['behavior'])}")
