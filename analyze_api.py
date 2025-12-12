# analyze_api.py
# TODO: Add rate limiting to prevent API abuse
# TODO: Implement proper logging system for production
# TODO: Add comprehensive request validation

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import joblib
import uvicorn
import os
import traceback
import time
import logging
import tensorflow as tf
import librosa
import soundfile as sf
import tempfile

# Import your modules
from text_embedding_module import TextEmbeddingModule
from explainability_engine import ExplainabilityEngine, analyze_prediction
from personalize_adapter import PersonalizedAdapter
from user_auth import authenticate_user, register_user, get_user_by_id
from data_storage import save_analysis, get_user_history, get_timeline_data, get_insights_summary

# Setup logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("analyze_api")

# Config paths
MODEL_PATH = "./models/fusion_baseline.h5"
SCALER_PATH = "./models/scaler.joblib"

# Init
app = FastAPI(title="BAYMAX: ModeM - Analyze API (Fusion + Explainability)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5174"],  # Vite/React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load text embedder (singleton)
text_embedder = TextEmbeddingModule(cache_dir="./emb_cache")

# Initialize ExplainabilityEngine (handles model & scaler loading)
try:
    explainer = ExplainabilityEngine(model_path=MODEL_PATH, scaler_path=SCALER_PATH)
    model = explainer.model
    scaler = explainer.scaler
    log.info("✓ Initialized ExplainabilityEngine with model and scaler")
except Exception as e:
    log.error(f"Error initializing explainability engine: {e}")
    explainer = None
    model = None
    scaler = None

# Define request & response schemas
class AnalyzeRequest(BaseModel):
    user_id: str
    date: str
    texts: List[str]
    # audio_embedding is a list of floats (pre-extracted by audio module)
    audio_embedding: List[float]
    # behavior vector numeric array (same order you defined)
    behavior: List[float]
    include_explain: Optional[bool] = True
    max_trigger_lag_days: Optional[int] = 7

class PatternConfidence(BaseModel):
    pattern: str
    probability: float

class AnalyzeResponse(BaseModel):
    user_id: str
    date: str
    patterns: List[PatternConfidence]
    burnout_score: float
    explainability: Optional[Dict[str, Any]] = None
    triggers: Optional[List[Dict[str, Any]]] = None
    meta: Dict[str, Any]

# Helper: scale + split concatenated vector
def prepare_inputs(audio_emb: np.ndarray, text_vec: np.ndarray, beh_vec: np.ndarray, scaler_obj):
    # concat
    concat = np.concatenate([audio_emb, text_vec, beh_vec]).reshape(1, -1).astype(np.float32)
    if scaler_obj is not None:
        scaled = scaler_obj.transform(concat)
    else:
        scaled = concat
    a_dim = audio_emb.shape[0]
    t_dim = text_vec.shape[0]
    audio_s = scaled[:, :a_dim]
    text_s = scaled[:, a_dim:a_dim+t_dim]
    beh_s = scaled[:, -beh_vec.shape[0]:]
    return audio_s.astype(np.float32), text_s.astype(np.float32), beh_s.astype(np.float32)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    start = time.time()
    try:
        # 1) Text day vector (use your text embedder caching)
        text_vec = text_embedder.build_day_feature_vector(req.user_id, req.date, req.texts)
        text_vec = text_vec.astype(np.float32)

        # 2) audio & behavior arrays
        audio_emb = np.array(req.audio_embedding, dtype=np.float32)
        beh_emb = np.array(req.behavior, dtype=np.float32)

        # Validate dims
        # Replace these with actual expected dims if different
        if audio_emb.ndim != 1 or text_vec.ndim != 1 or beh_emb.ndim != 1:
            raise ValueError("Input vectors must be 1D arrays")

        # 3) Prepare inputs and scale
        a_s, t_s, b_s = prepare_inputs(audio_emb, text_vec, beh_emb, scaler)

        # 4) Predict
        preds = model.predict([a_s, t_s, b_s])
        # Model returns [pattern_probs, burnout_score]
        pattern_probs = preds[0][0].tolist() if isinstance(preds[0], np.ndarray) else preds[0]
        burnout_score = float(preds[1][0][0] if isinstance(preds[1], np.ndarray) else preds[1])

        # Map pattern names (same order as model outputs)
        pattern_names = ["social_withdrawal", "sleep_disruption", "emotional_volatility", "cognitive_decline"]
        patterns_out = [{"pattern": name, "probability": float(prob)} for name, prob in zip(pattern_names, pattern_probs)]

        explain_payload = None
        triggers_payload = None

        if req.include_explain and explainer is not None:
            # 5) Explainability using ExplainabilityEngine
            fused_scaled = np.concatenate([a_s.flatten(), t_s.flatten(), b_s.flatten()])
            try:
                # Get feature importances
                feat_importances = explainer.explain_instance(fused_scaled, nsamples=50)
                
                # Create predictions dict for insight card
                predictions_dict = {
                    'patterns': {
                        'social_withdrawal': float(pattern_probs[0]),
                        'sleep_disruption': float(pattern_probs[1]),
                        'emotional_volatility': float(pattern_probs[2]),
                        'cognitive_decline': float(pattern_probs[3])
                    },
                    'burnout': burnout_score
                }
                
                # Generate insight card (will include trigger-response if available)
                insight_card = explainer.generate_insight_card(
                    feat_importances,
                    predictions_dict,
                    trigger_responses=None,  # Will add trigger responses below if found
                    top_k=5
                )
                
                explain_payload = {
                    "top_contributors": [
                        {"feature": feat, "score": float(score), 
                         "friendly_name": explainer.behavior_friendly_names.get(feat, None)}
                        for feat, score in feat_importances[:10]
                    ],
                    "insight_card": {
                        "title": insight_card.title,
                        "subtitle": insight_card.subtitle,
                        "description": insight_card.description,
                        "severity": insight_card.severity,
                        "recommendation": insight_card.recommendation
                    },
                    "modality_contributions": {
                        "audio": sum(score for feat, score in feat_importances if feat.startswith('audio')),
                        "text": sum(score for feat, score in feat_importances if feat.startswith('text') or feat in ['lexical_diversity', 'avg_sentence_length', 'sentence_count']),
                        "behavior": sum(score for feat, score in feat_importances if feat in explainer.behavior_friendly_names)
                    }
                }
            except Exception as ex:
                log.warning(f"Explainability failed: {ex}")
                import traceback
                log.warning(traceback.format_exc())
                explain_payload = {"error": "Explainability temporarily unavailable.", "details": str(ex)}

            # 6) Trigger-response mining from user's historical data
            try:
                user_df = None
                # Load user's historical data if available
                hist_path = f"./user_data/{req.user_id}_daily.csv"
                if os.path.exists(hist_path):
                    import pandas as pd
                    user_df = pd.read_csv(hist_path, parse_dates=["date"]).set_index("date")
                    
                if user_df is not None and len(user_df) >= 14:
                    # Define candidate triggers and targets
                    triggers = ["screen_time_hours", "avg_sleep_hours", "session_count", "break_frequency"]
                    targets = ["burnout_score", "emotional_volatility", "stress_level"]
                    
                    # Use ExplainabilityEngine to find correlations
                    trigger_responses = explainer.find_trigger_responses(
                        df=user_df,
                        triggers=triggers,
                        targets=targets,
                        threshold_corr=0.25,
                        threshold_pval=0.05
                    )
                    
                    if trigger_responses:
                        triggers_payload = [
                            {
                                "trigger": tr.trigger,
                                "target": tr.target,
                                "lag_days": tr.lag_days,
                                "correlation": tr.correlation,
                                "pvalue": tr.pvalue,
                                "confidence": tr.confidence,
                                "interpretation": tr.interpretation
                            }
                            for tr in trigger_responses[:5]  # Top 5 correlations
                        ]
                    else:
                        triggers_payload = []
                else:
                    triggers_payload = None
                    if user_df is not None:
                        log.info(f"Not enough historical data for user {req.user_id}: {len(user_df)} days (need 14+)")
            except Exception as ex2:
                log.warning(f"Trigger mining failed: {ex2}")
                import traceback
                log.warning(traceback.format_exc())
                triggers_payload = None

        meta = {
            "processing_time_s": round(time.time() - start, 3),
            "model_loaded": MODEL_PATH,
            "scaler_loaded": SCALER_PATH
        }

        response = AnalyzeResponse(
            user_id=req.user_id,
            date=req.date,
            patterns=[PatternConfidence(**p) for p in patterns_out],
            burnout_score=burnout_score,
            explainability=explain_payload,
            triggers=triggers_payload,
            meta=meta
        )
        
        # Save analysis to user history
        try:
            save_analysis(
                user_id=req.user_id,
                date=req.date,
                analysis_data={
                    "patterns": patterns_out,
                    "burnout_score": burnout_score,
                    "explainability": explain_payload,
                    "triggers": triggers_payload,
                    "meta": meta
                }
            )
            log.info(f"✓ Saved analysis for user {req.user_id} on {req.date}")
        except Exception as save_err:
            log.warning(f"Failed to save analysis: {save_err}")
        
        return response

    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# health endpoint
@app.get("/health")
def health():
    return {"status":"healthy", "model": MODEL_PATH if model is not None else None}


# ============================================
# Audio Upload Endpoint
# ============================================

class AudioEmbeddingResponse(BaseModel):
    audio_embedding: List[float]
    duration_seconds: float
    sample_rate: int

@app.post("/audio_upload", response_model=AudioEmbeddingResponse)
async def audio_upload(file: UploadFile = File(...)):
    """Upload audio file and get embedding (256-dim feature vector)"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Load audio using librosa
            y, sr = librosa.load(tmp_path, sr=22050, duration=10.0)
            
            # Extract 256 audio features (matching your model)
            # Using MFCCs, chroma, spectral features similar to speech_emotion_recognition.py
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma.T, axis=0)
            
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_mean = np.mean(mel.T, axis=0)[:20]  # Take first 20
            
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroid)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            
            # Combine features to get 256 dimensions
            # mfcc(40) + chroma(12) + mel(20) + other features
            features = np.concatenate([
                mfccs_mean,  # 40
                chroma_mean,  # 12
                mel_mean,  # 20
                [spectral_centroid_mean, spectral_rolloff_mean, zcr_mean]  # 3
            ])
            
            # Pad or truncate to exactly 256 dimensions
            if len(features) < 256:
                features = np.pad(features, (0, 256 - len(features)), mode='constant')
            else:
                features = features[:256]
            
            duration = float(len(y) / sr)
            
            return AudioEmbeddingResponse(
                audio_embedding=features.tolist(),
                duration_seconds=duration,
                sample_rate=sr
            )
        
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    except Exception as e:
        log.error(f"Audio upload error: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")


# ============================================
# Adapter Personalization Endpoints
# ============================================

class UpdateAdapterRequest(BaseModel):
    user_id: str
    audio_samples: List[List[float]]  # List of audio embeddings
    text_samples: List[List[str]]  # List of text arrays (each day's texts)
    behavior_samples: List[List[float]]  # List of behavior vectors
    target_patterns: List[List[float]]  # Ground truth patterns for each sample
    target_burnout: List[float]  # Ground truth burnout scores
    epochs: Optional[int] = 5
    batch_size: Optional[int] = 8

class AdapterStatusResponse(BaseModel):
    user_id: str
    status: str
    adapter_size_kb: float
    training_samples: int
    message: str

@app.post("/adapter/update_adapter", response_model=AdapterStatusResponse)
def update_adapter(req: UpdateAdapterRequest):
    """Train personalized adapter for a user with few-shot examples"""
    try:
        log.info(f"Training adapter for user {req.user_id} with {len(req.audio_samples)} samples")
        
        # Prepare training data
        X_audio, X_text, X_behavior = [], [], []
        y_patterns, y_burnout = [], []
        
        for i in range(len(req.audio_samples)):
            # Get text embedding for this sample
            date = f"adapter_train_{i}"
            text_vec = text_embedder.build_day_feature_vector(req.user_id, date, req.text_samples[i])
            
            audio_emb = np.array(req.audio_samples[i], dtype=np.float32)
            beh_emb = np.array(req.behavior_samples[i], dtype=np.float32)
            
            # Scale inputs
            a_s, t_s, b_s = prepare_inputs(audio_emb, text_vec, beh_emb, scaler)
            
            X_audio.append(a_s[0])
            X_text.append(t_s[0])
            X_behavior.append(b_s[0])
            y_patterns.append(req.target_patterns[i])
            y_burnout.append([req.target_burnout[i]])
        
        X_audio = np.array(X_audio)
        X_text = np.array(X_text)
        X_behavior = np.array(X_behavior)
        y_patterns = np.array(y_patterns)
        y_burnout = np.array(y_burnout)
        
        # Create and train adapter
        adapter = PersonalizedAdapter()
        
        history = adapter.train_user_adapter(
            user_id=req.user_id,
            audio_data=X_audio,
            text_data=X_text,
            behavior_data=X_behavior,
            pattern_labels=y_patterns,
            burnout_labels=y_burnout,
            epochs=req.epochs,
            batch_size=req.batch_size,
            validation_split=0.0  # No validation with few-shot
        )
        
        # Save adapter
        adapter.save_adapter_weights(req.user_id)
        adapter_path = f"./models/user_adapters/adapter_{req.user_id}.npz"
        adapter_size = os.path.getsize(adapter_path) / 1024  # KB
        
        return AdapterStatusResponse(
            user_id=req.user_id,
            status="success",
            adapter_size_kb=round(adapter_size, 2),
            training_samples=len(req.audio_samples),
            message=f"Adapter trained and saved to {adapter_path}"
        )
    
    except Exception as e:
        log.error(f"Adapter training error: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/adapter/load_adapter/{user_id}")
def load_adapter_endpoint(user_id: str):
    """Load a user's personalized adapter"""
    try:
        adapter_path = f"./models/user_adapters/adapter_{user_id}.npz"
        
        if not os.path.exists(adapter_path):
            raise HTTPException(status_code=404, detail=f"No adapter found for user {user_id}")
        
        adapter = PersonalizedAdapter()
        success = adapter.load_adapter_weights(user_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load adapter weights")
        
        adapter_size = os.path.getsize(adapter_path) / 1024
        
        return {
            "user_id": user_id,
            "status": "loaded",
            "adapter_path": adapter_path,
            "adapter_size_kb": round(adapter_size, 2),
            "message": "Adapter loaded successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Adapter loading error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/adapter/status/{user_id}")
def adapter_status(user_id: str):
    """Check if user has a trained adapter"""
    adapter_path = f"./models/user_adapters/adapter_{user_id}.npz"
    
    if os.path.exists(adapter_path):
        adapter_size = os.path.getsize(adapter_path) / 1024
        return {
            "user_id": user_id,
            "has_adapter": True,
            "adapter_path": adapter_path,
            "adapter_size_kb": round(adapter_size, 2)
        }
    else:
        return {
            "user_id": user_id,
            "has_adapter": False,
            "message": "No personalized adapter found"
        }


# ============================================
# Authentication Endpoints
# ============================================

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class AuthResponse(BaseModel):
    success: bool
    user: Optional[Dict[str, Any]] = None
    message: str

@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """Authenticate user with email and password"""
    try:
        user = authenticate_user(req.email, req.password)
        
        if user:
            return AuthResponse(
                success=True,
                user=user,
                message="Login successful"
            )
        else:
            return AuthResponse(
                success=False,
                message="Invalid email or password"
            )
    except Exception as e:
        log.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    """Register a new user"""
    try:
        user = register_user(req.email, req.password, req.name)
        
        if user:
            return AuthResponse(
                success=True,
                user=user,
                message="Registration successful"
            )
        else:
            return AuthResponse(
                success=False,
                message="User already exists"
            )
    except Exception as e:
        log.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/user/{user_id}")
def get_user(user_id: str):
    """Get user information by user_id"""
    try:
        user = get_user_by_id(user_id)
        
        if user:
            return user
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Get user error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Data Retrieval Endpoints
# ============================================

@app.get("/data/history/{user_id}")
def get_history(user_id: str, limit: Optional[int] = 30):
    """Get analysis history for a user"""
    try:
        history = get_user_history(user_id, limit=limit)
        return {
            "user_id": user_id,
            "count": len(history),
            "analyses": history
        }
    except Exception as e:
        log.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/timeline/{user_id}")
def get_timeline(user_id: str, days: int = 30):
    """Get timeline data for charts"""
    try:
        timeline = get_timeline_data(user_id, days=days)
        return timeline
    except Exception as e:
        log.error(f"Get timeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/insights/{user_id}")
def get_insights(user_id: str):
    """Get insights summary for a user"""
    try:
        insights = get_insights_summary(user_id)
        return insights
    except Exception as e:
        log.error(f"Get insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("analyze_api:app", host="0.0.0.0", port=8100, reload=True)
