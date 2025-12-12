# api_explainability.py
"""
FastAPI endpoints for Explainability & Insights
Add these endpoints to text_embedding_module.py or create as separate microservice
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from explainability_engine import ExplainabilityEngine, analyze_prediction, InsightCard

# Initialize
app = FastAPI(title="BAYMAX Explainability API")
engine = ExplainabilityEngine()

# Request/Response Models
class FeatureAttribution(BaseModel):
    feature: str
    score: float
    friendly_name: Optional[str] = None

class TriggerResponseResult(BaseModel):
    trigger: str
    target: str
    lag_days: int
    correlation: float
    pvalue: float
    confidence: str
    interpretation: str

class InsightCardResponse(BaseModel):
    title: str
    subtitle: str
    description: str
    severity: str
    recommendation: str
    contributors: List[Dict[str, float]]
    caveat: str

class ExplainRequest(BaseModel):
    fused_input: List[float]  # 1033 dims
    nsamples: int = 50

class InsightRequest(BaseModel):
    fused_input: List[float]  # 1033 dims
    predictions: Dict  # {'patterns': {...}, 'burnout': float}
    user_history: Optional[Dict] = None  # DataFrame as dict

class TriggerAnalysisRequest(BaseModel):
    user_history: Dict  # DataFrame as dict with columns
    triggers: List[str]
    targets: List[str]
    threshold_corr: float = 0.25
    threshold_pval: float = 0.05

# Endpoints

@app.get("/")
def root():
    return {
        "service": "BAYMAX Explainability API",
        "version": "1.0",
        "endpoints": [
            "/explain",
            "/insights",
            "/trigger_response",
            "/feature_groups"
        ]
    }

@app.post("/explain", response_model=List[FeatureAttribution])
def explain_prediction(request: ExplainRequest):
    """
    Get feature attributions for a prediction.
    
    Args:
        fused_input: Scaled fused input (1033 dims)
        nsamples: SHAP samples (more = better but slower)
    
    Returns:
        List of feature attributions sorted by importance
    """
    try:
        fused_input = np.array(request.fused_input, dtype=np.float32)
        
        if len(fused_input) != 1033:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 1033 features, got {len(fused_input)}"
            )
        
        # Get attributions
        importances = engine.explain_instance(fused_input, nsamples=request.nsamples)
        
        # Format response
        results = []
        for feat, score in importances[:20]:  # Top 20
            friendly = engine.behavior_friendly_names.get(feat, None)
            results.append(FeatureAttribution(
                feature=feat,
                score=float(score),
                friendly_name=friendly
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insights", response_model=InsightCardResponse)
def generate_insights(request: InsightRequest):
    """
    Generate comprehensive insight card with recommendations.
    
    Args:
        fused_input: Scaled input (1033 dims)
        predictions: Model predictions dict
        user_history: Optional DataFrame for trigger analysis
    
    Returns:
        InsightCard with analysis and recommendations
    """
    try:
        fused_input = np.array(request.fused_input, dtype=np.float32)
        
        # Convert user_history if provided
        import pandas as pd
        user_df = None
        if request.user_history:
            user_df = pd.DataFrame(request.user_history)
            if 'date' in user_df.columns:
                user_df['date'] = pd.to_datetime(user_df['date'])
                user_df.set_index('date', inplace=True)
        
        # Generate insight card
        card = analyze_prediction(fused_input, request.predictions, user_df)
        
        # Format response
        return InsightCardResponse(
            title=card.title,
            subtitle=card.subtitle,
            description=card.description,
            severity=card.severity,
            recommendation=card.recommendation,
            contributors=[
                {"feature": feat, "score": score} 
                for feat, score in card.contributors[:5]
            ],
            caveat=card.caveat
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trigger_response", response_model=List[TriggerResponseResult])
def analyze_triggers(request: TriggerAnalysisRequest):
    """
    Find significant trigger-response correlations in user history.
    
    Args:
        user_history: DataFrame with date-indexed data
        triggers: List of trigger column names
        targets: List of target column names
        threshold_corr: Minimum correlation
        threshold_pval: Maximum p-value
    
    Returns:
        List of significant trigger-response pairs
    """
    try:
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(request.user_history)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Find correlations
        results = engine.find_trigger_responses(
            df,
            request.triggers,
            request.targets,
            threshold_corr=request.threshold_corr,
            threshold_pval=request.threshold_pval
        )
        
        # Format response
        return [
            TriggerResponseResult(
                trigger=tr.trigger,
                target=tr.target,
                lag_days=tr.lag_days,
                correlation=tr.correlation,
                pvalue=tr.pvalue,
                confidence=tr.confidence,
                interpretation=tr.interpretation
            )
            for tr in results
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature_groups")
def get_feature_groups():
    """Get feature group information."""
    return {
        "total_features": len(engine.feature_names),
        "groups": {
            "audio": {
                "count": len(engine.feature_groups['audio']),
                "description": "Speech emotion recognition features"
            },
            "text": {
                "count": len(engine.feature_groups['text']),
                "description": "Text embeddings and lexical features",
                "extras": ["lexical_diversity", "avg_sentence_length", "sentence_count"]
            },
            "behavior": {
                "count": len(engine.feature_groups['behavior']),
                "features": list(engine.behavior_friendly_names.keys()),
                "friendly_names": engine.behavior_friendly_names
            }
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": engine.model is not None,
        "scaler_loaded": engine.scaler is not None,
        "shap_available": engine.get_explainer() is not None
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting BAYMAX Explainability API...")
    print("Docs available at: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)
