"""
Analysis Data Storage for BAYMAX
Stores user analysis history and retrieves data for visualization
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional

# Data directory
DATA_DIR = "./user_data"
os.makedirs(DATA_DIR, exist_ok=True)

def save_analysis(user_id: str, date: str, analysis_data: Dict) -> bool:
    """
    Save analysis result for a user on a specific date
    Creates/updates the user's data file
    """
    try:
        user_file = os.path.join(DATA_DIR, f"{user_id}_history.json")
        
        # Load existing data or create new
        if os.path.exists(user_file):
            with open(user_file, 'r') as f:
                history = json.load(f)
        else:
            history = {
                "user_id": user_id,
                "analyses": []
            }
        
        # Add new analysis with timestamp
        analysis_entry = {
            "date": date,
            "timestamp": datetime.now().isoformat(),
            **analysis_data
        }
        
        # Check if entry for this date exists, update or append
        date_exists = False
        for i, entry in enumerate(history["analyses"]):
            if entry.get("date") == date:
                history["analyses"][i] = analysis_entry
                date_exists = True
                break
        
        if not date_exists:
            history["analyses"].append(analysis_entry)
        
        # Sort by date (newest first)
        history["analyses"].sort(key=lambda x: x.get("date", ""), reverse=True)
        
        # Save to file
        with open(user_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return False

def get_user_history(user_id: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Get analysis history for a user
    Returns list of analyses, optionally limited to recent N entries
    """
    try:
        user_file = os.path.join(DATA_DIR, f"{user_id}_history.json")
        
        if not os.path.exists(user_file):
            return []
        
        with open(user_file, 'r') as f:
            history = json.load(f)
        
        analyses = history.get("analyses", [])
        
        if limit:
            return analyses[:limit]
        
        return analyses
    
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def get_analysis_by_date(user_id: str, date: str) -> Optional[Dict]:
    """Get a specific analysis by date"""
    try:
        history = get_user_history(user_id)
        
        for analysis in history:
            if analysis.get("date") == date:
                return analysis
        
        return None
    
    except Exception as e:
        print(f"Error getting analysis: {e}")
        return None

def get_timeline_data(user_id: str, days: int = 30) -> Dict:
    """
    Get timeline data for charts
    Returns patterns and burnout scores over time
    """
    try:
        history = get_user_history(user_id, limit=days)
        
        timeline = {
            "dates": [],
            "burnout_scores": [],
            "patterns": {
                "social_withdrawal": [],
                "sleep_disruption": [],
                "emotional_volatility": [],
                "cognitive_decline": []
            }
        }
        
        for analysis in reversed(history):  # Oldest first for timeline
            timeline["dates"].append(analysis.get("date", ""))
            timeline["burnout_scores"].append(analysis.get("burnout_score", 0))
            
            # Extract pattern probabilities
            patterns = analysis.get("patterns", [])
            pattern_dict = {p["pattern"]: p["probability"] for p in patterns if isinstance(p, dict)}
            
            for pattern_name in timeline["patterns"].keys():
                timeline["patterns"][pattern_name].append(
                    pattern_dict.get(pattern_name, 0)
                )
        
        return timeline
    
    except Exception as e:
        print(f"Error getting timeline: {e}")
        return {
            "dates": [],
            "burnout_scores": [],
            "patterns": {
                "social_withdrawal": [],
                "sleep_disruption": [],
                "emotional_volatility": [],
                "cognitive_decline": []
            }
        }

def get_insights_summary(user_id: str) -> Dict:
    """
    Get summary insights from recent analyses
    Returns average scores, trends, top contributors
    """
    try:
        history = get_user_history(user_id, limit=7)  # Last 7 days
        
        if not history:
            return {
                "avg_burnout": 0,
                "trend": "stable",
                "top_patterns": [],
                "top_contributors": []
            }
        
        # Calculate averages
        burnout_scores = [a.get("burnout_score", 0) for a in history]
        avg_burnout = sum(burnout_scores) / len(burnout_scores) if burnout_scores else 0
        
        # Determine trend
        if len(burnout_scores) >= 2:
            recent_avg = sum(burnout_scores[:3]) / min(3, len(burnout_scores))
            older_avg = sum(burnout_scores[3:]) / max(1, len(burnout_scores) - 3)
            
            if recent_avg > older_avg + 0.1:
                trend = "increasing"
            elif recent_avg < older_avg - 0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Get most common patterns
        pattern_counts = {}
        for analysis in history:
            for pattern in analysis.get("patterns", []):
                if isinstance(pattern, dict):
                    name = pattern.get("pattern", "")
                    prob = pattern.get("probability", 0)
                    if prob > 0.5:  # Only count significant patterns
                        pattern_counts[name] = pattern_counts.get(name, 0) + 1
        
        top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Get most common contributors
        contributor_scores = {}
        for analysis in history:
            explainability = analysis.get("explainability", {})
            contributors = explainability.get("top_contributors", [])
            
            for contrib in contributors:
                if isinstance(contrib, dict):
                    feature = contrib.get("feature", "")
                    score = contrib.get("score", 0)
                    if feature not in contributor_scores:
                        contributor_scores[feature] = []
                    contributor_scores[feature].append(score)
        
        # Average contributor scores
        avg_contributors = {
            feature: sum(scores) / len(scores) 
            for feature, scores in contributor_scores.items()
        }
        
        top_contributors = sorted(
            avg_contributors.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            "avg_burnout": round(avg_burnout, 2),
            "trend": trend,
            "top_patterns": [{"pattern": p[0], "count": p[1]} for p in top_patterns],
            "top_contributors": [{"feature": f, "avg_score": round(s, 2)} for f, s in top_contributors],
            "total_analyses": len(history)
        }
    
    except Exception as e:
        print(f"Error getting insights: {e}")
        return {
            "avg_burnout": 0,
            "trend": "error",
            "top_patterns": [],
            "top_contributors": []
        }

if __name__ == "__main__":
    # Test the storage system
    print("Testing analysis data storage...")
    
    # Sample analysis
    test_analysis = {
        "patterns": [
            {"pattern": "sleep_disruption", "probability": 0.85},
            {"pattern": "social_withdrawal", "probability": 0.72}
        ],
        "burnout_score": 0.78,
        "explainability": {
            "top_contributors": [
                {"feature": "sleepHours", "score": 0.45},
                {"feature": "screenTime", "score": 0.32}
            ]
        }
    }
    
    # Save test analysis
    success = save_analysis("demo_user_123", "2025-12-12", test_analysis)
    print(f"✓ Save analysis: {'Success' if success else 'Failed'}")
    
    # Retrieve history
    history = get_user_history("demo_user_123")
    print(f"✓ Retrieved {len(history)} analyses")
    
    # Get timeline
    timeline = get_timeline_data("demo_user_123")
    print(f"✓ Timeline has {len(timeline['dates'])} data points")
    
    # Get insights
    insights = get_insights_summary("demo_user_123")
    print(f"✓ Insights summary: avg_burnout={insights['avg_burnout']}, trend={insights['trend']}")
    
    print("\n✓ Data storage system ready!")
