"""
Text Embedding Module for Mental Health Analysis
Provides text embedding, feature extraction, and semantic analysis capabilities.

Author: BAYMAX ModeM Team
Date: December 2025

TODO: Implement caching for repeated text embeddings
TODO: Support batch processing for multiple texts at once
TODO: Add multilingual support (currently English only)
"""

import os
from typing import List, Dict, Optional, Tuple
import warnings

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')


class TextEmbeddingModule:
    """
    A comprehensive text embedding module for analyzing text data.
    
    Features:
    - Generates embeddings using DistilBERT
    - Computes lexical diversity and sentence statistics
    - Aggregates daily text features
    - Calculates semantic drift between days
    - Caches embeddings for efficiency
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        batch_size: int = 16,
        cache_dir: str = "./emb_cache",
        device: Optional[str] = None
    ):
        """
        Initialize the text embedding module.
        
        Args:
            model_name: HuggingFace model name (default: distilbert-base-uncased)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing texts
            cache_dir: Directory to cache computed embeddings
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to inference mode
        
        # Load stopwords
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            print("Downloading NLTK data...")
            import nltk
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words("english"))
        
        print(f"Model loaded successfully on {self.device}")
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text by removing extra whitespace.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        return " ".join(text.strip().split())
    
    def compute_lexical_diversity(self, text: str) -> float:
        """
        Compute lexical diversity (unique words / total words).
        
        Args:
            text: Input text string
            
        Returns:
            Lexical diversity score (0.0 to 1.0)
        """
        tokens = [
            t.lower() for t in word_tokenize(text)
            if t.isalpha() and t.lower() not in self.stop_words
        ]
        if len(tokens) == 0:
            return 0.0
        return len(set(tokens)) / len(tokens)
    
    def text_to_embeddings(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert texts to embeddings using mean pooling.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length (uses default if None)
            batch_size: Batch size (uses default if None)
            
        Returns:
            Numpy array of shape (len(texts), hidden_dim)
        """
        if max_length is None:
            max_length = self.max_length
        if batch_size is None:
            batch_size = self.batch_size
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                # Get model output
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
                
                # Mean pooling with attention mask
                mask = attention_mask.unsqueeze(-1).type(torch.float32)  # (batch, seq_len, 1)
                summed = torch.sum(last_hidden * mask, dim=1)  # (batch, hidden)
                lengths = torch.clamp(torch.sum(mask, dim=1), min=1e-8)  # (batch, 1)
                pooled = summed / lengths  # (batch, hidden)
                
                all_embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def aggregate_day_embeddings(self, embeddings: np.ndarray) -> Dict:
        """
        Aggregate embeddings for a single day.
        
        Args:
            embeddings: Numpy array of shape (n_entries, dim)
            
        Returns:
            Dictionary with mean, std, and count statistics
        """
        if embeddings.shape[0] == 0:
            dim = self.model.config.hidden_size
            return {
                "mean": np.zeros(dim, dtype=np.float32),
                "std": np.zeros(dim, dtype=np.float32),
                "count": 0
            }
        
        return {
            "mean": embeddings.mean(axis=0),
            "std": embeddings.std(axis=0),
            "count": embeddings.shape[0]
        }
    
    @staticmethod
    def semantic_drift(prev_mean: np.ndarray, curr_mean: np.ndarray) -> float:
        """
        Calculate semantic drift between two day centroids.
        
        Args:
            prev_mean: Previous day's mean embedding
            curr_mean: Current day's mean embedding
            
        Returns:
            Cosine distance (1 - cosine_similarity) in range [0, 2]
        """
        if prev_mean is None or curr_mean is None:
            return 0.0
        
        similarity = cosine_similarity(
            prev_mean.reshape(1, -1),
            curr_mean.reshape(1, -1)
        )[0, 0]
        
        return float(1.0 - similarity)
    
    def compute_entry_features(self, text: str) -> Dict:
        """
        Compute linguistic features for a single text entry.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with lexical diversity and average sentence length
        """
        lexical_diversity = self.compute_lexical_diversity(text)
        
        sentences = sent_tokenize(text)
        if sentences:
            sentence_lengths = [len(word_tokenize(s)) for s in sentences]
            avg_sent_len = float(np.mean(sentence_lengths))
        else:
            avg_sent_len = 0.0
        
        return {
            "lexical_diversity": lexical_diversity,
            "avg_sent_len": avg_sent_len
        }
    
    def cache_embeddings(
        self,
        user_id: str,
        date_str: str,
        texts: List[str],
        overwrite: bool = False
    ) -> np.ndarray:
        """
        Compute and cache embeddings for a user on a specific date.
        
        Args:
            user_id: User identifier
            date_str: Date string (e.g., "2025-12-08")
            texts: List of text entries
            overwrite: If True, recompute even if cached
            
        Returns:
            Numpy array of embeddings
        """
        cache_key = f"{user_id}__{date_str}.npy"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        # Load from cache if exists
        if os.path.exists(cache_path) and not overwrite:
            return np.load(cache_path)
        
        # Compute embeddings
        preprocessed_texts = [self.preprocess_text(t) for t in texts]
        embeddings = self.text_to_embeddings(preprocessed_texts)
        
        # Save to cache
        np.save(cache_path, embeddings)
        
        return embeddings
    
    def load_cached_embeddings(self, user_id: str, date_str: str) -> Optional[np.ndarray]:
        """
        Load cached embeddings if they exist.
        
        Args:
            user_id: User identifier
            date_str: Date string
            
        Returns:
            Numpy array of embeddings or None if not found
        """
        cache_path = os.path.join(self.cache_dir, f"{user_id}__{date_str}.npy")
        if not os.path.exists(cache_path):
            return None
        return np.load(cache_path)
    
    def day_text_features(
        self,
        user_id: str,
        date_str: str,
        texts: List[str]
    ) -> Dict:
        """
        Compute comprehensive features for all texts on a given day.
        
        Args:
            user_id: User identifier
            date_str: Date string
            texts: List of text entries for the day
            
        Returns:
            Dictionary with embedding stats and linguistic features
        """
        # Get or compute embeddings
        embeddings = self.cache_embeddings(user_id, date_str, texts)
        
        # Aggregate embeddings
        agg = self.aggregate_day_embeddings(embeddings)
        
        # Compute entry features
        entry_features = [self.compute_entry_features(t) for t in texts]
        
        if entry_features:
            mean_ld = float(np.mean([f["lexical_diversity"] for f in entry_features]))
            mean_sent_len = float(np.mean([f["avg_sent_len"] for f in entry_features]))
        else:
            mean_ld, mean_sent_len = 0.0, 0.0
        
        return {
            "emb_mean": agg["mean"],
            "emb_std": agg["std"],
            "entry_count": agg["count"],
            "lexical_diversity": mean_ld,
            "avg_sentence_length": mean_sent_len
        }
    
    def build_day_feature_vector(
        self,
        user_id: str,
        date_str: str,
        texts: List[str]
    ) -> np.ndarray:
        """
        Build a complete feature vector for a day's texts.
        
        Args:
            user_id: User identifier
            date_str: Date string
            texts: List of text entries
            
        Returns:
            Numpy array of shape (hidden_dim + 3,) containing:
            - Mean embedding vector (768 dims for DistilBERT)
            - Entry count
            - Lexical diversity
            - Average sentence length
        """
        features = self.day_text_features(user_id, date_str, texts)
        
        # Main embedding vector
        text_vec = features["emb_mean"]  # shape (768,)
        
        # Additional statistical features
        extras = np.array([
            features["entry_count"],
            features["lexical_diversity"],
            features["avg_sentence_length"]
        ], dtype=np.float32)
        
        # Concatenate all features
        day_vector = np.concatenate([text_vec, extras])  # shape (771,)
        
        return day_vector
    
    def compute_multi_day_drift(
        self,
        user_id: str,
        date_texts_list: List[Tuple[str, List[str]]]
    ) -> List[float]:
        """
        Compute semantic drift across multiple consecutive days.
        
        Args:
            user_id: User identifier
            date_texts_list: List of (date_str, texts) tuples in chronological order
            
        Returns:
            List of drift scores (one less than number of days)
        """
        drifts = []
        prev_mean = None
        
        for date_str, texts in date_texts_list:
            features = self.day_text_features(user_id, date_str, texts)
            curr_mean = features["emb_mean"]
            
            if prev_mean is not None:
                drift = self.semantic_drift(prev_mean, curr_mean)
                drifts.append(drift)
            
            prev_mean = curr_mean
        
        return drifts
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        return self.model.config.hidden_size


# Convenience function for quick initialization
def create_text_embedder(cache_dir: str = "./emb_cache") -> TextEmbeddingModule:
    """
    Create a text embedding module with default settings.
    
    Args:
        cache_dir: Directory for caching embeddings
        
    Returns:
        Initialized TextEmbeddingModule
    """
    return TextEmbeddingModule(cache_dir=cache_dir)


# Example usage
if __name__ == "__main__":
    # Initialize the module
    embedder = create_text_embedder()
    
    # Example texts
    sample_texts = [
        "Today was a wonderful day. I felt happy and energetic.",
        "I accomplished all my goals and feel proud of myself.",
        "Looking forward to tomorrow with excitement."
    ]
    
    # Compute embeddings
    print("\n" + "="*60)
    print("Example: Computing embeddings for sample texts")
    print("="*60)
    
    embeddings = embedder.text_to_embeddings(sample_texts)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    
    # Compute day features
    print("\n" + "="*60)
    print("Example: Computing day-level features")
    print("="*60)
    
    day_features = embedder.day_text_features("user_123", "2025-12-08", sample_texts)
    print(f"\nEntry count: {day_features['entry_count']}")
    print(f"Lexical diversity: {day_features['lexical_diversity']:.4f}")
    print(f"Avg sentence length: {day_features['avg_sentence_length']:.2f}")
    
    # Build complete feature vector
    feature_vector = embedder.build_day_feature_vector("user_123", "2025-12-08", sample_texts)
    print(f"\nComplete feature vector shape: {feature_vector.shape}")
    
    # Compute semantic drift
    print("\n" + "="*60)
    print("Example: Computing semantic drift")
    print("="*60)
    
    day1_texts = ["I feel happy and energetic today."]
    day2_texts = ["I'm feeling sad and tired."]
    
    date_texts = [
        ("2025-12-07", day1_texts),
        ("2025-12-08", day2_texts)
    ]
    
    drifts = embedder.compute_multi_day_drift("user_123", date_texts)
    print(f"\nSemantic drift between days: {drifts[0]:.4f}")
    
    print("\n" + "="*60)
    print("Text Embedding Module ready for use!")
    print("="*60)


# ============================================================================
# FastAPI Server Implementation
# ============================================================================

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")


if FASTAPI_AVAILABLE:
    
    # Pydantic models for request/response
    class TextsRequest(BaseModel):
        """Request model for text processing endpoints."""
        user_id: str
        date: str
        texts: List[str]
        overwrite: Optional[bool] = False
    
    class DriftRequest(BaseModel):
        """Request model for semantic drift computation."""
        user_id: str
        date_texts: List[List[str]]  # list of lists of texts in chronological order
        dates: List[str]
    
    class FeatureResponse(BaseModel):
        """Response model for day features."""
        emb_mean: List[float]
        emb_std: List[float]
        entry_count: int
        lexical_diversity: float
        avg_sentence_length: float
    
    class VectorResponse(BaseModel):
        """Response model for feature vector."""
        vector: List[float]
        shape: List[int]
    
    
    def create_api_app(embedder: Optional[TextEmbeddingModule] = None) -> FastAPI:
        """
        Create a FastAPI application for the text embedding module.
        
        Args:
            embedder: Existing TextEmbeddingModule instance, or None to create a new one
            
        Returns:
            FastAPI application instance
        """
        app = FastAPI(
            title="BAYMAX: ModeM - Text Embedding API",
            description="API for text embedding and semantic analysis for mental health monitoring",
            version="1.0.0"
        )
        
        # Initialize embedder (singleton pattern)
        if embedder is None:
            embedder = TextEmbeddingModule(cache_dir="./emb_cache")
        
        @app.get("/")
        def root():
            """Root endpoint with API information."""
            return {
                "service": "BAYMAX: ModeM - Text Embedding API",
                "version": "1.0.0",
                "status": "running",
                "embedding_dim": embedder.get_embedding_dimension(),
                "endpoints": [
                    "/embed_text",
                    "/day_features",
                    "/build_day_vector",
                    "/multi_day_drift",
                    "/semantic_drift"
                ]
            }
        
        @app.get("/health")
        def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "model": embedder.model_name}
        
        @app.post("/embed_text")
        def embed_text(req: TextsRequest):
            """
            Compute and cache embeddings for given texts.
            
            Args:
                req: TextsRequest containing user_id, date, texts, and optional overwrite flag
                
            Returns:
                Dictionary with embedding shape and status
            """
            try:
                embs = embedder.cache_embeddings(
                    req.user_id, 
                    req.date, 
                    req.texts, 
                    overwrite=req.overwrite
                )
                return {
                    "status": "ok",
                    "shape": list(embs.shape),
                    "embedding_dim": embs.shape[1] if len(embs.shape) > 1 else 0,
                    "num_texts": embs.shape[0]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/day_features", response_model=FeatureResponse)
        def day_features(req: TextsRequest):
            """
            Compute comprehensive features for a day's texts.
            
            Args:
                req: TextsRequest containing user_id, date, and texts
                
            Returns:
                Dictionary with embedding statistics and linguistic features
            """
            try:
                feat = embedder.day_text_features(req.user_id, req.date, req.texts)
                
                # Convert numpy arrays to lists for JSON serialization
                return {
                    "emb_mean": feat["emb_mean"].tolist(),
                    "emb_std": feat["emb_std"].tolist(),
                    "entry_count": int(feat["entry_count"]),
                    "lexical_diversity": float(feat["lexical_diversity"]),
                    "avg_sentence_length": float(feat["avg_sentence_length"])
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/build_day_vector", response_model=VectorResponse)
        def build_day_vector(req: TextsRequest):
            """
            Build complete feature vector for a day (embedding + statistics).
            
            Args:
                req: TextsRequest containing user_id, date, and texts
                
            Returns:
                Dictionary with feature vector and its shape
            """
            try:
                vec = embedder.build_day_feature_vector(req.user_id, req.date, req.texts)
                return {
                    "vector": vec.tolist(),
                    "shape": list(vec.shape)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/multi_day_drift")
        def multi_day_drift(req: DriftRequest):
            """
            Compute semantic drift across multiple consecutive days.
            
            Args:
                req: DriftRequest containing user_id, dates, and corresponding texts
                
            Returns:
                Dictionary with drift scores between consecutive days
            """
            try:
                # Build list of (date, texts) tuples
                date_texts = list(zip(req.dates, req.date_texts))
                drifts = embedder.compute_multi_day_drift(req.user_id, date_texts)
                
                return {
                    "drifts": drifts,
                    "num_days": len(req.dates),
                    "num_drifts": len(drifts),
                    "avg_drift": float(np.mean(drifts)) if drifts else 0.0,
                    "max_drift": float(np.max(drifts)) if drifts else 0.0
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/semantic_drift")
        def semantic_drift_endpoint(user_id: str, date1: str, date2: str):
            """
            Compute semantic drift between two specific dates.
            
            Args:
                user_id: User identifier
                date1: First date
                date2: Second date
                
            Returns:
                Dictionary with drift score
            """
            try:
                # Load cached embeddings for both dates
                emb1 = embedder.load_cached_embeddings(user_id, date1)
                emb2 = embedder.load_cached_embeddings(user_id, date2)
                
                if emb1 is None or emb2 is None:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Embeddings not found for {user_id} on {date1} or {date2}"
                    )
                
                # Compute means
                mean1 = emb1.mean(axis=0)
                mean2 = emb2.mean(axis=0)
                
                # Calculate drift
                drift = embedder.semantic_drift(mean1, mean2)
                
                return {
                    "user_id": user_id,
                    "date1": date1,
                    "date2": date2,
                    "drift": float(drift)
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.delete("/cache/{user_id}/{date}")
        def delete_cache(user_id: str, date: str):
            """
            Delete cached embeddings for a specific user and date (GDPR compliance).
            
            Args:
                user_id: User identifier
                date: Date string
                
            Returns:
                Dictionary with deletion status
            """
            try:
                cache_path = os.path.join(embedder.cache_dir, f"{user_id}__{date}.npy")
                
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    return {
                        "status": "deleted",
                        "user_id": user_id,
                        "date": date
                    }
                else:
                    return {
                        "status": "not_found",
                        "user_id": user_id,
                        "date": date
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    
    def run_api_server(
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        embedder: Optional[TextEmbeddingModule] = None
    ):
        """
        Run the FastAPI server.
        
        Args:
            host: Host address to bind to
            port: Port number to listen on
            reload: Enable auto-reload on code changes (development mode)
            embedder: Existing TextEmbeddingModule instance
        """
        app = create_api_app(embedder)
        uvicorn.run(app, host=host, port=port, reload=reload)


# Example API usage
def example_api_usage():
    """Example of how to use the API programmatically."""
    import requests
    
    # Example request to build day vector
    request_data = {
        "user_id": "user_123",
        "date": "2025-12-08",
        "texts": [
            "Today was a wonderful day.",
            "I felt happy and energetic.",
            "Looking forward to tomorrow."
        ]
    }
    
    # This would be called with the server running
    # response = requests.post("http://localhost:8000/build_day_vector", json=request_data)
    # print(response.json())
    
    print("\nExample API request:")
    print("POST http://localhost:8000/build_day_vector")
    print(f"Body: {request_data}")
    print("\nTo start the server, run: python text_embedding_module.py --serve")


if __name__ == "__main__":
    import sys
    
    if "--serve" in sys.argv or "--api" in sys.argv:
        # Run as API server
        print("="*60)
        print("Starting BAYMAX: ModeM Text Embedding API Server")
        print("="*60)
        
        if not FASTAPI_AVAILABLE:
            print("\nError: FastAPI not installed!")
            print("Install with: pip install fastapi uvicorn")
            sys.exit(1)
        
        # Parse optional port argument
        port = 8000
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
        
        # Create embedder and run server
        embedder = create_text_embedder()
        print(f"\nServer will be available at: http://localhost:{port}")
        print("API documentation: http://localhost:{port}/docs")
        print("="*60)
        
        run_api_server(port=port, embedder=embedder)
    else:
        # Run example usage
        # Initialize the module
        embedder = create_text_embedder()
        
        # Example texts
        sample_texts = [
            "Today was a wonderful day. I felt happy and energetic.",
            "I accomplished all my goals and feel proud of myself.",
            "Looking forward to tomorrow with excitement."
        ]
        
        # Compute embeddings
        print("\n" + "="*60)
        print("Example: Computing embeddings for sample texts")
        print("="*60)
        
        embeddings = embedder.text_to_embeddings(sample_texts)
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
        
        # Compute day features
        print("\n" + "="*60)
        print("Example: Computing day-level features")
        print("="*60)
        
        day_features = embedder.day_text_features("user_123", "2025-12-08", sample_texts)
        print(f"\nEntry count: {day_features['entry_count']}")
        print(f"Lexical diversity: {day_features['lexical_diversity']:.4f}")
        print(f"Avg sentence length: {day_features['avg_sentence_length']:.2f}")
        
        # Build complete feature vector
        feature_vector = embedder.build_day_feature_vector("user_123", "2025-12-08", sample_texts)
        print(f"\nComplete feature vector shape: {feature_vector.shape}")
        
        # Compute semantic drift
        print("\n" + "="*60)
        print("Example: Computing semantic drift")
        print("="*60)
        
        day1_texts = ["I feel happy and energetic today."]
        day2_texts = ["I'm feeling sad and tired."]
        
        date_texts = [
            ("2025-12-07", day1_texts),
            ("2025-12-08", day2_texts)
        ]
        
        drifts = embedder.compute_multi_day_drift("user_123", date_texts)
        print(f"\nSemantic drift between days: {drifts[0]:.4f}")
        
        print("\n" + "="*60)
        print("Text Embedding Module ready for use!")
        print("="*60)
        print("\nTo start the API server, run:")
        print("  python text_embedding_module.py --serve")
        print("  python text_embedding_module.py --serve --port 8080")
        print("="*60)
