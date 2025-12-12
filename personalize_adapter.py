# personalize_adapter.py
"""
Personalized Adapter Training for BAYMAX
Allows fine-tuning the adapter layers for individual users with minimal data.
"""

import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

MODEL_PATH = "./models/fusion_baseline.h5"
SCALER_PATH = "./models/scaler.joblib"
USER_ADAPTERS_DIR = "./models/user_adapters"

os.makedirs(USER_ADAPTERS_DIR, exist_ok=True)

class PersonalizedAdapter:
    """Handler for user-specific adapter training."""
    
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """Initialize with base model."""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load_resources()
    
    def _load_resources(self):
        """Load model and scaler."""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path, compile=False, safe_mode=False)
            print(f"✓ Loaded base model from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print(f"✓ Loaded scaler from {self.scaler_path}")
        else:
            print(f"⚠️ Scaler not found: {self.scaler_path}")
    
    def freeze_base_model(self):
        """Freeze all layers except adapter layers."""
        adapter_count = 0
        frozen_count = 0
        
        for layer in self.model.layers:
            if "adapter" in layer.name.lower():
                layer.trainable = True
                adapter_count += 1
            else:
                layer.trainable = False
                frozen_count += 1
        
        print(f"✓ Frozen {frozen_count} base layers, {adapter_count} adapter layers trainable")
        return adapter_count
    
    def prepare_for_training(self, learning_rate=1e-3):
        """Prepare model for adapter training."""
        self.freeze_base_model()
        
        # Compile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                "pattern_probs": "binary_crossentropy",
                "burnout_score": "mse"
            },
            metrics={
                "pattern_probs": "AUC",
                "burnout_score": "mae"
            }
        )
        print(f"✓ Model compiled with lr={learning_rate}")
    
    def train_user_adapter(
        self, 
        user_id, 
        audio_data, 
        text_data, 
        behavior_data,
        pattern_labels, 
        burnout_labels,
        epochs=10, 
        batch_size=8,
        validation_split=0.2
    ):
        """
        Train personalized adapter for a specific user.
        
        Args:
            user_id: User identifier
            audio_data: Audio embeddings (N, 256)
            text_data: Text embeddings (N, 771)
            behavior_data: Behavior features (N, 6)
            pattern_labels: Pattern labels (N, 4)
            burnout_labels: Burnout scores (N, 1)
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Training personalized adapter for user: {user_id}")
        print(f"{'='*60}")
        
        # Validate data shapes
        assert audio_data.shape[1] == 256, f"Audio dim mismatch: {audio_data.shape}"
        assert text_data.shape[1] == 771, f"Text dim mismatch: {text_data.shape}"
        assert behavior_data.shape[1] == 6, f"Behavior dim mismatch: {behavior_data.shape}"
        
        n_samples = len(audio_data)
        print(f"Training samples: {n_samples}")
        
        if n_samples < 10:
            print("⚠️ Warning: Very few samples (<10). Results may be unreliable.")
        
        # Prepare model
        self.prepare_for_training()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f"{USER_ADAPTERS_DIR}/adapter_{user_id}_best.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train
        history = self.model.fit(
            [audio_data, text_data, behavior_data],
            {
                "pattern_probs": pattern_labels,
                "burnout_score": burnout_labels
            },
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Training completed for user {user_id}")
        
        # Save adapter weights
        self.save_adapter_weights(user_id)
        
        return history
    
    def save_adapter_weights(self, user_id):
        """Save only adapter layer weights."""
        adapter_weights = {}
        
        for layer in self.model.layers:
            if "adapter" in layer.name.lower():
                for weight in layer.weights:
                    adapter_weights[weight.name] = weight.numpy()
        
        if adapter_weights:
            save_path = f"{USER_ADAPTERS_DIR}/adapter_{user_id}.npz"
            np.savez(save_path, **adapter_weights)
            print(f"✓ Saved adapter weights to {save_path}")
        else:
            print("⚠️ No adapter layers found to save")
    
    def load_adapter_weights(self, user_id):
        """Load user-specific adapter weights."""
        adapter_path = f"{USER_ADAPTERS_DIR}/adapter_{user_id}.npz"
        
        if not os.path.exists(adapter_path):
            print(f"⚠️ No adapter found for user {user_id}")
            return False
        
        # Load weights
        adapter_data = np.load(adapter_path)
        
        # Apply to model
        for layer in self.model.layers:
            if "adapter" in layer.name.lower():
                for i, weight in enumerate(layer.weights):
                    if weight.name in adapter_data:
                        weight.assign(adapter_data[weight.name])
        
        print(f"✓ Loaded personalized adapter for user {user_id}")
        return True
    
    def predict_with_user_adapter(self, user_id, audio_emb, text_vec, beh_vec):
        """
        Make prediction with user-specific adapter.
        
        Args:
            user_id: User identifier
            audio_emb: Audio embedding (256,)
            text_vec: Text vector (771,)
            beh_vec: Behavior vector (6,)
        
        Returns:
            Tuple of (patterns, burnout)
        """
        # Load user adapter if available
        self.load_adapter_weights(user_id)
        
        # Make prediction
        patterns, burnout = self.model.predict(
            [
                audio_emb.reshape(1, -1),
                text_vec.reshape(1, -1),
                beh_vec.reshape(1, -1)
            ],
            verbose=0
        )
        
        return patterns[0], burnout[0][0]

# Convenience functions

def train_user_adapter_from_csv(user_id, data_csv, epochs=10):
    """Train adapter from CSV file containing user data."""
    import pandas as pd
    
    df = pd.read_csv(data_csv)
    
    # Extract features (adjust column names as needed)
    audio_cols = [c for c in df.columns if c.startswith('audio_')]
    text_cols = [c for c in df.columns if c.startswith('text_')]
    beh_cols = [c for c in df.columns if c.startswith('beh_')]
    
    audio_data = df[audio_cols].values
    text_data = df[text_cols].values
    behavior_data = df[beh_cols].values
    
    # Extract labels
    pattern_cols = [c for c in df.columns if c.startswith('pattern_')]
    pattern_labels = df[pattern_cols].values if pattern_cols else np.zeros((len(df), 4))
    
    burnout_labels = df['burnout'].values if 'burnout' in df.columns else np.zeros(len(df))
    burnout_labels = burnout_labels.reshape(-1, 1)
    
    # Train
    adapter = PersonalizedAdapter()
    history = adapter.train_user_adapter(
        user_id,
        audio_data,
        text_data,
        behavior_data,
        pattern_labels,
        burnout_labels,
        epochs=epochs
    )
    
    return history

if __name__ == "__main__":
    print("BAYMAX Personalized Adapter Training")
    print("=" * 60)
    
    # Example: Create synthetic user data for demonstration
    user_id = "demo_user_123"
    n_samples = 20
    
    # Generate synthetic data
    audio_data = np.random.randn(n_samples, 256).astype(np.float32)
    text_data = np.random.randn(n_samples, 771).astype(np.float32)
    behavior_data = np.random.rand(n_samples, 6).astype(np.float32)
    
    pattern_labels = np.random.rand(n_samples, 4).astype(np.float32)
    burnout_labels = np.random.rand(n_samples, 1).astype(np.float32)
    
    print(f"\nExample: Training adapter for {user_id} with {n_samples} samples")
    print("Note: Using synthetic data for demonstration")
    
    # Initialize adapter trainer
    adapter = PersonalizedAdapter()
    
    # Train (commented out to avoid accidental execution)
    # history = adapter.train_user_adapter(
    #     user_id,
    #     audio_data,
    #     text_data,
    #     behavior_data,
    #     pattern_labels,
    #     burnout_labels,
    #     epochs=5,
    #     batch_size=8
    # )
    
    print("\n✓ Personalized adapter system ready")
