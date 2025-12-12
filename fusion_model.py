# fusion_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import joblib
import os

# CONFIG — adjust dims to match your system
AUDIO_DIM = 256
TEXT_DIM = 771
BEH_DIM = 6
FUSION_INPUT_DIM = AUDIO_DIM + TEXT_DIM + BEH_DIM
ADAPTER_BOTTLENECK = 64
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_fusion_model(audio_dim=AUDIO_DIM, text_dim=TEXT_DIM, beh_dim=BEH_DIM, adapter=True):
    audio_in = layers.Input(shape=(audio_dim,), name="audio_emb")
    text_in  = layers.Input(shape=(text_dim,), name="text_emb")
    beh_in   = layers.Input(shape=(beh_dim,), name="beh_emb")
    
    fused = layers.Concatenate(name="concat_embeddings")([audio_in, text_in, beh_in])
    x = layers.Dense(1024, activation="relu", name="dense_1")(fused)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.2, name="drop_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.2, name="drop_2")(x)
    
    if adapter:
        a = layers.Dense(ADAPTER_BOTTLENECK, activation="relu", name="adapter_down")(x)
        a = layers.Dense(512, activation=None, name="adapter_up")(a)
        # scaled residual (use a simple Dense layer with fixed scaling to avoid Lambda deserialization issues)
        # Multiply by 0.1 using a Dense layer with weight constraint
        a_scaled = layers.Dense(512, activation=None, use_bias=False, 
                                trainable=True, name="adapter_scale",
                                kernel_initializer=tf.keras.initializers.Constant(0.1))(a)
        x = layers.Add(name="adapter_res")([x, a_scaled])
    
    # Pattern head (multi-label)
    pattern_out = layers.Dense(4, activation="sigmoid", name="pattern_probs")(x)
    # Regression head (burnout)
    burnout_out = layers.Dense(1, activation="sigmoid", name="burnout_score")(x)
    
    model = models.Model(inputs=[audio_in, text_in, beh_in], outputs=[pattern_out, burnout_out], name="fusion_model")
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-5),
                  loss={"pattern_probs":"binary_crossentropy", "burnout_score":"mse"},
                  metrics={"pattern_probs":"AUC"})
    return model

# Toy pipeline: saves and loads scaler
from sklearn.preprocessing import StandardScaler

def save_scaler(scaler, path="./models/scaler.joblib"):
    joblib.dump(scaler, path)

def load_scaler(path="./models/scaler.joblib"):
    return joblib.load(path)

if __name__ == "__main__":
    m = build_fusion_model()
    m.summary()
    # save model architecture (not weights) for reference
    m.save(os.path.join(MODEL_DIR, "fusion_model_arch.h5"))
    print("Fusion model architecture saved.")
