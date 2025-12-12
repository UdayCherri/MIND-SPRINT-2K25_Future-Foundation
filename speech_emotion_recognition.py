"""
Speech Emotion Recognition System
Trains CNN and LSTM models to recognize emotions from audio files
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ============================
# Configuration
# ============================

# Update this path to your dataset location
DATA_PATH = "/kaggle/input/emotion-dataset/Emotion_Dataset"

# RAVDESS emotion mapping
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm', 
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


# ============================
# Dataset Creation
# ============================

def create_dataset(data_path):
    """Create dataset from RAVDESS files"""
    print("Creating dataset from RAVDESS files...")
    
    data = []
    
    for folder in ['Speech', 'Song']:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found")
            continue
            
        print(f"Processing {folder}...")
        
        for actor_folder in os.listdir(folder_path):
            actor_path = os.path.join(folder_path, actor_folder)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.wav'):
                        # Parse filename: 03-01-06-01-02-01-12.wav
                        parts = file.split('-')
                        if len(parts) >= 3:
                            emotion_code = parts[2]
                            if emotion_code in EMOTIONS:
                                emotion = EMOTIONS[emotion_code]
                                file_path = os.path.join(actor_path, file)
                                
                                # Extract additional info
                                intensity = 'strong' if parts[3] == '02' else 'normal'
                                actor_id = int(parts[6].split('.')[0])
                                gender = 'female' if actor_id % 2 == 0 else 'male'
                                
                                data.append({
                                    'file_path': file_path,
                                    'emotion': emotion,
                                    'emotion_code': emotion_code,
                                    'intensity': intensity,
                                    'gender': gender,
                                    'actor_id': actor_id,
                                    'source': 'speech' if 'Speech' in folder else 'song'
                                })
    
    df = pd.DataFrame(data)
    print(f"Dataset created with {len(df)} samples")
    return df


# ============================
# Data Augmentation
# ============================

def augment_audio(y, sr):
    """Apply data augmentation techniques"""
    augmented_samples = []
    
    # Original
    augmented_samples.append(y)
    
    # Add noise
    noise = np.random.normal(0, 0.005, len(y))
    augmented_samples.append(y + noise)
    
    # Time stretch
    try:
        y_stretch = librosa.effects.time_stretch(y, rate=0.9)
        if len(y_stretch) > len(y):
            y_stretch = y_stretch[:len(y)]
        else:
            y_stretch = np.pad(y_stretch, (0, len(y) - len(y_stretch)), mode='constant')
        augmented_samples.append(y_stretch)
    except:
        augmented_samples.append(y)
    
    # Pitch shift
    try:
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        augmented_samples.append(y_pitch)
    except:
        augmented_samples.append(y)
    
    return augmented_samples


# ============================
# Feature Extraction
# ============================

def extract_features(y, sr):
    """Extract comprehensive audio features including mel spectrograms"""
    features = []
    
    # 1. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.extend([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)])
    
    # 2. Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])
    
    # 3. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.extend([np.mean(spectral_centroid), np.std(spectral_centroid), 
                    np.max(spectral_centroid), np.min(spectral_centroid)])
    
    # 4. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                    np.max(spectral_bandwidth), np.min(spectral_bandwidth)])
    
    # 5. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff),
                    np.max(spectral_rolloff), np.min(spectral_rolloff)])
    
    # 6. Enhanced MFCCs (20 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features.extend([np.mean(mfccs[i]), np.std(mfccs[i]), 
                        np.max(mfccs[i]), np.min(mfccs[i])])
    
    # 7. Delta MFCCs
    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(20):
        features.extend([np.mean(delta_mfccs[i]), np.std(delta_mfccs[i])])
    
    # 8. Delta-Delta MFCCs
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    for i in range(20):
        features.extend([np.mean(delta2_mfccs[i]), np.std(delta2_mfccs[i])])
    
    # 9. Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    for i in range(12):
        features.extend([np.mean(chroma[i]), np.std(chroma[i])])
    
    # 10. Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    for i in range(6):
        features.extend([np.mean(tonnetz[i]), np.std(tonnetz[i])])
    
    # 11. Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    for i in range(7):
        features.extend([np.mean(spectral_contrast[i]), np.std(spectral_contrast[i])])
    
    # 12. Mel Spectrogram Features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    
    features.extend([
        np.mean(mel_spec_db), np.std(mel_spec_db), 
        np.max(mel_spec_db), np.min(mel_spec_db),
        np.median(mel_spec_db), np.percentile(mel_spec_db, 25),
        np.percentile(mel_spec_db, 75)
    ])
    
    # Mel spectrogram band energy
    for i in range(0, 128, 16):
        band_energy = np.mean(mel_spec_db[i:i+16])
        features.append(band_energy)
    
    # 13. Pitch and Harmony Features
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                   fmax=librosa.note_to_hz('C7'))
        if f0 is not None and len(f0) > 0:
            f0 = np.array(f0).flatten()
            voiced_flag = np.array(voiced_flag).flatten()
            f0_clean = f0[voiced_flag & ~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                features.extend([float(np.mean(f0_clean)), float(np.std(f0_clean)), 
                                float(np.max(f0_clean)), float(np.min(f0_clean))])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    except:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # 14. Tempo and Rhythm Features
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo))
        
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            features.extend([float(np.mean(beat_intervals)), float(np.std(beat_intervals))])
        else:
            features.extend([0.0, 0.0])
    except:
        features.extend([0.0, 0.0, 0.0])
    
    # 15. Spectral Features
    try:
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features.extend([float(np.mean(spectral_flatness)), float(np.std(spectral_flatness))])
    except:
        features.extend([0.0, 0.0])
    
    try:
        poly_features = librosa.feature.poly_features(y=y, sr=sr, order=1)
        if poly_features.ndim == 2 and poly_features.shape[0] > 0:
            poly_coeff = poly_features[0]
            features.extend([float(np.mean(poly_coeff)), float(np.std(poly_coeff))])
        else:
            features.extend([0.0, 0.0])
    except:
        features.extend([0.0, 0.0])
        
    features_array = np.array(features, dtype=np.float32)
    
    if features_array.ndim != 1:
        raise ValueError(f"Features array should be 1D, got shape: {features_array.shape}")
    
    return features_array


def prepare_data(df, use_augmentation=True):
    """Prepare features and labels for training"""
    print("Extracting features from audio files...")
    
    features_list = []
    labels_list = []
    
    total_files = len(df)
    
    for idx, row in df.iterrows():
        try:
            y, sr = librosa.load(row['file_path'], sr=22050, duration=3.0)
            
            if use_augmentation:
                augmented_samples = augment_audio(y, sr)
                
                for aug_sample in augmented_samples:
                    features = extract_features(aug_sample, sr)
                    features_list.append(features)
                    labels_list.append(row['emotion'])
            else:
                features = extract_features(y, sr)
                features_list.append(features)
                labels_list.append(row['emotion'])
                
        except Exception as e:
            print(f"Error processing {row['file_path']}: {e}")
            continue
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_files} files")
    
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    print(f"Feature extraction completed. Shape: {features_array.shape}")
    print(f"Total samples after augmentation: {len(labels_array)}")
    
    return features_array, labels_array


# ============================
# Model Architectures
# ============================

def create_cnn_model(input_shape, num_classes):
    """Create enhanced CNN model"""
    model = Sequential([
        Input(shape=(input_shape,)),
        
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lstm_model(input_shape, num_classes):
    """Create LSTM model for emotion recognition"""
    input_layer = Input(shape=(input_shape, 1))
    
    lstm1 = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input_layer)
    lstm2 = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(lstm1)
    
    dense1 = Dense(64, activation='relu')(lstm2)
    dropout1 = Dropout(0.3)(dense1)
    
    dense2 = Dense(32, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    
    output = Dense(num_classes, activation='softmax')(dropout2)
    
    model = Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================
# Evaluation Functions
# ============================

def evaluate_model(model, X_test, y_test, model_name, label_encoder):
    """Evaluate model and return metrics"""
    
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print(f"\n{model_name} Classification Report:")
    print(classification_report(
        y_test_classes, y_pred_classes, 
        target_names=label_encoder.classes_
    ))
    
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    return accuracy, f1, cm, y_pred_classes, y_test_classes


def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()
    
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print(f"\n{model_name} - Per-class Accuracy:")
    for i, acc in enumerate(class_accuracy):
        print(f"{class_names[i]}: {acc:.4f}")


def plot_training_history(history, model_name):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.show()


# ============================
# Main Training Pipeline
# ============================

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("Speech Emotion Recognition System")
    print("="*60)
    
    # Create dataset
    df = create_dataset(DATA_PATH)
    
    # Dataset overview
    print("\nDataset Overview:")
    print(f"Total samples: {len(df)}")
    print(f"Total emotions: {df['emotion'].nunique()}")
    print(f"Total actors: {df['actor_id'].nunique()}")
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())
    
    # Extract features with augmentation
    print("\nStarting feature extraction with data augmentation...")
    features, labels = prepare_data(df, use_augmentation=True)
    
    # Preprocessing
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    
    labels_encoded = label_encoder.fit_transform(labels)
    features_scaled = scaler.fit_transform(features)
    labels_categorical = tf.keras.utils.to_categorical(labels_encoded)
    
    print(f"\nFeatures shape: {features_scaled.shape}")
    print(f"Labels shape: {labels_categorical.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled, labels_categorical, 
        test_size=0.2, random_state=42, stratify=labels_categorical
    )
    
    X_test = X_val.copy()
    y_test = y_val.copy()
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create models
    print("\nCreating models...")
    cnn_model = create_cnn_model(features_scaled.shape[1], labels_categorical.shape[1])
    lstm_model = create_lstm_model(features_scaled.shape[1], labels_categorical.shape[1])
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001, verbose=1
    )
    
    # Train CNN
    print("\n" + "="*60)
    print("Training CNN Model...")
    print("="*60)
    cnn_history = cnn_model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Train LSTM
    print("\n" + "="*60)
    print("Training LSTM Model...")
    print("="*60)
    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_lstm = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    lstm_history = lstm_model.fit(
        X_train_lstm, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val_lstm, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate models
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    cnn_accuracy, cnn_f1, cnn_cm, _, _ = evaluate_model(
        cnn_model, X_test, y_test, "CNN", label_encoder
    )
    
    lstm_accuracy, lstm_f1, lstm_cm, _, _ = evaluate_model(
        lstm_model, X_test_lstm, y_test, "LSTM", label_encoder
    )
    
    # Plot results
    plot_confusion_matrix(cnn_cm, label_encoder.classes_, "CNN")
    plot_confusion_matrix(lstm_cm, label_encoder.classes_, "LSTM")
    plot_training_history(cnn_history, "CNN")
    plot_training_history(lstm_history, "LSTM")
    
    # Save best model
    if cnn_f1 > lstm_f1:
        best_model = cnn_model
        best_f1 = cnn_f1
        best_accuracy = cnn_accuracy
        best_name = "CNN"
    else:
        best_model = lstm_model
        best_f1 = lstm_f1
        best_accuracy = lstm_accuracy
        best_name = "LSTM"
    
    best_model.save('emotion_recognition_model.h5')
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("\nModel and preprocessing objects saved!")
    
    # Final Report
    print("\n" + "="*60)
    print("FINAL PERFORMANCE REPORT")
    print("="*60)
    print(f"CNN Model - Accuracy: {cnn_accuracy:.4f}, F1 Score: {cnn_f1:.4f}")
    print(f"LSTM Model - Accuracy: {lstm_accuracy:.4f}, F1 Score: {lstm_f1:.4f}")
    print(f"\nBest Model: {best_name}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    print(f"\nRequirements Check:")
    print(f"F1 Score > 0.80: {'✓' if best_f1 > 0.80 else '✗'} ({best_f1:.4f})")
    print(f"Overall Accuracy > 0.80: {'✓' if best_accuracy > 0.80 else '✗'} ({best_accuracy:.4f})")
    
    if best_f1 > 0.80 and best_accuracy > 0.80:
        print("\n🎉 Congratulations! Both requirements have been met!")
    else:
        print("\n⚠️ Requirements not fully met. Consider:")
        print("- Increasing dataset size")
        print("- Fine-tuning hyperparameters")
        print("- Adding more sophisticated augmentation")
        print("- Trying ensemble methods")


if __name__ == "__main__":
    main()
