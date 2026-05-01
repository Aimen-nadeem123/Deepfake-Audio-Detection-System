import os
import numpy as np
import librosa
import random

from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# SETTINGS
# -------------------------------
BASE_PATH = "data"
MAX_LEN = 2 * 22050  # 2 sec

# -------------------------------
# AUGMENTATION
# -------------------------------
def augment_audio(audio):
    if random.random() < 0.3:
        noise = np.random.randn(len(audio))
        audio = audio + 0.005 * noise

    if random.random() < 0.3:
        shift = np.random.randint(len(audio))
        audio = np.roll(audio, shift)

    return audio

# -------------------------------
# FEATURE EXTRACTION (SEQUENCE)
# -------------------------------
def extract_features(file_path, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        audio, _ = librosa.effects.trim(audio)

        # Fix length
        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]
        else:
            audio = np.pad(audio, (0, MAX_LEN - len(audio)))

        if augment:
            audio = augment_audio(audio)

        audio = librosa.util.normalize(audio)

        # 🔥 KEEP TIME DIMENSION
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = mfcc.T  # (time_steps, 40)

        return mfcc

    except:
        return None

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(folder, augment=False):
    X, y = [], []

    for label in ["real", "fake"]:
        path = os.path.join(BASE_PATH, folder, label)

        for file in os.listdir(path):
            file_path = os.path.join(path, file)

            feat = extract_features(file_path)
            if feat is not None:
                X.append(feat)
                y.append(1 if label == "real" else 0)

                # augmentation
                if augment:
                    aug_feat = extract_features(file_path, augment=True)
                    if aug_feat is not None:
                        X.append(aug_feat)
                        y.append(1 if label == "real" else 0)

    return X, np.array(y)

# -------------------------------
# LOAD
# -------------------------------
X_train, y_train = load_data("training", augment=True)
X_val, y_val = load_data("validation")
X_test, y_test = load_data("testing")

# -------------------------------
# PAD SEQUENCES
# -------------------------------
X_train = pad_sequences(X_train, padding='post', dtype='float32')
X_val = pad_sequences(X_val, padding='post', dtype='float32')
X_test = pad_sequences(X_test, padding='post', dtype='float32')

print("Train shape:", X_train.shape)

# -------------------------------
# MODEL (CNN + BiLSTM + Attention)
# -------------------------------
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# CNN
x = Conv1D(64, 3, activation='relu')(input_layer)
x = MaxPooling1D(2)(x)

x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)

# BiLSTM
x = Bidirectional(LSTM(64, return_sequences=True))(x)

# Attention
attn = Attention()([x, x])
x = GlobalAveragePooling1D()(attn)

# Dense
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# TRAIN
# -------------------------------
early = EarlyStopping(patience=5, restore_best_weights=True)
lr = ReduceLROnPlateau(patience=3, factor=0.5)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[early, lr],
    class_weight={0: 2.0, 1: 1.0}  # 🔥 handle imbalance
)

# -------------------------------
# TEST
# -------------------------------
probs = model.predict(X_test)
pred = (probs > 0.4).astype(int)  # 🔥 tuned threshold

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# -------------------------------
# SAVE
# -------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/audio_model_final.h5")

print("✅ MODEL SAVED")
print("✅ TRAINING COMPLETE")