import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import pandas as pd
from datetime import datetime
import os
from tensorflow.keras.models import load_model

# 🔥 RECORDING LIBRARY
from audiorecorder import audiorecorder

# -------------------------------
# CONFIG + STYLE
# -------------------------------
st.set_page_config(page_title="Deepfake Audio Detection", layout="centered")

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: white;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
.stFileUploader {
    border: 2px dashed #444;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model("models/audio_model_final.keras")
scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(delta.T, axis=0),
        np.mean(delta2.T, axis=0)
    ])

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    audio, _ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)

    chunk_size = 2 * sr
    predictions = []

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]

        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        feat = extract_features(chunk, sr)
        feat = scaler.transform([feat])
        feat = np.expand_dims(feat, axis=2)

        pred = model.predict(feat, verbose=0)[0][0]
        predictions.append(float(pred))

    # 🔥 SMART DECISION LOGIC
    max_pred = max(predictions)
    min_pred = min(predictions)

    if max_pred > 0.9:
        label = "🎙 REAL Audio"
        confidence = max_pred

    elif min_pred < 0.1:
        label = "🤖 FAKE Audio"
        confidence = 1 - min_pred

    else:
        real_count = sum(p > 0.5 for p in predictions)
        fake_count = sum(p <= 0.5 for p in predictions)

        if real_count > fake_count:
            label = "🎙 REAL Audio"
            confidence = real_count / len(predictions)
        else:
            label = "🤖 FAKE Audio"
            confidence = fake_count / len(predictions)

    return label, confidence, predictions

# -------------------------------
# SAVE HISTORY (TERMINAL + CSV ONLY)
# -------------------------------
def save_history(source, filename, label, confidence):
    os.makedirs("data", exist_ok=True)

    data = {
        "source": source,
        "file": filename,
        "result": label,
        "confidence (%)": round(confidence * 100, 2),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 🔥 PRINT IN TERMINAL
    print("\n===== NEW PREDICTION =====")
    for key, value in data.items():
        print(f"{key}: {value}")
    print("==========================\n")

    # 🔥 SAVE TO CSV
    df = pd.DataFrame([data])
    file_path = "data/history.csv"

    if os.path.exists(file_path):
        old = pd.read_csv(file_path)
        df = pd.concat([old, df])

    df.to_csv(file_path, index=False)

# -------------------------------
# UI
# -------------------------------
st.title("🎙 Deepfake Audio Detection System")
st.markdown("---")

# ===============================
# 🔹 UPLOAD AUDIO
# ===============================
st.subheader("📂 Upload Audio File")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    if st.button("🔍 Predict Uploaded Audio"):
        with st.spinner("Analyzing..."):
            label, confidence, preds = predict_audio(file_path)

        st.success(label)
        st.write(f"Confidence: {confidence*100:.2f}%")
        st.write("Chunks:", [round(p, 3) for p in preds])

        save_history("upload", uploaded_file.name, label, confidence)

# ===============================
# 🔹 RECORD AUDIO
# ===============================
st.markdown("---")
st.subheader("🎤 Record Real-Time Audio")

audio = audiorecorder("Click to record", "Recording...")

if len(audio) > 0:
    audio_bytes = audio.export().read()
    st.audio(audio_bytes, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        record_path = tmp.name

    if st.button("🎯 Predict Recorded Audio"):
        with st.spinner("Analyzing recording..."):
            label, confidence, preds = predict_audio(record_path)

        st.success(label)
        st.write(f"Confidence: {confidence*100:.2f}%")
        st.write("Chunks:", [round(p, 3) for p in preds])

        save_history("recorded", "recorded_audio.wav", label, confidence)
