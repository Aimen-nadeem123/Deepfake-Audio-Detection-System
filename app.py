import streamlit as st
import numpy as np
import librosa
import tempfile
import pandas as pd
from datetime import datetime
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from audiorecorder import audiorecorder

# -------------------------------
# CONFIG + THEME STYLE
# -------------------------------
st.set_page_config(page_title="Deepfake Audio Detection", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at center, #0a192f 0%, #020c1b 100%);
        color: #64ffda;
    }
    .header-container {
        display: flex; flex-direction: row; align-items: center; justify-content: center; 
        padding: 2rem; background: rgba(10, 25, 47, 0.8); border: 1px solid #64ffda33;
        border-radius: 20px; margin-bottom: 2rem; gap: 30px; 
    }
    .site-title {
        font-family: 'Courier New', Courier, monospace; font-size: 3rem; 
        font-weight: bold; color: #64ffda; text-shadow: 0 0 15px #64ffda77; margin: 0;
    }
    .result-panel {
        background: rgba(100, 255, 218, 0.05); border: 1px solid #64ffda33;
        border-radius: 15px; padding: 20px; margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<div class="header-container">
    <div style="width: 100px;">
        <svg viewBox="0 0 100 120">
            <rect x="10" y="40" width="15" height="40" fill="#64ffda"><animate attributeName="height" values="40;80;40" dur="0.6s" repeatCount="indefinite" /><animate attributeName="y" values="40;20;40" dur="0.6s" repeatCount="indefinite" /></rect>
            <rect x="40" y="20" width="15" height="80" fill="#64ffda"><animate attributeName="height" values="80;120;80" dur="0.8s" repeatCount="indefinite" /><animate attributeName="y" values="20;0;20" dur="0.8s" repeatCount="indefinite" /></rect>
            <rect x="70" y="30" width="15" height="60" fill="#64ffda"><animate attributeName="height" values="60;100;60" dur="0.5s" repeatCount="indefinite" /><animate attributeName="y" values="30;10;30" dur="0.5s" repeatCount="indefinite" /></rect>
        </svg>
    </div>
    <h1 class="site-title">DEEPFAKE AUDIO DETECTION</h1>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# ORIGINAL CORE LOGIC
# -------------------------------
@st.cache_resource
def get_model():
    return load_model("models/audio_model_final.h5")

model = get_model()

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return mfcc.T  

def predict_audio(file_path):
    # --- Exact Original Processing ---
    audio, sr = librosa.load(file_path, sr=22050)
    audio, _ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)

    chunk_size = 2 * sr
    features = []

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        feat = extract_features(chunk, sr)
        features.append(feat)

    features = pad_sequences(features, padding='post', dtype='float32')
    probs = model.predict(features, verbose=0).flatten()

    # --- Exact Original Decision Logic ---
    max_pred = np.max(probs)
    min_pred = np.min(probs)

    if max_pred > 0.9:
        label = "🎙 REAL Audio"
        confidence = max_pred
    elif min_pred < 0.1:
        label = "🤖 FAKE Audio"
        confidence = 1 - min_pred
    else:
        real_count = np.sum(probs > 0.4)
        fake_count = np.sum(probs <= 0.4)
        if real_count > fake_count:
            label = "🎙 REAL Audio"
            confidence = real_count / len(probs)
        else:
            label = "🤖 FAKE Audio"
            confidence = fake_count / len(probs)

    color = "#64ffda" if "REAL" in label else "#ff4b4b"
    return label, confidence, probs, color

# -------------------------------
# MAIN LAYOUT
# -------------------------------
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.markdown("### 📂 ANALYSIS PORT")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])
    st.markdown("---")
    st.markdown("### 🎤 LIVE FEED")
    recorded_audio = audiorecorder("START RECORDING", "STOP RECORDING")
    if recorded_audio:
        st.info(f"✅ Recording Captured: {len(recorded_audio)/1000:.1f}s")

with right_col:
    st.markdown("### 🔍 RESULTS & PREVIEW")
    
    # Process Upload
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("RUN SCAN ON UPLOAD"):
            with st.spinner("Analyzing..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    label, confidence, preds, color = predict_audio(tmp.name)
                st.markdown(f"<div class='result-panel'><h1 style='color:{color};'>{label}</h1><h3>Confidence: {confidence*100:.2f}%</h3><p>Chunks: {[round(p, 3) for p in preds]}</p></div>", unsafe_allow_html=True)

    # Process Recording
    if len(recorded_audio) > 0:
        if uploaded_file: st.markdown("---")
        audio_bytes = recorded_audio.export().read()
        st.audio(audio_bytes)
        if st.button("VERIFY RECORDED SIGNAL"):
            with st.spinner("Analyzing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    label, confidence, preds, color = predict_audio(tmp.name)
                st.markdown(f"<div class='result-panel'><h1 style='color:{color};'>{label}</h1><h3>Confidence: {confidence*100:.2f}%</h3><p>Chunks: {[round(p, 3) for p in preds]}</p></div>", unsafe_allow_html=True)
    
    if not uploaded_file and len(recorded_audio) == 0:
        st.write("Waiting for input...")

st.markdown("<br><p style='text-align: center; opacity: 0.3;'>SECURED TERMINAL ACCESS</p>", unsafe_allow_html=True)
