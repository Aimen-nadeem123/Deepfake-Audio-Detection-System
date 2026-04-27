import os
import librosa
import soundfile as sf
import numpy as np
import random
import shutil

# -------------------------------
# Paths
# -------------------------------
NEW_FAKE_PATH = "new_fakes"         # Folder with new fake audio files
TRAIN_FAKE_PATH = "data/training/fake"
VAL_FAKE_PATH = "data/validation/fake"

# -------------------------------
# Target counts
# -------------------------------
TARGET_TRAIN_FAKE = 7338
TARGET_VAL_FAKE = 1417

# -------------------------------
# Audio processing parameters
# -------------------------------
TARGET_DURATION = 2.0   # seconds
SR = 22050

# -------------------------------
# Augmentation functions
# -------------------------------
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)

def time_stretch(audio, rate=1.1):
    return librosa.effects.time_stretch(audio, rate=rate)

# -------------------------------
# Process a single file
# -------------------------------
def process_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SR)
    # Trim/pad to TARGET_DURATION
    desired_len = int(TARGET_DURATION * sr)
    if len(audio) > desired_len:
        audio = audio[:desired_len]
    else:
        audio = np.pad(audio, (0, max(0, desired_len - len(audio))))
    # Randomly apply one augmentation
    aug_choice = random.choice(["none", "noise", "pitch", "stretch"])
    if aug_choice == "noise":
        audio = add_noise(audio)
    elif aug_choice == "pitch":
        audio = pitch_shift(audio, sr)
    elif aug_choice == "stretch":
        audio = time_stretch(audio, rate=random.uniform(0.9,1.1))
        # Pad again if needed
        audio = np.pad(audio, (0, max(0, desired_len - len(audio))))
        audio = audio[:desired_len]
    return audio

# -------------------------------
# Copy files to target folder
# -------------------------------
def fill_target_folder(src_folder, target_folder, target_count):
    existing_files = os.listdir(target_folder)
    current_count = len(existing_files)
    files_needed = target_count - current_count
    print(f"Current files: {current_count}, Target: {target_count}, Need to add: {files_needed}")

    new_files = os.listdir(src_folder)
    added = 0
    while added < files_needed:
        for f in new_files:
            if added >= files_needed:
                break
            src_path = os.path.join(src_folder, f)
            audio = process_audio(src_path)
            # Save with new unique name
            new_filename = f"fake_{current_count + added + 1}.wav"
            target_path = os.path.join(target_folder, new_filename)
            sf.write(target_path, audio, SR)
            added += 1
    print(f"Added {added} files to {target_folder}")

# -------------------------------
# Run
# -------------------------------
fill_target_folder(NEW_FAKE_PATH, TRAIN_FAKE_PATH, TARGET_TRAIN_FAKE)
fill_target_folder(NEW_FAKE_PATH, VAL_FAKE_PATH, TARGET_VAL_FAKE)

print("✅ All new fake audios processed and added.")