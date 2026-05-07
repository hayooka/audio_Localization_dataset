# SoundSense — Wearable Assistive System for Hearing-Impaired Users

This repository contains the full pipeline for the **sound source localization** component of a wearable assistive system designed for deaf and hearing-impaired users. It covers everything from raw audio recording to a live web interface running on a Raspberry Pi.

The system combines three modules:
- **Sound Source Localization** — GRU model, 24 directions × 360°, running in real time at 16 ms latency
- **Sound Recognition** — YAMNet classifies 500+ environmental sound categories
- **Speech-to-Text** — Google Cloud STT with Arabic (Kuwaiti dialect) and English support

> **Paper:** *A Non-Prosthetic Assistive System for Persons with Hearing Losses: Design and Experimental Investigation*  
> F. AlHayek, R. Alsubaiei, M. Alsahhaf, G. Alajmi, A. Almutairi, K. Youssef, S. Said, S. Alkork — American University of the Middle East

---

## What This Repository Contains

```
audio_Localization_dataset/
│
├── 0_Dataset/
│   ├── (24 angles)dataset/        ← Raw WAV recordings (R1: 5 min, R2: 3 min per angle)
│   └── features/
│       ├── train_DATA16.csv       ← Pre-extracted features, 16 ms frames (R1)
│       ├── test_DATA16.csv        ← Pre-extracted features, 16 ms frames (R2)
│       ├── train_DATA50.csv       ← Pre-extracted features, 50 ms frames (R1)
│       └── test_DATA50.csv        ← Pre-extracted features, 50 ms frames (R2)
│
├── 1_data_collection/
│   ├── record_4mic_ReSpeaker.py   ← Record from ReSpeaker on a PC/laptop
│   └── Raspberry_recor_Respeaker.py ← Record from ReSpeaker on a Raspberry Pi
│
├── 2_data_to_features/
│   └── features_to_csv.py         ← Convert WAV recordings → 895-dim feature CSVs
│
├── 3_training/
│   ├── train_ALL_features.py      ← CNN training (any frame size — used for Table I)
│   ├── train_ALL_features_m2.py   ← CNN training, 16 ms, seq=1 (Table II baseline)
│   └── sequence_audio_train.py    ← GRU training, 16 ms, seq=32 (best model)
│
├── 4_results/                     ← Auto-created when training runs
│   ├── models/                    ← Saved .pt model checkpoints
│   └── plots/                     ← All output figures
│
├── 5_inference/
│   ├── __init__.py                ← predict() function for offline use
│   └── _features.py               ← Feature extraction (shared with UI)
│
├── 6_user_interface/
│   ├── soundsense_server.py       ← FastAPI server (localization + YAMNet + STT)
│   ├── soundsense_v5.html         ← Mobile web UI (compass, captions, haptic alerts)
│   └── _features.py               ← Feature extraction copy for deployment
│
└── notebooks/
    ├── 01_localization_experiment.ipynb  ← All localization figures (Tables I–III)
    └── 02_system_results.ipynb           ← Sound recognition + STT figures (Tables IV–VII)
```

---

## Requirements

Install all dependencies in one command:

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib jupyter \
            fastapi uvicorn pyaudio tensorflow-hub google-cloud-speech
```

> Tested with Python 3.10, PyTorch 2.x. GPU is optional — training works on CPU.

---

## Step-by-Step: Reproducing the Full Pipeline

### Step 1 — Record Audio Data

Place the ReSpeaker XVF3800 at the centre of the room. Position a loudspeaker **1.75 m** away.  
Record at each of the **24 angles** (0°, 15°, 30°, …, 345°):

- **R1 (training):** 5 minutes per angle, same speech content across all angles
- **R2 (testing):** 3 minutes per angle, different speech content per angle

**On a PC/laptop:**
```bash
python 1_data_collection/record_4mic_ReSpeaker.py
```

**On a Raspberry Pi:**
```bash
python 1_data_collection/Raspberry_recor_Respeaker.py
```

Save recordings in this folder structure:
```
0_Dataset/(24 angles)dataset/<angle>/speakerM/5min/mic_right.wav
                                                    mic_front.wav
                                                    mic_left.wav
                                                    mic_back.wav
                              <angle>/speakerM/3min/mic_right.wav
                                                    ...
```

---

### Step 2 — Extract Features

Convert the raw WAV files into feature CSVs. This produces one CSV for training (R1) and one for testing (R2):

```bash
python 2_data_to_features/features_to_csv.py
```

**Output:** `0_Dataset/features/train_DATA16.csv` and `test_DATA16.csv`

Each row is one 16 ms audio frame described by **895 features**:

| Feature Group | Dimensions | Description |
|---|---|---|
| IPD scalar | 3 | Inter-channel phase difference (3 mic pairs) |
| IPD-Mel | 120 | IPD weighted per mel band (3 pairs × 40 bands) |
| GCC-PHAT TDOA | 6 | Time delay of arrival — 6 mic pairs |
| GCC Strength | 6 | Correlation peak sharpness — 6 mic pairs |
| GCC vectors | 600 | Full GCC curve — 6 pairs × 100 lags |
| Log-Mel | 160 | Log-mel spectrogram — 4 mics × 40 bands |
| **Total** | **895** | |

> If the CSVs already exist in `0_Dataset/features/`, skip this step.

---

### Step 3 — Train the Localization Model

#### Option A: GRU — Best model (recommended)

The GRU processes 32 consecutive frames (512 ms context window) and achieves **99.22% accuracy**, **MAE 0.4°**.

```bash
python 3_training/sequence_audio_train.py \
    --seq-len 32 \
    --stride 16 \
    --epochs 50
```

Saved to: `4_results/models/GRU/audioLOC_sequence_<run_id>.pt`

#### Option B: CNN — Feature comparison experiment (Table I in paper)

The CNN processes a single frame. Run once per feature subset to reproduce Table I:

```bash
python 3_training/train_ALL_features.py
```

#### Option C: CNN 16 ms baseline (Table II in paper)

```bash
python 3_training/train_ALL_features_m2.py
```

---

### Step 4 — View Results in Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Figures | Contents |
|---|---|---|
| `01_localization_experiment.ipynb` | Fig. 1–7 | Feature breakdown, Table I–III, model progression, confusion matrix |
| `02_system_results.ipynb` | Fig. 1–6 | Sound recognition, STT tables, WER analysis, radar chart |

Set `RETRAIN = True` at the top of notebook 01 to re-run training from scratch.  
Set `RETRAIN = False` (default) to display the paper's reported results instantly.

---

### Step 5 — Run the Real-Time System

The full system (localization + sound recognition + STT) runs as a web server on a Raspberry Pi 5.

**1. Copy your trained GRU model to the UI folder:**
```bash
cp 4_results/models/GRU/audioLOC_sequence_<run_id>.pt 6_user_interface/audioLOC_GRU.pt
```

**2. Set your Google Cloud credentials** (needed for STT):  
Edit line 41 of `6_user_interface/soundsense_server.py` with the path to your credentials JSON file.

**3. Start the server:**
```bash
cd 6_user_interface
python soundsense_server.py
```

**4. Open the web interface** on any phone or laptop on the same Wi-Fi network:
```
http://<raspberry-pi-ip>:8000
```

The interface shows:
- A live **compass arrow** pointing to the sound source direction
- **Sound labels** with confidence scores (YAMNet)
- **Live speech transcription** in Arabic or English
- **Haptic vibration alerts** for critical sounds (sirens, alarms)

---

## Key Results

### Sound Source Localization

| Model | Frame | Seq | Accuracy | MAE |
|---|---|---|---|---|
| CNN — All features | 50 ms | 1 | 91.07% | 6.6° |
| CNN — All features | 16 ms | 1 | 76.34% | 12.7° |
| CNN — All features | 16 ms | 2 | 83.04% | 12.4° |
| **GRU — All features** | **16 ms** | **32** | **99.22%** | **0.4°** |

**Real-time (live audio from ReSpeaker):**

| Model | Exact Acc | ±15° Acc | MAE |
|---|---|---|---|
| CNN (seq=2) | 35.4% | 82.9% | 16.8° |
| **GRU (seq=32)** | **46.1%** | **90.7%** | **9.5°** |

### Sound Recognition (YAMNet)

| Category | Accuracy |
|---|---|
| Speech | 100% |
| Alarms | 95% |
| Animals | 92% |
| Tools | 90% |
| Footsteps | 88% |

### Speech-to-Text (Google STT)

| Language | Word Error Rate | Latency |
|---|---|---|
| English | 3.70% | 4–8 s |
| Arabic (incl. Kuwaiti dialect) | 15.38% | 4–8 s |

---

## Hardware

| Component | Details |
|---|---|
| Microphone array | ReSpeaker XVF3800 — 4 MEMS mics, circular, USB |
| Processing unit | Raspberry Pi 5 |
| Power | 5V portable battery |
| Speaker distance | 1.75 m from array |
| Coverage | 360° — 24 angles in 15° steps |
| Audio format | 16-bit PCM, 16 kHz, 4 channels |

---

## Team

| Member | Contribution |
|---|---|
| Farah AlHayek | Sound Source Localization |
| Reemas Alsubaiei | Sound Source Localization |
| Malak Alsahhaf | Sound Recognition |
| Ghina Alajmi | Speech-to-Text |
| Arwa Almutairi | Hardware & Integration |
| Dr. Khaled Youssef | Supervisor |
| Dr. Samer Said | Supervisor |
| Dr. Samer Alkork | Supervisor |
