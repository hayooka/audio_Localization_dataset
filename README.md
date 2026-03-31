# ALGuide — Sound Source Localization Dataset & System

A complete pipeline for sound source localization using a **ReSpeaker 4-microphone array** (front, right, back, left), covering **24 angles** across a full 360°, processed with feature extraction and a CNN classifier.

---

## Dataset

Audio data was collected using a ReSpeaker 4-microphone array connected to a Raspberry Pi 5. The speaker was placed **1.75 metres** from the ReSpeaker and moved to each angle while the ReSpeaker remained fixed. Recordings were captured at **24 angles** (0° to 345° in 15° increments) in a complex acoustic environment with a high probability of echo.

Two recording sessions were made per angle:
- **5 minutes** — same speech content across all angles (used for training)
- **3 minutes** — unique speech recording per angle (used for testing)

All recordings are raw WAV files (16-bit, 16 kHz, one file per microphone channel).

```
(24 angles)dataset/
└── 0/ 15/ 30/ ... 345/
    └── speakerM/
        ├── 5min/   ← mic_front.wav  mic_right.wav  mic_back.wav  mic_left.wav
        └── 3min/   ← mic_front.wav  mic_right.wav  mic_back.wav  mic_left.wav
```

> Dataset files are stored locally and excluded from version control (see `.gitignore`).

---

## Project Structure

```
audio_Localization_dataset/
├── Respaeker/
│   └── data_collection/
│       ├── collecting_data_set.py    ← ReSpeaker 4-mic recorder
│       └── data_collection.py        ← older 2-mic recorder
├── Tasks/
│   └── localization/
│       ├── Respeaker_localization.py ← early real-time GCC-PHAT + Gammatone localization (4 directions)
│       └── loc_code.py               ← older 2-mic version
├── feature_extraction/
│   ├── audionTOfeatures.py           ← WAV → IPD, GCC-PHAT, log-mel → CSV (30ms chunks)
│   └── 6_plot.py                     ← feature separability analysis & plots
├── training/
│   ├── data_processing.py            ← augmentation, early stopping utilities
│   ├── cnn.py                        ← CNN on all 175 features (IPD + GCC + log-mel)
│   ├── CNNonIPD.py                   ← CNN on IPD features only
│   ├── CNNonGCCtdoa.py               ← CNN on GCC-TDOA features only
│   ├── CNNonGCCStrength.py           ← CNN on GCC strength features only
│   ├── CNNonLOGMEL                   ← CNN on log-mel features only
│   └── model_E1.pt                   ← saved model (S2: trained on 80% of 5min, tested on 3min)
├── inference/
│   ├── realtime_inference.py         ← live prediction using trained model
│   └── test.py                       ← offline inference script
├── notebooks/
│   └── analyze_features.ipynb        ← feature analysis, RMS distribution, silence detection
├── Results/                          ← saved result plots (one per feature set)
└── Dataset/
    └── .CVS/                         ← extracted feature CSVs (train/test)
```

---

## Pipeline

1. **Record** → `Respaeker/data_collection/` — capture 5min + 3min WAV per angle
2. **Extract features** → `feature_extraction/audionTOfeatures.py` — 30ms chunks → IPD (3) + GCC-TDOA (6) + GCC-strength (6) + log-mel (160) = **175 features** → CSV
3. **Analyse** → `notebooks/analyze_features.ipynb` — inspect RMS distribution, silence, feature separability
4. **Train** → `training/cnn.py` (or per-feature variants) — 1D CNN, 24-class classification, results saved to `Results/`
5. **Infer** → `inference/realtime_inference.py` — live direction prediction

---

## Features

| Feature | Dims | Method |
|---|---|---|
| IPD | 3 | Hilbert phase difference between mic pairs |
| GCC-PHAT TDOA | 6 | Phase-weighted cross-correlation peak (ms) |
| GCC Strength | 6 | Correlation peak sharpness (confidence) |
| Log-mel spectrogram | 160 | 40 mel bands × 4 mics, 1024-pt FFT |
| **Total** | **175** | |

---

## Experiments

| Scenario | Train | Test | Purpose |
|---|---|---|---|
| S1 | 80% of 5min | 20% of 5min | Internal validation |
| S2 | 80% of 5min | 3min (held-out) | Generalization to different speech content |

---

## Team

| Member | Task |
|---|---|
| Farah | Sound Source Localization |
| Reemas | Sound Source Localization |
