# SoundSense — Sound Source Localization

A complete pipeline for real-time sound source localization using a **ReSpeaker XVF3800 6-microphone array**, covering **24 angles** across a full 360° in 15° steps, powered by a **GRU sequence model**.

Designed as an assistive system for deaf and hearing-impaired users — runs on a Raspberry Pi and serves a mobile-friendly web UI over Wi-Fi.

---

## Quick Start (installable package)

```bash
pip install git+https://github.com/hayooka/audio_Localization_dataset.git
```

```python
from audioloc import predict

angle, chunks = predict('mic_right.wav', 'mic_front.wav',
                        'mic_left.wav',  'mic_back.wav')
print(f'Direction: {angle}°')
```

---

## Real-Time Server (Raspberry Pi)

```bash
cd 6_user_interface
pip install fastapi uvicorn pyaudio torch numpy scipy tensorflow-hub google-cloud-speech
python soundsense_server.py
```

Open `http://<rpi-ip>:8000` on any device on the same Wi-Fi.

> Requires a Google Cloud Speech-to-Text credentials JSON on the Pi.
> Set the path in `soundsense_server.py` line 41.

---

## Project Structure

```
audio_Localization_dataset/
├── 0_Dataset/                        ← raw WAV recordings (24 angles × 2 durations)
├── 1_data_collection/
│   ├── record_4mic_ReSpeaker.py      ← 6-ch ReSpeaker recorder (uses ch 2–5)
│   ├── Raspberry_recor_Respeaker.py  ← RPi-optimised recorder
│   └── record_2mic_old.py            ← legacy 2-mic recorder
├── 2_data_to_features/
│   ├── features_to_csv.py            ← WAV → 895 features → CSV (16ms chunks)
│   └── 6_plot.py                     ← feature separability plots
├── 3_training/
│   ├── train_ALL_features.py         ← CNN baseline
│   ├── sequence_audio_train.py       ← GRU sequence model (main model)
│   ├── audioLOC.pt                   ← CNN weights
│   └── audioLOC_GCCTDOA.pt          ← GCC-TDOA CNN weights
├── 4_Results/
│   ├── models/                       ← saved model checkpoints
│   └── plots/                        ← accuracy / polar / per-angle plots
├── 5_inference/
│   ├── __init__.py                   ← predict() + predict_realtime()
│   └── _features.py                  ← feature extraction (shared with UI)
├── 6_user_interface/
│   ├── soundsense_server.py          ← FastAPI backend (GRU + YAMNet + STT)
│   ├── soundsense_v5.html            ← PWA frontend (compass, captions, alerts)
│   ├── _features.py                  ← feature extraction (copy for deployment)
│   ├── audioLOC_GRU.pt              ← GRU model weights
│   ├── manifest.json                 ← PWA manifest (Add to Home Screen)
│   └── sw.js                         ← Service Worker (background notifications)
├── notebooks/
│   └── analyze_features.ipynb        ← EDA and feature analysis
├── pyproject.toml
└── README.md
```

---

## Pipeline

1. **Record** — `1_data_collection/record_4mic_ReSpeaker.py` — 5 min train + 3 min test WAV per angle
2. **Extract** — `2_data_to_features/features_to_csv.py` — 16ms chunks → 895 features → CSV
3. **Train** — `3_training/sequence_audio_train.py` — GRU sequence model, 24-class
4. **Deploy** — `6_user_interface/soundsense_server.py` — live inference + YAMNet + STT

---

## Features

| Feature | Dims | Description |
|---|---|---|
| IPD | 3 | Hilbert phase difference (3 mic pairs) |
| IPD-mel | 120 | Phase difference weighted over 40 mel bands |
| GCC-PHAT TDOA | 6 | Cross-correlation peak delay (ms), 6 pairs |
| GCC Strength | 6 | Correlation peak sharpness, 6 pairs |
| GCC vector | 600 | Full 100-sample GCC curve, 6 pairs |
| Log-mel | 160 | 40 mel bands × 4 mics |
| **Total** | **895** | |

---

## Experiments

| Scenario | Train | Test | Purpose |
|---|---|---|---|
| S1 | 80% of 5 min | 20% of 5 min | Internal validation |
| S2 | 5 min (full) | 3 min (held-out) | Generalization to unseen conditions |

---

## Results

**Per-feature CNN models (30 ms chunks, S2):**

| Feature Set | Acc | MAE | RMSE |
|---|---|---|---|
| GCC-TDOA | 90.10% | 7.8° | 30.2° |
| GCC Strength | 89.44% | 8.7° | 32.3° |
| IPD | 83.36% | 12.5° | 36.9° |
| Log-Mel | 51.21% | 37.5° | 63.9° |
| RMS | 16.10% | 70.4° | 89.9° |

**ALL-features sequence models (16 ms chunks, cross-recording split):**

| Model | Acc | MAE | RMSE |
|---|---|---|---|
| CNN (seq=2, 32ms context) | 83.04% | 12.4° | 37.1° |
| **GRU (seq=32, 512ms context)** | **99.19%** | **0.5°** | **7.4°** |

---

## Real-Time UI Features

- **Compass** — live DOA arrow updating every 16ms
- **Sound classification** — YAMNet dual-mic agreement filter (reduces false positives)
- **Speech-to-text** — Google Cloud Speech streaming (Arabic / English)
- **Alerts** — danger sound detection with push notifications + haptic vibration
- **PWA** — installable on iOS / Android (Add to Home Screen)
- **Two views** — Detailed (full info) and Light (large text, accessible)

---

## Hardware

- **Mic array:** ReSpeaker XVF3800 (6-ch USB)
- **Host:** Raspberry Pi 5
- **Speaker distance:** 1.75 m from array
- **Angles:** 24 positions, 0°–345° in 15° steps
- **Format:** 16-bit, 16 kHz, 6-channel WAV

<!-- Add photos of the ReSpeaker and setup here -->
<!-- ![ReSpeaker XVF3800](docs/images/respeaker.jpg) -->
<!-- ![Setup overview](docs/images/setup.jpg) -->

---

## Deployment

To run the real-time server on a Raspberry Pi, see **[SETUP_PI.md](SETUP_PI.md)** for full step-by-step instructions covering installation, credentials, auto-start on boot, and troubleshooting.

---

## Team

| Member | Task |
|---|---|
| Farah | Sound Source Localization |
| Reemas | Sound Source Localization |
