# ALGuide — Sound Source Localization

A complete pipeline for sound source localization using a **ReSpeaker 4-microphone array**, covering **24 angles** across a full 360° in 15° steps, powered by a pretrained 1D CNN classifier.

---

## Quick Start

**Install:**
```bash
pip install git+https://github.com/hayooka/audio_Localization_dataset.git
```

**WAV files** (your own recordings):
```python
from audioloc import predict

angle, chunks = predict('mic_right.wav', 'mic_front.wav',
                        'mic_left.wav',  'mic_back.wav')
print(f'Direction: {angle}°')
```

**Real-time** (live ReSpeaker input):
```bash
pip install "audioloc[realtime]"   # adds sounddevice
```
```python
from audioloc import predict_realtime

predict_realtime(device=25)        # device index from your system
```

> First call downloads the pretrained ALL-features weights once to `~/.audioloc/`.
> **Requires:** Python ≥ 3.8, PyTorch, NumPy, SciPy

---

## Project Structure

```
audio_Localization_dataset/
├── 1_data_collection/
│   ├── record_4mic_ReSpeaker.py  ← 4-mic recorder (6-ch ReSpeaker, ch 2–5)
│   └── record_2mic_old.py        ← older 2-mic recorder
├── 2_data_to_features/
│   ├── features_to_csv.py        ← WAV → features → CSV (30ms chunks)
│   └── 6_plot.py                 ← feature separability analysis & plots
├── 3_inference/                  ← installable audioloc package
│   ├── __init__.py               ← predict() + predict_realtime()
│   └── _features.py              ← feature extraction from WAV / live audio
├── 4_training/
│   ├── data_processing.py        ← augmentation, early stopping
│   ├── train_ALL_features.py     ← CNN on all 895 features  ← main model
│   ├── train_IPD.py              ← CNN on IPD only
│   ├── train_GCC_TDOA.py         ← CNN on GCC-TDOA only
│   ├── train_GCC_Strength.py     ← CNN on GCC strength only
│   ├── train_LogMel.py           ← CNN on log-mel only
│   ├── train_RMS.py              ← CNN on RMS only
│   └── audioLOC.pt               ← pretrained weights (ALL features, S2)
├── results/                      ← plots & summaries per feature set
└── pyproject.toml                ← pip package config
```

---

## Pipeline

1. **Record** → `1_data_collection/record_4mic_ReSpeaker.py` — 5 min + 3 min WAV per angle
2. **Extract** → `2_data_to_features/features_to_csv.py` — 30ms chunks → 895 features → CSV
3. **Train** → `4_training/train_ALL_features.py` — 1D CNN, 24-class, saves `audioLOC.pt`
4. **Use** → `from audioloc import predict` or `predict_realtime()`

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
| S2 | 80% of 5 min | 3 min (held-out) | Generalization to unseen speech |

### Results — 16 ms chunks

**Per-feature models (30 ms, S1 / S2):**

| Feature Set | S1 Acc | S2 Acc | S2 MAE | S2 RMSE |
|---|---|---|---|---|
| GCC-TDOA | 99.83% | 90.10% | 7.8° | 30.2° |
| GCC Strength | 99.71% | 89.44% | 8.7° | 32.3° |
| IPD | 97.82% | 83.36% | 12.5° | 36.9° |
| Log-Mel | 89.33% | 51.21% | 37.5° | 63.9° |
| RMS | 20.94% | 16.10% | 70.4° | 89.9° |

**ALL-features models (16 ms, cross-recording split):**

| Model | Acc | MAE | RMSE |
|---|---|---|---|
| CNN (seq=2, 32ms context) | 83.04% | 12.4° | 37.1° |
| GRU (seq=32, 512ms context) | 99.19% | 0.5° | 7.4° |

---

## Hardware

- **Mic array:** ReSpeaker USB 4-mic array
- **Host:** Raspberry Pi 5
- **Distance:** speaker placed 1.75 m from ReSpeaker
- **Angles:** 24 positions, 0°–345° in 15° steps
- **Format:** 16-bit, 16 kHz, mono WAV per channel

---

## Team

| Member | Task |
|---|---|
| Farah | Sound Source Localization |
| Reemas | Sound Source Localization |
