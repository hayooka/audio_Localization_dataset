# ALGuide — Sound Source Localization Dataset & System

A complete pipeline for sound source localization using a **ReSpeaker 4-microphone array** (front, back, left, right), covering 8 angles across a full 360°.

---

## Dataset

- **Angles**: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
- **Audio format**: 16-bit, 16kHz, mono WAV, one file per microphone channel
- **Durations**: 3min and 5min recordings per angle
- **Microphones**: mic_front, mic_back, mic_left, mic_right

```
Dataset1/
└── 0/ 45/ 90/ 135/ 180/ 225/ 270/ 315/
    └── speakerM/
        ├── 3min/   ← mic_front.wav, mic_back.wav, mic_left.wav, mic_right.wav
        └── 5min/   ← mic_front.wav, mic_back.wav, mic_left.wav, mic_right.wav
```

> Dataset files are stored locally and excluded from version control (see `.gitignore`).

---

## Project Structure

```
audio_Localization_dataset/
├── Respaeker/
│   └── data_collection/
│       ├── collecting_data_set.py  ← ReSpeaker 4-mic recorder (7 min per angle)
│       └── data_collection.py      ← 2-mic recorder (older version)
├── Tasks/
│   └── localization/
│       ├── Respeaker_localization.py ← real-time 4-mic gammatone localization
│       └── loc_code.py               ← real-time 2-mic localization (older)
├── feature_extraction/
│   ├── audionTOfeatures.py           ← WAVs → RMS, IPD, GCC-PHAT → CSV for ML
│   └── 6_plot.py                     ← method separability analysis & plots
├── inference/
│   └── test.py                       ← live prediction with trained Keras model
└── analyze_audio.ipynb               ← full dataset quality & analysis notebook
```

---

## Pipeline

1. **Record** → `Respaeker/data_collection/` — capture audio per angle using ReSpeaker
2. **Analyze** → `analyze_audio.ipynb` — inspect dataset quality (RMS, silence, GCC-PHAT, TDOA)
3. **Extract features** → `feature_extraction/` — compute RMS, IPD, GCC-PHAT; export CSV for ML training
4. **Localize** → `Tasks/localization/` — real-time direction detection using gammatone filter bank
5. **Infer** → `inference/` — live prediction using a trained Keras model

---

## Features Used for Localization

| Feature | Method | Notes |
|---------|--------|-------|
| RMS | Energy per mic | Baseline |
| TDOA (raw) | Cross-correlation | Coarse delay estimate |
| IPD | Hilbert phase difference | Phase-based |
| Spectrogram energy | STFT | Frequency-domain |
| GCC-PHAT TDOA | Phase-weighted GCC | Best separability |
| GCC Strength | Correlation peak ratio | Confidence metric |

---

## Team

| Member | Task |
|--------|------|
| Farah | Sound Source Localization |
| Reemas | Sound Source Localization |
