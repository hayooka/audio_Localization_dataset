# ALGuide — Audio Sound Localization Dataset & System

A complete pipeline for sound source localization using a **ReSpeaker 4-microphone array** (front, back, left, right), covering 8 angles across a full 360°.

---

## Dataset

- **Angles**: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
- **Audio format**: 16-bit, 16kHz, mono WAV, one file per microphone channel
- **Durations**: 3min and 5min recordings per angle
- **Microphones**: mic_front, mic_back, mic_left, mic_right

---

## Project Structure

```
audio_Localization_dataset/
├── Dataset1/
│   └── 0/ 45/ 90/ 135/ 180/ 225/ 270/ 315/
│       └── speakerM/
│           ├── 3min/   ← mic_front.wav, mic_back.wav, mic_left.wav, mic_right.wav
│           └── 5min/   ← mic_front.wav, mic_back.wav, mic_left.wav, mic_right.wav
├── Respaeker/
│   ├── data_collection/
│   │   ├── collecting_data_set.py  ← ReSpeaker 4-mic recorder (7 min per angle)
│   │   └── data_collection.py      ← 2-mic recorder (older version)
│   └── Standard_mode.py            ← full PySide6 GUI combining all features
├── Tasks/
│   ├── localization/
│   │   ├── Respeaker_localization.py ← real-time 4-mic gammatone localization
│   │   └── loc_code.py               ← real-time 2-mic localization (older)
│   └── recognition/
│       ├── Respeaker_YAMNET.py       ← environmental sound classification (YAMNet)
│       └── RespeakerToSTT.py         ← speech-to-text (Vosk)
├── feature_extraction/
│   ├── audionTOfeatures.py           ← WAVs → RMS, IPD, GCC-PHAT → CSV for ML
│   └── 6_plot.py                     ← method separability analysis & plots
├── inference/
│   └── test.py                       ← live prediction with trained Keras model
└── analyze_audio.ipynb               ← full dataset quality & analysis notebook
```

---

## Pipeline

1. **Record** → `data_collection/` — capture audio per angle using ReSpeaker
2. **Extract features** → `feature_extraction/` — compute RMS, IPD, GCC-PHAT; export CSV for ML training
3. **Localize** → `localization/` — real-time direction detection using gammatone filter bank
4. **Recognize** → `recognition/` — sound classification and speech-to-text
5. **Infer** → `inference/` — live prediction using a trained Keras model
6. **GUI** → `app/` — full desktop application combining all features

---

## Team

| Branch | Member | Task |
|--------|--------|------|
| `farah` | Farah | Sound Source Localization |
| `reemas` | Reemas | Sound Source Localization |
| `ghada` | Ghada | Speech-to-Text |
| `alia` | Alia | Speech-to-Text |
| `mariam` | Mariam | Sound Recognition |

> **Note:**
> - **Speech-to-Text**: Ghada, Alia
> - **Sound Source Localization**: Farah, Reemas
> - **Sound Recognition**: Mariam
