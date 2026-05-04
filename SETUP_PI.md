# SoundSense — Raspberry Pi Setup Guide

Everything you need to run the real-time assistive interface on a Raspberry Pi with a ReSpeaker XVF3800.

---

## What you need

- Raspberry Pi 4 or 5 (4 GB RAM recommended)
- ReSpeaker XVF3800 USB mic array (6-channel)
- Google Cloud credentials JSON file (for speech-to-text)
- The Pi and your phone/laptop on the same Wi-Fi network

---

## Step 1 — Flash & connect

Install **Raspberry Pi OS (64-bit)** via Raspberry Pi Imager.  
Enable SSH during setup so you can connect without a monitor:

```bash
ssh pi@<your-pi-ip>
```

---

## Step 2 — Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv portaudio19-dev libsndfile1
```

---

## Step 3 — Clone the repository

```bash
git clone https://github.com/hayooka/audio_Localization_dataset.git
cd audio_Localization_dataset/6_user_interface
```

---

## Step 4 — Create a virtual environment and install packages

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install fastapi uvicorn pyaudio numpy scipy torch
pip install tensorflow-hub
pip install google-cloud-speech
```

> **Note:** PyTorch for RPi — if the above fails, install the ARM wheel:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Step 5 — Add your Google Cloud credentials

Copy your credentials JSON file to the Pi:

```bash
# Run this from your laptop (not the Pi)
scp your-credentials.json pi@<your-pi-ip>:/home/pi/google_credentials.json
```

Then open `soundsense_server.py` and update line 41 to match the path:

```python
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/pi/google_credentials.json'
```

> If you named your Pi user differently (e.g. `karim`), adjust the path accordingly.

---

## Step 6 — Plug in the ReSpeaker

Plug the ReSpeaker XVF3800 into a USB port, then verify it's detected:

```bash
python3 -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
```

Look for a line containing `respeaker` or `xvf`. Note its index number.

---

## Step 7 — Download the GRU model

The model downloads automatically on first run from GitHub Releases.  
Or copy it manually if you have it:

```bash
mkdir -p ~/.soundsense
cp audioLOC_GRU.pt ~/.soundsense/audioLOC_GRU.pt
```

---

## Step 8 — Run the server

```bash
source venv/bin/activate   # if not already activated
python soundsense_server.py
```

The server auto-detects the ReSpeaker. If it doesn't, pass the device index manually:

```bash
python soundsense_server.py --device 2
```

You should see:

```
Audio device index: 2
[GRU] Ready — seq=32 frames (512ms context)
[YAMNet] Ready — 521 classes
[STT] Ready — language=ar-KW
Open  http://<rpi-ip>:8000  in any browser on the same Wi-Fi.
```

---

## Step 9 — Open the UI

On your phone or laptop, open:

```
http://<your-pi-ip>:8000
```

On **Android/iPhone**, tap **Add to Home Screen** to install it as an app (PWA).

---

## Optional — Run on boot automatically

To start the server automatically when the Pi powers on:

```bash
sudo nano /etc/systemd/system/soundsense.service
```

Paste the following (adjust paths if your username is not `pi`):

```ini
[Unit]
Description=SoundSense Real-Time Server
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/audio_Localization_dataset/6_user_interface
ExecStart=/home/pi/audio_Localization_dataset/6_user_interface/venv/bin/python soundsense_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable soundsense
sudo systemctl start soundsense
```

Check it's running:

```bash
sudo systemctl status soundsense
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ReSpeaker not found` | Run with `--device <index>` (see Step 6) |
| `No module named pyaudio` | `sudo apt install portaudio19-dev` then reinstall pyaudio |
| YAMNet not loading | `pip install tensorflow-hub` inside the venv |
| STT not working | Check credentials path and that the JSON file exists on the Pi |
| Can't open the UI | Make sure your phone is on the same Wi-Fi as the Pi |
| Model not found | Copy `audioLOC_GRU.pt` to `~/.soundsense/` (see Step 7) |

---

## Command reference

```bash
# Basic run (auto-detect ReSpeaker)
python soundsense_server.py

# Specify device index
python soundsense_server.py --device 2

# Change port (default 8000)
python soundsense_server.py --port 9000

# Adjust RMS silence threshold (default 50)
python soundsense_server.py --rms 30
```
