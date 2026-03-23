import sys
import os
import queue
import json
import numpy as np
import sounddevice as sd
import pyaudio
import usb.core
import usb.util
import time
from vosk import Model, KaldiRecognizer
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QFrame, QHBoxLayout, QGridLayout, QScrollArea,
    QPushButton, QMessageBox, QTextEdit
)
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtCore import Qt, QThread, Signal, QTimer

MODEL_PATH = r"C:\Users\rimas\.spyder-py3\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"

# ================= SPEECH THREAD =================
class SpeechThread(QThread):
    text_signal   = Signal(str)
    error_signal  = Signal(str)
    status_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.is_running = True

    def run(self):
        try:
            if not os.path.exists(MODEL_PATH):
                self.error_signal.emit(f"Model not found at: {MODEL_PATH}")
                return
            q = queue.Queue()
            self.status_signal.emit("Initializing microphone...")

            def callback(indata, frames, time, status):
                q.put(bytes(indata))

            model = Model(MODEL_PATH)
            rec = KaldiRecognizer(model, 16000)
            self.status_signal.emit("Listening... Speak now!")

            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                   channels=1, callback=callback):
                while self.is_running:
                    try:
                        data = q.get(timeout=1)
                        if rec.AcceptWaveform(data):
                            text = json.loads(rec.Result()).get("text", "")
                            if text:
                                self.text_signal.emit(f"🗣 {text}")
                    except queue.Empty:
                        continue
        except Exception as e:
            self.error_signal.emit(f"Speech recognition error: {str(e)}")

    def stop(self):
        self.is_running = False


# ================= SOUND RECOGNITION THREAD =================
class SoundRecognitionThread(QThread):
    result_signal = Signal(str)
    error_signal  = Signal(str)
    status_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.sample_rate = 16000
        self.duration = 0.96

    def run(self):
        try:
            self.status_signal.emit("Loading sound recognition...")
            try:
                import tensorflow as tf
                import tensorflow_hub as hub
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                self.status_signal.emit("Loading YAMNet model...")
                model = hub.load("https://tfhub.dev/google/yamnet/1")
                class_names = []
                with open(model.class_map_path().numpy(), "r") as f:
                    for line in f.readlines()[1:]:
                        class_names.append(line.split(",")[2].strip())
                self.status_signal.emit("Sound recognition active")
                self.result_signal.emit("🔊 Sound Recognition Started\nWaiting for sounds...\n")

                while self.is_running:
                    audio = sd.rec(int(self.duration * self.sample_rate),
                                   samplerate=self.sample_rate, channels=1, dtype="float32")
                    sd.wait()
                    waveform = np.squeeze(audio)
                    scores, _, _ = model(waveform)
                    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
                    top_indices = np.argsort(mean_scores)[-5:][::-1]
                    result_text = "🔊 Detected Sounds:\n"
                    has_sound = False
                    for i in top_indices:
                        if mean_scores[i] > 0.1:
                            result_text += f"  • {class_names[i]}: {mean_scores[i]*100:.1f}%\n"
                            has_sound = True
                    self.result_signal.emit(result_text if has_sound else "🔇 No clear sounds detected...")

            except ImportError:
                import random
                sounds = ["Speech", "Music", "Silence", "Noise", "Clapping", "Laughter"]
                self.status_signal.emit("Sound recognition (simulation mode)")
                while self.is_running:
                    detected = random.sample(sounds, 3)
                    result_text = "🔊 Detected Sounds (Simulation):\n"
                    for s in detected:
                        result_text += f"  • {s}: {random.uniform(60,95):.1f}%\n"
                    self.result_signal.emit(result_text)
                    self.msleep(2000)

        except Exception as e:
            self.error_signal.emit(f"Sound recognition error: {str(e)}")

    def stop(self):
        self.is_running = False


# ================= RESPEAKER LOCALIZATION THREAD =================
class ReSpeakerLocalizationThread(QThread):
    result_signal = Signal(str)
    error_signal  = Signal(str)
    status_signal = Signal(str)

    PARAMETERS = {
        "VERSION":   (48, 0,  3, "ro", "uint8"),
        "DOA_VALUE": (20, 18, 4, "ro", "uint16"),
    }
    TIMEOUT = 100000

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.dev = None

    def find_device(self, vid=0x2886, pid=0x001A):
        return usb.core.find(idVendor=vid, idProduct=pid)

    def read(self, name):
        data = self.PARAMETERS.get(name)
        if not data:
            return None
        response = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 0x80 | data[1], data[0], data[2] + 1, self.TIMEOUT)
        return response.tolist() if data[4] == 'uint16' else None

    def get_direction(self, angle):
        if 315 <= angle or angle < 45:   return "FRONT"
        elif 45 <= angle < 135:           return "RIGHT"
        elif 135 <= angle < 225:          return "BACK"
        elif 225 <= angle < 315:          return "LEFT"
        return "UNKNOWN"

    def run(self):
        try:
            self.status_signal.emit("Initializing ReSpeaker...")
            self.dev = self.find_device()
            if not self.dev:
                self.error_signal.emit("ReSpeaker device not found.")
                return
            self.status_signal.emit("ReSpeaker connected")
            self.result_signal.emit("📍 Localization Started\n")

            stable_count, required_stable = 0, 3
            last_raw_direction = last_printed_direction = None

            while self.is_running:
                result = self.read("DOA_VALUE")
                if result:
                    angle, speech = result[1], result[3]
                    if speech == 1:
                        direction = self.get_direction(angle)
                        stable_count = stable_count + 1 if direction == last_raw_direction else 0
                        if stable_count >= required_stable and direction != last_printed_direction:
                            arrow = {"FRONT": "⬆️", "RIGHT": "➡️", "BACK": "⬇️", "LEFT": "⬅️"}.get(direction, "")
                            self.result_signal.emit(f"{arrow} Direction: {direction} ({angle}°)")
                            last_printed_direction = direction
                        last_raw_direction = direction
                self.msleep(50)

        except Exception as e:
            self.error_signal.emit(f"ReSpeaker error: {str(e)}")

    def stop(self):
        self.is_running = False
        if self.dev:
            usb.util.dispose_resources(self.dev)


# ================= EMOTION RECOGNITION THREAD =================
class EmotionRecognitionThread(QThread):
    result_signal = Signal(str)
    error_signal  = Signal(str)
    status_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.is_running = True

    def run(self):
        self.status_signal.emit("Emotion recognition (Under Development)")
        self.result_signal.emit("🧠 Emotion Recognition\n" + "="*30 + "\n🚧 UNDER DEVELOPMENT\nComing soon!")
        while self.is_running:
            self.msleep(1000)

    def stop(self):
        self.is_running = False


# ================= MAIN UI =================
class BusinessALGuide(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALGuide · Business AI Assistant")
        self.resize(1300, 900)
        self.setStyleSheet("""
            QWidget { background: #F4F7FB; font-family: 'Segoe UI', Arial, sans-serif; }
            QLabel  { background: transparent; border: none; }
            QScrollArea { border: none; background: transparent; }
            QTextEdit {
                background-color: white; border: 2px solid #2563EB;
                border-radius: 12px; padding: 15px; font-size: 14px;
                font-family: 'Consolas', 'Courier New', monospace; color: #1F2937;
            }
            QPushButton { font-size: 14px; font-weight: 600; }
        """)
        self.speech_thread = self.sound_thread = self.localization_thread = self.emotion_thread = None
        self.setup_ui()
        self.add_logo()
        QTimer.singleShot(500, self.show_welcome_message)

    def show_welcome_message(self):
        self.output_text.setHtml("""
        <div style='text-align:center;'>
            <h2 style='color:#2563EB;'>✨ Welcome to ALGuide ✨</h2>
            <p style='color:#4B5563;'>Click any button below to start:</p>
            <ul style='text-align:left;display:inline-block;'>
                <li>🎤 Speech-to-Text</li><li>🔊 Sound Recognition</li>
                <li>📍 Sound Source Localization</li><li>🧠 Emotion Recognition (coming soon)</li>
            </ul>
        </div>""")

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        self.content = QWidget(); self.content.setStyleSheet("background: transparent;")
        self.layout = QVBoxLayout(self.content)
        self.layout.setContentsMargins(80, 30, 80, 40); self.layout.setSpacing(25)
        scroll.setWidget(self.content); main_layout.addWidget(scroll)
        self.setup_header(); self.setup_development_bar(); self.setup_title()
        self.setup_capabilities_grid(); self.setup_status_area(); self.setup_output_area()

    def setup_header(self):
        header = QHBoxLayout()
        brand = QLabel("""<div style="line-height:1.05;">
            <div style="font-size:38px;font-weight:800;color:#0A2E65;">ALGuide</div>
            <div style="font-size:14px;font-style:italic;color:#2563EB;">A guide who leads you when you lose your way...</div>
        </div>""")
        brand.setTextFormat(Qt.RichText)
        header.addSpacing(100); header.addWidget(brand); header.addStretch()
        self.layout.addLayout(header)

    def setup_development_bar(self):
        bar = QLabel("🔧 Currently Under Development · Version 1.0 Beta")
        bar.setAlignment(Qt.AlignCenter)
        bar.setStyleSheet("background-color:#2563EB;color:white;font-size:15px;font-weight:600;"
                          "padding:12px;border-radius:8px;margin-top:20px;")
        self.layout.addWidget(bar)

    def setup_title(self):
        title = QLabel("Core Capabilities"); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:36px;font-weight:700;color:#0F2B4B;margin:20px 0 10px 0;")
        self.layout.addWidget(title)

    def setup_capabilities_grid(self):
        grid = QGridLayout(); grid.setSpacing(25)
        btn_style = """QPushButton {
            background-color:white;border-radius:12px;font-size:16px;font-weight:600;
            color:#0F2B4B;border:2px solid #E5E7EB;text-align:left;padding:20px;min-height:80px;}
            QPushButton:hover {background-color:#E6F0FF;border-color:#2563EB;}
            QPushButton:pressed {background-color:#DBEAFE;}"""
        caps = [
            ("🎤 Speech-to-Text",          self.start_speech,             "Convert speech to text"),
            ("⏹ Stop All",                 self.stop_all,                 "Stop all active processes"),
            ("🔊 Sound Recognition",        self.start_sound_recognition,  "Identify sounds"),
            ("📍 Sound Source Localization", self.start_localization,       "Direction detection"),
            ("🧠 Emotion Recognition",      self.start_emotion_recognition,"Under Development"),
            ("⚙️ Settings",                 self.open_settings,            "Settings"),
        ]
        for i, (text, handler, tooltip) in enumerate(caps):
            btn = QPushButton(text); btn.setMinimumHeight(100)
            btn.setStyleSheet(btn_style); btn.setToolTip(tooltip)
            btn.clicked.connect(handler); grid.addWidget(btn, i // 2, i % 2)
        self.layout.addLayout(grid)

    def setup_status_area(self):
        frame = QFrame(); frame.setStyleSheet(
            "QFrame{background-color:white;border-radius:12px;border:1px solid #E5E7EB;margin-top:20px;}")
        layout = QVBoxLayout(frame)
        lbl = QLabel("System Status"); lbl.setStyleSheet("font-size:18px;font-weight:600;color:#0F2B4B;padding:5px;")
        self.status_display = QLabel("✅ Ready")
        self.status_display.setStyleSheet(
            "font-size:14px;color:#2563EB;padding:10px;background-color:#F3F4F6;border-radius:8px;font-weight:500;")
        layout.addWidget(lbl); layout.addWidget(self.status_display)
        self.layout.addWidget(frame)

    def setup_output_area(self):
        frame = QFrame(); frame.setStyleSheet(
            "QFrame{background-color:white;border-radius:12px;border:1px solid #E5E7EB;margin-top:10px;}")
        layout = QVBoxLayout(frame)
        lbl = QLabel("Output"); lbl.setStyleSheet("font-size:18px;font-weight:600;color:#0F2B4B;padding:5px;")
        self.output_text = QTextEdit(); self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(300)
        layout.addWidget(lbl); layout.addWidget(self.output_text)
        self.layout.addWidget(frame)

    def add_logo(self):
        self.logo = QLabel(self.content)
        pixmap = QPixmap()
        for path in ["logo.png", "assets/logo.png"]:
            if os.path.exists(path):
                pixmap.load(path); break
        if not pixmap.isNull():
            self.logo.setPixmap(pixmap.scaled(260, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.logo.setText("⚡ AL")
            self.logo.setStyleSheet("font-size:48px;font-weight:bold;color:#2563EB;"
                                    "background-color:white;border-radius:130px;padding:20px;"
                                    "border:2px solid #2563EB;")
            self.logo.setAlignment(Qt.AlignCenter); self.logo.setMinimumSize(260, 260)
        self.logo.resize(260, 260); self.logo.move(40, -40); self.logo.raise_()

    # ===== HANDLERS =====
    def start_speech(self):
        self.stop_all()
        if not os.path.exists(MODEL_PATH):
            self.show_error("Model Not Found", f"Vosk model not found at:\n{MODEL_PATH}"); return
        self.speech_thread = SpeechThread()
        self.speech_thread.text_signal.connect(self.append_output)
        self.speech_thread.error_signal.connect(self.handle_error)
        self.speech_thread.status_signal.connect(self.update_status)
        self.speech_thread.start()
        self.output_text.clear(); self.append_output("🎤 Speech Recognition Started\nWaiting for speech...\n")

    def start_sound_recognition(self):
        self.stop_all()
        self.sound_thread = SoundRecognitionThread()
        self.sound_thread.result_signal.connect(self.append_output)
        self.sound_thread.error_signal.connect(self.handle_error)
        self.sound_thread.status_signal.connect(self.update_status)
        self.sound_thread.start(); self.output_text.clear()

    def start_localization(self):
        self.stop_all()
        self.localization_thread = ReSpeakerLocalizationThread()
        self.localization_thread.result_signal.connect(self.append_output)
        self.localization_thread.error_signal.connect(self.handle_error)
        self.localization_thread.status_signal.connect(self.update_status)
        self.localization_thread.start(); self.output_text.clear()

    def start_emotion_recognition(self):
        if self.emotion_thread and self.emotion_thread.isRunning():
            self.emotion_thread.stop(); self.emotion_thread.quit(); self.emotion_thread.wait()
        self.emotion_thread = EmotionRecognitionThread()
        self.emotion_thread.result_signal.connect(self.append_output)
        self.emotion_thread.error_signal.connect(self.handle_error)
        self.emotion_thread.status_signal.connect(self.update_status)
        self.emotion_thread.start(); self.output_text.clear()

    def open_settings(self):
        self.stop_all(); self.output_text.clear()
        self.append_output("⚙️ Settings\n" + "="*30 + "\nComing soon!")

    def stop_all(self):
        for thread in [self.speech_thread, self.sound_thread,
                       self.localization_thread, self.emotion_thread]:
            if thread and thread.isRunning():
                thread.stop(); thread.quit(); thread.wait(2000)
                if thread.isRunning(): thread.terminate(); thread.wait()
        self.speech_thread = self.sound_thread = self.localization_thread = self.emotion_thread = None
        self.update_status("✅ Ready"); self.append_output("\n⏹ All processes stopped\n")

    def append_output(self, text):
        self.output_text.append(text)
        cursor = self.output_text.textCursor()
        cursor.movePosition(QTextCursor.End); self.output_text.setTextCursor(cursor)

    def update_status(self, status):
        self.status_display.setText(f"📡 {status}")

    def handle_error(self, error_message):
        self.update_status("❌ Error"); self.append_output(f"\n⚠️ Error: {error_message}\n")
        self.show_error("Error", error_message)

    def show_error(self, title, message):
        msg = QMessageBox(); msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title); msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok); msg.exec()

    def closeEvent(self, event):
        self.stop_all(); event.accept()


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    window = BusinessALGuide(); window.show()
    sys.exit(app.exec())
