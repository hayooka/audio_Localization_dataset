import pyaudio
import numpy as np
from gammatone.gtgram import gtgram
import joblib
from tensorflow.keras.models import load_model

RATE      = 48000
FRAME_MS  = 20
CHUNK     = int(RATE * FRAME_MS / 1000)
NUM_FILTERS = 20
MIC1_INDEX  = 1
MIC2_INDEX  = 13

p = pyaudio.PyAudio()
stream1 = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                 input_device_index=MIC1_INDEX, frames_per_buffer=CHUNK)
stream2 = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                 input_device_index=MIC2_INDEX, frames_per_buffer=CHUNK)

print("Recording live audio... Ctrl+C to stop")

data1 = stream1.read(CHUNK, exception_on_overflow=False)
data2 = stream2.read(CHUNK, exception_on_overflow=False)

left  = np.frombuffer(data1, dtype=np.int16).astype(np.float32)
right = np.frombuffer(data2, dtype=np.int16).astype(np.float32)

PL_f = np.mean(gtgram(left,  RATE, CHUNK/RATE, CHUNK/RATE, NUM_FILTERS, 50), axis=1)
PR_f = np.mean(gtgram(right, RATE, CHUNK/RATE, CHUNK/RATE, NUM_FILTERS, 50), axis=1)

features = np.concatenate((PL_f, PR_f))

scaler = joblib.load("scaler.save")
features_scaled = scaler.transform([features])

model = load_model("my_trained_model.h5")
prediction = model.predict(features_scaled)
pred_label = np.argmax(prediction)

print(f"Predicted Direction / Label: {pred_label}")

stream1.close(); stream2.close(); p.terminate()
