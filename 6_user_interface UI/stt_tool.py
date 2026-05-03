import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import csv
import os

import sounddevice as sd
from google.cloud import speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\farah\Downloads\data-fabric-495218-c3-c0146f2a4600.json"

LOG_FILE = "test_log.csv"

CHUNK_SIZE = 1600  # 100ms at 16000Hz


class MultilingualSTTTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Multilingual Speech-to-Text Testing Tool")
        self.root.geometry("1200x760")

        self.sample_rate = 16000
        self.channels = 1

        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self.start_time = None

        self.final_text = ""
        self.last_text = ""
        self.last_duration = 0.0

        self.total_tests = 0
        self.correct = 0
        self.incorrect = 0
        self.partial = 0

        self.build_ui()
        self.load_previous_log()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        title = tk.Label(self.root, text="Multilingual Speech-to-Text Testing Tool", font=("Arial", 22, "bold"))
        title.pack(pady=10)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(8, 0))

        lang_frame = tk.LabelFrame(left_frame, text="Language Selection", padx=10, pady=10)
        lang_frame.pack(fill="x", pady=5)

        tk.Label(lang_frame, text="Select Language:").grid(row=0, column=0, sticky="w")

        self.language_var = tk.StringVar(value="English")
        self.language_box = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=["English", "Arabic"],
            state="readonly",
            width=20
        )
        self.language_box.grid(row=1, column=0, sticky="w", pady=5)

        controls_frame = tk.LabelFrame(left_frame, text="Controls", padx=10, pady=10)
        controls_frame.pack(fill="x", pady=5)

        self.start_button = tk.Button(
            controls_frame,
            text="Start Real-Time Test",
            width=18,
            command=self.start_recording
        )
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = tk.Button(
            controls_frame,
            text="Stop",
            width=18,
            command=self.stop_recording,
            state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=5)

        self.submit_button = tk.Button(
            controls_frame,
            text="Submit Evaluation",
            width=18,
            command=self.submit_evaluation
        )
        self.submit_button.grid(row=0, column=2, padx=5)

        output_frame = tk.LabelFrame(left_frame, text="Current Test Output", padx=10, pady=10)
        output_frame.pack(fill="x", pady=5)

        tk.Label(output_frame, text="Recognized Text:", font=("Arial", 11, "bold")).pack(anchor="w")

        self.text_output = tk.Text(output_frame, height=4, wrap="word", font=("Arial", 11))
        self.text_output.tag_config("interim", foreground="gray")
        self.text_output.pack(fill="x", pady=5)

        self.duration_label = tk.Label(output_frame, text="Processing Duration: -", font=("Arial", 10))
        self.duration_label.pack(anchor="w", pady=3)

        self.status_label = tk.Label(output_frame, text="Status: Ready", font=("Arial", 10))
        self.status_label.pack(anchor="w", pady=3)

        eval_frame = tk.LabelFrame(left_frame, text="Manual Evaluation", padx=10, pady=10)
        eval_frame.pack(fill="x", pady=5)

        tk.Label(eval_frame, text="Was the result correct?", font=("Arial", 11, "bold")).pack(anchor="w")

        self.evaluation_var = tk.StringVar(value="Correct")

        tk.Radiobutton(eval_frame, text="Correct", variable=self.evaluation_var, value="Correct").pack(side="left", padx=10)
        tk.Radiobutton(eval_frame, text="Incorrect", variable=self.evaluation_var, value="Incorrect").pack(side="left", padx=10)
        tk.Radiobutton(eval_frame, text="Partially Correct", variable=self.evaluation_var, value="Partial").pack(side="left", padx=10)

        counter_frame = tk.LabelFrame(right_frame, text="Results & Counters", padx=20, pady=20)
        counter_frame.pack(fill="x", pady=5)

        self.total_label = tk.Label(counter_frame, text="Total Tests: 0", font=("Arial", 12, "bold"))
        self.total_label.pack(anchor="w", pady=8)

        self.correct_label = tk.Label(counter_frame, text="Correct: 0", font=("Arial", 12, "bold"))
        self.correct_label.pack(anchor="w", pady=8)

        self.incorrect_label = tk.Label(counter_frame, text="Incorrect: 0", font=("Arial", 12, "bold"))
        self.incorrect_label.pack(anchor="w", pady=8)

        self.partial_label = tk.Label(counter_frame, text="Partially Correct: 0", font=("Arial", 12, "bold"))
        self.partial_label.pack(anchor="w", pady=8)

        self.accuracy_label = tk.Label(counter_frame, text="Accuracy: 0.00%", font=("Arial", 12, "bold"))
        self.accuracy_label.pack(anchor="w", pady=8)

        self.weighted_accuracy_label = tk.Label(counter_frame, text="Weighted Accuracy: 0.00%", font=("Arial", 12, "bold"))
        self.weighted_accuracy_label.pack(anchor="w", pady=8)

        log_frame = tk.LabelFrame(right_frame, text="Test Log", padx=10, pady=10)
        log_frame.pack(fill="both", expand=True, pady=5)

        log_scrollbar = tk.Scrollbar(log_frame)
        log_scrollbar.pack(side="right", fill="y")

        self.log_text = tk.Text(log_frame, height=18, wrap="word", font=("Consolas", 9), yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)

        log_scrollbar.config(command=self.log_text.yview)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        if self.is_recording:
            return

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.is_recording = True
        self.start_time = time.time()
        self.final_text = ""
        self.last_text = ""

        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, "Listening...\n")
        self.status_label.config(text="Status: Listening in real-time...")

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=self.audio_callback
        )
        self.stream.start()

        threading.Thread(target=self.streaming_recognize, daemon=True).start()

    def audio_generator(self):
        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())
            except queue.Empty:
                pass

    def streaming_recognize(self):
        client = speech.SpeechClient()

        language_code = "en-US" if self.language_var.get() == "English" else "ar-KW"

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=language_code,
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
        )

        try:
            responses = client.streaming_recognize(streaming_config, self.audio_generator())

            for response in responses:
                if not self.is_recording:
                    break
                for result in response.results:
                    transcript = result.alternatives[0].transcript
                    is_final = result.is_final

                    if is_final:
                        self.final_text += transcript + " "
                        self.last_text = self.final_text.strip()

                    self.root.after(0, lambda f=self.final_text, t=transcript, fin=is_final: self.update_realtime_output(f, t, fin))

        except Exception as e:
            if self.is_recording:
                self.root.after(0, lambda: messagebox.showerror("Google STT Error", str(e)))
                self.root.after(0, lambda: self.status_label.config(text="Status: Error"))
                self.root.after(0, self.stop_recording)

    def update_realtime_output(self, final_text, interim_transcript, is_final):
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, final_text)

        if not is_final:
            self.text_output.insert(tk.END, interim_transcript, "interim")

        self.last_duration = time.time() - self.start_time
        self.duration_label.config(text=f"Processing Duration: {self.last_duration:.2f} sec")
        self.status_label.config(text="Status: Real-time transcription running...")

    def on_close(self):
        self.stop_recording()
        self.root.destroy()

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.last_duration = time.time() - self.start_time
        self.last_text = self.final_text.strip()

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Stopped")

    def submit_evaluation(self):
        if not self.last_text:
            messagebox.showwarning("No Result", "Please run a test before submitting evaluation.")
            return

        result = self.evaluation_var.get()

        self.total_tests += 1

        if result == "Correct":
            self.correct += 1
        elif result == "Incorrect":
            self.incorrect += 1
        elif result == "Partial":
            self.partial += 1

        self.update_counters()
        self.add_to_log(result)
        self.save_to_csv(result)
        self.last_text = ""

    def update_counters(self):
        if self.total_tests > 0:
            accuracy = (self.correct / self.total_tests) * 100
            weighted_accuracy = ((self.correct + 0.5 * self.partial) / self.total_tests) * 100
        else:
            accuracy = 0
            weighted_accuracy = 0

        self.total_label.config(text=f"Total Tests: {self.total_tests}")
        self.correct_label.config(text=f"Correct: {self.correct}")
        self.incorrect_label.config(text=f"Incorrect: {self.incorrect}")
        self.partial_label.config(text=f"Partially Correct: {self.partial}")
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")
        self.weighted_accuracy_label.config(text=f"Weighted Accuracy: {weighted_accuracy:.2f}%")

    def add_to_log(self, result):
        log_entry = (
            f"Test {self.total_tests} | "
            f"Language: {self.language_var.get()} | "
            f"Text: {self.last_text} | "
            f"Duration: {self.last_duration:.2f}s | "
            f"Result: {result}\n"
        )

        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)

    def save_to_csv(self, result):
        file_exists = os.path.exists(LOG_FILE)

        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Test", "Language", "Text", "Duration", "Result"])

            writer.writerow([
                self.total_tests,
                self.language_var.get(),
                self.last_text,
                f"{self.last_duration:.2f}",
                result
            ])

    def load_previous_log(self):
        if not os.path.exists(LOG_FILE):
            self.update_counters()
            return

        try:
            with open(LOG_FILE, "r", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)

                for row in reader:
                    result = row.get("Result", "")
                    language = row.get("Language", "")
                    text = row.get("Text", "")
                    duration = row.get("Duration", "")

                    self.total_tests += 1

                    if result == "Correct":
                        self.correct += 1
                    elif result == "Incorrect":
                        self.incorrect += 1
                    elif result == "Partial":
                        self.partial += 1

                    log_entry = (
                        f"Test {self.total_tests} | "
                        f"Language: {language} | "
                        f"Text: {text} | "
                        f"Duration: {duration}s | "
                        f"Result: {result}\n"
                    )

                    self.log_text.insert(tk.END, log_entry)

            self.log_text.see(tk.END)
            self.update_counters()

        except Exception as e:
            messagebox.showwarning("Log Load Error", f"Could not load previous log:\n{e}")
            self.update_counters()


if __name__ == "__main__":
    root = tk.Tk()
    app = MultilingualSTTTool(root)
    root.mainloop()
