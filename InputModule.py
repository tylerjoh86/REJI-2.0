import numpy as np
import sounddevice as sd
import time 
import sys
import threading
from openwakeword.model import Model
from openwakeword.utils import download_models
from silero_vad import load_silero_vad
from faster_whisper import WhisperModel
import torch
import queue

# ---------------- CONFIGURATION ----------------
SAMPLE_RATE = 16000              # All components expect 16kHz
CHUNK_SIZE = 512                 # ~32 ms per chunk at 16kHz
WAKE_WORD = "hey_jarvis"           # Change to your wake word (e.g., "hey-mic", "computer")
WW_THRESHOLD = 0.8               # Wake-word confidence threshold

SILENCE_TIMEOUT = 1.2
RESPONSE_WINDOW = 5.0            # Seconds of no speech before stopping recording
WHISPER_MODEL_SIZE = "large-v3"    # Options: tiny, base, small, medium, large-v3
USE_GPU = True                   # Set True if you have CUDA; faster-whisper will use it automatically if available
# ------------------------------------------------

class InputModule:
    def __init__(self, orchestrator):

        self.orchestrator = orchestrator

        self.raw_queue = queue.Queue(maxsize=200)

    def start(self):
        capture_t = threading.Thread(target=self.capture_loop, daemon=True)
        capture_t.start()
        self.is_paused = threading.Event()
        self.running = True


    def pause(self):
        self.is_paused.set()

    def resume(self):
        with self.raw_queue.mutex:
            self.raw_queue.queue.clear()

        self.is_paused.clear()
    
    def capture_loop(self):
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SIZE
        ) as stream:
            while self.running:
                chunk, _ = stream.read(CHUNK_SIZE)

                if self.is_paused.is_set():
                    continue

                if not self.raw_queue.full():
                    ts = time.time()
                    # flatten to 1D float32
                    data = np.asarray(chunk).flatten()
                    self.raw_queue.put(data)