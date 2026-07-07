import numpy as np
import sounddevice as sd
import time 
import sys
from openwakeword.model import Model
from openwakeword.utils import download_models
from silero_vad import load_silero_vad
from faster_whisper import WhisperModel
import torch

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
        download_models()

        self.orchestrator = orchestrator
        self.state = "LISTENING"         # LISTENING | RECORDING
        self.audio_buffer = []
        self.last_speech_time = None

        print("Loading OpenWakeWord...")

        self.oww = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
        
        print("Loading Silero VAD...")
        self.vad_model = load_silero_vad()

        print(f"Loading Faster-Whisper ({WHISPER_MODEL_SIZE})...")
        device = "cuda" if USE_GPU else "cpu"
        compute_type = "float16" if USE_GPU else "int8"
        self.whisper = WhisperModel(
            device=device,
            compute_type=compute_type,
            model_size_or_path=WHISPER_MODEL_SIZE
        )

    def control_loop(self):
        print(f"\nListening for wake word: {WAKE_WORD}...")
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SIZE
        ) as stream:
            
            while not self.orchestrator.end_event.is_set():
                chunk, overflow = stream.read(CHUNK_SIZE)
                if overflow:
                    continue

                chunk1d = chunk.flatten()

                probs = self.oww.predict((chunk1d * 32767).astype(np.int16))
                ww_prob = float(probs.get(WAKE_WORD, 0.0))

                if self.state == "LISTENING" and ww_prob > WW_THRESHOLD:
                    print(f"Wake word detected. (confidence:{ww_prob:.2f})")
                    self.start_recording()
                        

                if self.state == "RECORDING":
                    chunk_tensor = torch.from_numpy(chunk1d)
                    speech_confidence = self.vad_model(chunk_tensor, SAMPLE_RATE)
                    print(speech_confidence)
                    self.audio_buffer.append(chunk1d)

                    if speech_confidence > 0.5:
                        self.last_speech_time = time.time()
                    else:
                        if (time.time() - self.last_speech_time) > SILENCE_TIMEOUT:
                            self.stop_and_transcribe()
                            print("finished recording.")

                if self.state == "WAITING":
                    chunk_tensor = torch.from_numpy(chunk1d)
                    speech_confidence = self.vad_model(chunk_tensor, SAMPLE_RATE)
                    print(speech_confidence)
                    self.audio_buffer.append(chunk1d)

                    if speech_confidence > 0.5:
                        self.start_recording()
                    else:
                        if (time.time() - self.last_speech_time) > RESPONSE_WINDOW:
                            print("finished recording.")
                            self.orchestrator.end_event.set()



    
    def start_recording(self):
        print("now recording...")
        self.state = "RECORDING"
        self.audio_buffer = []
        self.last_speech_time = time.time()

    
    def stop_and_transcribe(self):
        if not self.audio_buffer:
            return
        
        print("transcribing...")
        audio = np.concatenate(self.audio_buffer, axis=0)

        try:
            transcript = self.transcribe(audio)
            print(f"User: {transcript.strip()}\n")
            self.orchestrator.process_turn_from_text(transcript)
        except Exception as e:
            print("transcription ERROR:", e, file=sys.stderr)
        finally:
            self.state = "LISTENING"

    
    def transcribe(self, audio):
        segments, info = self.whisper.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=False
        )

        text = " ".join(seg.text.strip() for seg in segments)
        return text