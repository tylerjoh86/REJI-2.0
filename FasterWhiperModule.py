from faster_whisper import WhisperModel
import sys
import numpy as np


WHISPER_MODEL_SIZE = "large-v3"    # Options: tiny, base, small, medium, large-v3
USE_GPU = True                   # Set True if you have CUDA; faster-whisper will use it automatically if available

class FasterWhisperModule:

    def __init__(self):
        print(f"Loading Faster-Whisper ({WHISPER_MODEL_SIZE})...")
        device = "cuda" if USE_GPU else "cpu"
        compute_type = "float16" if USE_GPU else "int8"
        self.whisper = WhisperModel(
            device=device,
            compute_type=compute_type,
            model_size_or_path=WHISPER_MODEL_SIZE
        )

    def transcribe(self, audio_buffer):
        if not audio_buffer:
            return ""
        
        print("transcribing...")
        audio = np.concatenate(audio_buffer, axis=0)

        try:
            segments, info = self.whisper.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=False
            )

            transcript = " ".join(seg.text.strip() for seg in segments)
            print(f"User: {transcript.strip()}\n")
            return transcript
        except Exception as e:
            print("transcription ERROR:", e, file=sys.stderr)
            return None