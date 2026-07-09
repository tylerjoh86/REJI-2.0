from silero_vad import load_silero_vad
import numpy as np
import torch

class SeleroVADModule:

    def __init__(self):
        self.SAMPLE_RATE = 16000
        self.speech_confidence_threshold = 0.5

        print("Loading Silero VAD...")
        self.vad_model = load_silero_vad()

    def is_speech(self, chunk1d):
        chunk_tensor = torch.from_numpy(chunk1d)
        speech_confidence = self.vad_model(chunk_tensor, self.SAMPLE_RATE)

        if speech_confidence > self.speech_confidence_threshold:
            return True
        else:
            return False