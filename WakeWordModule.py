from openwakeword.model import Model
from openwakeword.utils import download_models
import numpy as np


class WakeWordModule:

    def __init__(self, settings):
        self.WAKE_WORD = settings.ww_engine.get('wake_word')
        print(self.WAKE_WORD)
        self.WW_THRESHOLD = settings.ww_engine.get('ww_threshold')

        download_models([self.WAKE_WORD])
        
        print("Loading OpenWakeWord...")
        self.oww = Model(wakeword_models=[self.WAKE_WORD], inference_framework="onnx")


    def reset(self):
        self.oww = Model(wakeword_models=[self.WAKE_WORD], inference_framework="onnx")

    def is_wake_word(self, chunk1d):
        probs = self.oww.predict((chunk1d * 32767).astype(np.int16))
        ww_prob = float(probs.get(self.WAKE_WORD, 0.0))

        if ww_prob > self.WW_THRESHOLD:
            print(f"Wake word detected. confidence {ww_prob}")
            return True
        else:
            return False
