from piper.voice import PiperVoice
import numpy as np

class PiperTTSModule:
    def __init__(self, settings):

        self.MODEL_PATH = settings.tts.get('model_path')

        self.voice = PiperVoice.load(self.MODEL_PATH)

    def load_audio(self, utterance):
        if not utterance is None:
            for audio_bytes in self.voice.synthesize_stream_raw(utterance):
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                return audio_chunk
        else:
            return None