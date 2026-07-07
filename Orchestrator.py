import threading
import sounddevice as sd
import numpy as np
import queue
import OllamaModule
import InputModule
import PiperTTSModule
import time

class orchestrator:
    def __init__(self):
        self.tts_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.tts = PiperTTSModule.PiperTTSModule(self.audio_queue)
        self.stt = InputModule.InputModule(self)
        self.llm = OllamaModule.OllamaModule(self)

        self.end_event = threading.Event()
        
   
    def start(self):
        tts_t = threading.Thread(target=self.tts_feeder, daemon=True)
        output_t = threading.Thread(target=self.speaker_loop, daemon=True)
        audio_input_t = threading.Thread(target=self.stt.control_loop, daemon=False)
        #text_input_t = threading.Thread(target=self.input_loop, daemon=False)

        tts_t.start()
        output_t.start()
        audio_input_t.start()
        #text_input_t.start()

    

    def process_turn_from_text(self, text):
        transcript = text
        print("sending transcript")
        self.llm.process_response(transcript)

   
    def tts_feeder(self):
        while True:
            utterance = self.tts_queue.get()
            self.tts.load_audio(utterance)

    
    def speaker_loop(self):
        with sd.OutputStream(samplerate=16000, channels=1, dtype='int16') as stream:
            while True:
                audio_chunk = self.audio_queue.get()
                stream.write(audio_chunk)
                self.audio_queue.task_done()

    
    def text_input_loop(self):
        user_input = ""
        while not self.end_event.is_set():
            user_input = ""
            user_input = input("say something to REJI...\n")
            self.process_turn_from_text(user_input)
            if (user_input == "/bye"):
                break



if __name__ == "__main__":
    print("starting...")
    orchestrator = orchestrator()
    orchestrator.start()