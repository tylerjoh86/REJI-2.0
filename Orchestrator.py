import threading
import sounddevice as sd
import numpy as np
import queue
import OllamaModule
import InputModule
import PiperTTSModule
import WakeWordModule
import SeleroVADModule
import FasterWhiperModule
import time
from enum import Enum
import time
from pynput import keyboard

SILENCE_TIMEOUT = 1.2
RESPONSE_WINDOW = 5.0

class State(Enum):
    LISTENING     = "LISTENING"
    WAITING       = "WAITING"
    RECORDING     = "RECORDING"
    TRANSCRIBING  = "TRANSCRIBING"
    SPEAKING      = "SPEAKING"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    RESETING      = "RESETING"

class orchestrator:
    def __init__(self):
        #queues
        self.tts_queue     = queue.Queue()
        self.audio_queue   = queue.Queue()
        
        #threading events
        self.end_event              = threading.Event()
        self.audio_finished_event   = threading.Event()
        self.reset_event            = threading.Event()

        #modules
        self.tts           = PiperTTSModule.PiperTTSModule(self.audio_queue)
        self.audioInput    = InputModule.InputModule(self)
        self.llm           = OllamaModule.OllamaModule(self)
        self.ww_engine     = WakeWordModule.WakeWordModule()
        self.vad           = SeleroVADModule.SeleroVADModule()
        self.stt           = FasterWhiperModule.FasterWhisperModule()

        self.state = State.LISTENING
        self.last_speech_time = None
        self.audio_buffer = []
        self.saved_chunk = None

        
   
    def start(self):
        tts_t           = threading.Thread(target=self.tts_feeder, daemon=True)
        output_t        = threading.Thread(target=self.speaker_loop, daemon=True)
        control_t       = threading.Thread(target=self.control_loop, daemon=False)
        #text_input_t   = threading.Thread(target=self.input_loop, daemon=False)

        tts_t.start()
        output_t.start()      
        self.audioInput.start()
        control_t.start()

        #text_input_t.start()

        print(f"Listening for wakeword {self.ww_engine.WAKE_WORD}...")

    
    def control_loop(self):
        
        while True:
            if self.end_event.is_set():
                self.state = State.SHUTTING_DOWN

            if self.reset_event.is_set():
                self.state = State.RESETING
                self.reset_event.clear()

            if self.audio_finished_event.is_set():
                self.audio_finished_event.clear()
                self.state = State.WAITING

            elif self.state == State.LISTENING:
                audioChunk = self.audioInput.raw_queue.get()
                if self.ww_engine.is_wake_word(audioChunk):
                    self.state = State.WAITING
                    self.last_speech_time = time.time()
                    print("listening -> waiting")


            elif self.state == State.WAITING:
                audioChunk = self.audioInput.raw_queue.get()
                if self.vad.is_speech(audioChunk):
                    self.state = State.RECORDING
                    self.audio_buffer.append(audioChunk)
                    print("waiting -> recording")

                elif (time.time() - self.last_speech_time) > RESPONSE_WINDOW:
                    self.state = State.RESETING
                    print("waiting -> reseting")


            elif self.state == State.RECORDING:
                audioChunk = self.audioInput.raw_queue.get()
                self.audio_buffer.append(audioChunk)
                if self.vad.is_speech(audioChunk):
                    self.last_speech_time = time.time()
                elif (time.time() - self.last_speech_time) > SILENCE_TIMEOUT:
                    self.state = State.TRANSCRIBING
                    print("recording -> transcribing")
                    self.audioInput.pause()
                
            
            elif self.state == State.TRANSCRIBING:
                transcription = self.stt.transcribe(self.audio_buffer)
                self.audio_buffer = []
                
                if transcription is None:
                    print("orchestrator error, resetting")
                    self.state = State.RESETING
                    print("transcribing -> listening")
                    print(f"Listening for wakeword {self.ww_engine.WAKE_WORD}...")
                
                elif transcription == "":
                    self.state = State.RESETING
                    print("transcribing -> waiting")
                else:
                    self.state = State.SPEAKING
                    self.last_speech_time = time.time()
                    print("transcribing -> speaking")
                    

            elif self.state == State.SPEAKING:
                self.tts_queue.put(self.llm.process_response(transcription))
                self.tts_queue.put(None)
                self.audio_finished_event.wait()
                self.audio_finished_event.clear()
                self.state = State.WAITING
                self.last_speech_time = time.time()
                self.audioInput.resume()

            elif self.state == State.SHUTTING_DOWN:
                print("shutting down...")
                break

            elif self.state == State.RESETING:
                print("reseting...")
                self.llm.clear_history()
                while not self.audioInput.raw_queue.empty():
                    try:
                        self.audioInput.raw_queue.get_nowait()
                        self.audioInput.raw_queue.task_done()  
                    except queue.Empty:
                        break
                self.ww_engine = WakeWordModule.WakeWordModule()
                self.last_speech_time = None
                self.audio_buffer = []
                self.saved_chunk = None
                self.state = State.LISTENING
                print("reseting -> listening")

            time.sleep(0.01)



   
    def tts_feeder(self):
        while True:
            utterance = self.tts_queue.get()
            self.tts.load_audio(utterance)

    
    def speaker_loop(self):
        with sd.OutputStream(samplerate=16000, channels=1, dtype='int16') as stream:
            while True:
                audio_chunk = self.audio_queue.get()
                if not audio_chunk is None:
                    stream.write(audio_chunk)
                    self.audio_queue.task_done()
                else:
                    print("no audio chunks detected. returning to waiting mode")
                    self.audio_finished_event.set()

    
    def text_input_loop(self):
        while True:
            user_input = input("enter to exit\n")
            if user_input:
                self.end_event.set()





if __name__ == "__main__":
    print("starting...")
    orchestrator = orchestrator()
    orchestrator.start()