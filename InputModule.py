import numpy as np
import sounddevice as sd
import time 
import threading
import queue

class InputModule:
    def __init__(self, settings, audio_finished_event):
        self.audio_finished_event = audio_finished_event

        self.sample_rate = settings.audio.get('sample_rate')
        self.chunk_size = settings.audio.get('chunk_size')

        self.raw_queue = queue.Queue(maxsize=200)
        self.audio_out_queue = queue.Queue()

    def start(self):
        capture_t = threading.Thread(target=self.capture_loop, daemon=True)
        output_t = threading.Thread(target=self.speaker_loop, daemon=True)
        
        capture_t.start()
        output_t.start()
        
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
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.chunk_size
        ) as stream:
            while self.running:
                chunk, _ = stream.read(self.chunk_size)

                if self.is_paused.is_set():
                    continue

                if not self.raw_queue.full():
                    # flatten to 1D float32
                    data = np.asarray(chunk).flatten()
                    self.raw_queue.put(data)

    def add_to_queue(self, audio_chunk):
        self.audio_out_queue.put(audio_chunk)

    def speaker_loop(self):
        with sd.OutputStream(samplerate=16000, channels=1, dtype='int16') as stream:
            while True:
                audio_chunk = self.audio_out_queue.get()
                if not audio_chunk is None:
                    stream.write(audio_chunk)
                    self.audio_out_queue.task_done()
                else:
                    print("no audio chunks detected. returning to waiting mode")
                    self.audio_finished_event.set()