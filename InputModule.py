import numpy as np
import sounddevice as sd
import time 
import threading
import queue

class InputModule:
    def __init__(self, settings):
        self.sample_rate = settings.audio.get('sample_rate')
        self.chunk_size = settings.audio.get('chunk_size')


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
                    ts = time.time()
                    # flatten to 1D float32
                    data = np.asarray(chunk).flatten()
                    self.raw_queue.put(data)