import yaml

class AppConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance
    
    def load(self, yaml_path="config.yaml"):
        if self._loaded:
            return
        
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        self.audio = data.get("audio", {})
        self.stt = data.get("faster_whisper", {})
        self.tts = data.get("piper_tts", {})
        self.vad = data.get("vad", {})
        self.llm = data.get("ollama", {})
        self.ww_engine = data.get("open_wake_word", {})
        self.orchestration = data.get("orchestration", {})

        self._loaded = True

    

