from Orchestrator import orchestrator
from AppConfig import AppConfig
import OllamaModule
import InputModule
import PiperTTSModule
import WakeWordModule
import SeleroVADModule
import FasterWhiperModule

if __name__ == "__main__":
    print("starting...")
    settings = AppConfig()
    settings.load("config.yaml")       

    orchestrator  = orchestrator(
        settings     =settings,
        tts          =PiperTTSModule.PiperTTSModule,
        audioInput   =InputModule.InputModule,
        llm          =OllamaModule.OllamaModule,
        ww_engine    =WakeWordModule.WakeWordModule,
        vad          =SeleroVADModule.SeleroVADModule,
        stt          =FasterWhiperModule.FasterWhisperModule
    )

    orchestrator.start()