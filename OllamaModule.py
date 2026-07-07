import requests
from ollama import chat
from ollama import ChatResponse
import json
import uuid

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "end_chat",
            "descriptoin": "Ends chat with user.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

class OllamaModule:
    

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.session = requests.Session()
        self.tts_queue = self.orchestrator.tts_queue

        self.history = []

        self.history.append({
            "role": "system",
            "content": "You are JARVIS, an AI voice assistant. Strictly talk in Text to speech compatible format. You are loosely based off of Reginald Jeeves"
        })

    def run_tool(self, name: str, args: dict):
        if name == "end_chat":
            print("shutting down")
            self.orchestrator.end_event.set()
            return "succesful"

        raise ValueError(f"unknown fuction: {name}")
          
   
    def process_response(self, transcript):
        
        self.history.append({
            "role": "user",
            "content": transcript
        })
        
        print("transcript received")
        
        #loops until model is finished calling tools
        while True:
            response: ChatResponse = chat(
                model='qwen3.6-27b-unlocked',
                messages=self.history,
                tools=TOOLS,
                stream=False,
                options={
                    "thinking": False
                }
            )

            message = response.message
            
            #prints REJI's response
            print("REJI: " + message.content)
            
            assistant_message = {
                "role": "assistant",
                "content": message.content or ""
            }
            if message.tool_calls:
                assistant_message["tool_calls"] = message.tool_calls

            self.history.append(assistant_message)
            
            if not message.tool_calls:
                break

            for i, tool_call in enumerate(message.tool_calls):
                func_name = tool_call["function"]["name"]
                args_raw = tool_call["function"].get("arguments", {})
                call_id = tool_call.get("id") or f"call_{i}_{uuid.uuid4().hex[:8]}"
                
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = args_raw or {}
                
                result = self.run_tool(func_name, args)

                self.history.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(result)
                })

        print("sending response to TTS...")

        for m in reversed(self.history):
            if m["role"] == "assistant" and not m.get("tool_calls"):
                final_text = (m.get("content") or "").strip()
                break
        self.tts_queue.put(final_text)       