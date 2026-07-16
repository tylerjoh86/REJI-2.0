import requests
from ollama import chat
from ollama import ChatResponse
import json
import uuid
import re

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
    

    def __init__(self, settings, output_queue, end_event):
        self.model = settings.llm.get('model')
        
        self.session = requests.Session()
        self.tts_queue = output_queue
        self.end_event = end_event

        self.clear_history()

        self.tts_buffer = ""

    def clear_history(self):
        self.history = [
            {
                "role": "system",
                "content": (
                    "You are JARVIS, an AI voice assistant. Strictly talk in Text to speech compatible format. "
                    "You are loosely based off of Reginald Jeeves."
                    "Do not creating long monologueing responses as they will likely tire the user"
                )
            }
        ]   

    def run_tool(self, name: str, args: dict):
        if name == "end_chat":
            self.end_event.set()
            return "succesful"

        raise ValueError(f"unknown fuction: {name}")
    
    def _stream_to_tts(self, tokens: str):
        """
        Appends incoming tokens to a buffer and pushes complete chunks
        to the TTS queue as soon as safe boundaries appear.
        """
        if not tokens or not tokens.strip():
            return

        self.tts_buffer += tokens

        # Split on sentence-ending punctuation with space after
        parts = re.split(r'(?<=[.!?;])\s+', self.tts_buffer)

        if len(parts) == 1:
            # No complete boundary yet; keep buffering unless it gets long
            if len(self.tts_buffer.strip()) > 300:
                self.tts_queue.put(self.tts_buffer.strip())
                self.tts_buffer = ""
            return

        # All but last are likely complete utterances
        complete_parts = parts[:-1]
        remainder = parts[-1].strip()

        for p in complete_parts:
            if p.strip():
                self.tts_queue.put(p.strip())

        self.tts_buffer = remainder
          
   
    def process_response(self, transcript):
        
        self.history.append({
            "role": "user",
            "content": transcript
        })
        
        print("Ollama: transcript received. streaming response...")

        final_text = ""
        
        #loops until model is finished calling tools
        while True:
            turn_content = ""
            tool_calls_in_turn = []

            response = chat(
                model='qwen3.5:9b',
                messages=self.history,
                tools=TOOLS,
                stream=True,
                options={"thinking": False}
            )
            for chunk in response:
                msg = chunk.message

                # Stream text content to TTS buffer immediately
                if msg.content:
                    turn_content += msg.content
                    self._stream_to_tts(msg.content)

                # Collect tool calls from this turn
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_in_turn.append(tc)
            
            #prints REJI's response
            print("REJI: " + turn_content)
            
            assistant_message = {
                "role": "assistant",
                "content": turn_content or ""
            }
            if tool_calls_in_turn:
                assistant_message["tool_calls"] = tool_calls_in_turn

            self.history.append(assistant_message)

            final_text += turn_content
            
            if not tool_calls_in_turn:
                break

            for i, tool_call in enumerate(tool_calls_in_turn):
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

        if self.tts_buffer.strip():
            self.tts_queue.put(self.tts_buffer.strip())
            self.tts_buffer = ""

        print("Ollama: streaming complete.")