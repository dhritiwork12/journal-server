import os
import json
import wave
import struct
import asyncio
from typing import List
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

SAMPLE_RATE = 8000       # must match ESP32
CHUNK_BYTES = 96000      # transcribe every ~2 seconds of audio

# ─────────────────────────────────────────────
# Write proper WAV file from raw PCM bytes
# ─────────────────────────────────────────────
def write_wav(filename: str, pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

# ─────────────────────────────────────────────
# Transcribe a WAV file using Groq Whisper
# ─────────────────────────────────────────────
def transcribe_audio(filename: str) -> str:
    try:
        from deepgram import DeepgramClient, PrerecordedOptions
        dg = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
        with open(filename, "rb") as f:
            buffer_data = f.read()
        payload = {"buffer": buffer_data}
        options = PrerecordedOptions(
            model="nova-2",
            language="en",
            punctuate=True,
        )
        response = dg.listen.prerecorded.v("1").transcribe_file(payload, options)
        return response.results.channels[0].alternatives[0].transcript
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""
# ─────────────────────────────────────────────
# WebSocket: Real-time transcription
# ESP32 sends raw 16-bit PCM at 8kHz
# We transcribe every CHUNK_BYTES and send back
# ─────────────────────────────────────────────
import os
import json
import wave
import asyncio
import requests
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

SAMPLE_RATE = 16000
CHUNK_BYTES = 96000

def write_wav(filename: str, pcm_bytes: bytes):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

def transcribe_audio(filename: str) -> str:
    try:
        with open(filename, "rb") as f:
            audio_data = f.read()
        response = requests.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true&language=en",
            headers={
                "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
                "Content-Type": "audio/wav"
            },
            data=audio_data
        )
        result = response.json()
        return result["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

@app.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("ESP32 connected")

    audio_buffer = bytearray()

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)

            if len(audio_buffer) >= CHUNK_BYTES:
                chunk = bytes(audio_buffer)
                audio_buffer.clear()

                try:
                    write_wav("temp_chunk.wav", chunk)
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(None, transcribe_audio, "temp_chunk.wav")
                    if text and len(text.split()) > 2:
                        print(f"[partial] {text}")
                        await websocket.send_json({
                            "transcript": text,
                            "is_final": False
                        })
                except Exception as e:
                    print(f"Chunk error: {e}")

    except WebSocketDisconnect:
        print("ESP32 disconnected")
        if len(audio_buffer) > 1000:
            try:
                write_wav("temp_final.wav", bytes(audio_buffer))
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, transcribe_audio, "temp_final.wav")
                if text:
                    print(f"Final: {text}")
            except Exception as e:
                print(f"Final error: {e}")

@app.post("/journal")
async def create_journal(transcript: dict):
    text = transcript.get("text", "")
    date = transcript.get("date", datetime.now().strftime("%Y-%m-%d"))

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a journal assistant. Turn the user's voice transcript into a title, coherent journal body, and 3 key themes. Respond ONLY in JSON format like: {\"title\": \"...\", \"body\": \"...\", \"themes\": [\"...\", \"...\", \"...\"]}"
            },
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"}
    )

    structured = json.loads(completion.choices[0].message.content)

    entry = {
        "title": structured.get("title", "New Entry"),
        "body": structured.get("body", text),
        "themes": ", ".join(structured.get("themes", [])),
        "date": date,
        "time": datetime.now().strftime("%H:%M"),
        "transcript": text
    }

    supabase.table("entries").insert(entry).execute()
    print(f"Saved: {entry['title']}")
    return entry

@app.get("/entries")
async def get_entries(password: str = None):
    if password != os.getenv("JOURNAL_PASSWORD"):
        return {"error": "Unauthorized"}
    response = supabase.table("entries").select("*").order("date", desc=True).execute()
    return response.data

@app.get("/")
def root():
    return {"status": "Journal server running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)