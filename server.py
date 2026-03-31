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
CHUNK_BYTES = 32000      # transcribe every ~2 seconds of audio

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
        with open(filename, "rb") as f:
            result = groq_client.audio.transcriptions.create(
                file=(filename, f.read()),
                model="whisper-large-v3",
                response_format="json",
            )
        return result.text.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

# ─────────────────────────────────────────────
# WebSocket: Real-time transcription
# ESP32 sends raw 16-bit PCM at 8kHz
# We transcribe every CHUNK_BYTES and send back
# ─────────────────────────────────────────────
@app.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("ESP32 connected")

    audio_buffer = bytearray()
    full_transcript = ""

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)

            # Transcribe every ~2 seconds of audio
            if len(audio_buffer) >= CHUNK_BYTES:
                chunk = bytes(audio_buffer)
                audio_buffer.clear()

                # Write proper WAV and transcribe
                write_wav("temp_chunk.wav", chunk)
                text = transcribe_audio("temp_chunk.wav")

                if text:
                    full_transcript += " " + text
                    print(f"[partial] {text}")
                    await websocket.send_json({
                        "transcript": text,
                        "is_final": False
                    })

    except WebSocketDisconnect:
        print("ESP32 disconnected")

        # Transcribe any remaining audio
        if len(audio_buffer) > 1000:
            write_wav("temp_final.wav", bytes(audio_buffer))
            text = transcribe_audio("temp_final.wav")
            if text:
                full_transcript += " " + text

        print(f"Full transcript: {full_transcript}")

# ─────────────────────────────────────────────
# POST /journal
# Takes transcript text, generates journal entry
# ─────────────────────────────────────────────
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
        "date": date
    }

    supabase.table("entries").insert(entry).execute()
    print(f"Saved journal: {entry['title']}")
    return entry

# ─────────────────────────────────────────────
# GET /entries
# ─────────────────────────────────────────────
@app.get("/entries")
async def get_entries():
    response = supabase.table("entries").select("*").order("date", desc=True).execute()
    return response.data

# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Journal server running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)