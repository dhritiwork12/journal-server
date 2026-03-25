import os
import json
import base64
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

# Initialize App
app = FastAPI()

# Enable CORS for your GitHub Pages dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Data Models
class JournalEntry(BaseModel):
    title: str
    body: str
    themes: List[str]

# ─────────────────────────────────────────────
# WebSocket: Real-time Transcription
# ─────────────────────────────────────────────
@app.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = bytearray()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_data.extend(data)
            
            # Every 3 seconds of audio (~96kb), get a partial transcript
            if len(audio_data) > 96000:
                # Save temp file for Groq
                with open("temp.wav", "wb") as f:
                    f.write(audio_data)
                
                with open("temp.wav", "rb") as f:
                    transcription = groq.audio.transcriptions.create(
                        file=("temp.wav", f.read()),
                        model="whisper-large-v3",
                        response_format="json",
                    )
                
                await websocket.send_json({
                    "transcript": transcription.text,
                    "is_final": False
                })
    except WebSocketDisconnect:
        print("Client disconnected")

# ─────────────────────────────────────────────
# POST /journal
# Process transcript and save to Supabase
# ─────────────────────────────────────────────
@app.post("/journal")
async def create_journal(transcript: dict):
    text = transcript.get("text", "")
    
    # Use Groq to Structure the Entry
    completion = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful journal assistant. Summarize the user's voice transcript into a title, a coherent journal body, and 3 key themes. Respond ONLY in JSON format."},
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"}
    )
    
    structured_data = json.loads(completion.choices[0].message.content)
    
    # Save to Supabase
    entry_to_save = {
        "title": structured_data.get("title", "New Entry"),
        "body": structured_data.get("body", text),
        "themes": ", ".join(structured_data.get("themes", [])),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    supabase.table("entries").insert(entry_to_save).execute()
    return entry_to_save

# ─────────────────────────────────────────────
# GET /entries
# Get all journal entries (SECURE)
# ─────────────────────────────────────────────
@app.get("/entries")
async def get_entries(password: str = None):
    # Security Check: Compare URL password with Render Environment Variable
    if password != os.getenv("JOURNAL_PASSWORD"):
        return {"error": "Unauthorized"}
        
    response = supabase.table("entries").select("*").order("date", desc=True).execute()
    return response.data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)