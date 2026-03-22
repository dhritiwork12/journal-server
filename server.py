import asyncio
import json
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import websockets
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&model=nova-2"
    "&interim_results=true"
    "&punctuate=true"
)

@app.websocket("/transcribe")
async def transcribe(esp32: WebSocket):
    await esp32.accept()
    print("ESP32 connected")

    try:
        async with websockets.connect(
            DEEPGRAM_URL,
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        ) as deepgram:
            print("Connected to Deepgram")

            async def send_audio():
                try:
                    while True:
                        audio_chunk = await esp32.receive_bytes()
                        await deepgram.send(audio_chunk)
                except WebSocketDisconnect:
                    await deepgram.send(json.dumps({"type": "CloseStream"}))
                    print("ESP32 disconnected")

            async def receive_transcript():
                try:
                    async for message in deepgram:
                        data = json.loads(message)
                        if data.get("type") == "Results":
                            transcript = (
                                data["channel"]["alternatives"][0]["transcript"]
                            )
                            is_final = data["is_final"]
                            if transcript.strip():
                                payload = json.dumps({
                                    "transcript": transcript,
                                    "is_final": is_final
                                })
                                await esp32.send_text(payload)
                                print(f"{'[FINAL]' if is_final else '[partial]'} {transcript}")
                except Exception as e:
                    print(f"Deepgram error: {e}")

            await asyncio.gather(send_audio(), receive_transcript())

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        print("Session ended")


class JournalRequest(BaseModel):
    date: str
    entries: List[dict]

@app.post("/journal")
async def compile_journal(request: JournalRequest):
    stitched = ""
    for entry in request.entries:
        stitched += f"\n[{entry['time']}]\n{entry['transcript']}\n"

    prompt = f"""Here is everything I said on {request.date}, recorded throughout the day with timestamps:

{stitched}

Write a reflective, first-person journal entry based on this.
- Clean up any filler words or repetition
- Group related thoughts into paragraphs
- Keep my natural voice and tone
- Give the entry a short title
- At the end, add a Key Themes line with 3-5 words or short phrases

Format your response as JSON like this:
{{
  "title": "...",
  "body": "...",
  "themes": ["...", "...", "..."]
}}
Return only the JSON, nothing else."""

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )

    raw = response.choices[0].message.content

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

    print(f"Journal compiled for {request.date}: {result['title']}")
    return result


@app.get("/")
def root():
    return {"status": "Journal server running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)