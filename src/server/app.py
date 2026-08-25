"""
Step 4: WebSocket server -- the networked "nervous system" every future
concept (agent, RAG, TTS, barge-in) will build on top of.

Run from the VoxAgent ROOT folder (not inside src/):
    uvicorn src.server.app:app --port 8000

What this teaches, hands-on:
- Async WebSocket handling with FastAPI
- Why the Whisper call is explicitly offloaded to a thread pool
  (run_in_executor) instead of awaited directly -- so one client's slow
  transcription can't freeze every other connected client
- Binary frames (raw PCM audio) vs text/JSON frames (control messages/
  events) over the SAME WebSocket connection
- Per-connection session isolation using VoiceSession from session.py
"""

import asyncio
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

from src.server.session import VoiceSession, SAMPLE_RATE

app = FastAPI()

# Loaded ONCE at server startup, shared across every connected client.
# Unlike VoiceSession, this is safe to share: transcribe() takes audio in
# and returns text out with no memory of previous calls.
MODEL_SIZE = "base"  # bump to "small" once you're on a faster connection
print(f"Loading Whisper '{MODEL_SIZE}' model (shared across all sessions)...")
asr_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Whisper ready. Server accepting connections.")


def blocking_transcribe(audio_int16) -> str:
    """
    Plain, synchronous, blocking function -- deliberately NOT async.
    This is what gets handed to run_in_executor below: it's fine for this
    to block for hundreds of milliseconds because it runs on a separate
    worker thread, not on the async event loop.
    """
    audio_float = audio_int16.astype("float32") / 32768.0
    segments, _ = asr_model.transcribe(audio_float, language="en", beam_size=1)
    return " ".join(seg.text.strip() for seg in segments)


@app.websocket("/ws")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()
    session = VoiceSession()          # fresh, isolated state for THIS client
    loop = asyncio.get_event_loop()
    print("Client connected.")

    try:
        while True:
            # Binary frame in: one 1024-byte raw PCM audio chunk from the client.
            chunk_bytes = await websocket.receive_bytes()

            # VAD is cheap (~1ms) -- safe to run inline without an executor.
            # Whisper is NOT cheap -- that's the one that gets offloaded below.
            completed_utterance = session.process_chunk(chunk_bytes)

            if completed_utterance is not None:
                # Text/JSON frame out: a control event, not audio.
                await websocket.send_json({
                    "type": "speech_end",
                    "duration_s": round(len(completed_utterance) / SAMPLE_RATE, 2),
                })

                t0 = time.time()
                # THE key line: run_in_executor hands blocking_transcribe to
                # a background thread pool. The event loop stays free to
                # keep receiving audio from THIS client and serving every
                # OTHER connected client while this transcription runs.
                text = await loop.run_in_executor(None, blocking_transcribe, completed_utterance)
                latency_ms = round((time.time() - t0) * 1000)

                await websocket.send_json({
                    "type": "transcript",
                    "text": text,
                    "asr_ms": latency_ms,
                })

    except WebSocketDisconnect:
        print("Client disconnected.")