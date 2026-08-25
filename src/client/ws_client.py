"""
Step 4 client: captures mic audio and streams it to the VoxAgent server
over a WebSocket, printing whatever transcript events the server sends back.

Run the SERVER FIRST, in its own terminal, from the VoxAgent root:
    uvicorn src.server.app:app --port 8000

Then, in a SECOND terminal (also from VoxAgent root):
    python src\\client\\ws_client.py

Install: pip install websockets
"""

import asyncio
import json
import queue

import numpy as np
import sounddevice as sd
import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # must match the server's VAD contract exactly
SERVER_URL = "ws://localhost:8000/ws"

mic_queue: "queue.Queue[bytes]" = queue.Queue()


def mic_callback(indata, frames, time_info, status):
    """Same rule as every prior concept: stay fast, just hand off to a queue."""
    chunk_int16 = indata.copy().flatten()
    mic_queue.put_nowait(chunk_int16.tobytes())  # raw bytes -- what actually travels over the wire


async def sender(ws):
    """Continuously drains the mic queue and sends each chunk as a binary WS frame."""
    loop = asyncio.get_event_loop()
    while True:
        # queue.get() is a blocking/synchronous call. Running it via
        # run_in_executor keeps it from freezing this coroutine's event
        # loop, which also needs to run `receiver` concurrently below.
        chunk_bytes = await loop.run_in_executor(None, mic_queue.get)
        await ws.send(chunk_bytes)  # binary frame: raw audio


async def receiver(ws):
    """Listens for JSON control/event messages the server sends back."""
    async for message in ws:
        data = json.loads(message)
        if data["type"] == "speech_end":
            print(f"[speech ended, {data['duration_s']}s captured -- transcribing...]")
        elif data["type"] == "transcript":
            print(f'>>> "{data["text"]}"  (server ASR: {data["asr_ms"]}ms)\n')


async def main():
    print(f"Connecting to {SERVER_URL} ...")
    async with websockets.connect(SERVER_URL) as ws:
        print("Connected. Speak into your mic (Ctrl+C to stop).\n")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=mic_callback,
        ):
            # sender and receiver run CONCURRENTLY on the same connection --
            # this IS the duplex pattern from Concept 1b, now over a network.
            await asyncio.gather(sender(ws), receiver(ws))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")