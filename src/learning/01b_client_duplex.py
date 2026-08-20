"""
Step 1b: Full-duplex client audio I/O
Run on your own machine (needs mic + speaker). Ideally use headphones while
testing — you have no AEC yet, so without headphones the mic will hear the
speaker and you'll see it in the printed logs.

Install: pip install sounddevice numpy

What this teaches:
- Capture and playback running CONCURRENTLY (not record-then-play)
- Queue-based decoupling: audio callbacks must never block on slow work
- The exact seam where the WebSocket sender/receiver will plug in (Concept 4)
- A naive echo demonstration you can literally hear, motivating AEC
"""

import queue
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_MS = 30
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 480 samples

# Outbound: mic -> (would go to WebSocket sender in Concept 4)
mic_out_queue: "queue.Queue[np.ndarray]" = queue.Queue()

# Inbound: (would arrive from WebSocket receiver / TTS in Concept 7) -> speaker
# This is the "jitter buffer" — it absorbs uneven delivery timing so playback
# stays gapless even if chunks arrive irregularly over the network.
playback_queue: "queue.Queue[np.ndarray]" = queue.Queue()


def mic_callback(indata, frames, time_info, status):
    """Fires every CHUNK_MS. Must be fast — never do network/disk I/O here directly."""
    if status:
        print("input status:", status)
    chunk = indata.copy().flatten()
    mic_out_queue.put_nowait(chunk)  # hand off immediately, don't block the callback


def speaker_callback(outdata, frames, time_info, status):
    """Pulls from the jitter buffer. If it's empty, play silence (underrun) instead of blocking."""
    if status:
        print("output status:", status)
    try:
        chunk = playback_queue.get_nowait()
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames - len(chunk)))
        outdata[:, 0] = chunk[:frames]
    except queue.Empty:
        outdata.fill(0)  # silence — this is what a network stall sounds like


def network_sender_stub():
    """
    Stand-in for Concept 4's WebSocket sender. Right now it just drains
    mic_out_queue and reports throughput — this is where ws.send(chunk.tobytes())
    will go once the server exists.
    """
    total_bytes = 0
    start = time.time()
    while True:
        chunk = mic_out_queue.get()
        total_bytes += chunk.nbytes
        elapsed = time.time() - start
        if elapsed > 1.0:
            kbps = (total_bytes * 8 / 1024) / elapsed
            print(f"[mic->network] {kbps:.1f} kbps effective")
            total_bytes = 0
            start = time.time()


def demo_local_loopback(seconds: float):
    """
    Feeds captured mic audio straight into the playback queue, with a small
    artificial delay -- simulating what an echo would sound/behave like if
    you had a real server round-trip. This is ONLY for understanding the
    duplex + jitter buffer mechanics. Real TTS audio replaces this in Concept 7.
    """
    def delayed_feed():
        while True:
            chunk = mic_out_queue.get()
            time.sleep(0.15)  # pretend "network + inference" latency
            playback_queue.put_nowait(chunk)

    threading.Thread(target=delayed_feed, daemon=True).start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
        blocksize=CHUNK_SAMPLES, callback=mic_callback,
    ), sd.OutputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
        blocksize=CHUNK_SAMPLES, callback=speaker_callback,
    ):
        print(f"Full-duplex running for {seconds}s. Speak — you'll hear yourself ~150ms delayed.")
        print("(This delayed self-echo is exactly the signal AEC has to cancel later.)")
        sd.sleep(int(seconds * 1000))


if __name__ == "__main__":
    threading.Thread(target=network_sender_stub, daemon=True).start()
    demo_local_loopback(8.0)
