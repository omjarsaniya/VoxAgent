"""
Step 1: Digital audio fundamentals + mic/speaker I/O
Run this on your own machine (needs a real mic/speaker) — not in a sandbox.

Install:  pip install sounddevice numpy

What this teaches, hands-on:
- Sample rate / bit depth / PCM
- Chunked (framed) audio capture — the pattern every downstream
  component (VAD, ASR, TTS playback) will reuse
- The exact byte math for buffer sizing
"""

import sounddevice as sd
import numpy as np

# --- Config: these constants are used across the ENTIRE project from here on ---
SAMPLE_RATE = 16000      # 16kHz: standard for speech models (Whisper, Silero VAD, Kokoro)
CHANNELS = 1              # mono: speech pipelines never need stereo
DTYPE = "int16"           # 16-bit PCM: standard bit depth for speech
CHUNK_MS = 30             # process audio in 30ms frames (matches Silero VAD's expected frame size)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # = 480 samples per chunk


def list_devices():
    """Always check this first on a new machine — device indices differ per system."""
    print(sd.query_devices())


def record_seconds(seconds: float) -> np.ndarray:
    """Records fixed-duration audio and returns it as a single int16 numpy array."""
    print(f"Recording {seconds}s... speak now.")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()  # blocks until recording finishes
    return audio.flatten()


def record_in_chunks(seconds: float):
    """
    This is the pattern real-time pipelines actually use: a callback fires
    every CHUNK_SAMPLES, handing you a small frame instead of one big blob.
    VAD, streaming ASR, and the WebSocket sender will all plug into this pattern.
    """
    chunks_collected = []

    def callback(indata, frames, time_info, status):
        if status:
            print("Stream status:", status)
        chunk = indata.copy().flatten()
        chunks_collected.append(chunk)
        # In the real pipeline: this is where you'd hand `chunk` to VAD.
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        print(f"chunk: {len(chunk)} samples, {chunk.nbytes} bytes, rms={rms:.1f}")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=CHUNK_SAMPLES,
        callback=callback,
    ):
        sd.sleep(int(seconds * 1000))

    return np.concatenate(chunks_collected)


def play(audio: np.ndarray):
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()


if __name__ == "__main__":
    print(f"Chunk size: {CHUNK_SAMPLES} samples = {CHUNK_SAMPLES * 2} bytes at {CHUNK_MS}ms\n")

    list_devices()

    audio = record_in_chunks(3.0)
    print(f"\nTotal captured: {len(audio)} samples = {audio.nbytes} bytes = {audio.nbytes / 1024:.1f} KB")

    print("\nPlaying back...")
    play(audio)
