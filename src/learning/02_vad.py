"""
Step 2: Voice Activity Detection with Silero VAD
Run on your own machine (needs mic). Install: pip install silero-vad torch

What this teaches, hands-on:
- Real neural VAD vs the naive RMS threshold from Concept 1
- Matching a model's exact input contract (512-sample windows at 16kHz)
- Endpointing: turning per-frame probabilities into clean speech-start /
  speech-end EVENTS using built-in hysteresis (min_silence_duration_ms)
- This is the exact signal that will later drive the turn-taking state
  machine (Concept 8) and barge-in (Concept 9)
"""

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad, VADIterator

SAMPLE_RATE = 16000

# IMPORTANT: Silero VAD requires EXACTLY 512 samples per window at 16kHz (32ms).
# This is different from Concept 1's 480-sample (30ms) chunk -- a real example
# of matching a model's contract rather than reusing a "close enough" constant.
VAD_CHUNK_SAMPLES = 512

model = load_silero_vad()

vad_iterator = VADIterator(
    model,
    sampling_rate=SAMPLE_RATE,
    threshold=0.5,               # confidence cutoff for "this window is speech"
    min_silence_duration_ms=400,  # sustained silence required before declaring speech ENDED
    speech_pad_ms=100,            # pad detected boundaries so words aren't clipped
)


def audio_callback(indata, frames, time_info, status):
    if status:
        print("status:", status)

    # Silero expects float32 in [-1, 1], not int16 -- this conversion matters,
    # feeding raw int16 values in would silently produce garbage predictions.
    int16_chunk = indata.copy().flatten()
    float_chunk = int16_chunk.astype(np.float32) / 32768.0
    tensor_chunk = torch.from_numpy(float_chunk)

    # VADIterator maintains internal state across calls -- it's not stateless
    # per-frame classification, it's a running endpointing state machine.
    event = vad_iterator(tensor_chunk, return_seconds=True)

    if event:
        if "start" in event:
            print(f"SPEECH STARTED  at {event['start']:.2f}s")
        if "end" in event:
            print(f"SPEECH ENDED    at {event['end']:.2f}s")


if __name__ == "__main__":
    print(f"VAD window: {VAD_CHUNK_SAMPLES} samples = {VAD_CHUNK_SAMPLES / SAMPLE_RATE * 1000:.0f}ms")
    print("Listening for 20s. Speak a sentence, pause, speak again...\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=VAD_CHUNK_SAMPLES,
        callback=audio_callback,
    ):
        sd.sleep(20000)

    vad_iterator.reset_states()  # always reset between sessions -- state persists otherwise