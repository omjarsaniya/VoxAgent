"""
Step 3: Streaming (segment-based) ASR with faster-whisper
Run on your own machine (needs mic). Install: pip install faster-whisper

First run downloads the Whisper "small" model (~500MB, cached afterward under
your user profile) -- this happens automatically, just needs internet access
once.

What this teaches, hands-on:
- Combining Concept 2's VAD endpointing with actual transcription
- Why ASR inference must run on a WORKER THREAD, never inside the audio
  callback (reusing the queue pattern from Concept 1b -- callbacks must
  never block)
- Measuring real end-to-end latency: end-of-speech -> transcript ready,
  which is the number that actually determines how "fast" your agent feels
"""

import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512

# Hand-off queue: audio callback -> transcription worker thread.
# Same pattern as mic_out_queue in Concept 1b -- the callback must stay fast.
transcription_queue: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue()

vad_model = load_silero_vad()
vad_iterator = VADIterator(
    vad_model,
    sampling_rate=SAMPLE_RATE,
    threshold=0.5,
    min_silence_duration_ms=400,
    speech_pad_ms=100,
)

speech_buffer: list[np.ndarray] = []
in_speech = False


def audio_callback(indata, frames, time_info, status):
    """Runs on the real-time audio thread -- must stay fast. No ASR here."""
    global in_speech, speech_buffer

    chunk_int16 = indata.copy().flatten()
    float_chunk = chunk_int16.astype(np.float32) / 32768.0
    tensor_chunk = torch.from_numpy(float_chunk)
    event = vad_iterator(tensor_chunk, return_seconds=True)

    if in_speech:
        speech_buffer.append(chunk_int16)

    if event:
        if "start" in event:
            in_speech = True
            speech_buffer = [chunk_int16]
        if "end" in event:
            in_speech = False
            full_audio = np.concatenate(speech_buffer)
            speech_buffer = []
            # Hand off the whole utterance + a timestamp for latency measurement.
            # This is the ONLY thing the callback does with it -- no inference here.
            transcription_queue.put((full_audio, time.time()))


def transcription_worker():
    """Runs on its own thread. Safe to block/be slow here -- this is NOT the audio thread."""
    # If your connection is slow, try "tiny" (~75MB) or "base" (~145MB) first
    # to validate the pipeline quickly, then switch back to "small" (~500MB)
    # for real accuracy once you're not in a hurry.
    MODEL_SIZE = "base"
    print(f"Loading Whisper '{MODEL_SIZE}' model (int8)... first run downloads weights.")
    asr_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Whisper ready.\n")

    while True:
        audio_int16, speech_ended_at = transcription_queue.get()
        audio_float = audio_int16.astype(np.float32) / 32768.0

        t0 = time.time()
        # beam_size=1 = greedy decoding: faster, slightly less accurate than
        # the default beam_size=5. For real-time use, speed wins this tradeoff.
        segments, info = asr_model.transcribe(audio_float, language="en", beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments)
        t1 = time.time()

        duration_s = len(audio_int16) / SAMPLE_RATE
        compute_ms = (t1 - t0) * 1000
        end_to_end_ms = (t1 - speech_ended_at) * 1000

        print(f'>>> "{text.strip()}"')
        print(
            f"    audio: {duration_s:.2f}s | ASR compute: {compute_ms:.0f}ms | "
            f"speech-end -> transcript ready: {end_to_end_ms:.0f}ms\n"
        )


if __name__ == "__main__":
    threading.Thread(target=transcription_worker, daemon=True).start()

    print("Mic is live now (VAD is already running).")
    print("Whisper is loading/downloading in the background -- transcripts")
    print("won't print until it's ready, but nothing will time out while you wait.")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=VAD_CHUNK_SAMPLES,
            callback=audio_callback,
        ):
            while True:
                sd.sleep(1000)  # loop indefinitely instead of a fixed countdown
    except KeyboardInterrupt:
        print("\nStopped.")