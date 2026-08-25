"""
Per-connection session state for the voice server.

Each WebSocket connection gets its own VoiceSession instance. This matters
because VAD is STATEFUL -- VADIterator remembers recent audio history to do
its hysteresis-based endpointing (Concept 2). If two users shared one
VADIterator, one person's silence could reset the other's in-progress
speech detection. The underlying Silero MODEL, by contrast, has no
per-call state -- it's just math -- so it's safe (and efficient) to load
once and share across every session.
"""

import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512  # int16 samples per chunk -> 1024 bytes on the wire

# Loaded once at import time, shared read-only across all sessions.
_vad_model = load_silero_vad()


class VoiceSession:
    """One of these is created per WebSocket connection (see app.py)."""

    def __init__(self):
        # A fresh, independent VADIterator per session -- this is what
        # keeps multiple simultaneous users from corrupting each other's
        # speech-start/speech-end state.
        self.vad_iterator = VADIterator(
            _vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.5,
            min_silence_duration_ms=400,
            speech_pad_ms=100,
        )
        self.speech_buffer: list[np.ndarray] = []
        self.in_speech = False

    def process_chunk(self, chunk_bytes: bytes):
        """
        Feed one 1024-byte (512-sample int16) chunk into this session's VAD.
        Returns a completed utterance (np.ndarray) the moment speech ends,
        otherwise None. Same logic as Concept 3's audio_callback, just
        moved into a class so it's isolated per connection.
        """
        chunk_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        float_chunk = chunk_int16.astype(np.float32) / 32768.0
        tensor_chunk = torch.from_numpy(float_chunk)

        event = self.vad_iterator(tensor_chunk, return_seconds=True)

        if self.in_speech:
            self.speech_buffer.append(chunk_int16)

        if event:
            if "start" in event:
                self.in_speech = True
                self.speech_buffer = [chunk_int16]
            if "end" in event:
                self.in_speech = False
                full_audio = np.concatenate(self.speech_buffer)
                self.speech_buffer = []
                return full_audio

        return None