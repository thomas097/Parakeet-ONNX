import sys, os
sys.path.append(os.getcwd())

import numpy as np
from time import time_ns

from src import TranscriberWithEouModel, AudioBuffer, AudioReplayer

# Load quantized model and tokenizer
parakeet = TranscriberWithEouModel.from_pretrained(
    path="checkpoints/parakeet-realtime-eou",
    device="cpu",
    quant="uint8")

# Prefill buffer with audio
buffer = AudioBuffer()
replayer = AudioReplayer(
    buffer=buffer,
    filepath="examples/data/placatus.wav",
    samplerate=16000,
    channels=1,
    dtype="float32",
    chunk_size=2560) # 160ms
replayer.prefill()

frames = buffer.get_contents()
duration = buffer.size / 16000

# Measure runtimes
time_elapsed = []
transcript = ""

for _ in range(50):
    start_time = time_ns()
    transcript = ""

    for frame in frames:
        text = parakeet.transcribe(frame)
        transcript += text
        
    end_time = time_ns()

    time_elapsed.append(
        (end_time - start_time) * 1e-9
    )

print(f"Transcript: {transcript}")
print(f"Min:  {np.min(time_elapsed):.4f}")
print(f"Max:  {np.max(time_elapsed):.4f}")
print(f"Mean: {np.mean(time_elapsed):.4f}")
print(f"Std.: {np.std(time_elapsed):.4f}")
print(f"RTF: {np.mean(time_elapsed) / duration:.4f}")
