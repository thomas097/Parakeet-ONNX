import sys, os
sys.path.append(os.getcwd())

import time
from parakeet_eou import ParakeetEouModel, AudioBuffer, AudioRecorder

# Load quantized model and tokenizer
parakeet = ParakeetEouModel.from_pretrained(
    path="checkpoints/parakeet-realtime-eou",
    device="cpu",
    quant="uint8"
    )

# Prepare recording device
buffer = AudioBuffer()
recorder = AudioRecorder(
    buffer=buffer,
    samplerate=16000,
    channels=1,
    dtype="float32",
    chunk_size=2560 # 160ms
    )
recorder.start()

# Process in 160ms chunks for streaming
text_output = ""
while recorder.is_recording():
    frames = buffer.get_contents(clear=True)
    for frame in frames:
        text = parakeet.transcribe(frame)

        print(text, end="", flush=True)
        text_output += text

        if "[EOU]" in text:
            print()

        if "stop" in text:
            break

    time.sleep(0.05)

recorder.stop()
