import sys, os
sys.path.append(os.getcwd())

from src import ParakeetEOUModel
from src.utils import AudioBuffer, AudioReplayer

# Load quantized model and tokenizer
parakeet = ParakeetEOUModel.from_pretrained(
    path="checkpoints/parakeet-realtime-eou",
    device="cpu",
    quant="uint8")

# Prepare recording device
buffer = AudioBuffer()
replayer = AudioReplayer(
    buffer=buffer,
    filepath="examples/data/placatus.wav",
    samplerate=16000,
    channels=1,
    dtype="float32",
    chunk_size=2560) # 160ms

replayer.start()

# Process in 160ms chunks for streaming
text_output = ""
while not replayer.is_done():
    chunks = buffer.get_contents(clear=True)
    for chunk in chunks:
        text = parakeet.transcribe(chunk)

        if text == "":
            text = " _"

        print(text, end="", flush=True)
        text_output += text

        if "[EOU]" in text:
            print() # Newline
