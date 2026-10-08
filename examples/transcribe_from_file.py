import sys, os
sys.path.append(os.getcwd())

from src import TranscriberWithEouModel, AudioBuffer, AudioReplayer

# Load quantized model and tokenizer
parakeet = TranscriberWithEouModel.from_pretrained(
    path="checkpoints/parakeet-realtime-eou",
    device="cpu",
    quant="uint8")

# Prepare streaming replayer
buffer = AudioBuffer()
replayer = AudioReplayer(
    buffer=buffer,
    filepath="examples/data/placatus.wav",
    samplerate=16000,
    channels=1,
    dtype="float32",
    chunk_size=2560) # 160ms

replayer.start()

# Process in 160ms frames for streaming
text_output = ""
while not replayer.is_done():
    frames = buffer.get_contents(clear=True)
    for frame in frames:
        text = parakeet.transcribe(frame)

        if text == "":
            text = " _"

        print(text, end="", flush=True)
        text_output += text

        if "[EOU]" in text:
            print() # Newline
