# ⚡ Streaming STT

A lightweight, low-latency streaming speech-to-text (STT) engine, powered by [NVIDIA's Parakeet Realtime EOU-120M](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) model and the [ONNX Runtime](https://onnxruntime.ai/).

The project provides a stateful, chunk-based transcription pipeline designed for **real-time voice applications**, with built-in **end-of-utterance (EOU) detection** and optional **uint8 quantization** for efficient CPU inference.

#### Key features

* ⚡ Ultra-low latency streaming transcription — CPU-optimized streaming inference
* 🧩 Pure ONNX Runtime inference — no PyTorch or NVIDIA NeMo runtime required
* 🔄 Stateful streaming pipeline — feed audio chunks continuously without manually managing model state
* ✋ End-of-utterance (EOU) detection for interactive voice applications
* 📦 Cross-platform ONNX deployment with optional uint8 quantization

## 🚀 Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/thomas097/parakeet-ONNX.git
cd parakeet-onnx
```

2. Create and synchronize the `uv` virtual environment:

```bash
uv sync
```

### Download the model

Download the optimized ONNX model and tokenizer:

```bash
cd checkpoints/parakeet-realtime-eou

wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/decoder_joint.onnx
wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/encoder.onnx
wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/tokenizer.json
```

### Optional: uint8 quantization

For CPU deployments, uint8 quantization is recommended to reduce memory usage and improve inference performance.

From the project root, run:

```bash
python scripts/quantize_onnx_partial_uint8.py
```

When prompted for a model path, provide the path relative to the project root.

For example:

```text
checkpoints/parakeet-realtime-eou/encoder.onnx
```

## 🧪 Dependencies

The project has been tested extensively with:

```text
tokenizers==0.19.1
sounddevice==0.5.1
numpy==1.25.2
scipy==1.10.1
onnxruntime==1.19.2

# Only required for uint8 quantization
onnx==1.20.0
onnxruntime-tools==1.7.0
```

## Usage

The transcription engine accepts 16 kHz audio in 160ms chunks (2560 samples) and maintains the streaming state internally.

```python
from src import TranscriberWithEouModel

# Load the quantized model
transcriber = TranscriberWithEouModel.from_pretrained(
    path="checkpoints/parakeet-realtime-eou",
    device="cpu",
    quant="uint8",  # or None
)

# Audio chunks at 16 kHz.
# The reference implementation uses 160 ms chunks
# (2560 samples per chunk).
audio = ...

for chunk in audio:
    new_tokens = transcriber.transcribe(chunk)
    print(new_tokens)
```

The transcription state is maintained automatically, so callers only need to provide successive audio chunks.

## Examples
### 🎙️ Live transcription

Capture audio directly from the default microphone:

```bash
python transcribe_from_mic.py
```

The application continuously captures audio and emits transcription tokens as they become available.

### 📁 Offline / file streaming

Stream audio from a file through the same real-time pipeline:

```bash
python transcribe_from_file.py
```

Rather than performing traditional batch transcription, the example feeds audio into the engine chunk-by-chunk and emits results as they become available.

## ⚙️ Architecture

The system consists of a small streaming inference pipeline built around an optimized ONNX representation of the Parakeet Realtime EOU model.

```text
Audio
  │
  ▼
┌─────────────────────┐
│ Audio preprocessing │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ONNX Encoder        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Streaming Decoder   │
│ + EOU detection     │
└──────────┬──────────┘
           │
           ▼
     Transcription
```

The underlying model is derived from `NVIDIA Parakeet Realtime EOU-120M v1`, but the runtime pipeline is implemented independently using ONNX Runtime.

This allows the transcription engine to run without loading the original PyTorch/NeMo inference stack.

## 🙏 Attribution

This project builds on the work of:

* **NVIDIA** — Parakeet Realtime EOU-120M model and research
* **ONNX Runtime** — High-performance cross-platform inference runtime

The underlying Parakeet model is provided by NVIDIA. All rights to the original model remain with NVIDIA.

## 📄 License

The source code in this repository is distributed under the **Apache 2.0 License**.

The Parakeet Realtime EOU 120M-v1 model itself is governed by **NVIDIA's Open Model License**.

For details, see `LICENSE-NVIDIA-OPEN-MODEL` or the:

[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/?utm_source=chatgpt.com)
