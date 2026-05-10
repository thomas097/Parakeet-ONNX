from dataclasses import dataclass

@dataclass(frozen=True)
class ParakeetConfig:
    sample_rate: int = 16000
    min_buffer_size: float = 0.5
    max_buffer_size: float = 8.0
    pre_encode_cache: int = 9
    frames_per_chunk: int = 16
    samples_per_chunk: int = 2560
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128
    pre_emphasis: float = 0.97
    log_zero_guard: float = 5.9604645e-8
    fmax: float = 8000.0
    max_symbols: int = 3