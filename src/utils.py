import time
import threading
import librosa
import numpy as np
import soundfile as sf
import sounddevice as sd
from copy import deepcopy
from typing import Optional
from numpy.typing import NDArray

class AudioBuffer:
    """
    Thread-safe buffer for storing audio chunks.

    This class is designed to be used with real-time audio callbacks.
    Chunks can be appended incrementally and retrieved as NumPy arrays.
    """

    def __init__(self) -> None:
        """
        Initialize an empty AudioBuffer.

        Args:
            samplerate (int): Sample rate of the audio.
        """
        self._buffer = []
        self._lock = threading.Lock()

    def append(self, frame: NDArray) -> None:
        """
        Append a frame of audio data to the buffer.

        Args:
            frame (NDArray): Frame of audio samples with shape (frames, channels).
        """
        with self._lock:
            # Copy to avoid referencing internal buffer
            self._buffer.append(frame.copy())

    def clear(self) -> None:
        """
        Remove all audio data from the buffer.
        """
        with self._lock:
            self._buffer.clear()

    def get_contents(self, clear: bool = False) -> list[NDArray]:
        """
        Retrieve all buffered audio as a single NumPy array.
        Returns an empty array if no audio has been recorded.

        Args:
            clear (bool, optional): Whether to clear the buffer once contents are fetched.

        Returns:
            NDArray: Concatenated audio samples with shape (samples, channels)
        """
        with self._lock:
            contents = deepcopy(self._buffer)
            if clear:
                self._buffer = []  
            return contents
                         


class AudioRecorder:
    """
    Non-blocking audio recorder to stream microphone input
    into an AudioBuffer.

    Example:
    ```
    buffer = AudioBuffer()
    recorder = AudioRecorder(buffer)
    recorder.start()
    ...
    recorder.stop()
    ```
    """

    def __init__(
        self,
        buffer: AudioBuffer,
        samplerate: int = 16_000,
        channels: int = 1,
        dtype: str = "float32",
        chunk_size: int = 2560,
        device: Optional[int] = None,
    ) -> None:
        """
        Initialize an AudioRecorder instance. 

        Args:
            buffer (AudioBuffer):       Buffer to collect audio chunks.
            samplerate (int, optional): Sample rate in Hz. Defaults to 16000.
            channels (int, optional):   Number of input channels. Defaults to 1 (mono).
            dtype (str, optional):      Datatype of audio samples. Defaults to "float32".
            chunk_size (int, optional): Number of samples per audio chunk. Defaults to 1024.
            device (int, optional):     Input device index. Defaults to None (system default microphone).
        """
        self._buffer = buffer
        self._samplerate = samplerate
        self._channels = channels
        self._dtype = dtype
        self._chunk_size = chunk_size
        self._device = device

        self._stream: Optional[sd.InputStream] = None
        self._recording = False

    def _callback(self, indata: NDArray, *args, **kwargs) -> None:
        """
        Internal SoundDevice callback to augment the AudioBuffer.

        This function is called on a separate audio thread and must
        be fast and non-blocking.

        Args:
            indata (NDArray): Recorded audio data.
            *args, **kwargs: Not used.
        """
        self._buffer.append(indata)

    def start(self) -> None:
        """
        Start recording audio from the microphone.
        """
        if self._recording:
            return

        self._buffer.clear()
        self._recording = True

        self._stream = sd.InputStream(
            samplerate=self._samplerate,
            channels=self._channels,
            dtype=self._dtype,
            blocksize=self._chunk_size,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def is_recording(self) -> bool:
        """
        Tests whether the recorder is currently recording.

        Returns:
            bool: True when the recorder is running; False otherwise.
        """
        return self._recording

    def stop(self) -> None:
        """
        Stops recorder and closes the input stream.
        """
        if not self._recording:
            return

        self._recording = False
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class AudioReplayer:
    def __init__(
            self,
            buffer: AudioBuffer,
            filepath: str,
            samplerate=16000,
            channels=1,
            dtype="float32",
            chunk_size=2560
            ) -> None:
        """
        Initialize an AudioReplayer instance. 

        Args:
            buffer (AudioBuffer):       Buffer to collect audio chunks.
            filepath (str):             Path to audio file.
            samplerate (int, optional): Sample rate in Hz. Defaults to 16000.
            channels (int, optional):   Number of input channels. Defaults to 1 (mono).
            dtype (str, optional):      Datatype of audio samples. Defaults to "float32".
            chunk_size (int, optional): Number of samples per audio chunk. Defaults to 1024.
            device (int, optional):     Input device index. Defaults to None (system default microphone).
        """
        self.buffer = buffer
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.chunk_size = chunk_size

        print('ready!')
        self._data = self._load_audio_from_file(filepath)
        self._thread = None
        self._done = False

    def _load_audio_from_file(self, filepath: str) -> np.ndarray:
        data, sr = sf.read(filepath, dtype=self.dtype) #type:ignore

        # Ensure correct shape
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)

        # Convert to mono if needed
        if self.channels == 1 and data.shape[1] > 1:
            data = data[..., 0]

        # Resample
        if sr != self.samplerate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.samplerate, axis=0)

        return data
   
    def start(self):
        self._done = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        total_samples = self._data.shape[0]
        idx = 0

        print("Total samples:", total_samples)

        while idx < total_samples:
            frame = self._data[idx:idx + self.chunk_size]
            idx += self.chunk_size

            self.buffer.append(frame)

            # Simulate real-time streaming
            duration = len(frame) / self.samplerate
            time.sleep(duration)

        self._done = True

    def is_done(self):
        return self._done
