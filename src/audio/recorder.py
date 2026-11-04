import sounddevice as sd
import numpy as np
from src.audio import config

def record_audio(duration_sec: float, sample_rate: int = config.SAMPLE_RATE, channels: int = config.CHANNELS, dtype=config.DTYPE) -> np.ndarray:
    """
    Record audio from the default microphone.
    
    Args:
        duration_sec (float): Duration of the recording in seconds.
        sample_rate (int): Sampling rate in Hz.
        channels (int): Number of audio channels.
        dtype: NumPy data type for the samples.
        
    Returns:
        np.ndarray: The recorded audio data as a NumPy array.
    """
    print(f"Recording for {duration_sec} seconds...")
    # sd.rec is a non-blocking call, so we use sd.wait() to wait until recording is finished
    recording = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=channels, dtype=dtype)
    sd.wait()
    print("Recording complete.")
    return recording
