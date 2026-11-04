import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from typing import Tuple
from src.audio import config

SUPPORTED_SAMPLE_RATES = {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000}


class InvalidAudioError(ValueError):
    """Raised when an audio file or buffer is malformed, empty, or has unsupported format/sample rate."""
    pass


def save_wav(file_path: str, audio_data: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> None:
    """
    Save an audio NumPy array to a WAV file.
    
    Args:
        file_path (str): The path to save the .wav file.
        audio_data (np.ndarray): The audio data to save.
        sample_rate (int): The sample rate of the audio data.
    """
    wavfile.write(file_path, sample_rate, audio_data)


def load_wav(file_path: str) -> Tuple[int, np.ndarray]:
    """
    Load an audio NumPy array from a WAV file.
    
    Args:
        file_path (str): The path to the .wav file.
        
    Returns:
        Tuple[int, np.ndarray]: A tuple containing the sample rate and the audio data.
        
    Raises:
        InvalidAudioError: If the WAV file cannot be read or is malformed.
    """
    try:
        sample_rate, audio_data = wavfile.read(file_path)
    except Exception as e:
        raise InvalidAudioError(f"Malformed or unreadable WAV file: {e}") from e
    return sample_rate, audio_data


def validate_and_normalize_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    target_sr: int = config.SAMPLE_RATE
) -> Tuple[int, np.ndarray]:
    """
    Validates WAV audio properties (sample rate, channel count, bit depth/dtype, finite samples, non-empty)
    and resamples to target_sr (16kHz) using polyphase filtering when sample_rate != target_sr.

    Args:
        audio_data (np.ndarray): Audio samples loaded from WAV.
        sample_rate (int): Original sample rate in Hz.
        target_sr (int): Target sample rate in Hz (default 16000).

    Returns:
        Tuple[int, np.ndarray]: (target_sr, processed_1d_audio)
    """
    if not isinstance(sample_rate, (int, np.integer)) or int(sample_rate) not in SUPPORTED_SAMPLE_RATES:
        raise InvalidAudioError(f"Unsupported sample rate: {sample_rate} Hz. Supported rates: {sorted(SUPPORTED_SAMPLE_RATES)}")

    sample_rate = int(sample_rate)

    if not isinstance(audio_data, np.ndarray):
        raise InvalidAudioError("Audio data must be a numpy ndarray.")

    if audio_data.size == 0:
        raise InvalidAudioError("Audio data is empty.")

    # Validate dtype / bit depth before channel processing
    is_int16 = (audio_data.dtype == np.int16)
    is_float = np.issubdtype(audio_data.dtype, np.floating)
    if not is_int16 and not is_float:
        raise InvalidAudioError(
            f"Unsupported audio bit depth/dtype '{audio_data.dtype}'. Expected 16-bit PCM (int16) or normalized float."
        )

    if not np.all(np.isfinite(audio_data)):
        raise InvalidAudioError("Audio data contains NaN or infinite values.")

    # Validate channel count and convert stereo (2ch) to mono (1ch)
    if audio_data.ndim == 2:
        channels = audio_data.shape[1]
        if channels == 1:
            audio_1d = audio_data[:, 0]
        elif channels == 2:
            mono_float = np.mean(audio_data.astype(np.float64), axis=1)
            audio_1d = np.clip(np.round(mono_float), -32768, 32767).astype(np.int16) if is_int16 else mono_float.astype(audio_data.dtype)
        else:
            raise InvalidAudioError(f"Unsupported channel count: {channels}. Only mono (1) and stereo (2) are supported.")
    elif audio_data.ndim == 1:
        audio_1d = audio_data
    else:
        raise InvalidAudioError(f"Invalid audio array dimensions: {audio_data.ndim}D.")

    if len(audio_1d) == 0:
        raise InvalidAudioError("Audio data has zero samples.")

    if is_float and np.max(np.abs(audio_1d)) > 1.05:
        raise InvalidAudioError("Floating-point audio exceeds normalized [-1.0, 1.0] range.")

    # Resample to target_sr if needed using polyphase FIR resampling
    if sample_rate != target_sr:
        g = math.gcd(target_sr, sample_rate)
        up = target_sr // g
        down = sample_rate // g

        if is_int16:
            float_norm = audio_1d.astype(np.float64) / 32768.0
            resampled_float = resample_poly(float_norm, up, down)
            resampled_int16 = np.clip(np.round(resampled_float * 32768.0), -32768, 32767).astype(np.int16)
            return target_sr, resampled_int16
        else:
            resampled_float = resample_poly(audio_1d.astype(np.float64), up, down)
            resampled_float = np.clip(resampled_float, -1.0, 1.0).astype(np.float32)
            return target_sr, resampled_float

    return sample_rate, audio_1d
