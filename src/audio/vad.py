import numpy as np
import torch
from typing import Tuple, List, Dict
from src.audio.config import SAMPLE_RATE
from src.audio.exceptions import NoSpeechDetectedError

# Load the Silero VAD model globally so we don't reload it per function call
_model = None
_get_speech_timestamps = None

def _load_model():
    global _model, _get_speech_timestamps
    if _model is None:
        # Silero VAD requires torch.hub
        _model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )
        (_get_speech_timestamps, _, _, _, _) = utils

def detect_speech(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """
    Detect speech in an audio array and return the trimmed audio and speech timestamps.
    
    Args:
        audio_data (np.ndarray): The raw audio array (usually int16).
        sample_rate (int): The sample rate in Hz. Silero VAD expects 16000 Hz.
        
    Returns:
        Tuple[np.ndarray, List[Dict[str, int]]]: A tuple containing:
            - The trimmed audio array containing only the speech segments concatenated.
            - A list of dictionaries representing the speech timestamps: [{'start': idx, 'end': idx}]
            
    Raises:
        NoSpeechDetectedError: If the VAD model finds no speech in the audio.
    """
    if sample_rate != 16000:
        raise ValueError(f"Silero VAD expects a sample rate of 16000 Hz, got {sample_rate}")
        
    _load_model()
    
    # Silero VAD expects float32 tensor in range [-1.0, 1.0]
    if audio_data.dtype == np.int16:
        audio_float32 = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float32 = audio_data.astype(np.float32)
        
    # Convert to PyTorch tensor
    audio_tensor = torch.from_numpy(audio_float32)
    
    # Ensure it's a 1D tensor for Silero
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.squeeze()
        
    # Get speech timestamps
    speech_timestamps = _get_speech_timestamps(audio_tensor, _model, sampling_rate=sample_rate)
    
    if not speech_timestamps:
        raise NoSpeechDetectedError("No speech was detected in the provided audio.")
        
    # Trim the audio based on timestamps
    segments = []
    for ts in speech_timestamps:
        start = ts['start']
        end = ts['end']
        segments.append(audio_data[start:end])
        
    trimmed_audio = np.concatenate(segments)
    
    return trimmed_audio, speech_timestamps
