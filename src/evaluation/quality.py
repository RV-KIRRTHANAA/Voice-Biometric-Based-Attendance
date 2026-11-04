import numpy as np
from dataclasses import dataclass

@dataclass
class AudioQualityMetrics:
    duration_sec: float
    rms_level: float
    peak_amplitude: float
    clipping_ratio: float

def analyze_quality(audio_data: np.ndarray, sample_rate: int) -> AudioQualityMetrics:
    """Computes basic quality metrics on raw audio without ML overhead."""
    if len(audio_data) == 0:
        return AudioQualityMetrics(0.0, 0.0, 0.0, 0.0)
        
    duration = len(audio_data) / sample_rate
    
    # Ensure float for math
    audio_f = audio_data.astype(np.float64)
    
    # Peak amplitude (assuming int16 PCM is max 32768, normalize to [0,1])
    # If the input is int16:
    if audio_data.dtype == np.int16:
        max_val = 32768.0
    else:
        max_val = 1.0 # assume already normalized if not int16
        
    peak = np.max(np.abs(audio_f)) / max_val
    
    # RMS level
    rms = np.sqrt(np.mean((audio_f / max_val)**2))
    
    # Clipping ratio (percentage of samples at absolute maximum)
    # 0.99 threshold to account for floating point conversions
    clipped_samples = np.sum(np.abs(audio_f) >= (max_val * 0.99))
    clipping_ratio = clipped_samples / len(audio_data)
    
    return AudioQualityMetrics(
        duration_sec=float(duration),
        rms_level=float(rms),
        peak_amplitude=float(peak),
        clipping_ratio=float(clipping_ratio)
    )
