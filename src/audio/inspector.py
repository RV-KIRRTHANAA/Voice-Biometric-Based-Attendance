import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

def inspect_audio(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Inspect the properties of an audio array.
    
    Args:
        audio_data (np.ndarray): The audio data.
        sample_rate (int): The sampling rate in Hz.
        
    Returns:
        Dict[str, Any]: A dictionary containing audio properties.
    """
    # Determine the number of channels
    # If the array is 1D, it's mono (1 channel). If 2D, the second dimension is channels.
    channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
    
    num_samples = audio_data.shape[0]
    duration_sec = num_samples / sample_rate
    
    properties = {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "duration_sec": duration_sec,
        "num_samples": num_samples,
        "dtype": str(audio_data.dtype),
        "max_amplitude": float(np.max(audio_data)),
        "min_amplitude": float(np.min(audio_data))
    }
    
    return properties

def plot_waveform(audio_data: np.ndarray, sample_rate: int, output_path: str = "waveform.png") -> None:
    """
    Generate and save a waveform plot of the audio data.
    
    Args:
        audio_data (np.ndarray): The audio data.
        sample_rate (int): The sampling rate in Hz.
        output_path (str): The path to save the generated plot image.
    """
    num_samples = audio_data.shape[0]
    duration_sec = num_samples / sample_rate
    
    # Create an array of time values for the x-axis
    time_axis = np.linspace(0, duration_sec, num = num_samples)
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, audio_data, color='blue', alpha=0.7)
    plt.title("Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
