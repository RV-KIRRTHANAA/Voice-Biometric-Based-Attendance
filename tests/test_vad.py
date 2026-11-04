import pytest
import numpy as np
from src.audio.config import SAMPLE_RATE
from src.audio.vad import detect_speech
from src.audio.exceptions import NoSpeechDetectedError

def test_vad_no_speech():
    """Test that VAD correctly raises NoSpeechDetectedError on pure silence."""
    # Generate 2 seconds of pure zeros (silence)
    duration = 2.0
    silence = np.zeros(int(SAMPLE_RATE * duration), dtype=np.int16)
    
    with pytest.raises(NoSpeechDetectedError):
        detect_speech(silence, SAMPLE_RATE)

def test_vad_detects_synthetic_speech(monkeypatch):
    """
    Test the VAD logic handles the return format correctly.
    Since Silero VAD requires realistic human speech to actually trigger natively,
    we mock the internal model call for deterministic testing of our logic.
    """
    # Create mock audio (1 second)
    audio = np.ones(SAMPLE_RATE, dtype=np.int16)
    
    # Mock the internal _get_speech_timestamps
    def mock_get_timestamps(tensor, model, sampling_rate):
        # Pretend it found speech from sample 100 to 500
        return [{'start': 100, 'end': 500}]
        
    import src.audio.vad as vad_module
    monkeypatch.setattr(vad_module, "_get_speech_timestamps", mock_get_timestamps)
    
    # Force _model to be not None so it bypasses loading
    monkeypatch.setattr(vad_module, "_model", "mocked_model")
    
    trimmed, timestamps = vad_module.detect_speech(audio, SAMPLE_RATE)
    
    # Verify the timestamps are returned correctly
    assert len(timestamps) == 1
    assert timestamps[0]['start'] == 100
    assert timestamps[0]['end'] == 500
    
    # Verify the audio was trimmed correctly (length should be 400)
    assert len(trimmed) == 400
