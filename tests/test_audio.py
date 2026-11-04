import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.audio import config
from src.audio.asr import ASRService
from src.audio.wav_io import (
    InvalidAudioError,
    SUPPORTED_SAMPLE_RATES,
    load_wav,
    save_wav,
    validate_and_normalize_audio,
)
from src.audio.inspector import inspect_audio, plot_waveform

def generate_synthetic_audio(duration_sec: float, sample_rate: int, freq: float = 440.0) -> np.ndarray:
    """Generate a synthetic sine wave tone for testing without a microphone."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Generate a tone and scale it to 16-bit integer range
    audio = np.sin(2 * np.pi * freq * t)
    audio = (audio * 32767).astype(np.int16)
    return audio

def test_audio_properties_inspection():
    """Test that audio properties are correctly extracted from synthetic audio."""
    duration = 2.0
    audio = generate_synthetic_audio(duration, config.SAMPLE_RATE)
    
    properties = inspect_audio(audio, config.SAMPLE_RATE)
    
    assert properties["sample_rate_hz"] == config.SAMPLE_RATE
    assert properties["channels"] == 1
    assert properties["duration_sec"] == duration
    assert properties["num_samples"] == int(config.SAMPLE_RATE * duration)
    assert properties["dtype"] == "int16"

def test_wav_save_and_load(tmp_path):
    """Test saving and loading audio to/from a WAV file."""
    duration = 1.0
    audio = generate_synthetic_audio(duration, config.SAMPLE_RATE)
    
    file_path = tmp_path / "test_output.wav"
    
    # Save the synthetic audio
    save_wav(str(file_path), audio, config.SAMPLE_RATE)
    assert file_path.exists()
    
    # Load the audio back
    loaded_sr, loaded_audio = load_wav(str(file_path))
    
    assert loaded_sr == config.SAMPLE_RATE
    assert np.array_equal(audio, loaded_audio)

def test_plot_waveform(tmp_path):
    """Test that a waveform plot image is generated successfully."""
    audio = generate_synthetic_audio(1.0, config.SAMPLE_RATE)
    
    plot_path = tmp_path / "test_plot.png"
    plot_waveform(audio, config.SAMPLE_RATE, str(plot_path))
    
    assert plot_path.exists()
    assert os.path.getsize(str(plot_path)) > 0


# ---------------------------------------------------------------------------
# Phase 2: WAV Sample-Rate Validation & Resampling Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sr", [8000, 16000, 22050, 24000, 32000, 44100, 48000])
def test_validate_and_normalize_audio_supported_sample_rates(sr: int):
    """Supported sample rates must be validated and resampled to 16kHz mono int16."""
    duration = 1.0
    audio = generate_synthetic_audio(duration, sr, freq=440.0)
    out_sr, out_audio = validate_and_normalize_audio(audio, sr, target_sr=16000)

    assert out_sr == 16000
    assert out_audio.dtype == np.int16
    assert out_audio.ndim == 1
    assert abs(len(out_audio) - 16000) <= 2


@pytest.mark.parametrize("bad_sr", [0, -16000, 4000, 12345, 192000])
def test_validate_and_normalize_audio_unsupported_sample_rates(bad_sr: int):
    """Unsupported or invalid sample rates must raise InvalidAudioError."""
    audio = np.zeros(1600, dtype=np.int16)
    with pytest.raises(InvalidAudioError, match="Unsupported|Invalid"):
        validate_and_normalize_audio(audio, bad_sr, target_sr=16000)


def test_validate_and_normalize_audio_stereo_downmix():
    """2-channel stereo audio is downmixed to 1D mono int16."""
    left = np.full(16000, 10000, dtype=np.int16)
    right = np.full(16000, 20000, dtype=np.int16)
    stereo = np.stack([left, right], axis=1)

    out_sr, out_audio = validate_and_normalize_audio(stereo, 16000)
    assert out_sr == 16000
    assert out_audio.ndim == 1
    assert out_audio.dtype == np.int16
    assert np.all(out_audio == 15000)


def test_validate_and_normalize_audio_rejects_multichannel_and_bad_dtypes():
    """Reject >2 channels, empty arrays, uint8/int32 bit depths, and NaN/unnormalized floats."""
    # >2 channels
    with pytest.raises(InvalidAudioError, match="Unsupported channel count"):
        validate_and_normalize_audio(np.zeros((1600, 3), dtype=np.int16), 16000)

    # Empty array
    with pytest.raises(InvalidAudioError, match="empty"):
        validate_and_normalize_audio(np.array([], dtype=np.int16), 16000)

    # Unsupported uint8 (8-bit PCM) and int32 (32-bit PCM)
    with pytest.raises(InvalidAudioError, match="Unsupported audio bit depth"):
        validate_and_normalize_audio(np.zeros(1600, dtype=np.uint8), 16000)
    with pytest.raises(InvalidAudioError, match="Unsupported audio bit depth"):
        validate_and_normalize_audio(np.zeros(1600, dtype=np.int32), 16000)

    # Out-of-range float32
    with pytest.raises(InvalidAudioError, match="normalized"):
        validate_and_normalize_audio(np.full(1600, 5.0, dtype=np.float32), 16000)

    # NaN float32
    nan_audio = np.zeros(1600, dtype=np.float32)
    nan_audio[0] = np.nan
    with pytest.raises(InvalidAudioError, match="NaN or infinite"):
        validate_and_normalize_audio(nan_audio, 16000)


def test_load_wav_rejects_malformed_file(tmp_path):
    """Malformed WAV bytes must raise InvalidAudioError."""
    bad_wav = tmp_path / "corrupt.wav"
    bad_wav.write_bytes(b"NOT_A_VALID_RIFF_WAV_FILE")
    with pytest.raises(InvalidAudioError, match="Malformed or unreadable WAV"):
        load_wav(str(bad_wav))


# ---------------------------------------------------------------------------
# Phase 3: Whisper ASR Normalization Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_asr_singleton():
    prev_model = ASRService._model
    yield
    ASRService._model = prev_model


def _make_asr_with_mock_model():
    asr = ASRService()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": " one two three "}
    asr._model = mock_model
    return asr, mock_model


def test_asr_transcribe_int16_normalization_scales_by_32768():
    """int16 audio must be divided by 32768.0, never peak-normalized."""
    asr, mock_model = _make_asr_with_mock_model()

    # Include half-scale (16384 -> 0.5) and quiet LSB signal (2 -> 2/32768.0)
    int16_audio = np.array([0, 16384, -16384, 2, -32768], dtype=np.int16)
    text = asr.transcribe(int16_audio, sample_rate=16000)

    assert text == "one two three"
    passed_audio = mock_model.transcribe.call_args[0][0]
    assert passed_audio.dtype == np.float32
    expected = np.array([0.0, 0.5, -0.5, 2.0 / 32768.0, -1.0], dtype=np.float32)
    np.testing.assert_allclose(passed_audio, expected, rtol=1e-6)


def test_asr_transcribe_quiet_int16_not_amplified_to_full_scale():
    """Quiet int16 noise (e.g. value 1) must NOT be left unscaled at 1.0 or peak-normalized."""
    asr, mock_model = _make_asr_with_mock_model()
    quiet_int16 = np.array([1, -1, 0, 1], dtype=np.int16)
    asr.transcribe(quiet_int16, sample_rate=16000)

    passed_audio = mock_model.transcribe.call_args[0][0]
    assert np.max(np.abs(passed_audio)) == pytest.approx(1.0 / 32768.0)


def test_asr_transcribe_float32_input():
    """Normalized float32 audio in [-1.0, 1.0] is passed directly as float32."""
    asr, mock_model = _make_asr_with_mock_model()
    float_audio = np.array([0.0, 0.25, -0.75, 1.0, -1.0], dtype=np.float32)
    text = asr.transcribe(float_audio, sample_rate=16000)

    assert text == "one two three"
    passed_audio = mock_model.transcribe.call_args[0][0]
    assert passed_audio.dtype == np.float32
    np.testing.assert_allclose(passed_audio, float_audio, rtol=1e-6)


def test_asr_transcribe_silence():
    """Pure silence (all zeros) in int16 or float32 does not divide by zero."""
    asr, mock_model = _make_asr_with_mock_model()

    for dtype in (np.int16, np.float32):
        silence = np.zeros(16000, dtype=dtype)
        text = asr.transcribe(silence, sample_rate=16000)
        assert text == "one two three"
        passed_audio = mock_model.transcribe.call_args[0][0]
        assert passed_audio.dtype == np.float32
        assert np.all(passed_audio == 0.0)


def test_asr_transcribe_rejects_invalid_audio_and_sample_rate():
    """ASRService.transcribe rejects non-16kHz sample rates, empty arrays, NaN, out-of-range floats, and 2D arrays."""
    asr, _ = _make_asr_with_mock_model()
    valid_audio = np.zeros(1600, dtype=np.int16)

    # Wrong sample rate
    with pytest.raises(ValueError, match="16000 Hz"):
        asr.transcribe(valid_audio, sample_rate=48000)

    # Empty array
    with pytest.raises(ValueError, match="empty"):
        asr.transcribe(np.array([], dtype=np.int16), sample_rate=16000)

    # 2D stereo array
    with pytest.raises(ValueError, match="1D mono"):
        asr.transcribe(np.zeros((1600, 2), dtype=np.int16), sample_rate=16000)

    # Unnormalized float32 > 1.05
    with pytest.raises(ValueError, match="normalized"):
        asr.transcribe(np.full(1600, 100.0, dtype=np.float32), sample_rate=16000)

    # NaN in float32
    nan_arr = np.zeros(1600, dtype=np.float32)
    nan_arr[5] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        asr.transcribe(nan_arr, sample_rate=16000)

    # Unsupported dtype (int32)
    with pytest.raises(ValueError, match="Unsupported audio dtype"):
        asr.transcribe(np.zeros(1600, dtype=np.int32), sample_rate=16000)

