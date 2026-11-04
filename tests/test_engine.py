import pytest
import numpy as np

from src.audio.embedding import SpeakerEmbedding
from src.engine.results import FailureReason, AttendanceDecision, BiometricResult
from src.engine.biometric_engine import run_biometric_pipeline, validate_reference_embedding
from src.engine.policy import evaluate_attendance
from src.audio.exceptions import NoSpeechDetectedError

class MockLivenessResult:
    def __init__(self, is_live, score, threshold):
        self.is_live = is_live
        self.score = score
        self.threshold = threshold

class MockVerificationResult:
    def __init__(self, match, score, threshold):
        self.match = match
        self.score = score
        self.threshold = threshold

@pytest.fixture
def valid_ref():
    return SpeakerEmbedding(vector=np.ones(192), model_id="test", dimension=192)

def test_validate_reference():
    assert validate_reference_embedding(SpeakerEmbedding(vector=np.ones(192), model_id="m", dimension=192)) == True
    assert validate_reference_embedding(None) == False
    # Dimension mismatch
    assert validate_reference_embedding(SpeakerEmbedding(vector=np.ones(192), model_id="m", dimension=100)) == False

def test_engine_no_speech(monkeypatch, valid_ref):
    def mock_detect(*args, **kwargs):
        raise NoSpeechDetectedError()
    monkeypatch.setattr("src.engine.biometric_engine.detect_speech", mock_detect)
    
    dummy_audio = np.zeros(100)
    result = run_biometric_pipeline(dummy_audio, valid_ref)
    
    assert result.passed == False
    assert result.reason == FailureReason.NO_SPEECH_DETECTED
    
    # Check policy
    decision = evaluate_attendance(result)
    assert decision.marked == False
    assert decision.reason == FailureReason.NO_SPEECH_DETECTED

def test_engine_spoof_rejection(monkeypatch, valid_ref):
    def mock_detect(*args, **kwargs):
        return np.ones(16000), [] # 1 second
    monkeypatch.setattr("src.engine.biometric_engine.detect_speech", mock_detect)
    
    def mock_liveness(*args, **kwargs):
        return MockLivenessResult(is_live=False, score=0.1, threshold=0.5)
    monkeypatch.setattr("src.engine.biometric_engine.check_liveness", mock_liveness)
    
    # We explicitly track if extract_embedding is called
    called_extract = False
    def mock_extract(*args, **kwargs):
        nonlocal called_extract
        called_extract = True
    monkeypatch.setattr("src.engine.biometric_engine.extract_embedding", mock_extract)
    
    result = run_biometric_pipeline(np.zeros(100), valid_ref)
    
    assert result.passed == False
    assert result.reason == FailureReason.ANTI_SPOOF_FAILED
    assert result.liveness_score == 0.1
    # Ensure short-circuiting worked
    assert called_extract == False
    
    decision = evaluate_attendance(result)
    assert decision.marked == False

def test_engine_speaker_mismatch(monkeypatch, valid_ref):
    monkeypatch.setattr("src.engine.biometric_engine.detect_speech", lambda *a, **kw: (np.ones(16000), []))
    monkeypatch.setattr("src.engine.biometric_engine.check_liveness", lambda *a, **kw: MockLivenessResult(True, 0.9, 0.5))
    monkeypatch.setattr("src.engine.biometric_engine.extract_embedding", lambda *a, **kw: SpeakerEmbedding(vector=np.zeros(192), model_id="test", dimension=192))
    
    def mock_verify(*args, **kwargs):
        return MockVerificationResult(match=False, score=0.2, threshold=0.75)
    monkeypatch.setattr("src.engine.biometric_engine.verify_speaker", mock_verify)
    
    result = run_biometric_pipeline(np.zeros(100), valid_ref)
    assert result.passed == False
    assert result.reason == FailureReason.SPEAKER_MISMATCH
    assert result.similarity_score == 0.2

def test_engine_success(monkeypatch, valid_ref):
    monkeypatch.setattr("src.engine.biometric_engine.detect_speech", lambda *a, **kw: (np.ones(16000), []))
    monkeypatch.setattr("src.engine.biometric_engine.check_liveness", lambda *a, **kw: MockLivenessResult(True, 0.95, 0.5))
    monkeypatch.setattr("src.engine.biometric_engine.extract_embedding", lambda *a, **kw: SpeakerEmbedding(vector=np.ones(192), model_id="test", dimension=192))
    monkeypatch.setattr("src.engine.biometric_engine.verify_speaker", lambda *a, **kw: MockVerificationResult(match=True, score=0.9, threshold=0.75))
    
    result = run_biometric_pipeline(np.zeros(100), valid_ref)
    assert result.passed == True
    assert result.reason == FailureReason.SUCCESS
    
    decision = evaluate_attendance(result)
    assert decision.marked == True
    assert decision.reason == FailureReason.SUCCESS
