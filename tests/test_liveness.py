import pytest
import numpy as np
from src.audio.config import SAMPLE_RATE
from src.audio.liveness import check_liveness, LivenessResult

def test_check_liveness_mocked(monkeypatch):
    """
    Test the liveness scoring logic deterministically by mocking the HuggingFace model.
    """
    audio = np.ones(SAMPLE_RATE, dtype=np.int16)
    
    class MockExtractor:
        def __call__(self, audio, sampling_rate, return_tensors):
            return {"input_values": None}
            
    class MockModel:
        class Config:
            id2label = {'0': 'Bonafide', '1': 'Spoof'}
        
        def __init__(self):
            self.config = self.Config()
            
        def eval(self):
            pass
            
        def __call__(self, **kwargs):
            # Return a fake logits tensor where class 0 (Bonafide) is very high
            import torch
            from collections import namedtuple
            Output = namedtuple("Output", ["logits"])
            return Output(logits=torch.tensor([[10.0, -10.0]]))
            
    # Mock the loader
    import src.audio.liveness as liveness_module
    monkeypatch.setattr(liveness_module, "_load_liveness_model", lambda: (MockExtractor(), MockModel()))
    
    result = check_liveness(audio, SAMPLE_RATE, threshold=0.5)
    
    assert isinstance(result, LivenessResult)
    assert result.is_live == True
    # Softmax of [10, -10] gives ~1.0 for class 0
    assert pytest.approx(result.score, 0.001) == 1.0
    assert result.threshold == 0.5
    assert result.model_id == "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection"

def test_check_liveness_threshold_boundary(monkeypatch):
    """Test that a score exactly equal to threshold is accepted."""
    audio = np.ones(SAMPLE_RATE, dtype=np.int16)
    
    class MockExtractor:
        def __call__(self, *args, **kwargs):
            return {}
            
    class MockModel:
        config = type('Config', (), {'id2label': {'0': 'Bonafide'}})()
        def eval(self): pass
        def __call__(self, **kwargs):
            import torch
            from collections import namedtuple
            Output = namedtuple("Output", ["logits"])
            # Logits [0, 0] -> Softmax [0.5, 0.5]
            return Output(logits=torch.tensor([[0.0, 0.0]]))
            
    import src.audio.liveness as liveness_module
    monkeypatch.setattr(liveness_module, "_load_liveness_model", lambda: (MockExtractor(), MockModel()))
    
    # EXACT threshold boundary
    result = check_liveness(audio, SAMPLE_RATE, threshold=0.5)
    assert result.is_live == True
    assert pytest.approx(result.score, 0.001) == 0.5
