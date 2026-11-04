import pytest
import numpy as np
from src.audio.config import SAMPLE_RATE
from src.audio.embedding import extract_embedding, SpeakerEmbedding

def test_extract_embedding_mocked(monkeypatch):
    """
    Test the embedding extraction logic deterministically by mocking SpeechBrain.
    """
    # Create 1 second of mock audio
    audio = np.ones(SAMPLE_RATE, dtype=np.int16)
    
    # Create a mock classifier object
    class MockClassifier:
        def encode_batch(self, tensor):
            # Return a fake tensor of shape (1, 1, 192) like SpeechBrain does
            import torch
            return torch.ones((1, 1, 192))
            
    # Mock the loader function to return our MockClassifier
    import src.audio.embedding as embed_module
    monkeypatch.setattr(embed_module, "_load_classifier", lambda: MockClassifier())
    
    embedding = extract_embedding(audio, SAMPLE_RATE)
    
    assert isinstance(embedding, SpeakerEmbedding)
    assert embedding.dimension == 192
    assert embedding.vector.shape == (192,)
    assert embedding.model_id == "speechbrain/spkrec-ecapa-voxceleb"
    assert embedding.vector[0] == 1.0 # Verify our mock value survived the conversion
