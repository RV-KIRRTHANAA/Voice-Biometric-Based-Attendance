import pytest
import numpy as np
from src.audio.embedding import SpeakerEmbedding
from src.audio.verification import verify_speaker, VerificationResult

def create_mock_embedding(vector: np.ndarray, model_id: str = "test-model") -> SpeakerEmbedding:
    return SpeakerEmbedding(
        vector=vector,
        model_id=model_id,
        dimension=len(vector)
    )

def test_verify_identical_vectors():
    """Test that comparing a vector to itself yields exactly 1.0."""
    vec = np.array([1.0, 2.0, 3.0])
    emb1 = create_mock_embedding(vec)
    emb2 = create_mock_embedding(vec)
    
    result = verify_speaker(emb1, emb2, threshold=0.75)
    
    assert result.match == True
    assert pytest.approx(result.score, 0.0001) == 1.0
    assert result.threshold == 0.75

def test_verify_orthogonal_vectors():
    """Test that orthogonal vectors yield 0.0 similarity."""
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])
    emb1 = create_mock_embedding(vec1)
    emb2 = create_mock_embedding(vec2)
    
    result = verify_speaker(emb1, emb2, threshold=0.1)
    
    assert result.match == False
    assert pytest.approx(result.score, 0.0001) == 0.0

def test_verify_opposite_vectors():
    """Test that opposite vectors yield -1.0 similarity."""
    vec1 = np.array([1.0, 1.0])
    vec2 = np.array([-1.0, -1.0])
    emb1 = create_mock_embedding(vec1)
    emb2 = create_mock_embedding(vec2)
    
    result = verify_speaker(emb1, emb2, threshold=-0.5)
    
    # Score is -1.0, threshold is -0.5, so score is NOT >= threshold
    assert result.match == False
    assert pytest.approx(result.score, 0.0001) == -1.0

def test_verify_threshold_boundary():
    """Test that similarity exactly equal to threshold is accepted (score >= threshold)."""
    # Create vectors with known cosine similarity.
    # vec1 = [1, 0], vec2 = [cos(theta), sin(theta)]. similarity = cos(theta)
    # Let theta = 60 degrees, cos(60) = 0.5
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.5, np.sqrt(3)/2])
    
    emb1 = create_mock_embedding(vec1)
    emb2 = create_mock_embedding(vec2)
    
    # Set threshold exactly to the expected score (0.5)
    result = verify_speaker(emb1, emb2, threshold=0.5)
    
    assert pytest.approx(result.score, 0.0001) == 0.5
    assert result.match == True # >= 

def test_verify_mismatched_models():
    """Test that comparing embeddings from different models raises an error."""
    emb1 = create_mock_embedding(np.array([1.0, 2.0]), "model-A")
    emb2 = create_mock_embedding(np.array([1.0, 2.0]), "model-B")
    
    with pytest.raises(ValueError, match="Model mismatch"):
        verify_speaker(emb1, emb2)

def test_verify_mismatched_dimensions():
    """Test that comparing embeddings of different dimensions raises an error."""
    emb1 = create_mock_embedding(np.array([1.0, 2.0]), "model-A")
    emb2 = create_mock_embedding(np.array([1.0, 2.0, 3.0]), "model-A")
    
    with pytest.raises(ValueError, match="Dimension mismatch"):
        verify_speaker(emb1, emb2)
