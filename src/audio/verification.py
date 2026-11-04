from dataclasses import dataclass
from scipy.spatial.distance import cosine
from src.audio.embedding import SpeakerEmbedding
from src.audio.config import DEFAULT_SIMILARITY_THRESHOLD

@dataclass
class VerificationResult:
    """Result of a speaker verification attempt."""
    match: bool
    score: float
    threshold: float

def verify_speaker(
    reference_embedding: SpeakerEmbedding,
    current_embedding: SpeakerEmbedding,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> VerificationResult:
    """
    Compare a current speaker embedding against a reference embedding.
    
    Args:
        reference_embedding (SpeakerEmbedding): The enrolled/baseline voice fingerprint.
        current_embedding (SpeakerEmbedding): The new voice fingerprint to verify.
        threshold (float): The similarity threshold required for a match.
        
    Returns:
        VerificationResult: The score and match decision.
        
    Raises:
        ValueError: If embeddings are from different models or have mismatched dimensions.
    """
    
    # 1. Verify model compatibility
    if reference_embedding.model_id != current_embedding.model_id:
        raise ValueError(
            f"Model mismatch: Reference is {reference_embedding.model_id}, "
            f"but current is {current_embedding.model_id}"
        )
        
    # 2. Verify dimensionality
    if reference_embedding.dimension != current_embedding.dimension:
        raise ValueError(
            f"Dimension mismatch: Reference is {reference_embedding.dimension}, "
            f"but current is {current_embedding.dimension}"
        )
        
    # 3. Calculate Cosine Similarity
    # scipy's cosine() computes the *distance*, so similarity = 1 - distance
    distance = cosine(reference_embedding.vector, current_embedding.vector)
    similarity = 1.0 - distance
    
    # 4. Apply threshold logic
    is_match = bool(similarity >= threshold)
    
    return VerificationResult(
        match=is_match,
        score=float(similarity),
        threshold=float(threshold)
    )
