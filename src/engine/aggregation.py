from abc import ABC, abstractmethod
from typing import List
import numpy as np

from src.audio.embedding import SpeakerEmbedding

class EmbeddingAggregator(ABC):
    @abstractmethod
    def aggregate(self, embeddings: List[SpeakerEmbedding]) -> SpeakerEmbedding:
        """
        Combine multiple embeddings into a single reference embedding.
        
        Args:
            embeddings (List[SpeakerEmbedding]): The validated samples.
            
        Returns:
            SpeakerEmbedding: The aggregated result.
        """
        pass

class MeanEmbeddingAggregator(EmbeddingAggregator):
    """
    L2-normalizes each sample embedding, computes the mean vector, and L2-normalizes the result.
    """
    def aggregate(self, embeddings: List[SpeakerEmbedding]) -> SpeakerEmbedding:
        if not embeddings:
            raise ValueError("Cannot aggregate an empty list of embeddings.")
            
        # Verify model compatibility
        model_id = embeddings[0].model_id
        dimension = embeddings[0].dimension
        
        for e in embeddings:
            if e.model_id != model_id or e.dimension != dimension:
                raise ValueError("All embeddings must have the same model_id and dimension to aggregate.")
                
        normalized_vectors = []
        for e in embeddings:
            vec = np.asarray(e.vector, dtype=np.float64)
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 0:
                vec = vec / vec_norm
            normalized_vectors.append(vec)
            
        mean_vector = np.mean(normalized_vectors, axis=0)
        
        # L2 Normalization of the aggregated mean vector
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm
            
        return SpeakerEmbedding(
            vector=mean_vector.astype(embeddings[0].vector.dtype if np.issubdtype(embeddings[0].vector.dtype, np.floating) else np.float64),
            model_id=model_id,
            dimension=dimension
        )
