from abc import ABC, abstractmethod
from typing import Optional
from src.audio.embedding import SpeakerEmbedding

class ReferenceEmbeddingRepository(ABC):
    """
    Abstract repository for retrieving user reference embeddings.
    This abstraction ensures the core biometric application is completely 
    decoupled from the underlying storage mechanism (Files, Database, etc).
    """

    @abstractmethod
    def get_reference(self, student_id: str, tenant_id: str) -> Optional[SpeakerEmbedding]:
        """
        Retrieve the SpeakerEmbedding for a given student ID and tenant ID.
        Returns None if no reference is found.
        """
        pass
