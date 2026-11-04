import os
import json
import numpy as np
import io
from typing import Optional
from sqlalchemy.orm import Session

from src.audio.embedding import SpeakerEmbedding
from src.application.interfaces import ReferenceEmbeddingRepository
from src.infrastructure.database import ReferenceEmbeddingModel

class DatabaseReferenceEmbeddingRepository(ReferenceEmbeddingRepository):
    """
    Phase 10 Persistent Repository.
    Loads active reference embeddings from the SQLAlchemy Database.
    """
    
    def __init__(self, db: Session):
        self.db = db
        
    def get_reference(self, student_id: str, tenant_id: str) -> Optional[SpeakerEmbedding]:
        db_record = self.db.query(ReferenceEmbeddingModel).filter(
            ReferenceEmbeddingModel.user_id == student_id,
            ReferenceEmbeddingModel.tenant_id == tenant_id,
            ReferenceEmbeddingModel.is_active == True
        ).first()
        
        if not db_record:
            return None
            
        try:
            # We assume vector_blob is an NPY serialized string/bytes
            vector = np.load(io.BytesIO(db_record.vector_blob))
            
            return SpeakerEmbedding(
                vector=vector,
                model_id=db_record.model_id,
                dimension=db_record.dimension
            )
        except Exception:
            return None

