import uuid
import io
import numpy as np
from datetime import datetime
from sqlalchemy.orm import Session

from src.audio import config
from src.audio.wav_io import load_wav, validate_and_normalize_audio
from src.audio.vad import detect_speech
from src.audio.liveness import check_liveness
from src.audio.embedding import extract_embedding, SpeakerEmbedding
from src.audio.exceptions import NoSpeechDetectedError
from src.engine.results import FailureReason, EnrollmentStatus
from src.engine.aggregation import MeanEmbeddingAggregator
from src.infrastructure.database import (
    Enrollment, EnrollmentSample, ReferenceEmbeddingModel, AuditLog
)

class EnrollmentService:
    def __init__(self, db: Session):
        self.db = db
        self.aggregator = MeanEmbeddingAggregator()

    def _has_active_reference(self, user_id: str, tenant_id: str) -> bool:
        return self.db.query(ReferenceEmbeddingModel).filter(
            ReferenceEmbeddingModel.user_id == user_id,
            ReferenceEmbeddingModel.tenant_id == tenant_id,
            ReferenceEmbeddingModel.is_active == True
        ).first() is not None
        
    def get_enrollment_status(self, user_id: str, tenant_id: str) -> dict:
        has_active = self._has_active_reference(user_id, tenant_id)
        enrollment = self.db.query(Enrollment).filter(
            Enrollment.user_id == user_id,
            Enrollment.tenant_id == tenant_id,
            Enrollment.status == EnrollmentStatus.IN_PROGRESS.value
        ).first()
        
        if enrollment:
            samples = self.db.query(EnrollmentSample).filter(EnrollmentSample.enrollment_id == enrollment.id).count()
            return {
                "status": "IN_PROGRESS",
                "accepted_samples": samples,
                "required_samples": enrollment.required_samples,
                "has_active_reference": has_active
            }
            
        if has_active:
            return {
                "status": "ENROLLED",
                "accepted_samples": config.MIN_ENROLLMENT_SAMPLES,
                "required_samples": config.MIN_ENROLLMENT_SAMPLES,
                "has_active_reference": True
            }
            
        return {
            "status": "NOT_ENROLLED",
            "accepted_samples": 0,
            "required_samples": config.MIN_ENROLLMENT_SAMPLES,
            "has_active_reference": False
        }

    def start_enrollment(self, user_id: str, tenant_id: str) -> dict:
        has_active = self._has_active_reference(user_id, tenant_id)
        enrollment = self.db.query(Enrollment).filter(
            Enrollment.user_id == user_id,
            Enrollment.tenant_id == tenant_id,
            Enrollment.status == EnrollmentStatus.IN_PROGRESS.value
        ).first()
        
        if enrollment:
            samples = self.db.query(EnrollmentSample).filter(EnrollmentSample.enrollment_id == enrollment.id).count()
            return {
                "status": "IN_PROGRESS",
                "accepted_samples": samples,
                "required_samples": enrollment.required_samples,
                "has_active_reference": has_active
            }
            
        enrollment = Enrollment(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            status=EnrollmentStatus.IN_PROGRESS.value,
            required_samples=config.MIN_ENROLLMENT_SAMPLES,
            created_at=datetime.utcnow()
        )
        self.db.add(enrollment)
        self.db.commit()
        
        return {
            "status": "IN_PROGRESS",
            "accepted_samples": 0,
            "required_samples": config.MIN_ENROLLMENT_SAMPLES,
            "has_active_reference": has_active
        }

    def _get_active_enrollment(self, user_id: str, tenant_id: str) -> Enrollment:
        enrollment = self.db.query(Enrollment).filter(
            Enrollment.user_id == user_id,
            Enrollment.tenant_id == tenant_id,
            Enrollment.status == EnrollmentStatus.IN_PROGRESS.value
        ).first()
        
        if not enrollment:
            raise ValueError("No active enrollment in progress. Please start enrollment first.")
        return enrollment
        
    def process_enrollment_sample(self, user_id: str, tenant_id: str, audio_path: str):
        enrollment = self._get_active_enrollment(user_id, tenant_id)
        
        liveness_score = None
        liveness_threshold = config.DEFAULT_LIVENESS_THRESHOLD
        
        try:
            # 1. Load, Validate, and Normalize/Resample Audio to 16kHz
            sr, raw_audio = load_wav(audio_path)
            sr, raw_audio = validate_and_normalize_audio(raw_audio, sr, config.SAMPLE_RATE)

            # 2. VAD
            trimmed_audio, _ = detect_speech(raw_audio, sr)
            if len(trimmed_audio) < sr * 0.5:
                return self._log_and_reject(enrollment, user_id, tenant_id, FailureReason.AUDIO_TOO_SHORT)
                
            # 3. Anti-Spoof
            liveness_res = check_liveness(trimmed_audio, sr, liveness_threshold)
            liveness_score = liveness_res.score
            
            if not liveness_res.is_live:
                return self._log_and_reject(
                    enrollment, user_id, tenant_id, FailureReason.ANTI_SPOOF_FAILED,
                    liveness_score=liveness_score
                )
                
            # 4. ECAPA
            embedding = extract_embedding(trimmed_audio, sr)
            
            # 5. Save intermediate sample
            bio = io.BytesIO()
            np.save(bio, embedding.vector)
            
            sample = EnrollmentSample(
                id=str(uuid.uuid4()),
                enrollment_id=enrollment.id,
                embedding_vector=bio.getvalue(),
                model_id=embedding.model_id,
                dimension=embedding.dimension
            )
            self.db.add(sample)
            self.db.commit()
            
            self._log_audit(
                tenant_id=tenant_id, user_id=user_id, session_token=enrollment.id,
                result="SAMPLE_ACCEPTED", reason=FailureReason.SUCCESS,
                liveness_score=liveness_score
            )
            
            # 6. Check if enrollment is complete
            samples = self.db.query(EnrollmentSample).filter(EnrollmentSample.enrollment_id == enrollment.id).all()
            if len(samples) >= enrollment.required_samples:
                return self._complete_enrollment(enrollment, samples, user_id, tenant_id)
                
            return {
                "status": "SAMPLE_ACCEPTED",
                "accepted_samples": len(samples),
                "required_samples": enrollment.required_samples,
                "reason": "SUCCESS",
                "has_active_reference": self._has_active_reference(user_id, tenant_id)
            }
            
        except NoSpeechDetectedError:
            return self._log_and_reject(enrollment, user_id, tenant_id, FailureReason.NO_SPEECH_DETECTED)
        except Exception:
            return self._log_and_reject(enrollment, user_id, tenant_id, FailureReason.MODEL_ERROR)

    def _log_and_reject(self, enrollment, user_id, tenant_id, reason: FailureReason, liveness_score=None):
        self._log_audit(
            tenant_id=tenant_id, user_id=user_id, session_token=enrollment.id,
            result="SAMPLE_REJECTED", reason=reason, liveness_score=liveness_score
        )
        samples = self.db.query(EnrollmentSample).filter(EnrollmentSample.enrollment_id == enrollment.id).all()
        return {
            "status": "SAMPLE_REJECTED",
            "accepted_samples": len(samples),
            "required_samples": enrollment.required_samples,
            "reason": reason.value,
            "has_active_reference": self._has_active_reference(user_id, tenant_id)
        }
        
    def _complete_enrollment(self, enrollment: Enrollment, samples, user_id: str, tenant_id: str):
        # Reconstruct embeddings
        embeddings = []
        for s in samples:
            bio = io.BytesIO(s.embedding_vector)
            vec = np.load(bio)
            embeddings.append(SpeakerEmbedding(vector=vec, model_id=s.model_id, dimension=s.dimension))
            
        # Aggregate (L2-normalizes each individual sample and L2-normalizes the mean)
        reference = self.aggregator.aggregate(embeddings)
        
        # Versioning: Mark old ones inactive
        self.db.query(ReferenceEmbeddingModel).filter(
            ReferenceEmbeddingModel.user_id == user_id,
            ReferenceEmbeddingModel.tenant_id == tenant_id
        ).update({"is_active": False})
        
        bio = io.BytesIO()
        np.save(bio, reference.vector)
        
        # Save new active reference
        new_ref = ReferenceEmbeddingModel(
            id=str(uuid.uuid4()),
            user_id=user_id,
            tenant_id=tenant_id,
            enrollment_id=enrollment.id,
            vector_blob=bio.getvalue(),
            model_id=reference.model_id,
            dimension=reference.dimension,
            is_active=True
        )
        self.db.add(new_ref)
        
        # Mark enrollment complete
        enrollment.status = EnrollmentStatus.ENROLLED.value
        enrollment.completed_at = datetime.utcnow()
        
        # Delete intermediate samples (per user request)
        self.db.query(EnrollmentSample).filter(EnrollmentSample.enrollment_id == enrollment.id).delete()
        
        self.db.commit()
        
        self._log_audit(
            tenant_id=tenant_id, user_id=user_id, session_token=enrollment.id,
            result="ENROLLMENT_COMPLETED", reason=FailureReason.SUCCESS
        )
        
        return {
            "status": "ENROLLED",
            "accepted_samples": len(samples),
            "required_samples": enrollment.required_samples,
            "reason": "SUCCESS",
            "has_active_reference": True
        }

    def _log_audit(self, tenant_id: str, user_id: str, session_token: str, result: str, reason: FailureReason, liveness_score=None):
        log = AuditLog(
            tenant_id=tenant_id,
            user_id=user_id,
            session_token=session_token,
            result=result,
            reason=reason.value,
            liveness_score=liveness_score,
            request_id=str(uuid.uuid4())
        )
        self.db.add(log)
        self.db.commit()
