import uuid
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import update

from src.audio.wav_io import load_wav, validate_and_normalize_audio
from src.audio import config
from src.engine.biometric_engine import run_biometric_pipeline
from src.engine.policy import evaluate_attendance
from src.engine.results import AttendanceDecision, FailureReason, BiometricResult
from src.application.interfaces import ReferenceEmbeddingRepository
from src.infrastructure.database import AttendanceSessionModel, AttendanceRecord, AuditLog, Challenge

class AttendanceService:
    def __init__(self, repository: ReferenceEmbeddingRepository, db: Session):
        self.repository = repository
        self.db = db
        
    def generate_challenge(self, student_id: str, tenant_id: str, session_token: str) -> dict:
        session = self.db.query(AttendanceSessionModel).filter(
            AttendanceSessionModel.session_token == session_token,
            AttendanceSessionModel.tenant_id == tenant_id,
            AttendanceSessionModel.is_active == True
        ).first()
        
        if not session:
            raise ValueError("Invalid session")
            
        now = datetime.utcnow()
        if now < session.starts_at or now > session.ends_at:
            raise ValueError("Session is not currently active")
            
        phrase = "present sir " + " ".join([random.choice(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]) for _ in range(3)])
        challenge_id = str(uuid.uuid4())
        expires_at = now + timedelta(seconds=config.CHALLENGE_TTL_SECONDS)
        
        challenge = Challenge(
            id=challenge_id,
            tenant_id=tenant_id,
            user_id=student_id,
            session_id=session.id,
            expected_phrase=phrase,
            created_at=now,
            expires_at=expires_at,
            consumed=False
        )
        self.db.add(challenge)
        self.db.commit()
        
        return {
            "challenge_id": challenge_id,
            "phrase": phrase,
            "expires_at": expires_at.isoformat() + "Z"
        }

    def process_attendance(self, student_id: str, tenant_id: str, audio_path: str, session_token: str, challenge_id: Optional[str] = None) -> AttendanceDecision:
        request_id = str(uuid.uuid4())
        reference = None
        
        # Helper to log audit
        def log_audit(decision: AttendanceDecision):
            bio = decision.biometric_data
            log = AuditLog(
                tenant_id=tenant_id,
                user_id=student_id,
                session_token=session_token,
                request_id=request_id,
                result="PRESENT" if decision.marked else "FAILED",
                reason=decision.reason.value if hasattr(decision.reason, 'value') else str(decision.reason),
                similarity_score=bio.similarity_score,
                liveness_score=bio.liveness_score,
                model_id=reference.model_id if reference and getattr(reference, "model_id", None) else "speechbrain/spkrec-ecapa-voxceleb"
            )
            self.db.add(log)
            self.db.commit()

        # 1. Validate Session Time Bounds
        session = self.db.query(AttendanceSessionModel).filter(
            AttendanceSessionModel.session_token == session_token,
            AttendanceSessionModel.tenant_id == tenant_id,
            AttendanceSessionModel.is_active == True
        ).first()
        
        if not session:
             decision = AttendanceDecision(marked=False, reason=FailureReason.MODEL_ERROR, biometric_data=BiometricResult(passed=False, reason=FailureReason.MODEL_ERROR))
             decision.reason = "INVALID_SESSION"
             log_audit(decision)
             return decision
             
        now = datetime.utcnow()
        if now < session.starts_at or now > session.ends_at:
             decision = AttendanceDecision(marked=False, reason=FailureReason.MODEL_ERROR, biometric_data=BiometricResult(passed=False, reason=FailureReason.MODEL_ERROR))
             decision.reason = "SESSION_EXPIRED_OR_NOT_STARTED"
             log_audit(decision)
             return decision

        # 2. Challenge Verification
        expected_phrase = None
        if config.CHALLENGE_RESPONSE_ENABLED:
            if not challenge_id:
                decision = AttendanceDecision(
                    marked=False,
                    reason=FailureReason.CHALLENGE_INVALID,
                    biometric_data=BiometricResult(passed=False, reason=FailureReason.CHALLENGE_INVALID)
                )
                log_audit(decision)
                return decision

            # Atomically consume the challenge to prevent replay
            stmt = update(Challenge).where(
                Challenge.id == challenge_id,
                Challenge.tenant_id == tenant_id,
                Challenge.user_id == student_id,
                Challenge.session_id == session.id,
                Challenge.consumed == False
            ).values(consumed=True)
            
            result = self.db.execute(stmt)
            self.db.commit()
            
            if result.rowcount == 0:
                challenge = self.db.query(Challenge).filter(Challenge.id == challenge_id).first()
                reason = FailureReason.CHALLENGE_INVALID
                if challenge:
                    if challenge.consumed:
                        reason = FailureReason.CHALLENGE_USED
                    elif challenge.user_id != student_id or challenge.tenant_id != tenant_id or challenge.session_id != session.id:
                        reason = FailureReason.CHALLENGE_INVALID
                        
                decision = AttendanceDecision(marked=False, reason=reason, biometric_data=BiometricResult(passed=False, reason=reason))
                log_audit(decision)
                return decision
                
            challenge = self.db.query(Challenge).filter(Challenge.id == challenge_id).first()
            if now > challenge.expires_at:
                decision = AttendanceDecision(marked=False, reason=FailureReason.CHALLENGE_EXPIRED, biometric_data=BiometricResult(passed=False, reason=FailureReason.CHALLENGE_EXPIRED))
                log_audit(decision)
                return decision
            expected_phrase = challenge.expected_phrase

        # 3. Fetch Reference
        reference = self.repository.get_reference(student_id, tenant_id)
        if not reference:
            decision = AttendanceDecision(marked=False, reason=FailureReason.INVALID_REFERENCE, biometric_data=BiometricResult(passed=False, reason=FailureReason.INVALID_REFERENCE))
            log_audit(decision)
            return decision
            
        # 4. Load, Validate, and Normalize/Resample Audio to 16kHz
        try:
            sr, raw_audio = load_wav(audio_path)
            sr, raw_audio = validate_and_normalize_audio(raw_audio, sr, config.SAMPLE_RATE)
        except Exception:
             decision = AttendanceDecision(marked=False, reason=FailureReason.MODEL_ERROR, biometric_data=BiometricResult(passed=False, reason=FailureReason.MODEL_ERROR))
             log_audit(decision)
             return decision
             
        # 5. Invoke Biometric Engine
        biometric_res = run_biometric_pipeline(raw_audio, reference, expected_phrase=expected_phrase)
        decision = evaluate_attendance(biometric_res)
        
        # 6. Persist Attendance if successful
        if decision.marked:
            record = AttendanceRecord(
                tenant_id=tenant_id,
                user_id=student_id,
                session_id=session.id
            )
            self.db.add(record)
            try:
                self.db.commit()
            except IntegrityError:
                self.db.rollback()
                # Duplicate attendance
                decision.marked = False
                decision.reason = "ALREADY_MARKED"
        
        # 7. Always Audit
        log_audit(decision)
        
        return decision
