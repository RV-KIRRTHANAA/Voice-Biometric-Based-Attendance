import numpy as np
from src.audio import config
from src.audio.vad import detect_speech
from src.audio.liveness import check_liveness
from src.audio.embedding import extract_embedding, SpeakerEmbedding
from src.audio.verification import verify_speaker
from src.audio.exceptions import NoSpeechDetectedError
from src.engine.results import BiometricResult, FailureReason
from src.audio.asr import ASRService
from src.engine.challenge_verifier import ChallengeVerifier

asr_service = ASRService()
challenge_verifier = ChallengeVerifier()


def validate_reference_embedding(reference: SpeakerEmbedding) -> bool:
    """Check if the reference embedding is structurally sound."""
    if not isinstance(reference, SpeakerEmbedding):
        return False
    if not hasattr(reference, 'model_id') or not reference.model_id:
        return False
    if not hasattr(reference, 'dimension') or reference.dimension <= 0:
        return False
    if not hasattr(reference, 'vector') or len(reference.vector) != reference.dimension:
        return False
    return True

def run_biometric_pipeline(raw_audio: np.ndarray, reference_embedding: SpeakerEmbedding, expected_phrase: str = None) -> BiometricResult:
    """
    Orchestrates the raw audio through the ML pipeline.
    Execution order: VAD -> Anti-Spoof -> Embedding -> Verification.
    Short-circuits if anti-spoofing fails.
    """
    
    # 0. Validate Reference
    if not validate_reference_embedding(reference_embedding):
        return BiometricResult(passed=False, reason=FailureReason.INVALID_REFERENCE)
        
    try:
        # 1. VAD (Voice Activity Detection)
        try:
            trimmed_audio, _ = detect_speech(raw_audio, config.SAMPLE_RATE)
        except NoSpeechDetectedError:
            return BiometricResult(passed=False, reason=FailureReason.NO_SPEECH_DETECTED)
            
        # Optional: check if trimmed audio is too short (e.g. less than 0.5s)
        if len(trimmed_audio) < config.SAMPLE_RATE * 0.5:
            return BiometricResult(passed=False, reason=FailureReason.AUDIO_TOO_SHORT)
            
        # 2. Anti-Spoof (Liveness)
        liveness_res = check_liveness(trimmed_audio, config.SAMPLE_RATE, config.DEFAULT_LIVENESS_THRESHOLD)
        
        if not liveness_res.is_live:
            # Short-circuit: Do not waste compute extracting speaker embeddings
            return BiometricResult(
                passed=False, 
                reason=FailureReason.ANTI_SPOOF_FAILED,
                liveness_score=liveness_res.score,
                liveness_threshold=liveness_res.threshold
            )
            
        # 3. Embedding Extraction
        current_embedding = extract_embedding(trimmed_audio, config.SAMPLE_RATE)
        
        # 4. Speaker Verification
        try:
            verification_res = verify_speaker(
                reference_embedding, 
                current_embedding, 
                threshold=config.DEFAULT_SIMILARITY_THRESHOLD
            )
        except ValueError:
            # Raised if model_id or dimensions mismatch
            return BiometricResult(
                passed=False, 
                reason=FailureReason.MODEL_ERROR,
                liveness_score=liveness_res.score,
                liveness_threshold=liveness_res.threshold
            )
            
        if not verification_res.match:
            return BiometricResult(
                passed=False,
                reason=FailureReason.SPEAKER_MISMATCH,
                liveness_score=liveness_res.score,
                liveness_threshold=liveness_res.threshold,
                similarity_score=verification_res.score,
                similarity_threshold=verification_res.threshold
            )
            
        # 5. ASR Challenge Verification (if expected_phrase provided)
        if expected_phrase:
            try:
                transcribed = asr_service.transcribe(trimmed_audio, config.SAMPLE_RATE)
            except Exception:
                return BiometricResult(
                    passed=False,
                    reason=FailureReason.ASR_ERROR,
                    liveness_score=liveness_res.score,
                    liveness_threshold=liveness_res.threshold,
                    similarity_score=verification_res.score,
                    similarity_threshold=verification_res.threshold
                )
            
            if not challenge_verifier.verify_challenge(expected_phrase, transcribed):
                return BiometricResult(
                    passed=False,
                    reason=FailureReason.CHALLENGE_MISMATCH,
                    liveness_score=liveness_res.score,
                    liveness_threshold=liveness_res.threshold,
                    similarity_score=verification_res.score,
                    similarity_threshold=verification_res.threshold
                )

        # 6. Success
        return BiometricResult(
            passed=True,
            reason=FailureReason.SUCCESS,
            liveness_score=liveness_res.score,
            liveness_threshold=liveness_res.threshold,
            similarity_score=verification_res.score,
            similarity_threshold=verification_res.threshold
        )
        
    except Exception:
        # Catch unexpected crashes (e.g. out of memory, malformed array)
        return BiometricResult(passed=False, reason=FailureReason.MODEL_ERROR)
