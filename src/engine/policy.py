from src.engine.results import AttendanceDecision, BiometricResult, FailureReason

def evaluate_attendance(biometric_result: BiometricResult) -> AttendanceDecision:
    """
    Consumes the biometric signal and outputs the final business decision.
    In a real system, this is where grace periods, retries, or manager overrides
    could be logically evaluated. For this POC, it strictly adheres to the 
    biometric pass/fail status.
    """
    
    is_marked = biometric_result.passed
    reason = biometric_result.reason
    
    return AttendanceDecision(
        marked=is_marked,
        reason=reason,
        biometric_data=biometric_result
    )
