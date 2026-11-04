package com.example.voiceattendance.ui.attendance

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.voiceattendance.audio.AudioRecorder
import com.example.voiceattendance.audio.WavEncoder
import com.example.voiceattendance.data.model.SessionInfo
import com.example.voiceattendance.data.model.StudentInfo
import com.example.voiceattendance.data.network.ApiErrorParser
import com.example.voiceattendance.data.repository.AttendanceRepository
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import retrofit2.HttpException
import java.io.File

sealed class AttendanceState {
    object Idle : AttendanceState()
    object LoadingHome : AttendanceState()
    data class HomeLoaded(
        val student: StudentInfo,
        val sessions: List<SessionInfo>
    ) : AttendanceState()
    // Recording flow states
    data class ReadyToRecord(val session: SessionInfo) : AttendanceState()
    data class Recording(val session: SessionInfo) : AttendanceState()
    data class Uploading(val session: SessionInfo) : AttendanceState()
    data class Result(
        val session: SessionInfo,
        val isPresent: Boolean,
        val reason: String
    ) : AttendanceState()
    data class Error(val message: String) : AttendanceState()
}

/** Maps backend reason codes to human-readable strings. */
fun mapReason(raw: String): String = when (raw.uppercase()) {
    "SUCCESS", "MARKED_PRESENT"      -> "Attendance marked"
    "SPEAKER_MATCH"                  -> "Voice verified"
    "SPEAKER_MISMATCH"               -> "Voice did not match"
    "ANTI_SPOOF_FAILED",
    "LIVENESS_FAILED"                -> "Liveness check failed"
    "AUDIO_TOO_SHORT"                -> "Recording too short — please speak longer"
    "AUDIO_TOO_NOISY"                -> "Background too noisy — try a quieter spot"
    "ALREADY_MARKED"                 -> "Already marked for this class"
    "SESSION_EXPIRED_OR_NOT_STARTED" -> "Session is not currently active"
    "INVALID_SESSION"                -> "Invalid session"
    "NO_SPEECH_DETECTED"             -> "No speech detected — speak clearly"
    "MODEL_ERROR"                    -> "Processing error"
    "INVALID_REFERENCE"              -> "Voice not enrolled — please enroll first"
    "CHALLENGE_EXPIRED"              -> "Challenge expired — please try again"
    "CHALLENGE_INVALID"              -> "Invalid challenge — please try again"
    "CHALLENGE_USED"                 -> "Challenge already used — please try again"
    "CHALLENGE_MISMATCH"             -> "Spoken phrase did not match the challenge"
    "ASR_ERROR"                      -> "Could not transcribe speech — speak clearly"
    else                             -> raw.replace("_", " ").lowercase()
        .replaceFirstChar { it.uppercase() }
}

class AttendanceViewModel(
    private val repository: AttendanceRepository,
    private val audioRecorder: AudioRecorder,
    private val cacheDir: File
) : ViewModel() {
    companion object {
        const val CHALLENGE_RESPONSE_ENABLED = true
        const val CHALLENGE_TTL_SECONDS = 10
    }

    var challengeId: String? = null
    var challengePhrase: String? = null
    val challengeSecondsRemaining = MutableStateFlow(CHALLENGE_TTL_SECONDS)
    private var countdownJob: Job? = null

    private val _state = MutableStateFlow<AttendanceState>(AttendanceState.Idle)
    val state: StateFlow<AttendanceState> = _state

    fun reset() {
        countdownJob?.cancel()
        challengeId = null
        challengePhrase = null
        challengeSecondsRemaining.value = CHALLENGE_TTL_SECONDS
        _state.value = AttendanceState.Idle
    }

    fun loadHome() {
        countdownJob?.cancel()
        if (_state.value is AttendanceState.LoadingHome) return
        viewModelScope.launch {
            _state.value = AttendanceState.LoadingHome
            try {
                val student  = repository.getStudentInfo()
                val sessions = repository.getSessions()
                _state.value = AttendanceState.HomeLoaded(student, sessions)
            } catch (e: HttpException) {
                if (e.code() == 401) {
                    _state.value = AttendanceState.Error("Session expired, please login again.")
                } else {
                    _state.value = AttendanceState.Error(ApiErrorParser.parse(e))
                }
            } catch (e: Exception) {
                _state.value = AttendanceState.Error(e.message ?: "Failed to load")
            }
        }
    }

    fun selectSession(session: SessionInfo) {
        if (!session.isMarkable) return
        countdownJob?.cancel()
        challengeId = null
        challengePhrase = null
        challengeSecondsRemaining.value = CHALLENGE_TTL_SECONDS
        
        if (!CHALLENGE_RESPONSE_ENABLED) {
            // Challenge disabled: skip challenge fetch, leave challengeId null
            _state.value = AttendanceState.ReadyToRecord(session)
            return
        }
        
        viewModelScope.launch {
            repository.getChallenge(session.sessionToken).fold(
                onSuccess = { challenge ->
                    challengeId = challenge.challenge_id
                    challengePhrase = challenge.phrase
                    _state.value = AttendanceState.ReadyToRecord(session)

                    // Start countdown timer for ephemeral challenge
                    countdownJob?.cancel()
                    countdownJob = viewModelScope.launch {
                        for (s in CHALLENGE_TTL_SECONDS downTo 0) {
                            challengeSecondsRemaining.value = s
                            if (s == 0) {
                                val current = _state.value
                                if (current is AttendanceState.ReadyToRecord) {
                                    _state.value = AttendanceState.Error("Challenge expired. Please try again.")
                                }
                                break
                            }
                            delay(1000L)
                        }
                    }
                },
                onFailure = { err ->
                    val detail = when (err) {
                        is HttpException -> ApiErrorParser.parse(err)
                        else -> err.message ?: "Unknown error"
                    }
                    _state.value = AttendanceState.Error("Failed to load challenge: $detail")
                }
            )
        }
    }

    fun startRecording(session: SessionInfo) {
        _state.value = AttendanceState.Recording(session)
        markAttendance(session)
    }

    private fun markAttendance(session: SessionInfo) {
        val current = _state.value
        if (current is AttendanceState.Recording && current.session.sessionId == session.sessionId) {
            // Already started — do the actual recording
        } else {
            return
        }
        viewModelScope.launch {
            val wavFile = File(cacheDir, "att_.wav")
            try {
                val pcmData = audioRecorder.startRecording()
                if (pcmData.isEmpty()) {
                    _state.value = AttendanceState.Error("Recording was empty")
                    return@launch
                }

                _state.value = AttendanceState.Uploading(session)
                WavEncoder.writeWavFile(pcmData, wavFile, sampleRate = 16000, channels = 1, bitsPerSample = 16)

                val chId = challengeId
                if (CHALLENGE_RESPONSE_ENABLED && chId == null) {
                    _state.value = AttendanceState.Error("Challenge not loaded")
                    return@launch
                }
                val response = repository.markAttendance(session.sessionToken, chId, wavFile)
                _state.value = AttendanceState.Result(
                    session   = session,
                    isPresent = response.is_present,
                    reason    = mapReason(response.reason)
                )
            } catch (e: HttpException) {
                if (e.code() == 401) {
                    _state.value = AttendanceState.Error("Session expired, please login again.")
                } else {
                    _state.value = AttendanceState.Error(ApiErrorParser.parse(e))
                }
            } catch (e: Exception) {
                audioRecorder.stopRecording()
                _state.value = AttendanceState.Error(e.message ?: "Attendance failed")
            } finally {
                if (wavFile.exists()) wavFile.delete()
            }
        }
    }

    /** Reload home screen and refresh attendance status from backend. */
    fun returnToHome() {
        loadHome()
    }
}
