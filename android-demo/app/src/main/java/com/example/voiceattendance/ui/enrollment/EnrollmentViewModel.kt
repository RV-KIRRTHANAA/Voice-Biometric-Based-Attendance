package com.example.voiceattendance.ui.enrollment

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.voiceattendance.audio.AudioRecorder
import com.example.voiceattendance.audio.WavEncoder
import com.example.voiceattendance.data.network.ApiErrorParser
import com.example.voiceattendance.data.repository.EnrollmentRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import retrofit2.HttpException
import java.io.File

sealed class EnrollmentState {
    object Loading : EnrollmentState()
    object NotEnrolled : EnrollmentState()
    data class InProgress(
        val accepted: Int,
        val required: Int,
        val hasActiveReference: Boolean = false
    ) : EnrollmentState()
    object Recording : EnrollmentState()
    object Uploading : EnrollmentState()
    object Enrolled : EnrollmentState()
    data class Error(val message: String) : EnrollmentState()
}

class EnrollmentViewModel(
    private val repository: EnrollmentRepository,
    private val audioRecorder: AudioRecorder,
    private val cacheDir: File
) : ViewModel() {

    private val _state = MutableStateFlow<EnrollmentState>(EnrollmentState.Loading)
    val state: StateFlow<EnrollmentState> = _state

    fun reset() {
        _state.value = EnrollmentState.Loading
    }

    fun fetchStatus() {
        viewModelScope.launch {
            _state.value = EnrollmentState.Loading
            try {
                val response = repository.getEnrollmentStatus()
                _state.value = when (response.status) {
                    "NOT_ENROLLED" -> EnrollmentState.NotEnrolled
                    "IN_PROGRESS" -> EnrollmentState.InProgress(
                        response.accepted_samples,
                        response.required_samples,
                        response.has_active_reference
                    )
                    "ENROLLED" -> EnrollmentState.Enrolled
                    else -> EnrollmentState.Error("Unknown status: ${response.status}")
                }
            } catch (e: HttpException) {
                if (e.code() == 401) {
                    _state.value = EnrollmentState.Error("Session expired, please login again.")
                } else {
                    _state.value = EnrollmentState.Error(ApiErrorParser.parse(e))
                }
            } catch (e: Exception) {
                _state.value = EnrollmentState.Error(e.message ?: "Failed to fetch status")
            }
        }
    }

    fun startEnrollment() {
        viewModelScope.launch {
            _state.value = EnrollmentState.Loading
            try {
                val response = repository.startEnrollment()
                _state.value = EnrollmentState.InProgress(
                    response.accepted_samples,
                    response.required_samples,
                    response.has_active_reference
                )
            } catch (e: HttpException) {
                if (e.code() == 401) {
                    _state.value = EnrollmentState.Error("Session expired, please login again.")
                } else {
                    _state.value = EnrollmentState.Error(ApiErrorParser.parse(e))
                }
            } catch (e: Exception) {
                _state.value = EnrollmentState.Error(e.message ?: "Failed to start enrollment")
            }
        }
    }

    fun recordAndUpload() {
        if (_state.value is EnrollmentState.Recording || _state.value is EnrollmentState.Uploading) return
        
        viewModelScope.launch {
            val tempFile = File(cacheDir, "temp_enrollment.wav")
            try {
                _state.value = EnrollmentState.Recording
                val pcmData = audioRecorder.startRecording()
                
                if (pcmData.isEmpty()) {
                    _state.value = EnrollmentState.Error("Recording was empty")
                    return@launch
                }

                _state.value = EnrollmentState.Uploading
                WavEncoder.writeWavFile(pcmData, tempFile)
                
                val response = repository.processEnrollment(tempFile)
                
                when (response.status) {
                    "SAMPLE_ACCEPTED" -> {
                        _state.value = EnrollmentState.InProgress(
                            response.acceptedSamples,
                            response.requiredSamples,
                            response.hasActiveReference
                        )
                    }
                    "SAMPLE_REJECTED" -> {
                        _state.value = EnrollmentState.Error(response.reason)
                    }
                    "ENROLLED" -> {
                        _state.value = EnrollmentState.Enrolled
                    }
                    else -> {
                        _state.value = EnrollmentState.Error("Unknown status: ${response.status}")
                    }
                }
            } catch (e: HttpException) {
                if (e.code() == 401) {
                    _state.value = EnrollmentState.Error("Session expired, please login again.")
                } else {
                    _state.value = EnrollmentState.Error("Server Error: ${ApiErrorParser.parse(e)}")
                }
            } catch (e: Exception) {
                _state.value = EnrollmentState.Error(e.message ?: "An error occurred")
                audioRecorder.stopRecording()
            } finally {
                if (tempFile.exists()) {
                    tempFile.delete()
                }
            }
        }
    }
}
