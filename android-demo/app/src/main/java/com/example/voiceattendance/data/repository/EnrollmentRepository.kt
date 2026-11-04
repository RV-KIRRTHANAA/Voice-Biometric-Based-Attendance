package com.example.voiceattendance.data.repository

import com.example.voiceattendance.data.api.EnrollmentApi
import com.example.voiceattendance.data.model.EnrollmentResponse
import com.example.voiceattendance.data.model.EnrollmentStatusResponse
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File

class EnrollmentRepository(private val api: EnrollmentApi) {
    suspend fun processEnrollment(audioFile: File): EnrollmentResponse {
        val requestFile = audioFile.asRequestBody("audio/wav".toMediaTypeOrNull())
        val body = MultipartBody.Part.createFormData("audio_file", audioFile.name, requestFile)
        return api.processEnrollment(body)
    }

    suspend fun getEnrollmentStatus(): EnrollmentStatusResponse {
        return api.getEnrollmentStatus()
    }

    suspend fun startEnrollment(): EnrollmentStatusResponse {
        return api.startEnrollment()
    }
}
