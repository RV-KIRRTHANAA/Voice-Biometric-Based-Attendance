package com.example.voiceattendance.data.repository

import com.example.voiceattendance.data.api.AttendanceApi
import com.example.voiceattendance.data.model.AttendanceResponse
import com.example.voiceattendance.data.model.ChallengeResponse
import com.example.voiceattendance.data.model.SessionInfo
import com.example.voiceattendance.data.model.StudentInfo
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File

class AttendanceRepository(private val api: AttendanceApi) {
    suspend fun getChallenge(sessionToken: String): Result<ChallengeResponse> = runCatching {
        api.getChallenge(sessionToken)
    }

    suspend fun getStudentInfo(): StudentInfo = api.getStudentInfo()

    suspend fun getSessions(): List<SessionInfo> = api.getSessions()

    suspend fun markAttendance(sessionToken: String, challengeId: String?, audioFile: File): AttendanceResponse {
        val tokenBody = sessionToken.toRequestBody("text/plain".toMediaTypeOrNull())
        val challengeBody = challengeId?.toRequestBody("text/plain".toMediaTypeOrNull())
        val fileBody  = audioFile.asRequestBody("audio/wav".toMediaTypeOrNull())
        val filePart  = MultipartBody.Part.createFormData("audio_file", audioFile.name, fileBody)
        return api.markAttendance(tokenBody, challengeBody, filePart)
    }
}
