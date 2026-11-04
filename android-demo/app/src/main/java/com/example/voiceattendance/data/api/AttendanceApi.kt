package com.example.voiceattendance.data.api

import com.example.voiceattendance.data.model.AttendanceResponse
import com.example.voiceattendance.data.model.SessionInfo
import com.example.voiceattendance.data.model.StudentInfo
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.FormUrlEncoded
import retrofit2.http.Field
import com.example.voiceattendance.data.model.ChallengeResponse

interface AttendanceApi {
    @GET("api/v1/attendance/me")
    suspend fun getStudentInfo(): StudentInfo

    @GET("api/v1/attendance/sessions")
    suspend fun getSessions(): List<SessionInfo>

    @FormUrlEncoded
    @POST("api/v1/attendance/challenge")
    suspend fun getChallenge(
        @Field("session_token") sessionToken: String
    ): ChallengeResponse

    @Multipart
    @POST("api/v1/attendance/mark")
    suspend fun markAttendance(
        @Part("session_token") sessionToken: RequestBody,
        @Part("challenge_id") challengeId: RequestBody? = null,
        @Part audioFile: MultipartBody.Part
    ): AttendanceResponse
}
