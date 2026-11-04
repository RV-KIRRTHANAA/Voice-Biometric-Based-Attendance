package com.example.voiceattendance.data.api
import com.example.voiceattendance.data.model.EnrollmentResponse
import com.example.voiceattendance.data.model.EnrollmentStatusResponse
import retrofit2.http.GET
import okhttp3.MultipartBody
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface EnrollmentApi {
    @Multipart
    @POST("api/v1/enrollment/")
    suspend fun processEnrollment(
        @Part audioFile: MultipartBody.Part
    ): EnrollmentResponse

    @GET("api/v1/enrollment/status")
    suspend fun getEnrollmentStatus(): EnrollmentStatusResponse

    @POST("api/v1/enrollment/start")
    suspend fun startEnrollment(): EnrollmentStatusResponse
}

