package com.example.voiceattendance.data.network

import com.example.voiceattendance.BuildConfig
import com.example.voiceattendance.data.api.AttendanceApi
import com.example.voiceattendance.data.api.AuthApi
import com.example.voiceattendance.data.api.EnrollmentApi
import com.example.voiceattendance.data.repository.AuthRepository
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object ApiClient {
    private const val BASE_URL = BuildConfig.API_BASE_URL

    private var retrofit: Retrofit? = null

    fun initialize(authRepository: AuthRepository) {
        val logging = HttpLoggingInterceptor().apply {
            // Use BASIC level and redact Authorization so JWT tokens and binary WAV bodies are never logged
            redactHeader("Authorization")
            level = HttpLoggingInterceptor.Level.BASIC
        }

        val client = OkHttpClient.Builder()
            .addInterceptor(AuthInterceptor(authRepository))
            .addInterceptor(logging)
            .connectTimeout(120, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(120, TimeUnit.SECONDS)
            .build()

        retrofit = Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }

    val authApi: AuthApi by lazy { retrofit!!.create(AuthApi::class.java) }
    val enrollmentApi: EnrollmentApi by lazy { retrofit!!.create(EnrollmentApi::class.java) }
    val attendanceApi: AttendanceApi by lazy { retrofit!!.create(AttendanceApi::class.java) }
}
