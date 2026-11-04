package com.example.voiceattendance.data.network

import com.example.voiceattendance.data.repository.AuthRepository
import okhttp3.Interceptor
import okhttp3.Response

class AuthInterceptor(private val authRepository: AuthRepository) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val requestBuilder = chain.request().newBuilder()
        
        // Exclude token endpoint from authorization header
        if (!chain.request().url.encodedPath.contains("/auth/token")) {
            authRepository.getToken()?.let { token ->
                requestBuilder.addHeader("Authorization", "Bearer $token")
            }
        }
        
        return chain.proceed(requestBuilder.build())
    }
}
