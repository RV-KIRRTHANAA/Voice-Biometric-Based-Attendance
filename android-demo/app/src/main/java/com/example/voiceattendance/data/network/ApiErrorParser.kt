package com.example.voiceattendance.data.network

import org.json.JSONObject
import retrofit2.HttpException

object ApiErrorParser {
    fun parse(exception: HttpException): String {
        return try {
            val errorBody = exception.response()?.errorBody()?.string()
            if (!errorBody.isNullOrBlank()) {
                val jsonObject = JSONObject(errorBody)
                // FastAPI default error key is 'detail'
                jsonObject.optString("detail", "Unknown server error")
            } else {
                "HTTP Error: ${exception.code()}"
            }
        } catch (e: Exception) {
            "HTTP Error: ${exception.code()}"
        }
    }
}
