package com.example.voiceattendance.data.model

import com.google.gson.annotations.SerializedName

data class ChallengeResponse(
    @SerializedName("challenge_id") val challenge_id: String,
    @SerializedName("phrase")       val phrase: String,
    @SerializedName("expires_at")   val expires_at: String
)
