package com.example.voiceattendance.data.model
import com.google.gson.annotations.SerializedName

data class EnrollmentResponse(
    @SerializedName("status")               val status: String,
    @SerializedName("accepted_samples")     val acceptedSamples: Int,
    @SerializedName("required_samples")     val requiredSamples: Int,
    @SerializedName("reason")               val reason: String,
    @SerializedName("has_active_reference") val hasActiveReference: Boolean = false
)

data class EnrollmentStatusResponse(
    @SerializedName("status")               val status: String,
    @SerializedName("accepted_samples")     val accepted_samples: Int,
    @SerializedName("required_samples")     val required_samples: Int,
    @SerializedName("has_active_reference") val has_active_reference: Boolean = false
)
