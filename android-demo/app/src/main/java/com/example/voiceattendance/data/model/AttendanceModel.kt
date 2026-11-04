package com.example.voiceattendance.data.model

import com.google.gson.annotations.SerializedName

/**
 * Session descriptor returned by GET /attendance/sessions.
 * attendance_status is computed by the backend per authenticated student.
 * Android never computes or stores attendance state locally.
 */
data class SessionInfo(
    @SerializedName("session_id")         val sessionId: String,
    @SerializedName("session_token")      val sessionToken: String,
    @SerializedName("class_name")         val className: String,
    @SerializedName("starts_at")          val startsAt: String,
    @SerializedName("ends_at")            val endsAt: String,
    @SerializedName("attendance_status")  val attendanceStatus: String  // PRESENT | NOT_MARKED | UPCOMING | ENDED
) {
    val isPresent:  Boolean get() = attendanceStatus == "PRESENT"
    val isMarkable: Boolean get() = attendanceStatus == "NOT_MARKED"

    /** Short HH:MM - HH:MM range extracted from ISO-8601 strings */
    val timeRange: String get() {
        val s = startsAt.substringAfter("T").take(5)
        val e = endsAt.substringAfter("T").take(5)
        return "$s - $e"
    }
}

data class AttendanceResponse(
    @SerializedName("is_present") val is_present: Boolean,
    @SerializedName("reason")     val reason: String
)

data class StudentInfo(
    @SerializedName("student_id")   val studentId: String,
    @SerializedName("display_name") val displayName: String,
    @SerializedName("tenant_name")  val tenantName: String
)
