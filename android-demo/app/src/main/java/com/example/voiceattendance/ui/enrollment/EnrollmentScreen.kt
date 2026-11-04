package com.example.voiceattendance.ui.enrollment

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun EnrollmentScreen(viewModel: EnrollmentViewModel, onNavigateBack: () -> Unit) {
    val state by viewModel.state.collectAsState()

    LaunchedEffect(Unit) {
        viewModel.fetchStatus()
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("Voice Enrollment", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(32.dp))

        when (val s = state) {
            is EnrollmentState.Loading -> {
                CircularProgressIndicator()
            }
            is EnrollmentState.NotEnrolled -> {
                Text("Voice not enrolled")
                Spacer(modifier = Modifier.height(16.dp))
                Button(onClick = { viewModel.startEnrollment() }) { Text("Start Enrollment") }
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedButton(onClick = onNavigateBack) { Text("Back to Home") }
            }
            is EnrollmentState.InProgress -> {
                Text(if (s.hasActiveReference) "Re-enrollment in progress" else "Enrollment in progress")
                Text("Accepted samples: ${s.accepted} / ${s.required}")
                if (s.hasActiveReference) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        "Your previous voice profile remains active for attendance until re-enrollment completes.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))
                Button(onClick = { viewModel.recordAndUpload() }) { Text("Continue Enrollment") }
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedButton(onClick = onNavigateBack) { Text("Back to Home") }
            }
            is EnrollmentState.Recording -> {
                Text("Recording...")
                CircularProgressIndicator()
            }
            is EnrollmentState.Uploading -> {
                Text("Processing...")
                CircularProgressIndicator()
            }
            is EnrollmentState.Enrolled -> {
                Text("Voice enrolled", color = MaterialTheme.colorScheme.primary)
                Text("Your voice biometric is ready for attendance.")
                Spacer(modifier = Modifier.height(16.dp))
                Button(onClick = { viewModel.startEnrollment() }) { Text("Re-enroll voice") }
                Spacer(modifier = Modifier.height(16.dp))
                Button(onClick = onNavigateBack) { Text("Back to Home") }
            }
            is EnrollmentState.Error -> {
                Text(s.message, color = MaterialTheme.colorScheme.error)
                Spacer(modifier = Modifier.height(16.dp))
                Button(onClick = { viewModel.fetchStatus() }) { Text("Retry") }
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedButton(onClick = onNavigateBack) { Text("Back to Home") }
            }
        }
    }
}
