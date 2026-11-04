package com.example.voiceattendance

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.core.content.ContextCompat
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.voiceattendance.audio.AudioRecorder
import com.example.voiceattendance.data.network.ApiClient
import com.example.voiceattendance.data.repository.AttendanceRepository
import com.example.voiceattendance.data.repository.AuthRepository
import com.example.voiceattendance.data.repository.EnrollmentRepository
import com.example.voiceattendance.navigation.AppNavGraph
import com.example.voiceattendance.ui.attendance.AttendanceViewModel
import com.example.voiceattendance.ui.enrollment.EnrollmentViewModel
import com.example.voiceattendance.ui.login.AuthViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize Data Layer
        val authRepository = AuthRepository(applicationContext)
        ApiClient.initialize(authRepository)
        
        val enrollmentRepository = EnrollmentRepository(ApiClient.enrollmentApi)
        val attendanceRepository = AttendanceRepository(ApiClient.attendanceApi)
        val audioRecorder = AudioRecorder()

        val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val authViewModel: AuthViewModel = viewModel(factory = object : ViewModelProvider.Factory {
                        override fun <T : ViewModel> create(modelClass: Class<T>): T {
                            return AuthViewModel(authRepository) as T
                        }
                    })
                    
                    val enrollmentViewModel: EnrollmentViewModel = viewModel(factory = object : ViewModelProvider.Factory {
                        override fun <T : ViewModel> create(modelClass: Class<T>): T {
                            return EnrollmentViewModel(enrollmentRepository, audioRecorder, cacheDir) as T
                        }
                    })
                    
                    val attendanceViewModel: AttendanceViewModel = viewModel(factory = object : ViewModelProvider.Factory {
                        override fun <T : ViewModel> create(modelClass: Class<T>): T {
                            return AttendanceViewModel(attendanceRepository, audioRecorder, cacheDir) as T
                        }
                    })

                    val startDest = if (authRepository.getToken() != null) "home" else "login"

                    AppNavGraph(
                        authViewModel = authViewModel,
                        enrollmentViewModel = enrollmentViewModel,
                        attendanceViewModel = attendanceViewModel,
                        startDestination = startDest
                    )
                }
            }
        }
    }
}
