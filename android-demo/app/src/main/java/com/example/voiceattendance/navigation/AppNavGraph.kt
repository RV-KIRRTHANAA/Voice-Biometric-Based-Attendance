package com.example.voiceattendance.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.voiceattendance.ui.attendance.AttendanceScreen
import com.example.voiceattendance.ui.attendance.AttendanceViewModel
import com.example.voiceattendance.ui.enrollment.EnrollmentScreen
import com.example.voiceattendance.ui.enrollment.EnrollmentViewModel
import com.example.voiceattendance.ui.home.HomeScreen
import com.example.voiceattendance.ui.login.AuthViewModel
import com.example.voiceattendance.ui.login.LoginScreen

@Composable
fun AppNavGraph(
    navController: NavHostController = rememberNavController(),
    authViewModel: AuthViewModel,
    enrollmentViewModel: EnrollmentViewModel,
    attendanceViewModel: AttendanceViewModel,
    startDestination: String = "login"
) {
    NavHost(navController = navController, startDestination = startDestination) {
        composable("login") {
            LoginScreen(
                viewModel = authViewModel,
                onLoginSuccess = {
                    navController.navigate("home") {
                        popUpTo("login") { inclusive = true }
                    }
                }
            )
        }
        composable("home") {
            HomeScreen(
                authViewModel = authViewModel,
                onNavigateToEnrollment = { navController.navigate("enrollment") },
                onNavigateToAttendance = { navController.navigate("attendance") },
                onLogout = {
                    authViewModel.logout()
                    attendanceViewModel.reset()
                    enrollmentViewModel.reset()
                    navController.navigate("login") {
                        popUpTo("home") { inclusive = true }
                    }
                }
            )
        }
        composable("enrollment") {
            EnrollmentScreen(
                viewModel = enrollmentViewModel,
                onNavigateBack = { navController.popBackStack() }
            )
        }
        composable("attendance") {
            AttendanceScreen(
                viewModel = attendanceViewModel,
                onNavigateBack = { navController.popBackStack() }
            )
        }
    }
}
