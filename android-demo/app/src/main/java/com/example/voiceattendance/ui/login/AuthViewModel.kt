package com.example.voiceattendance.ui.login

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.voiceattendance.data.network.ApiClient
import com.example.voiceattendance.data.network.ApiErrorParser
import com.example.voiceattendance.data.repository.AuthRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import retrofit2.HttpException

sealed class LoginState {
    object Idle : LoginState()
    object Loading : LoginState()
    object Success : LoginState()
    data class Error(val message: String) : LoginState()
}

class AuthViewModel(private val authRepository: AuthRepository) : ViewModel() {
    private val _state = MutableStateFlow<LoginState>(LoginState.Idle)
    val state: StateFlow<LoginState> = _state

    fun login(studentId: String, password: String) {
        if (_state.value is LoginState.Loading) return
        
        viewModelScope.launch {
            _state.value = LoginState.Loading
            try {
                val response = ApiClient.authApi.login(studentId, password)
                authRepository.saveToken(response.accessToken)
                _state.value = LoginState.Success
            } catch (e: HttpException) {
                _state.value = LoginState.Error(ApiErrorParser.parse(e))
            } catch (e: Exception) {
                _state.value = LoginState.Error(e.message ?: "Login failed")
            }
        }
    }

    fun logout() {
        authRepository.clear()
        _state.value = LoginState.Idle
    }
}
