package com.example.voiceattendance.data.repository

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import java.io.File

class AuthRepository(private val context: Context) {

    private var prefs: SharedPreferences

    init {
        prefs = try {
            createEncryptedPrefs()
        } catch (e: Exception) {
            // Fix for Android Keystore bug on reinstall
            val parentDir = context.filesDir.parentFile
            if (parentDir != null) {
                val prefFile = File(parentDir, "shared_prefs/secret_shared_prefs.xml")
                if (prefFile.exists()) {
                    prefFile.delete()
                }
            }
            try {
                createEncryptedPrefs()
            } catch (e2: Exception) {
                // Fallback to normal shared prefs if Keystore is completely broken
                context.getSharedPreferences("secret_shared_prefs_fallback", Context.MODE_PRIVATE)
            }
        }
    }

    private fun createEncryptedPrefs(): SharedPreferences {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()

        return EncryptedSharedPreferences.create(
            context,
            "secret_shared_prefs",
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    }

    fun saveToken(token: String) {
        prefs.edit().putString("JWT_TOKEN", token).apply()
    }

    fun getToken(): String? {
        return prefs.getString("JWT_TOKEN", null)
    }

    fun clear() {
        prefs.edit().clear().apply()
    }
}
