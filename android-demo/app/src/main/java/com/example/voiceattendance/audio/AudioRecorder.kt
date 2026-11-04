package com.example.voiceattendance.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream

class AudioRecorder {
    
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    
    @SuppressLint("MissingPermission")
    suspend fun startRecording(): ByteArray = withContext(Dispatchers.IO) {
        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        
        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            throw Exception("AudioRecord initialization failed")
        }
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize * 2
        )
        
        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            throw Exception("AudioRecord not initialized")
        }
        
        val outputStream = ByteArrayOutputStream()
        val buffer = ByteArray(bufferSize)
        
        audioRecord?.startRecording()
        isRecording = true
        
        // Record for approx 4 seconds for time-bounded challenge response
        val startTime = System.currentTimeMillis()
        while (isRecording && (System.currentTimeMillis() - startTime) < 4000) {
            val readResult = audioRecord?.read(buffer, 0, bufferSize) ?: 0
            if (readResult > 0) {
                outputStream.write(buffer, 0, readResult)
            }
        }
        
        stopRecording()
        return@withContext outputStream.toByteArray()
    }
    
    fun stopRecording() {
        isRecording = false
        audioRecord?.apply {
            if (recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                stop()
            }
            release()
        }
        audioRecord = null
    }
}
