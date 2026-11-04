package com.example.voiceattendance.audio

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object WavEncoder {
    
    fun writeWavFile(pcmData: ByteArray, outputFile: File, sampleRate: Int = 16000, channels: Int = 1, bitsPerSample: Int = 16) {
        val byteRate = sampleRate * channels * (bitsPerSample / 8)
        val blockAlign = channels * (bitsPerSample / 8)
        val dataSize = pcmData.size
        val chunkSize = 36 + dataSize
        
        val header = ByteBuffer.allocate(44).apply {
            order(ByteOrder.LITTLE_ENDIAN)
            
            // RIFF chunk descriptor
            put("RIFF".toByteArray(Charsets.US_ASCII))
            putInt(chunkSize)
            put("WAVE".toByteArray(Charsets.US_ASCII))
            
            // "fmt " sub-chunk
            put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16) // Subchunk1Size (16 for PCM)
            putShort(1.toShort()) // AudioFormat (1 for PCM)
            putShort(channels.toShort())
            putInt(sampleRate)
            putInt(byteRate)
            putShort(blockAlign.toShort())
            putShort(bitsPerSample.toShort())
            
            // "data" sub-chunk
            put("data".toByteArray(Charsets.US_ASCII))
            putInt(dataSize)
        }.array()
        
        FileOutputStream(outputFile).use { fos ->
            fos.write(header)
            fos.write(pcmData)
        }
    }
}
