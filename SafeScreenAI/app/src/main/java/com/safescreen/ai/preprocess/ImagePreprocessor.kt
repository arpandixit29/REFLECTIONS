package com.safescreen.ai.preprocess

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * High-performance image preprocessor for ExecuTorch NPU model inputs.
 * Handles downscaling, normalization (ImageNet mean/std), and FloatBuffer layout conversion.
 */
object ImagePreprocessor {

    private const val MODEL_INPUT_SIZE = 224
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /**
     * Converts a raw screen Bitmap into a normalized 1x3x224x224 NCHW FloatBuffer tensor.
     */
    fun preprocessForModel(bitmap: Bitmap): FloatBuffer {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true)
        val buffer = ByteBuffer.allocateDirect(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        val intValues = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
        scaledBitmap.getPixels(intValues, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

        // Fill Planar (NCHW) Format
        val rOffset = 0
        val gOffset = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE
        val bOffset = 2 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE

        for (i in 0 until MODEL_INPUT_SIZE * MODEL_INPUT_SIZE) {
            val valPixel = intValues[i]
            val r = ((valPixel shr 16) and 0xFF) / 255.0f
            val g = ((valPixel shr 8) and 0xFF) / 255.0f
            val b = (valPixel and 0xFF) / 255.0f

            buffer.put(rOffset + i, (r - MEAN[0]) / STD[0])
            buffer.put(gOffset + i, (g - MEAN[1]) / STD[1])
            buffer.put(bOffset + i, (b - MEAN[2]) / STD[2])
        }

        buffer.rewind()
        return buffer
    }
}
