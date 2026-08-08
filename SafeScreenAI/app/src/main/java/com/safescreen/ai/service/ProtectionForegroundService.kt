package com.safescreen.ai.service

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import com.safescreen.ai.detector.ExecuTorchInferenceEngine
import com.safescreen.ai.postprocess.DecisionEngine
import com.safescreen.ai.preprocess.ImagePreprocessor
import com.safescreen.ai.settings.SafetyModeManager

class ProtectionForegroundService : Service() {

    private lateinit var inferenceEngine: ExecuTorchInferenceEngine
    private lateinit var decisionEngine: DecisionEngine
    private lateinit var safetyModeManager: SafetyModeManager

    override fun onCreate() {
        super.onCreate()
        safetyModeManager = SafetyModeManager(this)
        inferenceEngine = ExecuTorchInferenceEngine(this)
        decisionEngine = DecisionEngine(safetyModeManager)
        Log.i("ProtectionService", "SafeScreen Protection Service initialized with ExecuTorch NPU engine.")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
