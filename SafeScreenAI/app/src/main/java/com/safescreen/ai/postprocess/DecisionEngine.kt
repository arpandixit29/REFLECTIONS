package com.safescreen.ai.postprocess

import com.safescreen.ai.utils.ActionRequired
import com.safescreen.ai.utils.Constants
import com.safescreen.ai.settings.SafetyModeManager

data class DetectionResult(
    val nsfwScore: Float,
    val deepfakeScore: Float,
    val isScamScreenshot: Boolean,
    val detectedPIIRegions: List<PIIRegion>,
    val primaryAction: ActionRequired,
    val warningMessage: String?
)

data class PIIRegion(
    val type: String, // "Aadhaar", "PAN", "Passport", "CreditCard", "Phone"
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float
)

class DecisionEngine(private val safetyModeManager: SafetyModeManager) {

    fun evaluateFrame(
        nsfwScore: Float,
        deepfakeScore: Float,
        isScam: Boolean,
        piiRegions: List<PIIRegion>
    ): DetectionResult {

        val isChildMode = safetyModeManager.isChildSafetyModeEnabled()
        val isWomensMode = safetyModeManager.isWomensSafetyModeEnabled()

        // Adjusted sensitivity thresholds depending on active modes
        val blurThreshold = if (isChildMode || isWomensMode) 0.40f else Constants.NSFW_BLUR_THRESHOLD
        val blockThreshold = if (isChildMode) 0.70f else Constants.NSFW_BLOCK_THRESHOLD

        var action = ActionRequired.NONE
        var warning: String? = null

        // 1. NSFW / Violence evaluation
        if (nsfwScore >= blockThreshold) {
            action = ActionRequired.BLOCK
            warning = "BLOCKED: Harmful visual content detected (${(nsfwScore * 100).toInt()}%)"
        } else if (nsfwScore >= blurThreshold) {
            action = ActionRequired.BLUR
            warning = "BLURRED: NSFW/Explicit content detected (${(nsfwScore * 100).toInt()}%)"
        } 
        // 2. Deepfake evaluation
        else if (deepfakeScore >= Constants.DEEPFAKE_WARN_THRESHOLD) {
            action = ActionRequired.WARN_DEEPFAKE
            warning = "⚠ WARNING: Synthetic/Deepfake image detected (${(deepfakeScore * 100).toInt()}% confidence)"
        }
        // 3. Scam screenshot evaluation
        else if (isScam) {
            action = ActionRequired.WARN_SCAM
            warning = "⚠ SCAM ALERT: Suspected phishing/banking scam screenshot detected!"
        }
        // 4. Document Privacy PII redaction evaluation
        else if (piiRegions.isNotEmpty()) {
            action = ActionRequired.REDACT_PII
            warning = "PROTECTED: Sensitive document details redacted local-first"
        }

        return DetectionResult(
            nsfwScore = nsfwScore,
            deepfakeScore = deepfakeScore,
            isScamScreenshot = isScam,
            detectedPIIRegions = piiRegions,
            primaryAction = action,
            warningMessage = warning
        )
    }
}
