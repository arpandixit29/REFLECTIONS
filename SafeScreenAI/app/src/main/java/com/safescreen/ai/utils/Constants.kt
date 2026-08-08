package com.safescreen.ai.utils

object Constants {
    // Model files targeted for Qualcomm Hexagon NPU execution via ExecuTorch QNN Delegate
    const val NSFW_MODEL_PTE = "models/nsfw_vit_int8.pte"
    const val DEEPFAKE_MODEL_PTE = "models/deepfake_efficientnet_int8.pte"
    const val YOLO_FACE_PTE = "models/yolo_face_int8.pte"
    const val SCAM_OCR_PTE = "models/scam_ocr_int8.pte"

    // Default Thresholds
    const val NSFW_BLUR_THRESHOLD = 0.55f
    const val NSFW_BLOCK_THRESHOLD = 0.85f
    const val DEEPFAKE_WARN_THRESHOLD = 0.75f
    const val SCAM_WARN_THRESHOLD = 0.70f

    // Performance Targets
    const val TARGET_LATENCY_MS = 15L
    const val TARGET_FPS = 60

    // Shared Preferences / Security Keys
    const val PREFS_NAME = "safescreen_prefs"
    const val KEY_CHILD_MODE_ENABLED = "child_mode_enabled"
    const val KEY_CHILD_MODE_PIN = "child_mode_pin"
    const val KEY_WOMENS_MODE_ENABLED = "womens_mode_enabled"
    const val KEY_PROTECTION_LEVEL = "protection_level" // LOW, MEDIUM, HIGH
    const val KEY_BLUR_STRENGTH = "blur_strength" // 30, 60, 100
}

enum class ProtectionLevel {
    LOW, MEDIUM, HIGH
}

enum class ActionRequired {
    NONE, BLUR, BLOCK, WARN_DEEPFAKE, WARN_SCAM, REDACT_PII
}
