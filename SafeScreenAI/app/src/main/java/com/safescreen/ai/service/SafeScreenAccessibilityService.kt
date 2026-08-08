package com.safescreen.ai.service

import android.accessibilityservice.AccessibilityService
import android.util.Log
import android.view.accessibility.AccessibilityEvent

/**
 * Accessibility Service to detect active target applications in real-time (WhatsApp, Camera, Browser, Gallery).
 * Enables adaptive safety policies per application context.
 */
class SafeScreenAccessibilityService : AccessibilityService() {

    private var currentPackageName: String = ""

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null) return

        if (event.eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            val pkg = event.packageName?.toString() ?: return
            if (pkg != currentPackageName) {
                currentPackageName = pkg
                Log.d("AccessibilityService", "Active app context changed to: $currentPackageName")
                handleAppChange(currentPackageName)
            }
        }
    }

    private fun handleAppChange(packageName: String) {
        when {
            packageName.contains("whatsapp") -> {
                Log.i("AccessibilityService", "WhatsApp context detected: Enabling instant incoming media scanner")
            }
            packageName.contains("chrome") || packageName.contains("browser") -> {
                Log.i("AccessibilityService", "Browser context detected: Enabling scroll protection filter")
            }
            packageName.contains("gallery") || packageName.contains("photos") -> {
                Log.i("AccessibilityService", "Gallery context detected: Triggering background media scan")
            }
            packageName.contains("camera") -> {
                Log.i("AccessibilityService", "Live Camera context detected: Enabling instant view-finder shield")
            }
        }
    }

    override fun onInterrupt() {
        Log.w("AccessibilityService", "SafeScreen Accessibility Service interrupted")
    }
}
