package com.safescreen.ai.settings

import android.content.Context
import android.content.SharedPreferences
import com.safescreen.ai.utils.Constants

class SafetyModeManager(context: Context) {

    private val prefs: SharedPreferences = context.getSharedPreferences(Constants.PREFS_NAME, Context.MODE_PRIVATE)

    fun isChildSafetyModeEnabled(): Boolean {
        return prefs.getBoolean(Constants.KEY_CHILD_MODE_ENABLED, false)
    }

    fun setChildSafetyMode(enabled: Boolean, pin: String? = null) {
        val editor = prefs.edit()
        editor.putBoolean(Constants.KEY_CHILD_MODE_ENABLED, enabled)
        if (pin != null) {
            editor.putString(Constants.KEY_CHILD_MODE_PIN, pin)
        }
        editor.apply()
    }

    fun verifyChildModePin(pin: String): Boolean {
        val storedPin = prefs.getString(Constants.KEY_CHILD_MODE_PIN, "1234")
        return storedPin == pin
    }

    fun isWomensSafetyModeEnabled(): Boolean {
        return prefs.getBoolean(Constants.KEY_WOMENS_MODE_ENABLED, true)
    }

    fun setWomensSafetyMode(enabled: Boolean) {
        prefs.edit().putBoolean(Constants.KEY_WOMENS_MODE_ENABLED, enabled).apply()
    }

    fun getBlurStrength(): Int {
        return prefs.getInt(Constants.KEY_BLUR_STRENGTH, 60)
    }

    fun setBlurStrength(strength: Int) {
        prefs.edit().putInt(Constants.KEY_BLUR_STRENGTH, strength).apply()
    }
}
