package com.safescreen.ai.overlay

import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.RenderEffect
import android.graphics.Shader
import android.os.Build
import android.os.IBinder
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import com.safescreen.ai.postprocess.DetectionResult
import com.safescreen.ai.utils.ActionRequired

/**
 * High-priority WindowManager system overlay service displaying instant hardware-accelerated
 * blur, warning banners, and PII redactor blocks on top of screen content.
 */
class OverlayService : Service() {

    private lateinit var windowManager: WindowManager
    private var overlayContainer: FrameLayout? = null
    private var warningBannerView: View? = null
    private var blurOverlayView: View? = null

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        initOverlayViews()
    }

    private fun initOverlayViews() {
        overlayContainer = FrameLayout(this)
        
        val layoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else
                WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        )

        windowManager.addView(overlayContainer, layoutParams)
    }

    private var isUserTemporarilyViewing = false

    /**
     * Updates overlay elements instantly based on the latest AI decision frame.
     */
    fun updateOverlay(result: DetectionResult) {
        overlayContainer?.removeAllViews()

        if (isUserTemporarilyViewing) {
            showReenableBanner()
            return
        }

        when (result.primaryAction) {
            ActionRequired.BLUR -> {
                showBlurLayer(false, result.warningMessage ?: "Sensitive Content Blurred")
            }
            ActionRequired.BLOCK -> {
                showBlurLayer(true, result.warningMessage ?: "Content Blocked")
            }
            ActionRequired.WARN_DEEPFAKE -> {
                showWarningBanner("⚠ Deepfake Media Detected", result.warningMessage ?: "This image may be AI generated.")
            }
            ActionRequired.WARN_SCAM -> {
                showWarningBanner("⚠ Phishing / Scam Alert", result.warningMessage ?: "Potential financial fraud screenshot detected!")
            }
            ActionRequired.REDACT_PII -> {
                showPIIRedactions(result)
            }
            ActionRequired.NONE -> {
                // Clear overlays
            }
        }
    }

    private fun showBlurLayer(isFullBlock: Boolean, messageText: String) {
        val blurView = View(this)
        if (isFullBlock) {
            blurView.setBackgroundColor(Color.BLACK)
        } else {
            blurView.setBackgroundColor(Color.parseColor("#CC0B0F17"))
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val blurEffect = RenderEffect.createBlurEffect(50f, 50f, Shader.TileMode.MIRROR)
                blurView.setRenderEffect(blurEffect)
            }
        }

        val contentLayout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(48, 48, 48, 48)
        }

        val textNotice = TextView(this).apply {
            text = "🛡 SAFE SCREEN AI\n$messageText"
            setTextColor(Color.WHITE)
            textSize = 16f
            gravity = Gravity.CENTER
            setBackgroundColor(Color.parseColor("#DD111827"))
            setPadding(32, 24, 32, 24)
        }
        contentLayout.addView(textNotice)

        if (!isFullBlock) {
            val clickToViewBtn = android.widget.Button(this).apply {
                text = "👁 Click to View Content"
                setTextColor(Color.WHITE)
                textSize = 14f
                setBackgroundColor(Color.parseColor("#3B82F6"))
                setPadding(32, 16, 32, 16)
                setOnClickListener {
                    isUserTemporarilyViewing = true
                    overlayContainer?.removeAllViews()
                    showReenableBanner()
                }
            }
            val btnParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = 24
            }
            contentLayout.addView(clickToViewBtn, btnParams)
        }

        val layoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        )
        overlayContainer?.addView(blurView, layoutParams)

        val centerParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.CENTER
        )
        overlayContainer?.addView(contentLayout, centerParams)
    }

    private fun showReenableBanner() {
        val bannerLayout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setBackgroundColor(Color.parseColor("#EE1E293B"))
            setPadding(24, 16, 24, 16)
        }

        val label = TextView(this).apply {
            text = "👁 Viewing Unblurred Content  "
            setTextColor(Color.YELLOW)
            textSize = 13f
        }
        bannerLayout.addView(label)

        val reenableBtn = android.widget.Button(this).apply {
            text = "🛡 Re-enable Protection"
            setTextColor(Color.WHITE)
            textSize = 12f
            setBackgroundColor(Color.parseColor("#10B981"))
            setOnClickListener {
                isUserTemporarilyViewing = false
                overlayContainer?.removeAllViews()
            }
        }
        bannerLayout.addView(reenableBtn)

        val params = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP
        )
        overlayContainer?.addView(bannerLayout, params)
    }

    private fun showWarningBanner(title: String, subtitle: String) {
        val banner = TextView(this).apply {
            text = "$title\n$subtitle"
            setTextColor(Color.YELLOW)
            textSize = 14f
            setBackgroundColor(Color.parseColor("#EE1F2937"))
            setPadding(24, 20, 24, 20)
            gravity = Gravity.CENTER
        }

        val bannerParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP
        )
        overlayContainer?.addView(banner, bannerParams)
    }

    private fun showPIIRedactions(result: DetectionResult) {
        for (region in result.detectedPIIRegions) {
            val redactorBox = View(this).apply {
                setBackgroundColor(Color.BLACK)
            }
            val boxParams = FrameLayout.LayoutParams(
                region.width.toInt(),
                region.height.toInt()
            ).apply {
                leftMargin = region.x.toInt()
                topMargin = region.y.toInt()
            }
            overlayContainer?.addView(redactorBox, boxParams)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (overlayContainer != null) {
            windowManager.removeView(overlayContainer)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
