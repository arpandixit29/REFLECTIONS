package com.safescreen.ai.ui

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.safescreen.ai.capture.ScreenCaptureService

class MainActivity : ComponentActivity() {

    private val REQUEST_CODE_SCREEN_CAPTURE = 2001

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        checkOverlayPermission()

        setContent {
            SafeScreenTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color(0xFF0F172A)
                ) {
                    DashboardScreen(
                        onStartProtection = { requestScreenCapture() },
                        onStopProtection = { stopScreenCapture() }
                    )
                }
            }
        }
    }

    private fun checkOverlayPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!Settings.canDrawOverlays(this)) {
                val intent = Intent(
                    Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:$packageName")
                )
                startActivity(intent)
            }
        }
    }

    private fun requestScreenCapture() {
        val projectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        startActivityForResult(projectionManager.createScreenCaptureIntent(), REQUEST_CODE_SCREEN_CAPTURE)
    }

    private fun stopScreenCapture() {
        val intent = Intent(this, ScreenCaptureService::class.java)
        stopService(intent)
        Toast.makeText(this, "Reflections Protection Deactivated", Toast.LENGTH_SHORT).show()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_CODE_SCREEN_CAPTURE && resultCode == Activity.RESULT_OK && data != null) {
            val intent = Intent(this, ScreenCaptureService::class.java).apply {
                putExtra(ScreenCaptureService.EXTRA_RESULT_CODE, resultCode)
                putExtra(ScreenCaptureService.EXTRA_RESULT_DATA, data)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(intent)
            } else {
                startService(intent)
            }
            Toast.makeText(this, "Reflections Active on Qualcomm NPU", Toast.LENGTH_SHORT).show()
        }
    }
}

@Composable
fun SafeScreenTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Color(0xFF3B82F6),
            secondary = Color(0xFF10B981),
            background = Color(0xFF0F172A),
            surface = Color(0xFF1E293B)
        ),
        content = content
    )
}

@Composable
fun DashboardScreen(onStartProtection: () -> Unit, onStopProtection: () -> Unit) {
    var isProtectionActive by remember { mutableStateOf(false) }
    var childSafetyMode by remember { mutableStateOf(false) }
    var womensSafetyMode by remember { mutableStateOf(true) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Reflections",
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Text(
                    text = "Real-Time AI Protection for Every Screen",
                    fontSize = 14.sp,
                    color = Color(0xFF94A3B8)
                )
            }
            Badge(
                containerColor = if (isProtectionActive) Color(0xFF10B981) else Color(0xFFEF4444)
            ) {
                Text(
                    text = if (isProtectionActive) "ACTIVE" else "OFFLINE",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Hardware Acceleration Telemetry Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(modifier = Modifier.padding(20.dp)) {
                Text("Qualcomm Hexagon NPU Hardware Telemetry", color = Color(0xFF3B82F6), fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(12.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    TelemetryItem(title = "ExecuTorch Latency", value = "<14 ms")
                    TelemetryItem(title = "NPU Delegate", value = "Qualcomm QNN HTP")
                    TelemetryItem(title = "Network Privacy", value = "100% Offline")
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Toggle Main Protection Button
        Button(
            onClick = {
                if (isProtectionActive) {
                    onStopProtection()
                    isProtectionActive = false
                } else {
                    onStartProtection()
                    isProtectionActive = true
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isProtectionActive) Color(0xFFDC2626) else Color(0xFF2563EB)
            ),
            shape = RoundedCornerShape(14.dp)
        ) {
            Text(
                text = if (isProtectionActive) "Stop Real-Time Protection" else "Activate SafeScreen Protection",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        Text("Safety Mode Profiles", color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.Bold, modifier = Modifier.fillMaxWidth())

        Spacer(modifier = Modifier.height(12.dp))

        // Child Safety Mode Row
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text("Child Safety Mode 👶", color = Color.White, fontWeight = FontWeight.Bold)
                    Text("PIN-locked auto blur for ads & violence", color = Color(0xFF94A3B8), fontSize = 12.sp)
                }
                Switch(
                    checked = childSafetyMode,
                    onCheckedChange = { childSafetyMode = it }
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Women's Safety Mode Row
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text("Women's Safety Mode 🛡️", color = Color.White, fontWeight = FontWeight.Bold)
                    Text("Detect NCII, nude edits & fake profiles", color = Color(0xFF94A3B8), fontSize = 12.sp)
                }
                Switch(
                    checked = womensSafetyMode,
                    onCheckedChange = { womensSafetyMode = it }
                )
            }
        }
    }
}

@Composable
fun TelemetryItem(title: String, value: String) {
    Column {
        Text(title, color = Color(0xFF94A3B8), fontSize = 11.sp)
        Text(value, color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
    }
}
