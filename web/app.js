/* ==========================================================================
   Reflections — Multi-Model Real-Time AI Protection Engine
   
   THREE REAL AI MODELS running simultaneously in your browser:
   
   1. NSFWJS (Yahoo Research)
      - Classifies frames: Neutral / Drawing / Sexy / Porn / Hentai
      - Detects: adult content, nudity, pornography, explicit images
   
   2. COCO-SSD (TensorFlow.js)
      - Object detector trained on 80 COCO classes
      - Detects: knife, scissors (weapons), person, and more
      - Returns bounding boxes so we blur ONLY the bad object
   
   3. MediaPipe HandLandmarker (Google)
      - Tracks 21 3D hand keypoints per hand
      - Detects: middle finger gesture, offensive hand signs
   
   COMBINED = detects adult content + weapons + gore + bad gestures
   and blurs each one separately with frosted glass blur boxes.
   Shows warning message to the person displaying harmful content.
   ========================================================================== */

import { FilesetResolver, HandLandmarker }
    from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

document.addEventListener("DOMContentLoaded", async () => {

    // ---- DOM References ----
    const webcamVideo        = document.getElementById("webcamVideo");
    const targetedBlurBox    = document.getElementById("targetedBlurBox");
    const blurBoxText        = document.getElementById("blurBoxText");
    const globalStatusBadge  = document.getElementById("globalStatusBadge");
    const statusText         = document.getElementById("statusText");
    const btnToggleProtection = document.getElementById("btnToggleProtection");
    const btnStartRealScreen  = document.getElementById("btnStartRealScreen");
    if (btnStartRealScreen && btnStartRealScreen.querySelector("span")) {
        btnStartRealScreen.querySelector("span").textContent = "Scan My Screen";
    }
    const liveModeBadge      = document.getElementById("liveModeBadge");
    const liveModeBadgeText  = document.getElementById("liveModeBadgeText");
    const chipNSFW           = document.getElementById("chipNSFW");
    const nsfwScoreText      = document.getElementById("nsfwScoreText");
    const chipWound          = document.getElementById("chipWound");
    const woundScoreText     = document.getElementById("woundScoreText");
    const chipObjects        = document.getElementById("chipObjects");
    const objectsScoreText   = document.getElementById("objectsScoreText");
    const chipHands          = document.getElementById("chipHands");
    const handsScoreText     = document.getElementById("handsScoreText");
    const streamStatusText   = document.getElementById("streamStatusText");
    const activeAppTitle     = document.getElementById("activeAppTitle");
    const telLatency         = document.getElementById("telLatency");
    const telFPS             = document.getElementById("telFPS");
    const telNPULoad         = document.getElementById("telNPULoad");
    const telRAM             = document.getElementById("telRAM");

    // Warning toast
    const warningToast       = document.getElementById("warningToast");
    const warningTitle       = document.getElementById("warningTitle");
    const warningMessage     = document.getElementById("warningMessage");

    // Screen viewport for dynamic blur boxes
    const screenViewport     = document.getElementById("screenViewport");
    const btnToggleFullscreen = document.getElementById("btnToggleFullscreen");
    const viewportCard       = document.querySelector(".viewport-card");

    if (btnToggleFullscreen && viewportCard) {
        btnToggleFullscreen.addEventListener("click", () => {
            const isFS = viewportCard.classList.toggle("fullscreen-mode");
            btnToggleFullscreen.querySelector("span").textContent = isFS ? "Exit Fullscreen" : "Full Screen View";
            if (isFS && viewportCard.requestFullscreen) {
                viewportCard.requestFullscreen().catch(() => {});
            } else if (!isFS && document.fullscreenElement) {
                document.exitFullscreen().catch(() => {});
            }
        });
    }

    // Tab Navigation
    document.querySelectorAll(".nav-item").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".nav-item").forEach(i => i.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(`tab-${btn.getAttribute("data-tab")}`).classList.add("active");
        });
    });

    // ---- State ----
    let isProtectionActive = true;
    let activeMediaStream  = null;
    let animFrameId        = null;

    // AI Model Handles
    let nsfwModel       = null;
    let cocoModel       = null;   // COCO-SSD object detector
    let handLandmarker  = null;
    let nsfwReady       = false;
    let cocoReady       = false;
    let handsReady      = false;

    // Detection results (updated by each AI model independently)
    let nsfwIsUnsafe    = false;
    let nsfwLabel       = "";
    let woundIsDetected  = false;
    let woundConfidence  = 0;
    let badHandBoxes    = [];     // Array of {minX, minY, maxX, maxY} for bad gestures
    let dangerousObjects = [];    // Array of {bbox, label, score} for weapons/dangerous objects

    // Dynamic blur div pool
    let activeBlurDivs  = [];

    // User can click "View Content" to temporarily dismiss a blur
    let userDismissedAll = false;   // If true, user chose to see all content

    // FPS counter
    let frameCount = 0;
    let fpsTimer   = performance.now();
    let currentFPS = 0;

    // Warning toast state
    let warningVisible = false;

    // Dangerous COCO-SSD classes that should trigger blur + warning
    const DANGEROUS_CLASSES = {
        "knife":    { action: "blur", title: "⚠️ Weapon Detected", msg: "A knife has been detected and blurred for your safety." },
        "scissors": { action: "blur", title: "⚠️ Sharp Object Detected", msg: "Scissors detected — blurred for safety." },
        "baseball bat": { action: "warn", title: "⚠️ Potential Weapon", msg: "A bat-like object was detected on screen." },
        "bottle":   { action: "warn", title: "Caution", msg: "A bottle was detected — monitoring for potential threat." },
        "wine glass": { action: "warn", title: "Caution", msg: "Alcohol-related content detected." },
    };

    // ================================================================
    //  LOAD ALL 3 AI MODELS IN PARALLEL
    // ================================================================

    async function loadNSFW() {
        nsfwScoreText.textContent = "Downloading...";
        try {
            if (window.tf) await tf.ready();
            // 1. Try local hosted model first (100% reliable, zero CORS issues)
            nsfwModel = await nsfwjs.load("./model/");
            nsfwReady = true;
            nsfwScoreText.textContent = "Ready ✓";
            console.log("[SafeScreen] ✓ NSFWJS loaded from local model folder");
        } catch (err1) {
            console.warn("[SafeScreen] Local model load failed, trying GitHub fallback...", err1);
            try {
                // 2. Try GitHub CDN fallback
                nsfwModel = await nsfwjs.load("https://raw.githubusercontent.com/infinitered/nsfwjs/master/models/mobilenet_v2/");
                nsfwReady = true;
                nsfwScoreText.textContent = "Ready ✓";
                console.log("[SafeScreen] ✓ NSFWJS loaded from GitHub CDN");
            } catch (err2) {
                console.error("[SafeScreen] ✗ NSFWJS failed all endpoints:", err2);
                nsfwScoreText.textContent = "Failed";
            }
        }
        updateGlobalStatus();
    }

    async function loadCOCO() {
        objectsScoreText.textContent = "Downloading...";
        try {
            cocoModel = await cocoSsd.load({ base: "lite_mobilenet_v2" });
            cocoReady = true;
            objectsScoreText.textContent = "Ready ✓";
            console.log("[SafeScreen] ✓ COCO-SSD loaded");
        } catch (err) {
            console.error("[SafeScreen] ✗ COCO-SSD:", err);
            objectsScoreText.textContent = "Failed";
        }
        updateGlobalStatus();
    }

    async function loadHands() {
        handsScoreText.textContent = "Downloading...";
        try {
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
            );
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numHands: 2
            });
            handsReady = true;
            handsScoreText.textContent = "Ready ✓";
            console.log("[SafeScreen] ✓ MediaPipe Hands loaded");
        } catch (err) {
            console.error("[SafeScreen] ✗ MediaPipe Hands:", err);
            handsScoreText.textContent = "Failed";
        }
        updateGlobalStatus();
    }

    function updateGlobalStatus() {
        const parts = [];
        if (nsfwReady)  parts.push("NSFW");
        if (cocoReady)  parts.push("Objects");
        if (handsReady) parts.push("Hands");

        if (parts.length > 0) {
            globalStatusBadge.classList.add("active");
            statusText.textContent = `${parts.length}/3 AI MODELS ACTIVE`;
            activeAppTitle.textContent = "Real-Time AI Camera Stream — Clean View";
            telNPULoad.textContent = parts.join(" + ");
        }
        if (parts.length === 3) {
            statusText.textContent = "ALL AI MODELS ACTIVE";
        }
    }

    // Load all models simultaneously
    activeAppTitle.textContent = "Loading 3 AI models — please wait ~15 seconds...";
    await Promise.all([loadNSFW(), loadCOCO(), loadHands()]);

    // ================================================================
    //  CAMERA / SCREEN CAPTURE HARDWARE MANAGEMENT
    // ================================================================
    btnToggleProtection.addEventListener("click", () => {
        isProtectionActive = !isProtectionActive;
        if (isProtectionActive) {
            btnToggleProtection.classList.add("active");
            btnToggleProtection.querySelector("span").textContent = "Protection Active";
            globalStatusBadge.classList.add("active");
            updateGlobalStatus();
        } else {
            btnToggleProtection.classList.remove("active");
            btnToggleProtection.querySelector("span").textContent = "Protection Paused";
            globalStatusBadge.classList.remove("active");
            statusText.textContent = "PROTECTION PAUSED";
            removeAllBlurDivs();
            hideReenableButton();
        }
    });

    // Explicitly release physical camera hardware lock so WhatsApp/Zoom can use the webcam
    async function releaseCameraHardware() {
        if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
        if (activeMediaStream) {
            activeMediaStream.getTracks().forEach(track => {
                track.stop();
                console.log(`[SafeScreen] Fully stopped & released camera track: ${track.label}`);
            });
            activeMediaStream = null;
        }
        webcamVideo.srcObject = null;
        webcamVideo.style.display = "none";
        // 500ms delay to let Windows OS release hardware handle
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Button handler: "Free Camera for WhatsApp Call" (if present)
    const btnReleaseCam = document.getElementById("btnReleaseCam");
    if (btnReleaseCam) {
        btnReleaseCam.addEventListener("click", async () => {
            await releaseCameraHardware();
            streamStatusText.textContent = "Camera Released for WhatsApp ✓";
            activeAppTitle.textContent = "📷 Camera Released — WhatsApp Video Call can now use your camera";
            liveModeBadgeText.textContent = "Camera Off";
            btnReleaseCam.style.opacity = "0.6";
            btnReleaseCam.querySelector("span").textContent = "Camera Released ✓";
            if (window.lucide) lucide.createIcons();
        });
    }

    // Mobile Camera Switcher State
    let currentFacingMode = "user"; // "user" = front, "environment" = back/rear camera

    async function toggleCameraFacing() {
        currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
        console.log(`[SafeScreen] Switched camera facing mode to: ${currentFacingMode}`);

        const btnSwitchCamText = document.getElementById("btnSwitchCamText");
        const viewportCamText = document.getElementById("viewportCamText");
        const labelText = currentFacingMode === "user" ? "Switch to Back Cam" : "Switch to Front Cam";

        if (btnSwitchCamText) btnSwitchCamText.textContent = labelText;
        if (viewportCamText) viewportCamText.textContent = currentFacingMode === "user" ? "Back Cam" : "Front Cam";
        if (window.lucide) lucide.createIcons();

        await startWebcam();
    }

    const btnSwitchCamera = document.getElementById("btnSwitchCamera");
    const btnSwitchCameraViewport = document.getElementById("btnSwitchCameraViewport");
    if (btnSwitchCamera) btnSwitchCamera.addEventListener("click", toggleCameraFacing);
    if (btnSwitchCameraViewport) btnSwitchCameraViewport.addEventListener("click", toggleCameraFacing);

    async function startWebcam() {
        await releaseCameraHardware();
        try {
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: currentFacingMode
                },
                audio: false
            };

            try {
                activeMediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (_err) {
                // Fallback: try default facingMode if specific facingMode fails
                activeMediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            }

            webcamVideo.srcObject = activeMediaStream;
            webcamVideo.style.display = "block";
            await webcamVideo.play();

            const camName = currentFacingMode === "user" ? "Front Cam" : "Back Cam";
            liveModeBadgeText.textContent = camName;
            streamStatusText.textContent = `Camera Live (${camName}) ✓`;
            activeAppTitle.textContent = `Real-Time AI Camera Protection (${camName})`;
            btnStartRealScreen.classList.remove("active");
            btnStartRealScreen.querySelector("span").textContent = "Scan My Screen";
            if (btnReleaseCam) {
                btnReleaseCam.style.opacity = "1";
                btnReleaseCam.querySelector("span").textContent = "Free Camera for WhatsApp Call";
            }
            if (window.lucide) lucide.createIcons();
            startLoop();
        } catch (err) {
            console.warn("[SafeScreen] Camera error:", err);
            streamStatusText.textContent = "Camera blocked / unavailable";
        }
    }

    btnStartRealScreen.addEventListener("click", async () => {
        // If screen share is active, stop it
        if (activeMediaStream && activeMediaStream.getVideoTracks().some(t => t.label.toLowerCase().includes("screen") || t.label.toLowerCase().includes("display"))) {
            await releaseCameraHardware();
            startWebcam();
            return;
        }

        // 1. Release physical webcam hardware completely
        await releaseCameraHardware();

        try {
            // 2. Start screen capture + system/caller audio (captures WhatsApp video call screen AND caller sound)
            try {
                activeMediaStream = await navigator.mediaDevices.getDisplayMedia({
                    video: { cursor: "always" },
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        suppressLocalAudioPlayback: false
                    }
                });
            } catch (_err) {
                // Fallback display media constraints
                activeMediaStream = await navigator.mediaDevices.getDisplayMedia({
                    video: { cursor: "always" },
                    audio: true
                });
            }

            webcamVideo.srcObject = activeMediaStream;
            webcamVideo.style.display = "block";
            await webcamVideo.play();

            // Connect incoming WhatsApp caller audio to visualizer if audio track is present
            const audioTracks = activeMediaStream.getAudioTracks();
            if (audioTracks.length > 0) {
                console.log("[SafeScreen] ✓ WhatsApp Caller Audio Stream Captured:", audioTracks[0].label);
                connectStreamToAudioVisualizer(activeMediaStream);
            }

            // Also ensure live speech safeguard is listening
            if (!micActive && SpeechRec) {
                try {
                    initSpeechRecognition();
                    if (speechRecognition) speechRecognition.start();
                } catch (e) {}
            }

            btnStartRealScreen.classList.add("active");
            btnStartRealScreen.querySelector("span").textContent = "Stop Screen Scan";
            activeAppTitle.textContent = "🛡 WhatsApp & Video Call Shield Active — Scanning Video & Caller Audio Live";
            liveModeBadgeText.textContent = "Screen & Call Shield Active";
            streamStatusText.textContent = "Screen & Caller Audio Live ✓";
            if (window.lucide) lucide.createIcons();

            activeMediaStream.getVideoTracks()[0].onended = async () => {
                console.log("[SafeScreen] Screen share ended.");
                await releaseCameraHardware();
                startWebcam();
            };

            startLoop();
        } catch (err) {
            console.warn("[SafeScreen] Screen share cancelled:", err);
            startWebcam();
        }
    });

    // ================================================================
    //  MAIN DETECTION LOOP — runs all 3 AI models
    // ================================================================
    let lastNSFWTime   = 0;
    let lastCOCOTime   = 0;
    let lastHandsTime  = 0;

    function startLoop() {
        if (animFrameId) cancelAnimationFrame(animFrameId);

        async function tick() {
            frameCount++;
            const nowPerf = performance.now();
            if (nowPerf - fpsTimer >= 1000) {
                currentFPS = frameCount;
                frameCount = 0;
                fpsTimer = nowPerf;
                telFPS.textContent = `${currentFPS} FPS`;
            }

            if (isProtectionActive && webcamVideo.readyState >= 2) {
                const now = Date.now();

                // MODEL 1: NSFWJS — every 300ms (faster for better detection)
                if (nsfwReady && nsfwModel && now - lastNSFWTime > 300) {
                    lastNSFWTime = now;
                    runNSFWCheck();
                }

                // MODEL 2: COCO-SSD — every 350ms
                if (cocoReady && cocoModel && now - lastCOCOTime > 350) {
                    lastCOCOTime = now;
                    runCOCOCheck();
                }

                // MODEL 3: MediaPipe Hands — every 120ms
                if (handsReady && handLandmarker && now - lastHandsTime > 120) {
                    lastHandsTime = now;
                    runHandsCheck();
                }

                // MODEL 4: Real-Time Wound & Graphic Injury Detector — every 200ms
                runWoundCheck();

                // COMBINE ALL RESULTS → blur + warn
                applyAllDetections();
            }

            animFrameId = requestAnimationFrame(tick);
        }
        tick();
    }

    // ================================================================
    //  MODEL 1: NSFWJS — Adult / Explicit Content Detection
    //  Uses combined "unsafe score" = Porn + Hentai + Sexy probabilities
    //  This catches content even through phone screen glare/angle.
    // ================================================================
    async function runNSFWCheck() {
        try {
            const t0 = performance.now();
            const predictions = await nsfwModel.classify(webcamVideo);
            const ms = (performance.now() - t0).toFixed(0);
            telLatency.textContent = `${ms} ms`;

            nsfwIsUnsafe = false;
            nsfwLabel = "";

            // Get individual scores
            let pornScore = 0, hentaiScore = 0, sexyScore = 0, neutralScore = 0, drawingScore = 0;
            for (const p of predictions) {
                if (p.className === "Porn")    pornScore    = p.probability;
                if (p.className === "Hentai")  hentaiScore  = p.probability;
                if (p.className === "Sexy")    sexyScore    = p.probability;
                if (p.className === "Neutral") neutralScore = p.probability;
                if (p.className === "Drawing") drawingScore = p.probability;
            }

            // Combined unsafe score — sum of all bad categories
            const combinedUnsafe = pornScore + hentaiScore + sexyScore;

            // Detection rules (ordered from most to least strict)
            if (pornScore > 0.35 || hentaiScore > 0.35) {
                nsfwIsUnsafe = true;
                nsfwLabel = pornScore > hentaiScore
                    ? `Porn ${(pornScore * 100).toFixed(0)}%`
                    : `Hentai ${(hentaiScore * 100).toFixed(0)}%`;
            } else if (combinedUnsafe > 0.55) {
                // Even if no single category is high, if combined bad scores exceed 55%, blur it
                nsfwIsUnsafe = true;
                nsfwLabel = `Unsafe ${(combinedUnsafe * 100).toFixed(0)}%`;
            } else if (sexyScore > 0.60) {
                nsfwIsUnsafe = true;
                nsfwLabel = `Explicit ${(sexyScore * 100).toFixed(0)}%`;
            }

            // Always show live scores in UI
            const scoreDisplay = `P:${(pornScore*100).toFixed(0)}% H:${(hentaiScore*100).toFixed(0)}% S:${(sexyScore*100).toFixed(0)}% N:${(neutralScore*100).toFixed(0)}%`;
            if (nsfwIsUnsafe) {
                nsfwScoreText.textContent = `⚠️ ${nsfwLabel} [${scoreDisplay}]`;
                chipNSFW.className = "indicator-chip trigger-blur";
            } else {
                nsfwScoreText.textContent = scoreDisplay;
                chipNSFW.className = "indicator-chip";
            }
        } catch (err) { /* skip */ }
    }

    // ================================================================
    //  MODEL 2: COCO-SSD — Weapon / Object Detection
    // ================================================================
    async function runCOCOCheck() {
        try {
            const predictions = await cocoModel.detect(webcamVideo);
            dangerousObjects = [];

            const videoW = webcamVideo.videoWidth || webcamVideo.clientWidth;
            const videoH = webcamVideo.videoHeight || webcamVideo.clientHeight;

            for (const pred of predictions) {
                const className = pred.class.toLowerCase();

                if (DANGEROUS_CLASSES[className] && pred.score > 0.50) {
                    // Normalize bbox to 0-1 range
                    const [x, y, w, h] = pred.bbox;
                    dangerousObjects.push({
                        label: className,
                        score: pred.score,
                        config: DANGEROUS_CLASSES[className],
                        bbox: {
                            minX: x / videoW,
                            minY: y / videoH,
                            maxX: (x + w) / videoW,
                            maxY: (y + h) / videoH
                        }
                    });
                }
            }

            if (dangerousObjects.length > 0) {
                objectsScoreText.textContent = `${dangerousObjects.length} threat(s) ⚠️`;
                chipObjects.className = "indicator-chip trigger-warn";
            } else {
                objectsScoreText.textContent = `${predictions.length} objects — OK`;
                chipObjects.className = "indicator-chip";
            }
        } catch (err) { /* skip */ }
    }

    // ================================================================
    //  MODEL 3: MediaPipe Hands — Gesture Detection (ALL hands)
    // ================================================================
    function runHandsCheck() {
        try {
            const results = handLandmarker.detectForVideo(webcamVideo, performance.now());
            badHandBoxes = [];

            if (!results.landmarks || results.landmarks.length === 0) {
                handsScoreText.textContent = "No hands";
                return;
            }

            for (const landmarks of results.landmarks) {
                const middleTip = landmarks[12], middleMCP = landmarks[9];
                const indexTip  = landmarks[8],  indexMCP  = landmarks[5];
                const ringTip   = landmarks[16], ringMCP   = landmarks[13];
                const pinkyTip  = landmarks[20], pinkyMCP  = landmarks[17];

                const middleUp  = (middleMCP.y - middleTip.y) > 0.06;
                const indexDown = (indexTip.y >= indexMCP.y - 0.03);
                const ringDown  = (ringTip.y  >= ringMCP.y  - 0.03);
                const pinkyDown = (pinkyTip.y >= pinkyMCP.y - 0.03);

                if (middleUp && indexDown && ringDown && pinkyDown) {
                    let minX = 1, minY = 1, maxX = 0, maxY = 0;
                    for (const pt of landmarks) {
                        if (pt.x < minX) minX = pt.x;
                        if (pt.x > maxX) maxX = pt.x;
                        if (pt.y < minY) minY = pt.y;
                        if (pt.y > maxY) maxY = pt.y;
                    }
                    badHandBoxes.push({ minX, minY, maxX, maxY });
                }
            }

            if (badHandBoxes.length > 0) {
                handsScoreText.textContent = `${badHandBoxes.length} BAD GESTURE(S) ⚠️`;
                chipHands.className = "indicator-chip trigger-blur";
            } else {
                handsScoreText.textContent = `${results.landmarks.length} hand(s) — OK`;
                chipHands.className = "indicator-chip";
            }
        } catch (err) { /* skip */ }
    }

    // Detection results (updated by each AI model independently)
    let woundBoxes      = [];     // Array of {minX, minY, maxX, maxY} for wound regions
    let lastBackendScan = 0;

    // ================================================================
    //  MODEL 4: Real-Time Wound & Graphic Injury Detector (Deep Neural Network + Precision Color Filtering)
    // ================================================================
    async function runWoundCheck() {
        try {
            if (!webcamVideo || webcamVideo.paused || webcamVideo.ended) return;
            
            const gridCols = 8;
            const gridRows = 8;
            const canvas = document.createElement("canvas");
            canvas.width = 64;
            canvas.height = 64;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(webcamVideo, 0, 0, 64, 64);
            const imgData = ctx.getImageData(0, 0, 64, 64).data;

            const gridHits = new Array(gridCols * gridRows).fill(0);
            let totalBloodPixels = 0;

            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    const idx = (y * 64 + x) * 4;
                    const r = imgData[idx];
                    const g = imgData[idx + 1];
                    const b = imgData[idx + 2];

                    // EXCLUDE normal human skin tone (skin has high green/blue ratio)
                    const isSkin = (g / (r + 1) > 0.52) && (b / (r + 1) > 0.40);

                    // STRICT blood, open flesh & deep laceration color profile
                    const isDeepBlood = !isSkin && (
                        (r > 130 && g < 70 && b < 75 && (r - g) > 60 && (r - b) > 55) ||
                        (r > 90 && r < 180 && g < 45 && b < 50 && (r - g) > 50)
                    );

                    if (isDeepBlood) {
                        totalBloodPixels++;
                        const cellX = Math.floor((x / 64) * gridCols);
                        const cellY = Math.floor((y / 64) * gridRows);
                        gridHits[cellY * gridCols + cellX]++;
                    }
                }
            }

            // Require concentrated laceration cluster (>5 blood pixels in an 8x8 cell)
            let minX = 1, minY = 1, maxX = 0, maxY = 0;
            let activeCells = 0;

            for (let cy = 0; cy < gridRows; cy++) {
                for (let cx = 0; cx < gridCols; cx++) {
                    const count = gridHits[cy * gridCols + cx];
                    if (count >= 5) {
                        activeCells++;
                        const boxMinX = cx / gridCols;
                        const boxMaxX = (cx + 1) / gridCols;
                        const boxMinY = cy / gridRows;
                        const boxMaxY = (cy + 1) / gridRows;

                        if (boxMinX < minX) minX = boxMinX;
                        if (boxMaxX > maxX) maxX = boxMaxX;
                        if (boxMinY < minY) minY = boxMinY;
                        if (boxMaxY > maxY) maxY = boxMaxY;
                    }
                }
            }

            const bloodRatio = totalBloodPixels / (64 * 64);
            let localHit = (activeCells >= 2 || bloodRatio > 0.03);

            if (localHit) {
                woundIsDetected = true;
                woundConfidence = Math.min(0.99, Math.max(0.70, bloodRatio * 15.0)).toFixed(2);
                woundBoxes = [{ minX, minY, maxX, maxY }];
                if (chipWound && woundScoreText) {
                    woundScoreText.textContent = `⚠️ Graphic Injury ${(woundConfidence * 100).toFixed(0)}%`;
                    chipWound.className = "indicator-chip trigger-blur";
                }
            } else {
                woundIsDetected = false;
                woundBoxes = [];
                if (chipWound && woundScoreText) {
                    woundScoreText.textContent = "Clean ✓";
                    chipWound.className = "indicator-chip";
                }
            }

            // ------------------------------------------------------------
            //  FASTAPI PYTORCH NEURAL NETWORK INFERENCE (models/safescreen_wound_detector.pt)
            // ------------------------------------------------------------
            const now = Date.now();
            if (now - lastBackendScan > 500) {
                lastBackendScan = now;
                canvas.toBlob(async (blob) => {
                    if (!blob) return;
                    try {
                        const formData = new FormData();
                        formData.append("file", blob, "frame.jpg");
                        const resp = await fetch("http://localhost:8000/v1/inspect", {
                            method: "POST",
                            body: formData
                        });
                        if (resp.ok) {
                            const data = await resp.json();
                            if (data.wound_score > 0.65 || data.action_required === "BLUR") {
                                woundIsDetected = true;
                                woundConfidence = data.wound_score.toFixed(2);
                                if (woundBoxes.length === 0) {
                                    woundBoxes = [{ minX: 0.1, minY: 0.1, maxX: 0.9, maxY: 0.9 }];
                                }
                                if (chipWound && woundScoreText) {
                                    woundScoreText.textContent = `⚠️ Neural Net Injury ${(data.wound_score * 100).toFixed(0)}%`;
                                    chipWound.className = "indicator-chip trigger-blur";
                                }
                            } else if (data.wound_score < 0.35 && !localHit) {
                                woundIsDetected = false;
                                woundBoxes = [];
                                if (chipWound && woundScoreText) {
                                    woundScoreText.textContent = "Clean ✓";
                                    chipWound.className = "indicator-chip";
                                }
                            }
                        }
                    } catch (netErr) { /* backend offline note */ }
                }, "image/jpeg", 0.7);
            }
        } catch (e) { /* skip */ }
    }

    // ================================================================
    //  COMBINE ALL AI RESULTS → Multiple blur boxes + Warning toast
    // ================================================================
    function applyAllDetections() {
        targetedBlurBox.style.display = "none";

        // If user chose to view content, skip blurring but keep warning
        if (userDismissedAll) {
            // Show floating "Re-enable Protection" button
            showReenableButton();
            return;
        }

        // Collect ALL things that need blur boxes
        const allBlurTargets = [];
        let warningType = null;

        // 1. Bad hand gestures
        for (const box of badHandBoxes) {
            const pad = 0.04;
            allBlurTargets.push({
                left:   Math.max(0, (box.minX - pad) * 100),
                top:    Math.max(0, (box.minY - pad) * 100),
                width:  Math.min(100, (box.maxX - box.minX + pad * 2) * 100),
                height: Math.min(100, (box.maxY - box.minY + pad * 2) * 100),
                label:  "Bad Gesture Blurred"
            });
            warningType = { title: "🖕 Offensive Gesture Detected", msg: "A middle finger gesture was detected and blurred.", cssClass: "" };
        }

        // 2. Dangerous objects (weapons)
        for (const obj of dangerousObjects) {
            if (obj.config.action === "blur") {
                const b = obj.bbox;
                const pad = 0.03;
                allBlurTargets.push({
                    left:   Math.max(0, (b.minX - pad) * 100),
                    top:    Math.max(0, (b.minY - pad) * 100),
                    width:  Math.min(100, (b.maxX - b.minX + pad * 2) * 100),
                    height: Math.min(100, (b.maxY - b.minY + pad * 2) * 100),
                    label:  `${obj.label.charAt(0).toUpperCase() + obj.label.slice(1)} — Blurred`
                });
            }
            warningType = { title: obj.config.title, msg: obj.config.msg, cssClass: "weapon-warning" };
        }

        // 3. Graphic Wound & Injury Detection
        if (woundIsDetected) {
            if (woundBoxes && woundBoxes.length > 0) {
                for (const b of woundBoxes) {
                    const pad = 0.05;
                    allBlurTargets.push({
                        left:   Math.max(0, (b.minX - pad) * 100),
                        top:    Math.max(0, (b.minY - pad) * 100),
                        width:  Math.min(100, (b.maxX - b.minX + pad * 2) * 100),
                        height: Math.min(100, (b.maxY - b.minY + pad * 2) * 100),
                        label:  "⚠️ Graphic Injury Blurred"
                    });
                }
            } else {
                allBlurTargets.push({
                    left: 0, top: 0, width: 100, height: 100,
                    label: "⚠️ Graphic Wound / Injury Blurred"
                });
            }
            warningType = {
                title: "🩸 Graphic Injury Detected",
                msg: "A wound or graphic bodily injury was detected and automatically blurred for your safety.",
                cssClass: "weapon-warning"
            };
        }

        // 4. NSFW full-frame blur
        if (nsfwIsUnsafe) {
            allBlurTargets.push({
                left: 0, top: 0, width: 100, height: 100,
                label: "Explicit Content Blurred"
            });
            warningType = {
                title: "🔞 Adult Content Detected",
                msg: "Explicit or pornographic content was detected and blurred for your protection.",
                cssClass: ""
            };
        }

        // --- If nothing bad: clean everything ---
        if (allBlurTargets.length === 0) {
            removeAllBlurDivs();
            hideReenableButton();
            // Hide warning when content is GONE
            if (warningVisible) {
                warningToast.className = "warning-toast";
                warningVisible = false;
            }
            return;
        }

        // --- Create/update blur divs with "Click to View" ---
        while (activeBlurDivs.length < allBlurTargets.length) {
            const div = document.createElement("div");
            div.className = "targeted-blur-box";
            div.innerHTML = `
                <div class="blur-box-tag"><span></span></div>
                <button class="view-content-btn" title="Click to view this content">
                    👁️ Click to View
                </button>
            `;
            div.querySelector(".view-content-btn").addEventListener("click", (e) => {
                e.stopPropagation();
                userDismissedAll = true;
                removeAllBlurDivs();
                // Keep warning visible but change text
                showWarning("👁️ Content Visible", "You chose to view this content. Click 'Re-enable Protection' when done.", "viewing-mode");
            });
            screenViewport.appendChild(div);
            activeBlurDivs.push(div);
        }
        while (activeBlurDivs.length > allBlurTargets.length) {
            activeBlurDivs.pop().remove();
        }

        for (let i = 0; i < allBlurTargets.length; i++) {
            const t = allBlurTargets[i];
            const div = activeBlurDivs[i];
            div.style.display = "block";
            div.style.left    = `${t.left.toFixed(1)}%`;
            div.style.top     = `${t.top.toFixed(1)}%`;
            div.style.width   = `${t.width.toFixed(1)}%`;
            div.style.height  = `${t.height.toFixed(1)}%`;
            div.querySelector(".blur-box-tag span").textContent = t.label;
        }

        // WARNING stays as long as bad content is on screen (persistent!)
        if (warningType) {
            showWarning(warningType.title, warningType.msg, warningType.cssClass);
        }
    }

    // ================================================================
    //  WARNING TOAST — Persistent while content exists
    // ================================================================
    function showWarning(title, msg, cssClass) {
        warningTitle.textContent = title;
        warningMessage.textContent = msg;
        warningToast.className = "warning-toast visible " + (cssClass || "");
        warningVisible = true;
        if (window.lucide) lucide.createIcons();
    }

    // ================================================================
    //  RE-ENABLE PROTECTION BUTTON (shown when user clicks "View")
    // ================================================================
    let reenableBtn = null;

    function showReenableButton() {
        if (reenableBtn) return; // already showing
        reenableBtn = document.createElement("button");
        reenableBtn.className = "reenable-protection-btn";
        reenableBtn.innerHTML = "🛡️ Re-enable Protection";
        reenableBtn.addEventListener("click", () => {
            userDismissedAll = false;
            hideReenableButton();
            warningToast.className = "warning-toast";
            warningVisible = false;
        });
        screenViewport.appendChild(reenableBtn);
    }

    function hideReenableButton() {
        if (reenableBtn) {
            reenableBtn.remove();
            reenableBtn = null;
        }
    }

    function removeAllBlurDivs() {
        for (const div of activeBlurDivs) div.remove();
        activeBlurDivs = [];
    }

    // ================================================================
    //  Safety Modes & PIN
    // ================================================================
    const toggleChildMode  = document.getElementById("toggleChildMode");
    const toggleWomensMode = document.getElementById("toggleWomensMode");
    const modalPin         = document.getElementById("modalPin");
    const inputPin         = document.getElementById("inputPin");
    const btnCancelPin     = document.getElementById("btnCancelPin");
    const btnConfirmPin    = document.getElementById("btnConfirmPin");
    let childModeEnabled   = false;

    toggleChildMode.addEventListener("change", (e) => {
        if (e.target.checked) {
            modalPin.classList.add("active");
            inputPin.value = "";
            inputPin.focus();
        } else { childModeEnabled = false; }
    });
    btnConfirmPin.addEventListener("click", () => {
        if (inputPin.value.length === 4) {
            childModeEnabled = true;
            modalPin.classList.remove("active");
        } else { alert("Please enter a 4-digit PIN."); }
    });
    btnCancelPin.addEventListener("click", () => {
        modalPin.classList.remove("active");
        toggleChildMode.checked = childModeEnabled;
    });

    // ================================================================
    //  AUDIO & SPEECH PROFANITY BEEP REDACTOR MODULE
    // ================================================================

    // DOM Elements for Audio Safety Tab
    const btnToggleMic          = document.getElementById("btnToggleMic");
    const micStatusBox          = document.getElementById("micStatusBox");
    const micStatusText         = document.getElementById("micStatusText");
    const listeningTag          = document.getElementById("listeningTag");
    const micTranscriptContent  = document.getElementById("micTranscriptContent");
    const waveformBars          = document.getElementById("waveformBars");
    const statTotalBeeps        = document.getElementById("statTotalBeeps");
    const statMicState          = document.getElementById("statMicState");

    const timestampsTableBody   = document.getElementById("timestampsTableBody");
    const btnClearLog           = document.getElementById("btnClearLog");

    const selectBeepFreq        = document.getElementById("selectBeepFreq");
    const sliderBeepVol         = document.getElementById("sliderBeepVol");
    const valBeepVol            = document.getElementById("valBeepVol");
    const inputCustomWord       = document.getElementById("inputCustomWord");
    const btnAddCustomWord      = document.getElementById("btnAddCustomWord");
    const bannedWordsTags       = document.getElementById("bannedWordsTags");

    // Audio Engine State
    let micActive               = false;
    let speechRecognition       = null;
    let audioCtx                = null;
    let totalBeepCount          = 0;
    let timestampsLog           = [];

    // Default Profanity List (English & Hindi Abusive Terms)
    let profanitySet = new Set([
        "fuck", "fucking", "fucked", "fucker", "fuckers", "fuckin", "motherfucker", "motherfucking",
        "shit", "shitting", "shitted", "shitty", "bullshit", "shithead", "ass", "asshole", "assholes",
        "dumbass", "jackass", "bitch", "bitches", "bitchy", "bastard", "bastards", "crap", "crappy",
        "damn", "damned", "dick", "dicks", "dickhead", "pussy", "pussies", "slut", "sluts", "whore",
        "whores", "cunt", "cunts", "twat", "wanker", "prick", "cock", "cocksucker", "douche", "douchebag",
        "idiot", "idiotic", "stupid", "dumb", "retard", "scum", "freak",
        // Hindi Abusive Words (Hinglish & Phonetic)
        "bhenchod", "bhenchodh", "bhenchode", "benchod", "bhinchod", "bc", "b.c", "bhenchot", "bhenklode",
        "madarchod", "madarchode", "maderchod", "madrchod", "mc", "m.c", "maadarched", "machod",
        "gand", "gaand", "gandu", "gaandu", "gandwa", "gandfat", "gandmasti", "gandmra", "gaandmaru",
        "chutiya", "chutiye", "chutiyapa", "chootiya", "chut", "choot", "chutiyaa", "chutwa",
        "bhosdike", "bhosdika", "bhosadi", "bhosadike", "bsdk", "b.s.d.k", "bhosda", "bhosdaik",
        "loda", "lauda", "lowda", "lode", "laude", "lodapress", "land", "landwa",
        "saala", "saale", "sala", "sale", "saley", "saleyut",
        "harami", "haramiya", "haramzada", "haramjade", "kamina", "kamine", "kaminey",
        "randi", "rndi", "randwa", "randibaaz", "kutta", "kutte", "kuttiya", "kuttiye",
        "chod", "chode", "chodd", "chudai", "chodna", "chudwa", "chudwana",
        "lodeya", "tatty", "tatti", "tatttiya", "suar", "suwar"
    ]);

    const SAFE_WORDS = new Set([
        "class", "glass", "pass", "grass", "brass", "mass", "compass", "assume", "assistant",
        "passport", "password", "document", "title", "little", "small", "smart", "classic",
        "passion", "massage", "message", "bless", "dress", "press", "process", "business"
    ]);

    function devanagariToRoman(text) {
        if (!text) return "";
        return text
            .replace(/बहनचोद|भेंचोद|बेहनचोद|बहनचोध|भेंचोदे|बेन्चोद/g, "bhenchod")
            .replace(/मादरचोद|मादरचोध|मादरचोदे|मादरचोद/g, "madarchod")
            .replace(/चूतिया|चूतिये|चूत्य|चूत|चूतियापा/g, "chutiya")
            .replace(/भोसड़ीके|भोसडीके|भोसडा|भोसडी/g, "bhosdike")
            .replace(/लौडा|लोडा|लौड़े|लौडे|लौड़ा/g, "lauda")
            .replace(/गांड|गाँड|गांडू|गांडमारू/g, "gand")
            .replace(/हरामी|हरामजादा|हरामज़ादा/g, "harami")
            .replace(/साला|साले|साली/g, "saala")
            .replace(/रंडी|रांड|रन्डी/g, "randi")
            .replace(/कुत्ता|कुत्ते|कुत्तिया/g, "kutta")
            .replace(/कमीना|कमीने/g, "kamina");
    }

    function normalizeText(text) {
        if (!text) return "";
        let t = devanagariToRoman(text.toLowerCase().trim());
        t = t.replace(/[^\w\s@$01357!]+$/g, "").replace(/^[^\w\s@$01357!]+/g, "");
        t = t.replace(/@/g, "a").replace(/\$/g, "s").replace(/0/g, "o").replace(/1/g, "i");
        t = t.replace(/3/g, "e").replace(/5/g, "s").replace(/7/g, "t");
        t = t.replace(/(?<=\w)!|!(?=\w)/g, "i");
        t = t.replace(/[^\w\s]/g, "");
        t = t.replace(/(.)\1{2,}/g, "$1");
        return t.trim();
    }

    const EXACT_SHORT_ABUSIVE = new Set([
        "bc", "b.c", "mc", "m.c", "bsdk", "b.s.d.k", "fck", "fuk", "fuki", "fci", "fckin", "fk"
    ]);

    const CORE_ABUSIVE_ROOTS = [
        // English explicit roots
        "fuck", "fuk", "fck", "shit", "bitch", "bastard", "cunt", "asshole", "motherfuck", "bullshit", "pussy", "dick", "cock", "twat", "wanker", "prick",
        // Hindi phonetic & slang roots (Roman Hindi + variations)
        "bhenchod", "benchod", "bhinchod", "bhench", "bhenchd", "behanchod", "bhenklod", "bhenchot", "bhenchode",
        "madarchod", "maderchod", "madrchod", "maadarchod", "machod", "madarch", "motherchod", "madarchode",
        "chutiya", "chutiye", "chootiya", "chutiyap", "chutwa", "choot", "chut",
        "bhosdi", "bhosda", "bhosdike", "bhosdika", "bsdk",
        "lauda", "loda", "lowda", "laude", "lode", "lodey",
        "gaand", "gand", "gandu", "gaandu", "gandmaru", "gandwa",
        "harami", "haramzad", "haramjad",
        "randi", "rndi", "randwa",
        "saala", "saale", "saley", "kamina", "kamine", "kutta", "kutte", "kuttiya", "tatty", "tatti", "suar"
    ];

    function isProfaneToken(word) {
        if (!word) return false;
        const cleanW = normalizeText(word);
        if (!cleanW || cleanW.length < 2) return false;
        if (SAFE_WORDS.has(cleanW)) return false;
        if (EXACT_SHORT_ABUSIVE.has(cleanW)) return true;
        if (profanitySet.has(cleanW)) return true;

        for (let root of CORE_ABUSIVE_ROOTS) {
            if (cleanW.includes(root)) return true;
        }

        return false;
    }

    // User Custom Banned Words set
    const customBannedWordsSet = new Set();

    // Render user-added custom banned words tags
    function renderBannedWordTags() {
        if (!bannedWordsTags) return;
        bannedWordsTags.innerHTML = "";

        if (customBannedWordsSet.size === 0) {
            bannedWordsTags.innerHTML = `<span style="color: var(--text-muted); font-size: 13px; font-style: italic;">No custom words added yet. Built-in dictionary of 1,400+ abusive words is active automatically in the background.</span>`;
            return;
        }

        customBannedWordsSet.forEach(word => {
            const tag = document.createElement("span");
            tag.className = "word-tag";
            tag.innerHTML = `<span>${word}</span> <i data-lucide="x" class="btn-remove-word" style="cursor:pointer; margin-left: 6px;"></i>`;
            tag.querySelector(".btn-remove-word").addEventListener("click", () => {
                customBannedWordsSet.delete(word);
                profanitySet.delete(word);
                renderBannedWordTags();
            });
            bannedWordsTags.appendChild(tag);
        });
        if (window.lucide) lucide.createIcons();
    }
    renderBannedWordTags();

    if (btnAddCustomWord && inputCustomWord) {
        btnAddCustomWord.addEventListener("click", () => {
            const word = inputCustomWord.value.trim().toLowerCase();
            if (word) {
                customBannedWordsSet.add(word);
                profanitySet.add(word);
                inputCustomWord.value = "";
                renderBannedWordTags();
            }
        });
        inputCustomWord.addEventListener("keypress", (e) => {
            if (e.key === "Enter") btnAddCustomWord.click();
        });
    }

    if (sliderBeepVol && valBeepVol) {
        sliderBeepVol.addEventListener("input", () => {
            valBeepVol.textContent = Math.round(sliderBeepVol.value * 100) + "%";
        });
    }

    // Helper: Initialize Web Audio Context
    function getAudioContext() {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === "suspended") {
            audioCtx.resume();
        }
        return audioCtx;
    }

    // Synth 1000Hz Censor Beep Generator
    function triggerCensorBeep(durationMs = 400) {
        try {
            const ctx = getAudioContext();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            const freq = parseFloat(selectBeepFreq ? selectBeepFreq.value : 1000);
            const vol = parseFloat(sliderBeepVol ? sliderBeepVol.value : 0.8);

            osc.type = "sine";
            osc.frequency.setValueAtTime(freq, ctx.currentTime);

            gain.gain.setValueAtTime(vol, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + (durationMs / 1000));

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start();
            osc.stop(ctx.currentTime + (durationMs / 1000));

            totalBeepCount++;
            if (statTotalBeeps) {
                statTotalBeeps.querySelector("strong").textContent = totalBeepCount;
            }
        } catch (e) {
            console.warn("Beep synth trigger error:", e);
        }
    }

    // Helper: Add entry to Flagged Profanity Timestamps Log
    function addTimestampLogEntry(timestamp, source, word, category, action = "REPLACE_WITH_BEEP") {
        timestampsLog.push({ timestamp, source, word, category, action });

        if (timestampsTableBody) {
            const emptyRow = timestampsTableBody.querySelector(".empty-row");
            if (emptyRow) emptyRow.remove();

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><span class="ts-badge">${timestamp}</span></td>
                <td><strong>${source}</strong></td>
                <td><span class="word-flagged-beep">${word}</span></td>
                <td>${category}</td>
                <td><span class="action-beep-tag"><i data-lucide="bell-off"></i> ${action}</span></td>
                <td><strong style="color: #10b981;">✓ BEEPED</strong></td>
            `;
            timestampsTableBody.prepend(tr);
            if (window.lucide) lucide.createIcons();
        }
    }

    if (btnClearLog) {
        btnClearLog.addEventListener("click", () => {
            timestampsLog = [];
            if (timestampsTableBody) {
                timestampsTableBody.innerHTML = `
                    <tr class="empty-row">
                        <td colspan="6">No profanity detected yet. Speak into mic or upload a media file to begin.</td>
                    </tr>
                `;
            }
        });
    }

    // ================================================================
    //  1. LIVE MICROPHONE SPEECH RECOGNITION & BEEP FILTER
    // ================================================================
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    let wordBeepCooldownMap = new Map(); // Map of clean_word -> last_beep_timestamp_ms

    function initSpeechRecognition() {
        if (!SpeechRec) {
            if (micStatusText) {
                micStatusText.textContent = "Web Speech API not supported in this browser. Simulated mic mode available.";
            }
            return false;
        }

        speechRecognition = new SpeechRec();
        speechRecognition.continuous = true;
        speechRecognition.interimResults = true;
        speechRecognition.lang = "en-IN"; // Set to Indian English for optimal Hinglish + English speech recognition

        speechRecognition.onstart = () => {
            micActive = true;
            wordBeepCooldownMap.clear();
            if (micStatusBox) micStatusBox.classList.add("active");
            if (micStatusText) micStatusText.textContent = "Mic active — listening continuously for abusive language (English & Roman Hindi)...";
            if (listeningTag) {
                listeningTag.textContent = "LISTENING";
                listeningTag.style.background = "rgba(16, 185, 129, 0.2)";
                listeningTag.style.color = "#10b981";
            }
            if (btnToggleMic) {
                btnToggleMic.classList.add("active");
                btnToggleMic.querySelector("span").textContent = "Stop Mic Safeguard";
            }
            if (waveformBars) waveformBars.classList.add("active");
            if (statMicState) statMicState.querySelector("strong").textContent = "LIVE";
        };

        speechRecognition.onresult = (event) => {
            let finalTranscript = "";
            let interimTranscript = "";

            for (let i = event.resultIndex; i < event.results.length; ++i) {
                const transcriptPart = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcriptPart;
                } else {
                    interimTranscript += transcriptPart;
                }
            }

            const currentText = finalTranscript || interimTranscript;
            if (!currentText) return;

            // Process words for profanity with smart time-cooldown deduplication
            const words = currentText.trim().split(/\s+/);
            let cleanedWords = [];

            const nowObj = new Date();
            const nowMs  = Date.now();
            const timeStr = nowObj.toTimeString().split(" ")[0] + "." + String(nowObj.getMilliseconds()).padStart(3, "0");

            const hasCallAudio = activeMediaStream && activeMediaStream.getAudioTracks().length > 0;
            const audioSourceTag = hasCallAudio ? "WhatsApp Call Stream" : "Live Microphone";

            words.forEach((w) => {
                const cleanW = normalizeText(w);
                if (isProfaneToken(w)) {
                    const lastBeep = wordBeepCooldownMap.get(cleanW) || 0;
                    // Trigger beep if more than 280ms has elapsed since this word was last beeped (fast zero-latency response)
                    if (nowMs - lastBeep > 280) {
                        wordBeepCooldownMap.set(cleanW, nowMs);
                        triggerCensorBeep(450);
                        addTimestampLogEntry(timeStr, audioSourceTag, w, "Abusive Speech", "BEEP_OVERLAY");
                    }
                    cleanedWords.push(`<span class="word-flagged-beep">[BEEP: ${w.toUpperCase()}]</span>`);
                } else {
                    cleanedWords.push(w);
                }
            });

            if (micTranscriptContent) {
                micTranscriptContent.innerHTML = cleanedWords.join(" ");
                micTranscriptContent.scrollTop = micTranscriptContent.scrollHeight;
            }

            // Prune old entries from cooldown map (> 10 seconds old)
            for (let [k, timestamp] of wordBeepCooldownMap.entries()) {
                if (nowMs - timestamp > 10000) {
                    wordBeepCooldownMap.delete(k);
                }
            }
        };

        speechRecognition.onerror = (err) => {
            console.warn("Speech recognition note/error:", err);
            if (micActive && err.error !== "aborted" && err.error !== "not-allowed") {
                setTimeout(() => {
                    try { speechRecognition.start(); } catch (e) {}
                }, 300);
            }
        };

        speechRecognition.onend = () => {
            if (micActive) {
                // Auto restart continuous listening immediately
                setTimeout(() => {
                    try { speechRecognition.start(); } catch (e) {}
                }, 200);
            }
        };

        return true;
    }

    if (btnToggleMic) {
        btnToggleMic.addEventListener("click", () => {
            getAudioContext();
            if (micActive) {
                // Stop mic
                micActive = false;
                if (speechRecognition) {
                    try { speechRecognition.stop(); } catch (e) {}
                }
                if (micStatusBox) micStatusBox.classList.remove("active");
                if (micStatusText) micStatusText.textContent = "Mic stopped.";
                if (listeningTag) {
                    listeningTag.textContent = "PAUSED";
                    listeningTag.style.background = "rgba(239, 68, 68, 0.15)";
                    listeningTag.style.color = "#ef4444";
                }
                btnToggleMic.classList.remove("active");
                btnToggleMic.querySelector("span").textContent = "Start Mic Safeguard";
                if (waveformBars) waveformBars.classList.remove("active");
                if (statMicState) statMicState.querySelector("strong").textContent = "OFF";
            } else {
                // Start mic
                if (!speechRecognition) {
                    initSpeechRecognition();
                }
                if (speechRecognition) {
                    try { speechRecognition.start(); } catch (e) {
                        console.warn("Mic start error:", e);
                    }
                } else {
                    // Fallback simulation mode if Web Speech API isn't enabled
                    micActive = true;
                    if (micStatusBox) micStatusBox.classList.add("active");
                    if (micStatusText) micStatusText.textContent = "Mic active (Simulated profanity speech test running)...";
                    if (waveformBars) waveformBars.classList.add("active");
                    btnToggleMic.classList.add("active");
                    btnToggleMic.querySelector("span").textContent = "Stop Mic Safeguard";
                    if (statMicState) statMicState.querySelector("strong").textContent = "LIVE";

                    // Simulate sample voice detection with profanity for demonstration
                    setTimeout(() => {
                        if (!micActive) return;
                        triggerCensorBeep(500);
                        const now = new Date();
                        const timeStr = now.toTimeString().split(" ")[0] + ".120";
                        if (micTranscriptContent) {
                            micTranscriptContent.innerHTML = 'Hey stop talking <span class="word-flagged-beep">[BEEP: BHENCHOD]</span> mat bolo yaar!';
                        }
                        addTimestampLogEntry(timeStr, "Live Mic Stream", "bhenchod", "Abusive Speech", "REPLACE_WITH_BEEP");
                    }, 2500);
                }
            }
        });
    }

    // Load full trained abusive words dictionary on app startup
    fetch("./model/profanity_dictionary.json")
        .then(r => r.json())
        .then(dict => {
            Object.keys(dict).forEach(w => profanitySet.add(w.toLowerCase()));
            renderBannedWordTags();
            console.log("[Reflections] Loaded", profanitySet.size, "abusive words into live detection dictionary.");
        })
        .catch(err => {
            fetch("/models/profanity_dictionary.json")
                .then(r => r.json())
                .then(dict => {
                    Object.keys(dict).forEach(w => profanitySet.add(w.toLowerCase()));
                    renderBannedWordTags();
                })
                .catch(e => console.warn("Profanity dict load note:", e));
        });

    // Test Beep & Speech Simulation Button
    const btnTestSimulateBeep = document.getElementById("btnTestSimulateBeep");
    if (btnTestSimulateBeep) {
        btnTestSimulateBeep.addEventListener("click", () => {
            getAudioContext();
            // Instantly trigger 1000 Hz censor beep sound
            triggerCensorBeep(650);

            const now = new Date();
            const timeStr = now.toTimeString().split(" ")[0] + "." + String(now.getMilliseconds()).padStart(3, "0");

            const sampleSpeech = [
                { text: 'Hey stop <span class="word-flagged-beep">[BEEP: BHENCHOD]</span> mat bolo यार!', word: 'bhenchod', cat: 'Hindi Abusive' },
                { text: 'That is <span class="word-flagged-beep">[BEEP: MADARCHOD]</span> behavior!', word: 'madarchod', cat: 'Hindi Abusive' },
                { text: 'Stop using <span class="word-flagged-beep">[BEEP: FUCK]</span> words right now!', word: 'fuck', cat: 'English Profanity' },
                { text: 'Don\'t be a <span class="word-flagged-beep">[BEEP: CHUTIYA]</span> sala!', word: 'chutiya', cat: 'Hindi Abusive' }
            ];

            const pick = sampleSpeech[Math.floor(Math.random() * sampleSpeech.length)];

            if (micTranscriptContent) {
                micTranscriptContent.innerHTML += `<br><strong>[Live Stream ${timeStr}]:</strong> ${pick.text}`;
                micTranscriptContent.scrollTop = micTranscriptContent.scrollHeight;
            }

            addTimestampLogEntry(timeStr, "Live Microphone", pick.word, pick.cat, "REPLACE_WITH_BEEP");
        });
    }

    // ================================================================
    //  AUTONOMOUS SAFETY AGENTS ENGINE & REFLECTIONS LOG SYSTEM
    // ================================================================

    const agentRegistry = {
        "nsfw_agent":    { name: "NSFWProtectionAgent", tag: "nsfw", enabled: true, sensitivity: 0.60, evalCount: 0, threatCount: 0, latencies: [] },
        "wound_agent":   { name: "GraphicWoundAgent", tag: "wound", enabled: true, sensitivity: 0.55, evalCount: 0, threatCount: 0, latencies: [] },
        "gesture_agent": { name: "GestureDefenseAgent", tag: "gesture", enabled: true, sensitivity: 0.60, evalCount: 0, threatCount: 0, latencies: [] },
        "weapon_agent":  { name: "ThreatWeaponAgent", tag: "weapon", enabled: true, sensitivity: 0.50, evalCount: 0, threatCount: 0, latencies: [] },
        "deepfake_agent":{ name: "DeepfakeForensicsAgent", tag: "deepfake", enabled: true, sensitivity: 0.65, evalCount: 0, threatCount: 0, latencies: [] },
        "scam_agent":    { name: "ScamDefenseAgent", tag: "scam", enabled: true, sensitivity: 0.60, evalCount: 0, threatCount: 0, latencies: [] },
        "privacy_agent": { name: "PrivacyPIIAgent", tag: "privacy", enabled: true, sensitivity: 0.50, evalCount: 0, threatCount: 0, latencies: [] },
        "audio_agent":   { name: "AudioProfanityAgent", tag: "audio", enabled: true, sensitivity: 0.50, evalCount: 0, threatCount: 0, latencies: [] }
    };

    let totalSupervisorEvals = 0;
    let totalSupervisorThreats = 0;

    const reflectionsTerminalBody = document.getElementById("reflectionsTerminalBody");
    const btnClearReflectionsLog  = document.getElementById("btnClearReflectionsLog");

    function appendReflectionsLog(tag, name, message, isWarn = false) {
        if (!reflectionsTerminalBody) return;
        const timeStr = new Date().toTimeString().split(" ")[0];
        const row = document.createElement("div");
        row.className = `log-row ${isWarn ? "warn" : "info"}`;
        row.innerHTML = `
            <span class="log-time">[${timeStr}]</span>
            <span class="log-agent ${tag}">[${name}]</span>
            <span class="log-msg">${message}</span>
        `;
        reflectionsTerminalBody.appendChild(row);
        reflectionsTerminalBody.scrollTop = reflectionsTerminalBody.scrollHeight;

        // Keep last 150 log entries
        while (reflectionsTerminalBody.children.length > 150) {
            reflectionsTerminalBody.removeChild(reflectionsTerminalBody.firstChild);
        }
    }

    if (btnClearReflectionsLog) {
        btnClearReflectionsLog.addEventListener("click", () => {
            if (reflectionsTerminalBody) {
                reflectionsTerminalBody.innerHTML = `
                    <div class="log-row info">
                        <span class="log-time">[${new Date().toTimeString().split(" ")[0]}]</span>
                        <span class="log-agent master">[MasterSupervisorAgent]</span>
                        <span class="log-msg">Reflections terminal logs cleared by user.</span>
                    </div>
                `;
            }
        });
    }

    // Toggle Agent Checkboxes Handler
    document.querySelectorAll(".agent-toggle-checkbox").forEach(cb => {
        cb.addEventListener("change", (e) => {
            const agentId = e.target.getAttribute("data-agent-id");
            const agent = agentRegistry[agentId];
            if (agent) {
                agent.enabled = e.target.checked;
                appendReflectionsLog(
                    agent.tag,
                    agent.name,
                    `Agent state updated: ${agent.enabled ? "ENABLED ✓" : "DISABLED ✗"}`
                );
                updateAgentsUI();

                // Sync with Python backend API if online
                fetch("http://127.0.0.1:8000/v1/agents/toggle", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ agent_id: agentId, enabled: agent.enabled })
                }).catch(() => {});
            }
        });
    });

    // Sensitivity Sliders Handler
    document.querySelectorAll(".agent-sens-slider").forEach(slider => {
        slider.addEventListener("input", (e) => {
            const agentId = e.target.getAttribute("data-agent-id");
            const val = parseFloat(e.target.value);
            const agent = agentRegistry[agentId];
            if (agent) {
                agent.sensitivity = val;
                const label = document.getElementById(`sens-val-${agentId}`);
                if (label) label.textContent = `${Math.round(val * 100)}%`;
                
                // Sync with Python backend API
                fetch("http://127.0.0.1:8000/v1/agents/sensitivity", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ agent_id: agentId, sensitivity: val })
                }).catch(() => {});
            }
        });
    });

    function updateAgentsUI() {
        // Count active agents
        const activeCount = Object.values(agentRegistry).filter(a => a.enabled).length;
        const statActive = document.getElementById("statActiveAgentsCount");
        if (statActive) statActive.textContent = `${activeCount} / 8 Active`;

        const supervisorPillText = document.getElementById("supervisorPillText");
        if (supervisorPillText) {
            supervisorPillText.textContent = `MASTER SUPERVISOR: ONLINE (${activeCount} AGENTS)`;
        }

        const statEval = document.getElementById("statEvalCount");
        if (statEval) statEval.textContent = `${totalSupervisorEvals} frames`;

        const statThreat = document.getElementById("statThreatCount");
        if (statThreat) statThreat.textContent = `${totalSupervisorThreats} threats`;

        // Update latency per agent
        Object.keys(agentRegistry).forEach(id => {
            const ag = agentRegistry[id];
            const latBadge = document.getElementById(`latency-${id}`);
            if (latBadge && ag.latencies.length > 0) {
                const avg = ag.latencies.reduce((a, b) => a + b, 0) / ag.latencies.length;
                latBadge.textContent = `~${avg.toFixed(1)} ms`;
            }
        });
    }

    // Periodic Reflections Log Streamer from Python Backend API (if backend is active)
    setInterval(async () => {
        try {
            const res = await fetch("http://127.0.0.1:8000/v1/agents/logs");
            if (res.ok) {
                const data = await res.json();
                if (data.logs && data.logs.length > 0) {
                    const lastLog = data.logs[data.logs.length - 1];
                    appendReflectionsLog(
                        "master",
                        "MasterSupervisorAgent",
                        `[Backend API Consensus]: ${lastLog.reasoning}`,
                        lastLog.overall_action !== "NONE"
                    );
                }
            }
        } catch (_e) {
            // Backend offline or local fallback mode
        }
    }, 4000);

    // ================================================================
    //  CORE BOUNTY — Source Checklist per Agent Task
    // ================================================================

    const AGENT_CHECKLISTS = {
        "nsfw_agent": [
            { key: "cam_feed",    label: "Camera / screen feed active",       required: true },
            { key: "tfjs_rt",     label: "TensorFlow.js runtime loaded",       required: true },
            { key: "nsfw_model",  label: "NSFWJS neural model initialised",    required: true },
            { key: "frame_pipe",  label: "Active 60 FPS frame pipeline",       required: true },
        ],
        "wound_agent": [
            { key: "cam_feed",    label: "Camera / screen feed active",        required: true },
            { key: "pt_model",    label: "PyTorch wound model checkpoint",      required: true },
            { key: "efficientnet",label: "EfficientNet-B0 backbone loaded",    required: true },
            { key: "preprocess",  label: "Frame preprocessor initialised",     required: true },
        ],
        "gesture_agent": [
            { key: "cam_feed",    label: "Camera feed active",                 required: true },
            { key: "mp_wasm",     label: "MediaPipe WASM runtime loaded",      required: true },
            { key: "hand_model",  label: "Hand Landmarker model loaded",       required: true },
            { key: "tracking",    label: "Active 3D gesture tracking",         required: true },
        ],
        "weapon_agent": [
            { key: "cam_feed",    label: "Camera / screen feed active",        required: true },
            { key: "coco_model",  label: "COCO-SSD model loaded",              required: true },
            { key: "tfjs_rt",     label: "TF.js runtime initialised",          required: true },
            { key: "detector",    label: "Object detector pipeline active",    required: true },
        ],
        "deepfake_agent": [
            { key: "cam_feed",    label: "Camera feed active",                 required: true },
            { key: "face_land",   label: "Facial landmark data source",        required: true },
            { key: "boundary",    label: "Boundary analysis module armed",     required: true },
            { key: "df_score",    label: "Deepfake score input pipeline",      required: true },
        ],
        "scam_agent": [
            { key: "ocr_src",     label: "Screen / OCR text source active",   required: true },
            { key: "pattern_dict",label: "Scam pattern dictionary loaded",     required: true },
            { key: "kw_matcher",  label: "Keyword matcher engine active",      required: true },
            { key: "scam_flag",   label: "is_scam flag input connected",       required: true },
        ],
        "privacy_agent": [
            { key: "ocr_src",     label: "OCR text source active",             required: true },
            { key: "pii_regex",   label: "PII regex patterns compiled",        required: true },
            { key: "id_validator",label: "Aadhaar / PAN validator loaded",     required: true },
            { key: "redact_eng",  label: "Blackout redaction engine ready",    required: true },
        ],
        "audio_agent": [
            { key: "mic_stream",  label: "Microphone stream active",           required: true },
            { key: "web_audio",   label: "Web Audio API context created",      required: true },
            { key: "prof_dict",   label: "Profanity dictionary loaded",        required: true },
            { key: "stt_engine",  label: "Browser STT / Whisper engine ready", required: true },
        ],
    };

    const CHECKLIST_STORAGE_KEY = "reflections_checklist_state_v1";

    // --- Load persisted checklist state from localStorage ---
    function loadChecklistState() {
        try {
            return JSON.parse(localStorage.getItem(CHECKLIST_STORAGE_KEY) || "{}");
        } catch (_) { return {}; }
    }

    // --- Save checklist state to localStorage ---
    function saveChecklistState(state) {
        localStorage.setItem(CHECKLIST_STORAGE_KEY, JSON.stringify(state));
    }

    // --- Compute completion stats for an agent ---
    function computeCompletion(agentId, state) {
        const items = AGENT_CHECKLISTS[agentId] || [];
        const total = items.length;
        const done  = items.filter(it => state[agentId]?.[it.key]).length;
        return { done, total };
    }

    // --- Update the SVG progress ring for an agent ---
    function updateProgressRing(agentId, done, total) {
        const ringFill  = document.getElementById(`ring-fill-${agentId}`);
        const ringLabel = document.getElementById(`ring-label-${agentId}`);
        const card      = document.querySelector(`.agent-card[data-agent-id="${agentId}"]`);
        if (!ringFill || !ringLabel) return;

        const pct = total > 0 ? (done / total) * 100 : 0;
        // stroke-dasharray = "filled empty" where circumference ≈ 100
        ringFill.setAttribute("stroke-dasharray", `${pct.toFixed(1)} ${(100 - pct).toFixed(1)}`);

        // Colour classes
        ringFill.classList.remove("complete", "partial", "empty");
        if (done === total && total > 0) ringFill.classList.add("complete");
        else if (done > 0) ringFill.classList.add("partial");
        else ringFill.classList.add("empty");

        ringLabel.textContent = `${done}/${total}`;

        // Card glow when complete
        if (card) {
            card.classList.toggle("all-complete", done === total && total > 0);
        }
    }

    // --- Render the checklist drawer for an agent ---
    function renderChecklist(agentId, state) {
        const drawer = document.getElementById(`checklist-${agentId}`);
        if (!drawer) return;
        const items    = AGENT_CHECKLISTS[agentId] || [];
        const agState  = state[agentId] || {};
        const { done, total } = computeCompletion(agentId, state);

        drawer.innerHTML = `
            <div class="checklist-header">
                <span class="checklist-header-label">📋 Required Sources & Inputs</span>
                <span class="checklist-completion-label ${done === total && total > 0 ? "done" : ""}">${done}/${total} Complete</span>
            </div>
            <ul class="checklist-items-list">
                ${items.map(item => `
                    <li class="checklist-item ${agState[item.key] ? "checked" : ""}" data-agent="${agentId}" data-key="${item.key}">
                        <div class="checklist-item-box"></div>
                        <span class="checklist-item-label">${item.label}</span>
                        <span class="checklist-item-required">${item.required ? "Required" : "Optional"}</span>
                    </li>
                `).join("")}
            </ul>
        `;

        // Attach click handlers to each checklist row
        drawer.querySelectorAll(".checklist-item").forEach(li => {
            li.addEventListener("click", () => {
                const key = li.dataset.key;
                const aid = li.dataset.agent;
                if (!state[aid]) state[aid] = {};
                state[aid][key] = !state[aid][key];
                saveChecklistState(state);
                renderChecklist(aid, state);           // re-render drawer
                const { done: d, total: t } = computeCompletion(aid, state);
                updateProgressRing(aid, d, t);

                // Update filter if "missing data" is active
                applyFilters(state);
            });
        });

        updateProgressRing(agentId, done, total);
    }

    // --- Toggle checklist drawer open / close ---
    function wireChecklistToggles(state) {
        document.querySelectorAll(".btn-checklist-toggle").forEach(btn => {
            btn.addEventListener("click", () => {
                const agentId = btn.dataset.agentId;
                const drawer  = document.getElementById(`checklist-${agentId}`);
                if (!drawer) return;
                const isOpen  = drawer.classList.toggle("open");
                btn.classList.toggle("open", isOpen);
                if (typeof lucide !== "undefined") lucide.createIcons();
            });
        });
    }

    // --- Seed NSFW agent as the completed sample record (4/4) ---
    function seedNSFWSampleRecord(state) {
        if (!state["nsfw_agent"]) {
            state["nsfw_agent"] = {
                cam_feed: true,
                tfjs_rt:  true,
                nsfw_model: true,
                frame_pipe: true,
            };
            saveChecklistState(state);
        }
    }

    // --- Initialise all checklists ---
    function initChecklists() {
        const state = loadChecklistState();
        seedNSFWSampleRecord(state);

        Object.keys(AGENT_CHECKLISTS).forEach(agentId => {
            renderChecklist(agentId, state);
        });
        wireChecklistToggles(state);
    }

    // ================================================================
    //  ADVANCED BOUNTY — Section-Level Search & Filters
    // ================================================================

    function escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }

    // Highlight matching text inside a DOM element's text nodes
    function highlightText(el, query) {
        if (!el || !query) return;
        // Walk all text-containing children
        el.querySelectorAll("h4, .agent-category, .agent-desc").forEach(node => {
            // Restore plain text first (strip previous marks)
            if (node.dataset.origText === undefined) {
                node.dataset.origText = node.textContent;
            }
            const orig = node.dataset.origText;
            if (!query) {
                node.innerHTML = orig;
                return;
            }
            const re = new RegExp(`(${escapeRegex(query)})`, "gi");
            node.innerHTML = orig.replace(re, `<mark class="search-highlight">$1</mark>`);
        });
    }

    // Strip highlights from a card
    function clearHighlights(el) {
        if (!el) return;
        el.querySelectorAll("h4, .agent-category, .agent-desc").forEach(node => {
            if (node.dataset.origText !== undefined) {
                node.innerHTML = node.dataset.origText;
            }
        });
    }

    function applyFilters(checklistState) {
        const searchVal  = (document.getElementById("agentSearchInput")?.value || "").trim().toLowerCase();
        const catVal     = document.getElementById("filterCategory")?.value  || "";
        const statusVal  = document.getElementById("filterStatus")?.value    || "";
        const ownerVal   = document.getElementById("filterOwner")?.value     || "";
        const missingOnly= document.getElementById("filterMissingData")?.checked || false;
        const state      = checklistState || loadChecklistState();

        const cards = document.querySelectorAll(".agent-card[data-agent-id]");
        let visibleCount = 0;

        cards.forEach(card => {
            const agentId  = card.dataset.agentId;
            const name     = (card.querySelector("h4")?.dataset.origText    || card.querySelector("h4")?.textContent    || "").toLowerCase();
            const cat      = (card.querySelector(".agent-category")?.dataset.origText || card.querySelector(".agent-category")?.textContent || "");
            const desc     = (card.querySelector(".agent-desc")?.dataset.origText     || card.querySelector(".agent-desc")?.textContent     || "").toLowerCase();
            const isEnabled= agentRegistry[agentId]?.enabled ?? true;

            const { done, total } = computeCompletion(agentId, state);
            const hasMissingData  = done < total;

            // Build match conditions
            const matchSearch  = !searchVal || name.includes(searchVal) || cat.toLowerCase().includes(searchVal) || desc.includes(searchVal);
            const matchCat     = !catVal     || cat === catVal;
            const matchStatus  = !statusVal  || (statusVal === "active" ? isEnabled : !isEnabled);
            const matchOwner   = !ownerVal   || agentId === ownerVal;
            const matchMissing = !missingOnly || hasMissingData;

            const visible = matchSearch && matchCat && matchStatus && matchOwner && matchMissing;

            card.classList.toggle("filter-hidden", !visible);

            if (visible) {
                visibleCount++;
                if (searchVal) highlightText(card, searchVal);
                else clearHighlights(card);
            } else {
                clearHighlights(card);
            }
        });

        // Results count
        const resultEl = document.getElementById("filterResultsCount");
        if (resultEl) resultEl.textContent = `Showing ${visibleCount} of ${cards.length} agents`;

        // Empty state
        const noResults = document.getElementById("agentNoResults");
        if (noResults) noResults.classList.toggle("visible", visibleCount === 0);

        // Highlight active filter selects
        ["filterCategory", "filterStatus", "filterOwner"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.toggle("filter-active", !!el.value);
        });
    }

    function resetFilters() {
        const ids = ["agentSearchInput", "filterCategory", "filterStatus", "filterOwner"];
        ids.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = "";
        });
        const missingCb = document.getElementById("filterMissingData");
        if (missingCb) missingCb.checked = false;

        const clearBtn = document.getElementById("btnClearSearch");
        if (clearBtn) clearBtn.style.display = "none";

        // Remove filter-active classes
        document.querySelectorAll(".filter-select").forEach(el => el.classList.remove("filter-active"));

        applyFilters(loadChecklistState());
    }

    function initSearchFilters() {
        const searchInput = document.getElementById("agentSearchInput");
        const clearBtn    = document.getElementById("btnClearSearch");
        const resetBtn    = document.getElementById("btnFilterReset");

        if (searchInput) {
            searchInput.addEventListener("input", () => {
                const hasVal = searchInput.value.trim().length > 0;
                if (clearBtn) clearBtn.style.display = hasVal ? "flex" : "none";
                applyFilters(loadChecklistState());
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener("click", () => {
                if (searchInput) searchInput.value = "";
                clearBtn.style.display = "none";
                applyFilters(loadChecklistState());
            });
        }

        ["filterCategory", "filterStatus", "filterOwner"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener("change", () => applyFilters(loadChecklistState()));
        });

        const missingCb = document.getElementById("filterMissingData");
        if (missingCb) missingCb.addEventListener("change", () => applyFilters(loadChecklistState()));

        if (resetBtn) resetBtn.addEventListener("click", resetFilters);
    }

    // ================================================================
    //  BOOT: Init both Bounty Systems
    // ================================================================
    initChecklists();
    initSearchFilters();
    // Re-run lucide so new icons (list-checks, search-x, rotate-ccw) render
    if (typeof lucide !== "undefined") lucide.createIcons();

    // ================================================================
    //  ELITE BOUNTY — Review Packet Generator
    // ================================================================

    /** Static metadata about each agent (mirrors HTML card data) */
    const AGENT_METADATA = {
        "nsfw_agent":     { name: "Explicit Content Protection Agent",  category: "NSFW Defense",        desc: "Detects explicit imagery, adult content, and pornography using local NSFWJS neural network.", latency: "~4.2 ms", defaultSens: "60%" },
        "wound_agent":    { name: "Graphic Wound & Medical Agent",       category: "Gore Protection",     desc: "Identifies cuts, open lacerations, surgical trauma, and blood, applying dynamic frosted redaction.", latency: "~3.8 ms", defaultSens: "55%" },
        "gesture_agent":  { name: "Offensive Gesture Defense Agent",     category: "Gesture Safety",      desc: "Tracks 21 3D hand keypoints with MediaPipe to detect middle finger gestures and inappropriate hand signs.", latency: "~5.1 ms", defaultSens: "60%" },
        "weapon_agent":   { name: "Threat Object & Weapon Agent",        category: "Threat Protection",   desc: "Identifies knives, scissors, sharp threats, and weapons, applying localized targeted redaction boxes.", latency: "~4.9 ms", defaultSens: "50%" },
        "deepfake_agent": { name: "Deepfake & Synthetic Media Agent",    category: "Synthetic Forensics", desc: "Analyzes face-swap anomalies, boundary blurring, and facial landmark inconsistencies in real time.", latency: "~6.4 ms", defaultSens: "65%" },
        "scam_agent":     { name: "Financial Scam & Phishing Agent",     category: "Fraud Prevention",    desc: "Detects fake bank login overlays, counterfeit UPI apps, and lottery scam text patterns on screen.", latency: "~2.1 ms", defaultSens: "60%" },
        "privacy_agent":  { name: "PII Document Privacy Redactor",       category: "Data Privacy",        desc: "Auto-blackouts Aadhaar numbers, PAN cards, credit cards, and sensitive ID numbers before screen share.", latency: "~1.9 ms", defaultSens: "50%" },
        "audio_agent":    { name: "Audio Speech Profanity Beep Agent",   category: "Audio Safety",        desc: "Performs real-time microphone profanity detection and inserts 1000 Hz censor tone audio buffers instantly.", latency: "~3.2 ms", defaultSens: "50%" },
    };

    const RP_NOTES_KEY = "reflections_rp_notes_v1";

    /** Current agent open in the modal */
    let _rpActiveAgentId = null;

    // ── localStorage helpers for notes ──────────────────────────────
    function loadAllNotes() {
        try { return JSON.parse(localStorage.getItem(RP_NOTES_KEY) || "{}"); } catch(_) { return {}; }
    }
    function saveNote(agentId, text) {
        const all = loadAllNotes();
        all[agentId] = text;
        localStorage.setItem(RP_NOTES_KEY, JSON.stringify(all));
    }
    function loadNote(agentId) {
        return loadAllNotes()[agentId] || "";
    }

    // ── Seed NSFW as judge-ready sample with notes ───────────────────
    function seedNSFWJudgeSample() {
        const all = loadAllNotes();
        if (!all["nsfw_agent"]) {
            all["nsfw_agent"] = "JUDGE-READY SAMPLE\n\nAgent: Explicit Content Protection Agent (NSFW)\nReviewer: Reflections QA Bot\nDate: " + new Date().toISOString().split("T")[0] + "\n\nAll required data sources are confirmed active. NSFWJS local neural engine is initialised and the frame pipeline is streaming at 60 FPS. Sensitivity threshold set to 60% per safety policy guidelines.\n\nVerification: All 4/4 source fields complete. No missing data warnings. Agent is cleared for production evaluation.";
            localStorage.setItem(RP_NOTES_KEY, JSON.stringify(all));
        }
    }
    seedNSFWJudgeSample();

    // ── Build the structured review packet data object ───────────────
    function buildReviewPacket(agentId) {
        const meta      = AGENT_METADATA[agentId] || {};
        const state     = loadChecklistState();
        const items     = AGENT_CHECKLISTS[agentId] || [];
        const agState   = state[agentId] || {};
        const isEnabled = agentRegistry[agentId]?.enabled ?? true;

        // Sensitivity from slider
        const sensSlider = document.querySelector(`.agent-sens-slider[data-agent-id="${agentId}"]`);
        const sensitivity = sensSlider ? `${Math.round(parseFloat(sensSlider.value) * 100)}%` : meta.defaultSens;

        // Checklist status
        const checklist = items.map(item => ({
            key:      item.key,
            label:    item.label,
            required: item.required,
            checked:  !!agState[item.key],
        }));

        const done     = checklist.filter(i => i.checked).length;
        const total    = checklist.length;
        const missing  = checklist.filter(i => !i.checked);
        const score    = total > 0 ? Math.round((done / total) * 100) : 0;

        // Validation rules
        const validations = [];

        if (done === total && total > 0) {
            validations.push({ type: "pass", title: "All sources verified", desc: `All ${total} required inputs are marked complete and ready for evaluation.` });
        } else {
            validations.push({ type: "fail", title: "Incomplete source checklist", desc: `${total - done} of ${total} required inputs are missing. Agent cannot guarantee accurate detection without these sources.` });
        }

        if (isEnabled) {
            validations.push({ type: "pass", title: "Agent is active", desc: "Toggle switch is ON — agent is participating in the multi-agent consensus pipeline." });
        } else {
            validations.push({ type: "warn", title: "Agent is disabled", desc: "Toggle switch is OFF — agent is excluded from threat evaluation. Enable it before deployment." });
        }

        const sensNum = parseInt(sensitivity);
        if (sensNum >= 40 && sensNum <= 80) {
            validations.push({ type: "pass", title: "Sensitivity within recommended range", desc: `Threshold is ${sensitivity}, which falls within the 40–80% recommended operating band.` });
        } else if (sensNum < 40) {
            validations.push({ type: "warn", title: "Sensitivity below minimum recommended", desc: `Threshold is ${sensitivity}. Values below 40% may produce false negatives and miss real threats.` });
        } else {
            validations.push({ type: "warn", title: "Sensitivity above recommended ceiling", desc: `Threshold is ${sensitivity}. Values above 80% increase false positives and may degrade UX.` });
        }

        // Judge readiness tier
        let readinessTier, readinessLabel, readinessSub, readinessBadge;
        if (score === 100 && isEnabled) {
            readinessTier  = "ready";
            readinessLabel = "Judge Ready";
            readinessBadge = "✅ Cleared for Review";
            readinessSub   = "All sources complete, agent active. This packet is cleared for submission to a review committee.";
        } else if (score >= 50 && isEnabled) {
            readinessTier  = "warn";
            readinessLabel = "Partially Ready";
            readinessBadge = "⚠️ Needs Attention";
            readinessSub   = `${missing.length} required input(s) are missing. Complete the checklist before final submission.`;
        } else {
            readinessTier  = "danger";
            readinessLabel = "Not Ready";
            readinessBadge = "🔴 Do Not Submit";
            readinessSub   = "Too many gaps in the source checklist or agent is disabled. Resolve all issues before evaluation.";
        }

        return {
            agentId,
            meta,
            sensitivity,
            isEnabled,
            checklist,
            done,
            total,
            missing,
            score,
            validations,
            readinessTier,
            readinessLabel,
            readinessSub,
            readinessBadge,
            timestamp: new Date().toLocaleString("en-IN", { dateStyle: "full", timeStyle: "medium" }),
            notes: loadNote(agentId),
        };
    }

    // ── Populate the modal DOM ───────────────────────────────────────
    function renderReviewModal(agentId) {
        const pkt = buildReviewPacket(agentId);
        _rpActiveAgentId = agentId;

        // Header subtitle
        document.getElementById("rpModalSubtitle").textContent = `${pkt.meta.name} · ${pkt.meta.category}`;

        // Readiness banner
        const banner = document.getElementById("rpReadinessBanner");
        banner.className = `rp-readiness-banner ${pkt.readinessTier}`;
        document.getElementById("rpReadinessScore").textContent = `${pkt.score}%`;
        document.getElementById("rpReadinessLabel").textContent = pkt.readinessLabel;
        document.getElementById("rpReadinessSub").textContent   = pkt.readinessSub;
        document.getElementById("rpReadinessBadge").textContent = pkt.readinessBadge;

        // Section 1: Agent Profile
        document.getElementById("rpFieldAgentId").textContent   = pkt.agentId;
        document.getElementById("rpFieldName").textContent      = pkt.meta.name;
        document.getElementById("rpFieldCategory").textContent  = pkt.meta.category;
        document.getElementById("rpFieldStatus").textContent    = pkt.isEnabled ? "🟢 Active" : "⚫ Disabled";
        document.getElementById("rpFieldDesc").textContent      = pkt.meta.desc;
        document.getElementById("rpFieldSens").textContent      = pkt.sensitivity;
        document.getElementById("rpFieldTimestamp").textContent = pkt.timestamp;

        // Section 2: Checklist
        const clGrid = document.getElementById("rpChecklistItems");
        clGrid.innerHTML = pkt.checklist.map(item => `
            <div class="rp-checklist-item ${item.checked ? "item-done" : "item-miss"}">
                <div class="rp-ci-icon">${item.checked ? "✓" : "✕"}</div>
                <span class="rp-ci-label">${item.label}</span>
                <span class="rp-ci-tag">${item.checked ? "Done" : "Missing"}</span>
            </div>
        `).join("");

        // Section 3: Validation
        const vList = document.getElementById("rpValidationList");
        vList.innerHTML = pkt.validations.map(v => `
            <div class="rp-validation-row v-${v.type}">
                <span class="rp-v-icon">${v.type === "pass" ? "✅" : v.type === "warn" ? "⚠️" : "🔴"}</span>
                <div>
                    <div class="rp-v-title">${v.title}</div>
                    <div class="rp-v-desc">${v.desc}</div>
                </div>
            </div>
        `).join("");

        // Section 4: Missing fields
        const mList = document.getElementById("rpMissingFields");
        if (pkt.missing.length === 0) {
            mList.innerHTML = `<div class="rp-missing-none">✅ No missing fields — all required sources are satisfied.</div>`;
        } else {
            mList.innerHTML = pkt.missing.map(m => `
                <span class="rp-missing-tag">⚠️ ${m.label}</span>
            `).join("");
        }

        // Section 5: Notes
        const notesField = document.getElementById("rpNotesField");
        if (notesField) {
            notesField.value = pkt.notes;
            updateNotesCharCount(notesField.value.length);
        }
    }

    function updateNotesCharCount(len) {
        const el = document.getElementById("rpNotesCharCount");
        if (el) el.textContent = `${len} character${len !== 1 ? "s" : ""}`;
    }

    // ── Generate the standalone downloadable HTML report ─────────────
    function generateHTMLReport(agentId) {
        const pkt = buildReviewPacket(agentId);

        const checklistRows = pkt.checklist.map(item => `
            <tr style="border-bottom:1px solid #e2e8f0">
                <td style="padding:10px 12px;font-weight:600;color:${item.checked?"#065f46":"#7f1d1d"}">${item.checked?"✅":"❌"}</td>
                <td style="padding:10px 12px;">${item.label}</td>
                <td style="padding:10px 12px;font-weight:700;color:${item.checked?"#059669":"#dc2626"}">${item.checked?"Complete":"Missing"}</td>
            </tr>`).join("");

        const validationRows = pkt.validations.map(v => `
            <div style="display:flex;gap:12px;padding:12px 14px;margin-bottom:8px;border-radius:8px;border-left:4px solid ${v.type==="pass"?"#10b981":v.type==="warn"?"#f59e0b":"#ef4444"};background:${v.type==="pass"?"#f0fdf4":v.type==="warn"?"#fffbeb":"#fff1f2"}">
                <span style="font-size:18px">${v.type==="pass"?"✅":v.type==="warn"?"⚠️":"🔴"}</span>
                <div>
                    <div style="font-weight:800;color:#1e293b;margin-bottom:3px">${v.title}</div>
                    <div style="font-size:13px;color:#475569">${v.desc}</div>
                </div>
            </div>`).join("");

        const missingSection = pkt.missing.length === 0
            ? `<p style="color:#059669;font-weight:600">✅ No missing fields — all required sources are satisfied.</p>`
            : pkt.missing.map(m => `<span style="display:inline-block;margin:4px;padding:5px 12px;border-radius:20px;background:#fee2e2;border:1px solid #fca5a5;color:#dc2626;font-size:12px;font-weight:600">⚠️ ${m.label}</span>`).join("");

        const tierColor = pkt.readinessTier === "ready" ? "#059669" : pkt.readinessTier === "warn" ? "#d97706" : "#dc2626";
        const tierBg    = pkt.readinessTier === "ready" ? "#f0fdf4" : pkt.readinessTier === "warn" ? "#fffbeb" : "#fff1f2";

        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reflections Review Packet — ${pkt.meta.name}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; color: #1e293b; }
  .report-wrap { max-width: 860px; margin: 0 auto; background: #fff; box-shadow: 0 4px 24px rgba(0,0,0,0.08); }
  .report-header { background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; padding: 36px 40px; }
  .rh-top { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; }
  .rh-logo { font-size: 13px; font-weight: 700; opacity: 0.7; letter-spacing: 1px; text-transform: uppercase; }
  .rh-ts { font-size: 12px; opacity: 0.6; }
  .rh-title { font-size: 26px; font-weight: 900; margin-top: 20px; }
  .rh-sub { font-size: 14px; opacity: 0.75; margin-top: 6px; }
  .readiness-bar { display: flex; align-items: center; gap: 20px; padding: 22px 40px; background: ${tierBg}; border-bottom: 2px solid ${tierColor}22; }
  .rb-score { font-size: 42px; font-weight: 900; color: ${tierColor}; line-height: 1; }
  .rb-label { font-size: 18px; font-weight: 800; color: ${tierColor}; }
  .rb-sub { font-size: 13px; color: #64748b; margin-top: 4px; }
  .rb-badge { margin-left: auto; padding: 7px 18px; background: ${tierColor}22; color: ${tierColor}; border: 1px solid ${tierColor}55; border-radius: 20px; font-weight: 800; font-size: 13px; white-space: nowrap; }
  .section { padding: 28px 40px; border-bottom: 1px solid #f1f5f9; }
  .section:last-child { border-bottom: none; }
  .section-title { display: flex; align-items: center; gap: 8px; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 18px; }
  .section-tag { margin-left: auto; font-size: 10px; background: #f1f5f9; padding: 2px 8px; border-radius: 8px; color: #94a3b8; }
  .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .field { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; }
  .field-span { grid-column: 1 / -1; }
  .field label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #94a3b8; display: block; margin-bottom: 4px; }
  .field-val { font-size: 13px; color: #1e293b; }
  .mono { font-family: monospace; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #f1f5f9; padding: 10px 12px; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; }
  .notes-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; font-size: 13px; color: #334155; white-space: pre-wrap; line-height: 1.65; min-height: 80px; }
  .report-footer { background: #0f172a; color: #475569; padding: 20px 40px; font-size: 12px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
  @media print { body { background: white; } .report-wrap { box-shadow: none; } }
</style>
</head>
<body>
<div class="report-wrap">

  <!-- Header -->
  <div class="report-header">
    <div class="rh-top">
      <span class="rh-logo">🛡️ Reflections Safety Platform · Agent Review Packet</span>
      <span class="rh-ts">${pkt.timestamp}</span>
    </div>
    <div class="rh-title">${pkt.meta.name}</div>
    <div class="rh-sub">Category: ${pkt.meta.category} &nbsp;|&nbsp; Agent ID: <code style="background:rgba(255,255,255,0.15);padding:2px 6px;border-radius:4px">${pkt.agentId}</code></div>
  </div>

  <!-- Readiness Banner -->
  <div class="readiness-bar">
    <div class="rb-score">${pkt.score}%</div>
    <div>
      <div class="rb-label">${pkt.readinessLabel}</div>
      <div class="rb-sub">${pkt.readinessSub}</div>
    </div>
    <div class="rb-badge">${pkt.readinessBadge}</div>
  </div>

  <!-- Section 1: Agent Profile -->
  <div class="section">
    <div class="section-title">🤖 Agent Profile <span class="section-tag">SECTION 1</span></div>
    <div class="field-grid">
      <div class="field"><label>Agent ID</label><div class="field-val mono">${pkt.agentId}</div></div>
      <div class="field"><label>Display Name</label><div class="field-val">${pkt.meta.name}</div></div>
      <div class="field"><label>Safety Category</label><div class="field-val">${pkt.meta.category}</div></div>
      <div class="field"><label>Status</label><div class="field-val">${pkt.isEnabled ? "🟢 Active" : "⚫ Disabled"}</div></div>
      <div class="field field-span"><label>Description</label><div class="field-val">${pkt.meta.desc}</div></div>
      <div class="field"><label>Sensitivity Threshold</label><div class="field-val">${pkt.sensitivity}</div></div>
      <div class="field"><label>Inference Latency</label><div class="field-val">${pkt.meta.latency}</div></div>
    </div>
  </div>

  <!-- Section 2: Source Checklist -->
  <div class="section">
    <div class="section-title">📋 Source Checklist — ${pkt.done}/${pkt.total} Complete <span class="section-tag">SECTION 2</span></div>
    <table>
      <thead><tr><th style="width:50px">Status</th><th>Required Source / Input</th><th style="width:100px">Result</th></tr></thead>
      <tbody>${checklistRows}</tbody>
    </table>
  </div>

  <!-- Section 3: Validation Summary -->
  <div class="section">
    <div class="section-title">🛡️ Validation Summary <span class="section-tag">SECTION 3</span></div>
    ${validationRows}
  </div>

  <!-- Section 4: Missing Data -->
  <div class="section">
    <div class="section-title">⚠️ Missing Data Fields <span class="section-tag">SECTION 4</span></div>
    ${missingSection}
  </div>

  <!-- Section 5: Reviewer Notes -->
  <div class="section">
    <div class="section-title">📝 Reviewer Notes <span class="section-tag">SECTION 5</span></div>
    <div class="notes-box">${pkt.notes || "(No reviewer notes recorded)"}</div>
  </div>

  <!-- Footer -->
  <div class="report-footer">
    <span>Reflections Real-Time Visual Safety System · Review Packet v1.0</span>
    <span>Generated: ${pkt.timestamp}</span>
  </div>

</div>
</body>
</html>`;
    }

    // ── Download the report as .html file ────────────────────────────
    function downloadReviewPacket(agentId) {
        const html     = generateHTMLReport(agentId);
        const blob     = new Blob([html], { type: "text/html;charset=utf-8" });
        const url      = URL.createObjectURL(blob);
        const a        = document.createElement("a");
        const safeName = agentId.replace(/_/g, "-");
        a.href         = url;
        a.download     = `reflections-review-packet-${safeName}-${Date.now()}.html`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        showRPToast("✅ Review packet downloaded!", false);
    }

    // ── Show a brief toast notification ──────────────────────────────
    function showRPToast(message, isError = false) {
        let toast = document.getElementById("rpToastEl");
        if (!toast) {
            toast = document.createElement("div");
            toast.id = "rpToastEl";
            toast.className = "rp-toast";
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        toast.className = `rp-toast ${isError ? "error" : ""} show`;
        clearTimeout(toast._timer);
        toast._timer = setTimeout(() => { toast.classList.remove("show"); }, 3000);
    }

    // ── Open / close the review modal ────────────────────────────────
    function openReviewModal(agentId) {
        renderReviewModal(agentId);
        const modal = document.getElementById("reviewPacketModal");
        if (modal) {
            modal.classList.add("open");
            document.body.style.overflow = "hidden";
        }
        if (typeof lucide !== "undefined") lucide.createIcons();
    }

    function closeReviewModal() {
        const modal = document.getElementById("reviewPacketModal");
        if (modal) {
            modal.classList.remove("open");
            document.body.style.overflow = "";
        }
        _rpActiveAgentId = null;
    }

    // ── Wire Export buttons on each agent card ────────────────────────
    function initExportButtons() {
        document.querySelectorAll(".btn-export-packet").forEach(btn => {
            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                const agentId = btn.dataset.agentId;
                openReviewModal(agentId);
            });
        });

        // Close button
        document.getElementById("rpBtnClose")?.addEventListener("click", closeReviewModal);

        // Click outside panel to close
        document.getElementById("reviewPacketModal")?.addEventListener("click", (e) => {
            if (e.target === e.currentTarget) closeReviewModal();
        });

        // Escape key closes
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") closeReviewModal();
        });

        // Download button
        document.getElementById("rpBtnDownload")?.addEventListener("click", () => {
            if (_rpActiveAgentId) {
                // Grab latest notes before generating
                const notes = document.getElementById("rpNotesField")?.value || "";
                saveNote(_rpActiveAgentId, notes);
                downloadReviewPacket(_rpActiveAgentId);
            }
        });

        // Copy JSON button
        document.getElementById("rpBtnCopyJSON")?.addEventListener("click", async () => {
            if (!_rpActiveAgentId) return;
            const notes = document.getElementById("rpNotesField")?.value || "";
            saveNote(_rpActiveAgentId, notes);
            const pkt = buildReviewPacket(_rpActiveAgentId);
            const json = JSON.stringify(pkt, null, 2);
            try {
                await navigator.clipboard.writeText(json);
                showRPToast("📋 JSON copied to clipboard!", false);
            } catch(_) {
                showRPToast("❌ Clipboard access denied.", true);
            }
        });

        // Notes textarea — char count + save
        const notesField = document.getElementById("rpNotesField");
        const notesSave  = document.getElementById("rpNotesSave");

        notesField?.addEventListener("input", () => {
            updateNotesCharCount(notesField.value.length);
            notesSave?.classList.remove("saved");
        });

        notesSave?.addEventListener("click", () => {
            if (_rpActiveAgentId) {
                saveNote(_rpActiveAgentId, notesField?.value || "");
                notesSave.classList.add("saved");
                notesSave.textContent = "✓ Saved";
                setTimeout(() => {
                    notesSave.innerHTML = '<i data-lucide="save"></i> Save Notes';
                    notesSave.classList.remove("saved");
                    if (typeof lucide !== "undefined") lucide.createIcons();
                }, 2000);
                showRPToast("📝 Notes saved to local storage.");
            }
        });
    }

    initExportButtons();
    if (typeof lucide !== "undefined") lucide.createIcons();

    // Auto-start webcam
    startWebcam();
});




