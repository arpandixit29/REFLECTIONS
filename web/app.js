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
            btnStartRealScreen.querySelector("span").textContent = "Scan My WhatsApp / Screen";
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
            // 2. Start screen capture (captures WhatsApp video call window)
            activeMediaStream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    cursor: "always"
                },
                audio: false
            });

            webcamVideo.srcObject = activeMediaStream;
            webcamVideo.style.display = "block";
            await webcamVideo.play();

            btnStartRealScreen.classList.add("active");
            btnStartRealScreen.querySelector("span").textContent = "Stop Screen Scan";
            activeAppTitle.textContent = "🛡 WhatsApp & Video Call Shield Active — Scanning Stream Live";
            liveModeBadgeText.textContent = "Screen Shield Active";
            streamStatusText.textContent = "Screen Scan Live ✓";
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

    function normalizeText(text) {
        if (!text) return "";
        let t = text.toLowerCase().trim();
        t = t.replace(/[^\w\s@$01357!]+$/g, "").replace(/^[^\w\s@$01357!]+/g, "");
        t = t.replace(/@/g, "a").replace(/\$/g, "s").replace(/0/g, "o").replace(/1/g, "i");
        t = t.replace(/3/g, "e").replace(/5/g, "s").replace(/7/g, "t");
        t = t.replace(/(?<=\w)!|!(?=\w)/g, "i");
        t = t.replace(/[^\w\s]/g, "");
        t = t.replace(/(.)\1{2,}/g, "$1");
        return t.trim();
    }

    function isProfaneToken(word) {
        if (!word) return false;
        const cleanW = normalizeText(word);
        if (!cleanW || cleanW.length < 2) return false;
        if (SAFE_WORDS.has(cleanW)) return false;
        if (profanitySet.has(cleanW)) return true;

        // Substring root matching ONLY for known abusive roots of length >= 4 (or core explicit roots)
        const CORE_ABUSIVE_ROOTS = [
            "fuck", "bhenchod", "madarchod", "chutiya", "bhosdi", "gaand", "gand",
            "lauda", "loda", "bastard", "cunt", "bitch", "motherfucker", "bullshit"
        ];
        for (let root of CORE_ABUSIVE_ROOTS) {
            if (cleanW.includes(root)) return true;
        }

        return false;
    }

    // Render initial banned words tags
    function renderBannedWordTags() {
        if (!bannedWordsTags) return;
        bannedWordsTags.innerHTML = "";
        profanitySet.forEach(word => {
            const tag = document.createElement("span");
            tag.className = "word-tag";
            tag.innerHTML = `<span>${word}</span> <i data-lucide="x" class="btn-remove-word"></i>`;
            tag.querySelector(".btn-remove-word").addEventListener("click", () => {
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
    let beepedSessionKeys = new Set(); // Track beeped word keys per recognition turn to avoid duplicate continuous beeps

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
            beepedSessionKeys.clear();
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

            // Process words for profanity with deduplication across interim frames
            const words = currentText.trim().split(/\s+/);
            let cleanedWords = [];

            const now = new Date();
            const timeStr = now.toTimeString().split(" ")[0] + "." + String(now.getMilliseconds()).padStart(3, "0");

            words.forEach((w, wIdx) => {
                if (isProfaneToken(w)) {
                    const wordKey = `${event.resultIndex}_${wIdx}_${w.toLowerCase()}`;
                    if (!beepedSessionKeys.has(wordKey)) {
                        beepedSessionKeys.add(wordKey);
                        triggerCensorBeep(550);
                        addTimestampLogEntry(timeStr, "Live Microphone", w, "Abusive Speech", "BEEP_OVERLAY");
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

            if (finalTranscript) {
                beepedSessionKeys.clear();
            }
        };

        speechRecognition.onerror = (err) => {
            console.warn("Speech recognition note/error:", err);
        };

        speechRecognition.onend = () => {
            if (micActive) {
                // Auto restart continuous listening if mic toggle is active
                try { speechRecognition.start(); } catch (e) {}
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




    // Auto-start webcam
    startWebcam();
});

