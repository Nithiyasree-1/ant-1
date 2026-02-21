import React, { useEffect, useRef, useState, useCallback } from 'react';
import { ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [isReady, setIsReady] = useState(false);
  const [detector, setDetector] = useState(null);
  const [detections, setDetections] = useState([]);
  const [trackedBox, setTrackedBox] = useState(null);
  const [blurIntensity, setBlurIntensity] = useState(12);
  const [fps, setFps] = useState(0);
  const [isColorPop, setIsColorPop] = useState(false);
  const [isAutoTrack, setIsAutoTrack] = useState(false);
  const [facingMode, setFacingMode] = useState("user");
  const [statusMessage, setStatusMessage] = useState("Initializing AI...");

  // 1. Initialize AI Engine with Latest CDN
  useEffect(() => {
    let active = true;
    async function initAI() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const objDetector = await ObjectDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          scoreThreshold: 0.35,
        });
        if (active) {
          setDetector(objDetector);
          setStatusMessage("AI Ready. Click anywhere to start camera.");
        }
      } catch (err) {
        console.error("AI Init Error:", err);
        setStatusMessage("AI Sync Error. Please Refresh.");
      }
    }
    initAI();
    return () => { active = false; };
  }, []);

  // 2. Adaptive Camera Logic (The main fix)
  const startCamera = useCallback(async () => {
    setStatusMessage("Connecting to Camera...");
    try {
      // Stop any existing stream tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        setIsReady(false); // Set ready to false while camera is restarting
      }

      // Use flexible constraints to avoid hardware rejection
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: facingMode
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // Forced playback to overcome browser "Auto-pause"
        await videoRef.current.play();
        setIsReady(true);
        setStatusMessage("");
      }
    } catch (err) {
      console.error("Camera Error:", err);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setStatusMessage("Camera Blocked. Please click the 'Lock' icon in your URL bar and allow access.");
      } else {
        setStatusMessage("Hardware Error: Please try another browser (Chrome/Edge recommended).");
      }
    }
  }, [facingMode]); // Add facingMode to dependencies

  // Trigger camera on mount AND on click (to beat browser security)
  useEffect(() => {
    startCamera();
    window.addEventListener('click', startCamera);
    return () => window.removeEventListener('click', startCamera);
  }, [startCamera, facingMode]);

  // 3. Rendering Engine
  useEffect(() => {
    if (!isReady || !detector) return;
    let animationId;
    let frameCount = 0;
    let lastFpsUpdate = performance.now();

    const render = async (time) => {
      frameCount++;
      if (time - lastFpsUpdate > 1000) {
        setFps(Math.round((frameCount * 1000) / (time - lastFpsUpdate)));
        lastFpsUpdate = time;
        frameCount = 0;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || video.paused || video.ended) return;

      const ctx = canvas.getContext('2d', { alpha: false });
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const results = await detector.detectForVideo(video, performance.now());
      const currentDets = results.detections;
      setDetections(currentDets);

      let focus = null;
      if (trackedBox) {
        let maxIoU = 0;
        currentDets.forEach(d => {
          const iA = Math.max(trackedBox.originX, d.boundingBox.originX);
          const iB = Math.max(trackedBox.originY, d.boundingBox.originY);
          const iC = Math.min(trackedBox.originX + trackedBox.width, d.boundingBox.originX + d.boundingBox.width);
          const iD = Math.min(trackedBox.originY + trackedBox.height, d.boundingBox.originY + d.boundingBox.height);
          const inter = Math.max(0, iC - iA) * Math.max(0, iD - iB);
          const iou = inter / ((trackedBox.width * trackedBox.height) + (d.boundingBox.width * d.boundingBox.height) - inter);
          if (iou > maxIoU && iou > 0.15) { maxIoU = iou; focus = d.boundingBox; }
        });
        if (focus) setTrackedBox(focus);
      }

      if (!focus && isAutoTrack && currentDets.length > 0) {
        focus = currentDets.sort((a, b) => (b.boundingBox.width * b.boundingBox.height) - (a.boundingBox.width * a.boundingBox.height))[0].boundingBox;
        setTrackedBox(focus);
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (focus && blurIntensity > 0) {
        ctx.filter = `blur(${blurIntensity}px) brightness(0.7) ${isColorPop ? 'grayscale(100%)' : ''}`;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.filter = 'none';
        ctx.save();
        ctx.beginPath();
        ctx.rect(focus.originX - 10, focus.originY - 10, focus.width + 20, focus.height + 20);
        ctx.clip();
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
      } else {
        ctx.filter = isColorPop ? 'grayscale(100%)' : 'none';
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.filter = 'none';
      }

      currentDets.forEach(d => {
        const isFocus = focus && (d.boundingBox.originX === focus.originX);
        ctx.strokeStyle = isFocus ? '#facc15' : 'rgba(56, 189, 248, 0.5)';
        ctx.lineWidth = isFocus ? 4 : 2;
        ctx.strokeRect(d.boundingBox.originX, d.boundingBox.originY, d.boundingBox.width, d.boundingBox.height);
      });

      animationId = requestAnimationFrame(render);
    };

    animationId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationId);
  }, [isReady, detector, trackedBox, blurIntensity, isColorPop, isAutoTrack]);

  const takeSnapshot = () => {
    if (!canvasRef.current || !isReady) return;
    try {
      // Get image data
      const dataUrl = canvasRef.current.toDataURL('image/png', 1.0);

      // Create a unique filename
      const date = new Date();
      const filename = `AI_Focus_${date.getTime()}.png`;

      // Force download via temporary link
      const link = document.createElement('a');
      link.href = dataUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Status feedback
      const prevMsg = statusMessage;
      setStatusMessage("📸 PHOTO SAVED TO GALLERY");
      setTimeout(() => setStatusMessage(prevMsg), 3000);
    } catch (err) {
      console.error(err);
      setStatusMessage("Snapshot failed. Try again.");
    }
  };

  return (
    <div className="app-container">
      <div className="video-wrapper" onClick={(e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        const x = videoRef.current.videoWidth - ((e.clientX - rect.left) * (videoRef.current.videoWidth / rect.width));
        const y = (e.clientY - rect.top) * (videoRef.current.videoHeight / rect.height);
        detections.forEach(d => {
          if (x >= d.boundingBox.originX && x <= d.boundingBox.originX + d.boundingBox.width &&
            y >= d.boundingBox.originY && y <= d.boundingBox.originY + d.boundingBox.height) setTrackedBox(d.boundingBox);
        });
      }}>
        <video ref={videoRef} autoPlay playsInline muted hidden />
        <canvas ref={canvasRef} className="overlay-canvas" />
      </div>

      {!isReady && (
        <div className="loading-overlay">
          <div className="loader"></div>
          <p className="status-msg">{statusMessage}</p>
          {!detector && <p className="sub-msg">Syncing AI Core...</p>}
        </div>
      )}

      {isReady && statusMessage && (
        <div className="status-banner">
          {statusMessage}
        </div>
      )}

      <div className="ui-panel">
        <div className="panel-header">
          <h3>AI Focus Pro</h3>
          <span className="fps-badge">{fps} FPS</span>
        </div>
        <div className="panel-body">
          <div className="control-group">
            <label>Depth of Field</label>
            <input type="range" min="0" max="40" value={blurIntensity} onChange={e => setBlurIntensity(Number(e.target.value))} />
          </div>
          <div className="toggle-group">
            <input type="checkbox" checked={isColorPop} onChange={e => setIsColorPop(e.target.checked)} id="cp" />
            <label htmlFor="cp">Color Pop (B&W)</label>
          </div>
          <div className="toggle-group">
            <input type="checkbox" checked={isAutoTrack} onChange={e => setIsAutoTrack(e.target.checked)} id="at" />
            <label htmlFor="at">Auto Subject Search</label>
          </div>
          <button
            className="switch-camera-btn"
            onClick={() => setFacingMode(prev => prev === "user" ? "environment" : "user")}
          >
            🔄 Switch to {facingMode === "user" ? "Back" : "Front"} Camera
          </button>
          <button className="snapshot-btn" onClick={takeSnapshot}>CAPTURE SNAPSHOT</button>
        </div>
      </div>
    </div>
  );
}

export default App;
