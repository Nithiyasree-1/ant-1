import React, { useEffect, useRef, useState } from 'react';
import { ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [isCameraReady, setIsCameraReady] = useState(false);
  const [detector, setDetector] = useState(null);
  const [detections, setDetections] = useState([]);
  const [trackedBox, setTrackedBox] = useState(null);
  const [blurIntensity, setBlurIntensity] = useState(12);
  const [fps, setFps] = useState(0);
  const [isCameraActive, setIsCameraActive] = useState(true);
  const [isColorPop, setIsColorPop] = useState(false);
  const [isAutoTrack, setIsAutoTrack] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  // Helper: Calculate IoU
  const calculateIoU = (boxA, boxB) => {
    if (!boxA || !boxB) return 0;
    const xA = Math.max(boxA.originX, boxB.originX);
    const yA = Math.max(boxA.originY, boxB.originY);
    const xB = Math.min(boxA.originX + boxA.width, boxB.originX + boxB.width);
    const yB = Math.min(boxA.originY + boxA.height, boxB.originY + boxB.height);
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const boxAArea = boxA.width * boxA.height;
    const boxBArea = boxB.width * boxB.height;
    return interArea / (boxAArea + boxBArea - interArea);
  };

  // 1. Initialize AI Detector immediately on page load
  useEffect(() => {
    async function initAI() {
      try {
        console.log("Loading AI Vision engine...");
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
        );
        const objectDetector = await ObjectDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          scoreThreshold: 0.35,
        });
        setDetector(objectDetector);
        console.log("AI engine ready.");
      } catch (err) {
        console.error("AI Init Error:", err);
      }
    }
    initAI();
  }, []);

  // 2. Initialize Camera when user clicks "Launch"
  useEffect(() => {
    if (!hasStarted || !isCameraActive) return;

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => setIsCameraReady(true);
        }
      } catch (err) {
        alert("Please allow camera access to use this app.");
        console.error(err);
      }
    }
    startCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, [hasStarted, isCameraActive]);

  // 3. High-Performance Rendering Loop
  useEffect(() => {
    let animationId;
    let frameCount = 0;
    let fpsInterval = performance.now();

    const runLoop = async (time) => {
      // FPS Stats
      frameCount++;
      if (time - fpsInterval > 1000) {
        setFps(Math.round((frameCount * 1000) / (time - fpsInterval)));
        fpsInterval = time;
        frameCount = 0;
      }

      if (isCameraReady && detector && videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d', { alpha: false });

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const results = await detector.detectForVideo(video, performance.now());
        const currentDetections = results.detections;
        setDetections(currentDetections);

        let activeBox = null;
        if (trackedBox) {
          let maxIoU = 0;
          currentDetections.forEach(det => {
            const iou = calculateIoU(trackedBox, det.boundingBox);
            if (iou > maxIoU && iou > 0.25) {
              maxIoU = iou;
              activeBox = det.boundingBox;
            }
          });
          if (activeBox) setTrackedBox(activeBox);
        }

        // Auto selection fallback
        if (!activeBox && isAutoTrack && currentDetections.length > 0) {
          activeBox = currentDetections.reduce((prev, current) =>
            (prev.boundingBox.width * prev.boundingBox.height > current.boundingBox.width * current.boundingBox.height) ? prev : current
          ).boundingBox;
          setTrackedBox(activeBox);
        }

        // DRAWING
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (activeBox && blurIntensity > 0) {
          const filter = `blur(${blurIntensity}px) brightness(0.6) ${isColorPop ? 'grayscale(100%)' : ''}`;
          ctx.filter = filter;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          ctx.filter = 'none';

          ctx.save();
          ctx.beginPath();
          ctx.rect(activeBox.originX - 10, activeBox.originY - 10, activeBox.width + 20, activeBox.height + 20);
          ctx.clip();
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          ctx.restore();
        } else {
          ctx.filter = isColorPop ? 'grayscale(100%)' : 'none';
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          ctx.filter = 'none';
        }

        drawOverlays(ctx, currentDetections, activeBox);
      }
      animationId = requestAnimationFrame(runLoop);
    };

    if (hasStarted) animationId = requestAnimationFrame(runLoop);
    return () => cancelAnimationFrame(animationId);
  }, [hasStarted, isCameraReady, detector, trackedBox, blurIntensity, isColorPop, isAutoTrack]);

  const drawOverlays = (ctx, dets, tracked) => {
    dets.forEach(d => {
      const isMe = tracked && calculateIoU(tracked, d.boundingBox) > 0.8;
      const { originX, originY, width, height } = d.boundingBox;

      ctx.strokeStyle = isMe ? '#facc15' : 'rgba(56, 189, 248, 0.4)';
      ctx.lineWidth = isMe ? 4 : 2;
      ctx.strokeRect(originX, originY, width, height);

      if (isMe) {
        ctx.fillStyle = '#facc15';
        ctx.fillRect(originX, originY - 25, 100, 25);
        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 12px Inter';
        ctx.fillText("TRACKING", originX + 5, originY - 8);
      }
    });
  };

  const handleInteraction = (e) => {
    if (!videoRef.current || detections.length === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = videoRef.current.videoWidth - ((e.clientX - rect.left) * (videoRef.current.videoWidth / rect.width));
    const y = (e.clientY - rect.top) * (videoRef.current.videoHeight / rect.height);

    detections.forEach(d => {
      const b = d.boundingBox;
      if (x >= b.originX && x <= b.originX + b.width && y >= b.originY && y <= b.originY + b.height) {
        setTrackedBox(b);
      }
    });
  };

  if (!hasStarted) {
    return (
      <div className="landing-page">
        <div className="hero-content">
          <h1>Smart Focus <span className="ai-badge">AI</span></h1>
          <p>Cinematic AI Photography Suite</p>
          <button className="start-btn" onClick={() => setHasStarted(true)}>LAUNCH ENGINE</button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="video-wrapper" onClick={handleInteraction}>
        <video ref={videoRef} autoPlay playsInline muted hidden />
        <canvas ref={canvasRef} className="overlay-canvas" />
      </div>

      {(!isCameraReady || !detector) && (
        <div className="loading-overlay">
          <div className="loader"></div>
          <p>{!detector ? "AI Core Warming Up..." : "Initializng Camera..."}</p>
        </div>
      )}

      <div className="ui-panel">
        <div className="panel-header">
          <h3>System Status</h3>
          <span className="fps-badge">{fps} FPS</span>
        </div>

        <div className="panel-body">
          <div className="control-group">
            <label>Focus Softness</label>
            <input type="range" min="0" max="40" value={blurIntensity} onChange={e => setBlurIntensity(Number(e.target.value))} />
          </div>

          <div className="toggle-group">
            <input type="checkbox" checked={isColorPop} onChange={e => setIsColorPop(e.target.checked)} id="cp" />
            <label htmlFor="cp">Color Pop (B&W)</label>
          </div>

          <div className="toggle-group">
            <input type="checkbox" checked={isAutoTrack} onChange={e => setIsAutoTrack(e.target.checked)} id="at" />
            <label htmlFor="at">AI Auto-Focus</label>
          </div>

          <button className={`camera-toggle ${isCameraActive ? 'active' : ''}`} onClick={() => setIsCameraActive(!isCameraActive)}>
            {isCameraActive ? "Camera OFF" : "Camera ON"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
