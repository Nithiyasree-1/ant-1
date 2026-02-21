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
  const [error, setError] = useState(null);

  // 1. Initialize AI Engine (Runs instantly)
  useEffect(() => {
    let active = true;
    async function init() {
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
        if (active) setDetector(objDetector);
      } catch (err) {
        console.error("AI Error:", err);
        setError("AI Engine Syncing...");
      }
    }
    init();
    return () => { active = false; };
  }, []);

  // 2. Optimized Camera Startup
  const startCamera = useCallback(async () => {
    if (streamRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => setIsReady(true);
      }
      setError(null);
    } catch (err) {
      setError("Please Refresh or Allow Camera Access");
      console.error(err);
    }
  }, []);

  // Auto-trigger camera on first interaction or mount
  useEffect(() => {
    startCamera();
    window.addEventListener('click', startCamera);
    window.addEventListener('touchstart', startCamera);
    return () => {
      window.removeEventListener('click', startCamera);
      window.removeEventListener('touchstart', startCamera);
    };
  }, [startCamera]);

  // 3. Render Loop (Canvas Engine)
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
      if (!video || !canvas) return;

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
          if (iou > maxIoU && iou > 0.2) { maxIoU = iou; focus = d.boundingBox; }
        });
        if (focus) setTrackedBox(focus);
      }

      if (!focus && isAutoTrack && currentDets.length > 0) {
        focus = currentDets.sort((a, b) => (b.boundingBox.width * b.boundingBox.height) - (a.boundingBox.width * a.boundingBox.height))[0].boundingBox;
        setTrackedBox(focus);
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (focus && blurIntensity > 0) {
        ctx.filter = `blur(${blurIntensity}px) brightness(0.6) ${isColorPop ? 'grayscale(100%)' : ''}`;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.filter = 'none';
        ctx.save();
        ctx.beginPath();
        ctx.rect(focus.originX - 5, focus.originY - 5, focus.width + 10, focus.height + 10);
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
        ctx.strokeStyle = isFocus ? '#facc15' : 'rgba(56, 189, 248, 0.4)';
        ctx.lineWidth = isFocus ? 4 : 2;
        ctx.strokeRect(d.boundingBox.originX, d.boundingBox.originY, d.boundingBox.width, d.boundingBox.height);
      });

      animationId = requestAnimationFrame(render);
    };

    animationId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationId);
  }, [isReady, detector, trackedBox, blurIntensity, isColorPop, isAutoTrack]);

  const clickSelect = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = videoRef.current.videoWidth - ((e.clientX - rect.left) * (videoRef.current.videoWidth / rect.width));
    const y = (e.clientY - rect.top) * (videoRef.current.videoHeight / rect.height);
    detections.forEach(d => {
      const b = d.boundingBox;
      if (x >= b.originX && x <= b.originX + b.width && y >= b.originY && y <= b.originY + b.height) setTrackedBox(b);
    });
  };

  return (
    <div className="app-container">
      <div className="video-wrapper" onClick={clickSelect}>
        <video ref={videoRef} autoPlay playsInline muted hidden />
        <canvas ref={canvasRef} className="overlay-canvas" />
      </div>

      {(!isReady || !detector) && (
        <div className="loading-overlay">
          <div className="loader"></div>
          <p>{error || "Booting Smart Vision Engine..."}</p>
        </div>
      )}

      <div className="ui-panel">
        <div className="panel-header">
          <h3>Smart Focus AI</h3>
          <span className="fps-badge">{fps} FPS</span>
        </div>
        <div className="panel-body">
          <div className="control-group">
            <label>Focus Depth</label>
            <input type="range" min="0" max="40" value={blurIntensity} onChange={e => setBlurIntensity(Number(e.target.value))} />
          </div>
          <div className="toggle-group">
            <input type="checkbox" checked={isColorPop} onChange={e => setIsColorPop(e.target.checked)} id="cp" />
            <label htmlFor="cp">Color Pop (B&W)</label>
          </div>
          <div className="toggle-group">
            <input type="checkbox" checked={isAutoTrack} onChange={e => setIsAutoTrack(e.target.checked)} id="at" />
            <label htmlFor="at">Auto-Focus</label>
          </div>
          <button className="snapshot-btn" onClick={() => window.open(canvasRef.current.toDataURL(), '_blank')}>CAPTURE</button>
        </div>
      </div>
    </div>
  );
}

export default App;
