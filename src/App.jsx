import React, { useEffect, useRef, useState } from 'react';
import { ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
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
  const streamRef = useRef(null);

  // Helper: Calculate IoU (Intersection over Union)
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

  useEffect(() => {
    if (!hasStarted) return;
    async function initDetector() {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      const objectDetector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        scoreThreshold: 0.4,
      });
      setDetector(objectDetector);
    }
    initDetector();
  }, []);

  useEffect(() => {
    if (!hasStarted) return;
    async function setupCamera() {
      if (!isCameraActive) {
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        setIsCameraReady(false);
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            setIsCameraReady(true);
          };
        }
      } catch (err) {
        console.error("Error accessing webcam: ", err);
      }
    }
    setupCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [isCameraActive]);

  // Detection and Drawing loop
  useEffect(() => {
    let animationId;
    let lastTime = 0;
    let frameCount = 0;
    let fpsInterval = 0;

    if (isCameraReady && detector && videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d', { alpha: false }); // Optimization

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const runDetection = async (time) => {
        // Calculate FPS
        frameCount++;
        if (time - fpsInterval > 1000) {
          setFps(Math.round((frameCount * 1000) / (time - fpsInterval)));
          fpsInterval = time;
          frameCount = 0;
        }

        if (video.readyState >= 2) {
          const startTimeMs = performance.now();
          const result = await detector.detectForVideo(video, startTimeMs);
          const currentDetections = result.detections;
          setDetections(currentDetections);

          let updatedTrackedBox = null;

          // Step 5: Tracking Logic
          if (trackedBox) {
            let maxIoU = 0;
            let bestMatch = null;

            currentDetections.forEach((det) => {
              const iou = calculateIoU(trackedBox, det.boundingBox);
              if (iou > maxIoU && iou > 0.25) {
                maxIoU = iou;
                bestMatch = det.boundingBox;
              }
            });

            if (bestMatch) {
              updatedTrackedBox = bestMatch;
              setTrackedBox(bestMatch);
            }
          }

          // Auto-Focus Intelligence: Pick largest object if none tracked
          if (isAutoTrack && !updatedTrackedBox && currentDetections.length > 0) {
            let largestBox = currentDetections[0].boundingBox;
            let maxArea = largestBox.width * largestBox.height;

            currentDetections.forEach(det => {
              const area = det.boundingBox.width * det.boundingBox.height;
              if (area > maxArea) {
                maxArea = area;
                largestBox = det.boundingBox;
              }
            });
            updatedTrackedBox = largestBox;
            setTrackedBox(largestBox);
          }

          // Step 6: Background Rendering & Visual Effects
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          if (updatedTrackedBox && blurIntensity > 0) {
            // 1. Draw background with Filters (Blur + optional Grayscale)
            const bgFilter = `blur(${blurIntensity}px) brightness(0.7) ${isColorPop ? 'grayscale(100%)' : ''}`;
            ctx.filter = bgFilter;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            ctx.filter = 'none';

            // 2. Clear then Draw sharp version inside the tracked box
            const { originX, originY, width, height } = updatedTrackedBox;

            // Cinematic Zoom logic (optional preview effect)
            ctx.save();
            ctx.beginPath();
            ctx.rect(originX - 10, originY - 10, width + 20, height + 20);
            ctx.clip();
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            ctx.restore();
          } else {
            // Standard feed without tracking
            const standardFilter = isColorPop ? 'grayscale(100%)' : 'none';
            ctx.filter = standardFilter;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            ctx.filter = 'none';
          }

          // Step 3 & 4: Draw UI Overlays
          drawDetections(ctx, currentDetections, updatedTrackedBox);
        }
        animationId = requestAnimationFrame(runDetection);
      };

      animationId = requestAnimationFrame(runDetection);
    }
    return () => cancelAnimationFrame(animationId);
  }, [isCameraReady, detector, trackedBox, blurIntensity, isCameraActive]);

  const toggleCamera = () => {
    setIsCameraActive(prev => !prev);
    if (isCameraActive) {
      setTrackedBox(null);
      setDetections([]);
    }
  };

  const takeSnapshot = () => {
    if (!canvasRef.current) return;
    const link = document.createElement('a');
    link.download = `SmartFocus_${new Date().getTime()}.png`;
    link.href = canvasRef.current.toDataURL('image/png');
    link.click();
  };

  const handleCanvasClick = (e) => {
    if (!videoRef.current || detections.length === 0) return;

    const rect = videoRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const scaleX = videoRef.current.videoWidth / rect.width;
    const scaleY = videoRef.current.videoHeight / rect.height;

    // Video is mirrored on X axis in CSS
    const canvasX = videoRef.current.videoWidth - (x * scaleX);
    const canvasY = y * scaleY;

    let found = null;
    detections.forEach((det) => {
      const { originX, originY, width, height } = det.boundingBox;
      if (
        canvasX >= originX - 20 &&
        canvasX <= originX + width + 20 &&
        canvasY >= originY - 20 &&
        canvasY <= originY + height + 20
      ) {
        found = det.boundingBox;
      }
    });

    setTrackedBox(found);
  };

  const drawDetections = (ctx, currentDetections, currentTrackedBox) => {
    currentDetections.forEach((detection) => {
      const iou = calculateIoU(currentTrackedBox, detection.boundingBox);
      const isTracked = currentTrackedBox && iou > 0.8;

      const { originX, originY, width, height } = detection.boundingBox;
      const label = detection.categories[0].categoryName;

      // Only draw others if they aren't the tracked one
      if (!isTracked) {
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(originX, originY, width, height);
        ctx.setLineDash([]);
      } else {
        // Highlighting for tracked object
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 4;

        // Subject Glow
        ctx.shadowBlur = 20;
        ctx.shadowColor = '#facc15';
        ctx.strokeRect(originX, originY, width, height);
        ctx.shadowBlur = 0;

        // Label for tracked
        ctx.fillStyle = '#facc15';
        const text = `TRACKING: ${label.toUpperCase()}`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(originX, originY - 30, textWidth + 12, 30);

        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 14px Inter';
        ctx.fillText(text, originX + 6, originY - 10);
      }
    });
  };

  if (!hasStarted) {
    return (
      <div className="landing-page">
        <div className="hero-content">
          <h1>Smart Focus <span className="ai-badge">AI</span></h1>
          <p>Professional Object Tracking & Cinematic Background Blur</p>
          <div className="feature-dots">
            <span>• Real-time Detection</span>
            <span>• IoU Tracking</span>
            <span>• Visual FX</span>
          </div>
          <button className="start-btn" onClick={() => setHasStarted(true)}>
            LAUNCH SYSTEM
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="video-wrapper" onClick={handleCanvasClick}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="webcam-video"
          style={{ visibility: 'hidden', position: 'absolute' }}
        />
        <canvas
          ref={canvasRef}
          className="overlay-canvas"
        />
      </div>

      {(!isCameraReady || !detector) && (
        <div className="loading-overlay">
          <div className="loader"></div>
          <p>{!detector ? "Loading AI Models..." : "Waking up Camera..."}</p>
        </div>
      )}

      <div className="ui-panel">
        <div className="panel-header">
          <h1>Smart Focus AI</h1>
          <div className="fps-badge">{fps} FPS</div>
        </div>

        <div className="panel-body">
          <p className="status-text">
            {trackedBox ? "✓ Subject Locked" : "Click an object to focus"}
          </p>

          <div className="control-group">
            <label>Blur Intensity</label>
            <input
              type="range"
              min="0"
              max="40"
              value={blurIntensity}
              onChange={(e) => setBlurIntensity(parseInt(e.target.value))}
            />
          </div>

          <button
            className={`camera-toggle ${isCameraActive ? 'active' : 'inactive'}`}
            onClick={toggleCamera}
          >
            {isCameraActive ? '📷 Turn Camera OFF' : '📹 Turn Camera ON'}
          </button>

          <div className="extra-actions">
            <div className="toggle-group">
              <label className="switch">
                <input type="checkbox" checked={isColorPop} onChange={e => setIsColorPop(e.target.checked)} />
                <span className="slider round"></span>
              </label>
              <span>Color Pop (B&W)</span>
            </div>

            <div className="toggle-group">
              <label className="switch">
                <input type="checkbox" checked={isAutoTrack} onChange={e => setIsAutoTrack(e.target.checked)} />
                <span className="slider round"></span>
              </label>
              <span>AI Auto-Focus</span>
            </div>

            <button className="snapshot-btn" onClick={takeSnapshot} disabled={!isCameraActive}>
              📸 Capture Masterpiece
            </button>
          </div>
        </div>

        <div className="panel-footer">
          <p>Hackathon v1.0 • MediaPipe AI</p>
        </div>
      </div>
    </div>
  );
}

export default App;

