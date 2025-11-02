import { useState, useRef, useCallback, useEffect } from "react";
import {
  Video,
  Upload,
  Play,
  Square,
  RotateCcw,
  Heart,
  Pause,
  Mic,
} from "lucide-react";

export default function VideoCapture({ userData, onVideoProcessed }) {
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordedVideoURL, setRecordedVideoURL] = useState(null);
  const [uploadedVideoURL, setUploadedVideoURL] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("record");

  const [recordingTime, setRecordingTime] = useState(0);
  const [micLevel, setMicLevel] = useState(0);
  const recordingIntervalRef = useRef(null);

  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  const analyserRef = useRef(null);
  const audioContextRef = useRef(null);
  const rafRef = useRef(null);

  const [clickCount, setClickCount] = useState(0);
  const clickTimeoutRef = useRef(null);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      try {
        if (
          stream.getAudioTracks &&
          stream.getAudioTracks().length > 0 &&
          (window.AudioContext || window.webkitAudioContext)
        ) {
          const AC = window.AudioContext || window.webkitAudioContext;
          audioContextRef.current = new AC();
          const source = audioContextRef.current.createMediaStreamSource(
            stream
          );
          const analyser = audioContextRef.current.createAnalyser();
          analyser.fftSize = 256;
          source.connect(analyser);
          analyserRef.current = analyser;

          const bufferLength = analyser.frequencyBinCount;
          const dataArr = new Uint8Array(bufferLength);

          const updateMic = () => {
            analyser.getByteTimeDomainData(dataArr);
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
              const v = (dataArr[i] - 128) / 128;
              sum += v * v;
            }
            const rms = Math.sqrt(sum / bufferLength);
            setMicLevel(Math.min(1, rms * 1.6));
            rafRef.current = requestAnimationFrame(updateMic);
          };
          if (!rafRef.current) rafRef.current = requestAnimationFrame(updateMic);
        }
      } catch (e) {
        console.warn("Mic analyser not available", e);
      }
    } catch (err) {
      setError("Unable to access camera. Please check permissions.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (e) {}
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setMicLevel(0);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const startRecording = useCallback(() => {
    if (!streamRef.current) return;

    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }

    setRecordingTime(0);

    if (recordedVideoURL) {
      URL.revokeObjectURL(recordedVideoURL);
      setRecordedVideoURL(null);
    }

    chunksRef.current = [];
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    mediaRecorderRef.current = new MediaRecorder(streamRef.current);

    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      setRecordedVideoURL(url);
      stopCamera();

      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
    };

    try {
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setIsPaused(false);

      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);
    } catch (e) {
      setError("Unable to start recording in this browser.");
      console.error(e);
    }
  }, [recordedVideoURL, stopCamera]);

  const stopRecording = useCallback(() => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    setIsPaused(false);

    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
  }, []);

  const pauseRecording = useCallback(() => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === "recording"
    ) {
      mediaRecorderRef.current.pause();
      setIsPaused(true);
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
    }
  }, []);

  const resumeRecording = useCallback(() => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === "paused"
    ) {
      mediaRecorderRef.current.resume();
      setIsPaused(false);
      if (!recordingIntervalRef.current) {
        recordingIntervalRef.current = setInterval(() => {
          setRecordingTime((t) => t + 1);
        }, 1000);
      }
    }
  }, []);

  const handlePauseClick = useCallback(() => {
    if (clickTimeoutRef.current) {
      clearTimeout(clickTimeoutRef.current);
      clickTimeoutRef.current = null;
    }

    setClickCount((prevCount) => prevCount + 1);

    if (clickCount === 1) {
      stopRecording();
      setClickCount(0);
    } else {
      pauseRecording();
      clickTimeoutRef.current = setTimeout(() => {
        setClickCount(0);
      }, 300);
    }
  }, [clickCount, pauseRecording, stopRecording]);

  const recordAgain = useCallback(async () => {
    if (recordedVideoURL) {
      URL.revokeObjectURL(recordedVideoURL);
      setRecordedVideoURL(null);
    }

    setRecordingTime(0);

    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }

    setIsRecording(false);
    setIsPaused(false);

    await startCamera();
  }, [recordedVideoURL, startCamera]);

  const handleFileUpload = (event) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith("video/")) {
      if (file.size > 100 * 1024 * 1024) {
        setError("File size exceeds 100MB");
        return;
      }
      if (uploadedVideoURL) {
        URL.revokeObjectURL(uploadedVideoURL);
      }
      const url = URL.createObjectURL(file);
      setUploadedVideoURL(url);
      setError(null);
    } else {
      setError("Please select a valid video file.");
    }
  };

  const clearUploadedVideo = () => {
    if (uploadedVideoURL) {
      URL.revokeObjectURL(uploadedVideoURL);
    }
    setUploadedVideoURL(null);
  };

  const processVideo = async (videoURL) => {
    setIsProcessing(true);
    setProcessingProgress(0);
    setError(null);

    try {
      const response = await fetch(videoURL);
      const videoBlob = await response.blob();
      
      const formData = new FormData();
      formData.append('video', videoBlob, 'confidence-video.webm');

      setProcessingProgress(20);

      const apiResponse = await fetch('http://127.0.0.1:8000/confidence-analyzer/confidence-score', {
        method: 'POST',
        body: formData,
      });

      setProcessingProgress(60);

      if (!apiResponse.ok) {
        const errorData = await apiResponse.json();
        throw new Error(errorData.detail || `HTTP error! status: ${apiResponse.status}`);
      }

      const result = await apiResponse.json();
      
      setProcessingProgress(100);

      // Use the REAL result from backend - ADJUSTED FOR NEW FORMAT
      const backendResult = result.result;
      const combinedScore = backendResult.combined_confidence_score;
      
      const finalResult = {
        score: Math.round(combinedScore * 100), // Convert 0.785 to 78.5
        feedback: generateConfidenceFeedback(combinedScore), // Pass the decimal score
        processingTime: 2.4,
        detailedResults: backendResult, // Include full backend response
        breakdown: {
          videoScore: Math.round(backendResult.predictions.video_analysis.confidence_score * 100),
          audioScore: Math.round(backendResult.predictions.audio_analysis.confidence_score * 100),
          videoPrediction: backendResult.predictions.video_analysis.prediction,
          audioPrediction: backendResult.predictions.audio_analysis.prediction,
          videoModel: backendResult.predictions.video_analysis.model,
          audioModel: backendResult.predictions.audio_analysis.model,
          framesAnalyzed: backendResult.predictions.video_analysis.frames_analyzed,
          segmentsAnalyzed: backendResult.predictions.audio_analysis.audio_segments_analyzed
        }
      };

      setIsProcessing(false);
      onVideoProcessed(finalResult);

      console.log('✅ Video successfully analyzed:', result);

    } catch (err) {
      console.error('❌ Error analyzing video:', err);
      setError(`Failed to process video: ${err.message}`);
      setIsProcessing(false);
    }
  };

  const generateConfidenceFeedback = (score) => {
    // Score is now decimal (0.785) instead of percentage (78.5)
    const percentage = score * 100;
    
    if (percentage >= 90)
      return "Outstanding confidence! You display excellent self-assurance, strong body language, and commanding presence. Your vocal tone is steady and engaging.";
    if (percentage >= 80)
      return "High confidence levels! You show good self-assurance with strong eye contact and clear vocal delivery. Minor improvements in posture could enhance your presence.";
    if (percentage >= 70)
      return "Good confidence foundation! You demonstrate solid self-assurance. Work on maintaining consistent eye contact and speaking with more vocal variety.";
    if (percentage >= 60)
      return "Moderate confidence levels. Focus on improving your posture, maintaining steadier eye contact, and speaking with more conviction in your voice.";
    return "Building confidence needed. Practice speaking more slowly, maintain better posture, and work on steady eye contact to project greater self-assurance.";
  };

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        try {
          mediaRecorderRef.current.stop();
        } catch (e) {}
      }
      if (recordingIntervalRef.current)
        clearInterval(recordingIntervalRef.current);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (audioContextRef.current) {
        try {
          audioContextRef.current.close();
        } catch (e) {}
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const formatTime = (sec) => {
    const m = Math.floor(sec / 60).toString().padStart(2, "0");
    const s = Math.floor(sec % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  return (
    <div className="w-100">
      <div className="text-center mb-4">
        <h4 className="d-flex align-items-center justify-content-center gap-2">
          <Heart className="text-success" size={24} />
          Confidence Assessment
        </h4>
        <p className="text-muted">
          Welcome {userData.name}! Record a new video or upload an existing one
          for confidence analysis. Speak naturally for 30-60 seconds.
        </p>
      </div>

      {isProcessing ? (
        <div className="text-center">
          <Heart className="heart-pulse mb-3" size={48} />
          <h5>Analyzing Your Confidence...</h5>
          <div className="progress mb-3" style={{ height: "20px" }}>
            <div
              className="progress-bar bg-primary"
              role="progressbar"
              style={{ width: `${processingProgress}%` }}
              aria-valuenow={processingProgress}
              aria-valuemin="0"
              aria-valuemax="100"
            ></div>
          </div>
          <p className="text-muted">
            Evaluating body language, vocal patterns, eye contact, and overall
            presence...
          </p>
        </div>
      ) : (
        <div>
          <ul className="nav nav-pills mb-4 justify-content-center">
            <li className="nav-item">
              <button
                className={`nav-link ${activeTab === "record" ? "active" : ""}`}
                onClick={() => setActiveTab("record")}
              >
                Record Video
              </button>
            </li>
            <li className="nav-item">
              <button
                className={`nav-link ${activeTab === "upload" ? "active" : ""}`}
                onClick={() => setActiveTab("upload")}
              >
                Upload Video
              </button>
            </li>
          </ul>

          <div className="tab-content">
            <div className={`tab-pane ${activeTab === "record" ? "active" : ""}`}>
              <div
                className="mb-4 rounded position-relative"
                style={{
                  width: "100%",
                  maxHeight: "400px",
                  overflow: "hidden",
                  backgroundColor: "#000",
                  border: isRecording
                    ? "3px solid #dc3545"
                    : isPaused
                    ? "3px solid #ffc107"
                    : "3px solid #dee2e6",
                  boxShadow: isRecording
                    ? "0 0 20px rgba(220, 53, 69, 0.3)"
                    : isPaused
                    ? "0 0 20px rgba(255, 193, 7, 0.3)"
                    : "0 4px 12px rgba(0,0,0,0.1)",
                  transition: "all 0.3s ease",
                }}
              >
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  style={{
                    maxHeight: "400px",
                    objectFit: "cover",
                    width: "100%",
                    display: streamRef.current ? "block" : "none",
                  }}
                />

                {!streamRef.current && !recordedVideoURL && (
                  <div
                    className="d-flex align-items-center justify-content-center text-white"
                    style={{
                      height: "400px",
                      background: "linear-gradient(135deg, #b8bac6ff 0%, #636065ff 100%)",
                    }}
                  >
                    <div className="text-center">
                      <Video size={64} className="mb-3 opacity-75" />
                      <h5>Camera Preview</h5>
                      <p className="mb-0 opacity-75">
                        Click "Start Camera" to begin
                      </p>
                    </div>
                  </div>
                )}

                {(isRecording || isPaused) && (
                  <>
                    <div
                      style={{
                        position: "absolute",
                        top: "0",
                        left: "0",
                        right: "0",
                        background:
                          "linear-gradient(90deg, #dc3545 0%, #ff6b6b 100%)",
                        height: "4px",
                        zIndex: 10,
                      }}
                    />

                    <div
                      style={{
                        position: "absolute",
                        top: "15px",
                        left: "15px",
                        display: "flex",
                        alignItems: "center",
                        gap: "10px",
                        backgroundColor: "rgba(0,0,0,0.8)",
                        color: "white",
                        padding: "8px 12px",
                        borderRadius: "20px",
                        fontSize: "0.9em",
                        fontVariantNumeric: "tabular-nums",
                        zIndex: 10,
                        backdropFilter: "blur(10px)",
                      }}
                    >
                      <div
                        style={{
                          width: "10px",
                          height: "10px",
                          backgroundColor: isRecording ? "#dc3545" : "#ffc107",
                          borderRadius: "50%",
                          animation: isRecording ? "pulse 1.5s infinite" : "none",
                        }}
                      />
                      <span
                        style={{
                          fontWeight: "600",
                          color: isRecording ? "#dc3545" : "#ffc107",
                        }}
                      >
                        {isPaused ? "PAUSED" : "RECORDING"}
                      </span>
                      <span
                        style={{
                          marginLeft: "8px",
                          fontFamily: "monospace",
                          fontWeight: "600",
                        }}
                      >
                        {formatTime(recordingTime)}
                      </span>
                    </div>

                    <div
                      style={{
                        position: "absolute",
                        top: "15px",
                        right: "15px",
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        backgroundColor: "rgba(0,0,0,0.8)",
                        color: "white",
                        padding: "8px 12px",
                        borderRadius: "20px",
                        zIndex: 10,
                        backdropFilter: "blur(10px)",
                      }}
                    >
                      <Mic size={16} />
                      <div
                        style={{
                          width: "40px",
                          height: "4px",
                          backgroundColor: "rgba(255,255,255,0.3)",
                          borderRadius: "2px",
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: `${micLevel * 100}%`,
                            height: "100%",
                            backgroundColor:
                              micLevel > 0.7
                                ? "#28a745"
                                : micLevel > 0.3
                                ? "#ffc107"
                                : "#dc3545",
                            transition: "all 0.1s ease",
                            borderRadius: "2px",
                          }}
                        />
                      </div>
                    </div>

                    <div
                      style={{
                        position: "absolute",
                        bottom: "20px",
                        left: "50%",
                        transform: "translateX(-50%)",
                        display: "flex",
                        gap: "20px",
                        alignItems: "center",
                        zIndex: 10,
                      }}
                    >
                      {isRecording && !isPaused && (
                        <button
                          className="control-btn control-btn-pause"
                          onClick={handlePauseClick}
                        >
                          <Pause color="white" size={24} />
                        </button>
                      )}
                      {isPaused && (
                        <button
                          className="control-btn control-btn-resume"
                          onClick={resumeRecording}
                        >
                          <Play color="white" size={24} />
                        </button>
                      )}
                      <button
                        className="control-btn control-btn-stop"
                        onClick={stopRecording}
                      >
                        <Square color="white" size={20} />
                      </button>
                    </div>
                  </>
                )}
              </div>

              <div className="text-center mb-4">
                <div className="row justify-content-center">
                  <div className="col-md-8">
                    <div className="alert alert-info bg-light border-0">
                      <small>
                        <strong>💡 Pro Tips:</strong> Look directly at the camera,
                        speak clearly, maintain good posture, and use hand
                        gestures naturally for the best confidence assessment
                      </small>
                    </div>
                  </div>
                </div>
              </div>

              <div className="d-flex flex-wrap gap-3 justify-content-center mb-4">
                {!streamRef.current && !recordedVideoURL && (
                  <button
                    className="btn btn-primary btn-lg d-flex align-items-center"
                    onClick={startCamera}
                  >
                    <Video className="me-2" size={20} />
                    Start Camera
                  </button>
                )}
                {streamRef.current && !isRecording && !isPaused && (
                  <button
                    className="btn btn-danger btn-lg d-flex align-items-center"
                    onClick={startRecording}
                  >
                    <div
                      className="me-2"
                      style={{
                        width: "12px",
                        height: "12px",
                        backgroundColor: "white",
                        borderRadius: "2px",
                      }}
                    />
                    Start Recording
                  </button>
                )}
                {recordedVideoURL && (
                  <>
                    <button
                      className="btn btn-outline-secondary btn-lg d-flex align-items-center"
                      onClick={recordAgain}
                    >
                      <RotateCcw className="me-2" size={18} />
                      Record Again
                    </button>
                    <button
                      className="btn btn-success btn-lg d-flex align-items-center"
                      onClick={() => processVideo(recordedVideoURL)}
                    >
                      <Heart className="me-2" size={18} />
                      Analyze Confidence
                    </button>
                  </>
                )}
              </div>
            </div>

            <div className={`tab-pane ${activeTab === "upload" ? "active" : ""}`}>
              {!uploadedVideoURL ? (
                <div
                  className="border border-dashed border-primary rounded p-5 text-center mb-4 upload-area"
                  style={{
                    background:
                      "linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%)",
                    cursor: "pointer",
                    transition: "all 0.3s ease",
                  }}
                  onMouseEnter={(e) => (e.target.style.transform = "translateY(-2px)")}
                  onMouseLeave={(e) => (e.target.style.transform = "translateY(0)")}
                  onClick={() => document.getElementById("video-upload").click()}
                >
                  <Upload className="mb-3 text-primary" size={64} />
                  <label htmlFor="video-upload" className="d-block cursor-pointer">
                    <span className="h5 text-primary d-block">
                      Click to upload a video file
                    </span>
                    <p className="text-muted mt-2 mb-1">
                      Supports MP4, WebM, AVI (max 100MB)
                    </p>
                    <p className="text-muted small">
                      Best results with videos showing your face and upper body
                      clearly
                    </p>
                  </label>
                  <input
                    id="video-upload"
                    type="file"
                    accept="video/*"
                    onChange={handleFileUpload}
                    className="d-none"
                  />
                </div>
              ) : (
                <div className="mb-4">
                  <div className="position-relative">
                    <video
                      src={uploadedVideoURL}
                      controls
                      className="rounded"
                      style={{
                        maxHeight: "400px",
                        objectFit: "cover",
                        width: "100%",
                        border: "3px solid #dee2e6",
                      }}
                    />
                  </div>
                  <div className="d-flex flex-wrap gap-3 justify-content-center mt-4">
                    <button
                      className="btn btn-outline-secondary btn-lg"
                      onClick={clearUploadedVideo}
                    >
                      Change Video
                    </button>
                    <button
                      className="btn btn-success btn-lg d-flex align-items-center"
                      onClick={() => processVideo(uploadedVideoURL)}
                    >
                      <Heart className="me-2" size={18} />
                      Analyze Confidence
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="alert alert-danger mt-3" role="alert">
          {error}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
        .heart-pulse {
          animation: pulse 2s infinite;
          color: #dc3545;
        }
        .cursor-pointer {
          cursor: pointer;
        }
        .btn {
          transition: all 0.3s ease;
        }
        .btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        .btn-success {
          background-color: #6f42c1 !important;
          border-color: #6f42c1 !important;
        }

        .btn-success:hover {
          background-color: #5d34a8 !important;
          border-color: #5d34a8 !important;
        }

        .control-btn {
          width: 60px;
          height: 60px;
          border-radius: 50%;
          border: none;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background-color 0.2s;
        }
        
        .control-btn-stop {
          background-color: #343a40;
        }
        
        .control-btn-pause {
          background-color: #5b636a;
        }
        
        .control-btn-resume {
          background-color: #4a5a4a;
        }
      `}</style>
    </div>
  );
}