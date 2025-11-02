import React from "react";
import { Star, RotateCcw, Heart, Video, Mic, BarChart3 } from "lucide-react";

const Results = ({ userData, result, onStartOver }) => {
  const getScoreColor = (score) => {
    if (score >= 90) return "text-success";
    if (score >= 80) return "text-primary";
    if (score >= 70) return "text-warning";
    if (score >= 60) return "text-secondary";
    return "text-danger";
  };

  const getScoreBadgeVariant = (score) => {
    if (score >= 90) return "bg-success text-white";
    if (score >= 80) return "bg-primary text-white";
    if (score >= 70) return "bg-warning text-dark";
    return "bg-danger text-white";
  };

  const getConfidenceLevel = (score) => {
    if (score >= 90) return "Very Confident";
    if (score >= 80) return "Confident";
    if (score >= 70) return "Moderately Confident";
    if (score >= 60) return "Building Confidence";
    return "Needs Practice";
  };

  const getConfidenceEmoji = (score) => {
    if (score >= 90) return "🌟";
    if (score >= 80) return "💪";
    if (score >= 70) return "👍";
    if (score >= 60) return "📈";
    return "🎯";
  };

  const getProgressBarVariant = (score) => {
    if (score >= 90) return "bg-success";
    if (score >= 80) return "bg-primary";
    if (score >= 70) return "bg-warning";
    if (score >= 60) return "bg-secondary";
    return "bg-danger";
  };

  return (
    <div className="w-100">
      <div className="text-center mb-4">
        <h4 className="d-flex align-items-center justify-content-center gap-2">
          <Heart className="text-success" size={24} />
          Results for {userData.name}
        </h4>
        <p className="text-muted">
          Your confidence analysis is complete
        </p>
      </div>

      {/* Overall Score */}
      <div className="card mb-4">
        <div className="card-body text-center">
          <div className="display-1 mb-2">{getConfidenceEmoji(result.score)}</div>
          <div className={`display-4 fw-bold ${getScoreColor(result.score)} mb-2`}>
            {result.score}/100
          </div>
          <span
            className={`badge ${getScoreBadgeVariant(result.score)} px-3 py-2 fs-6`}
          >
            <Star className="me-1" size={16} />
            {getConfidenceLevel(result.score)}
          </span>
        </div>
      </div>

      {/* Detailed Breakdown */}
      {result.breakdown && (
        <div className="card mb-4">
          <div className="card-header bg-light">
            <h5 className="mb-0 d-flex align-items-center">
              <BarChart3 className="me-2" size={20} />
              Detailed Analysis
            </h5>
          </div>
          <div className="card-body">
            {/* Video Analysis */}
            <div className="mb-4">
              <div className="d-flex align-items-center mb-2">
                <Video className="me-2 text-primary" size={18} />
                <h6 className="mb-0">Video Analysis ({result.breakdown.videoModel})</h6>
              </div>
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span>Confidence: {result.breakdown.videoScore}%</span>
                <span className="badge bg-primary">{result.breakdown.videoPrediction}</span>
              </div>
              <div className="progress mb-2" style={{ height: "8px" }}>
                <div
                  className={`progress-bar ${getProgressBarVariant(result.breakdown.videoScore)}`}
                  style={{ width: `${result.breakdown.videoScore}%` }}
                ></div>
              </div>
              <small className="text-muted">
                Analyzed {result.breakdown.framesAnalyzed} frames
              </small>
            </div>

            {/* Audio Analysis */}
            <div>
              <div className="d-flex align-items-center mb-2">
                <Mic className="me-2 text-info" size={18} />
                <h6 className="mb-0">Audio Analysis ({result.breakdown.audioModel})</h6>
              </div>
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span>Confidence: {result.breakdown.audioScore}%</span>
                <span className="badge bg-info text-dark">{result.breakdown.audioPrediction}</span>
              </div>
              <div className="progress mb-2" style={{ height: "8px" }}>
                <div
                  className={`progress-bar ${getProgressBarVariant(result.breakdown.audioScore)}`}
                  style={{ width: `${result.breakdown.audioScore}%` }}
                ></div>
              </div>
              <small className="text-muted">
                Analyzed {result.breakdown.segmentsAnalyzed} audio segments
              </small>
            </div>
          </div>
        </div>
      )}

      {/* Feedback */}
      <div className="card mb-4">
        <div className="card-body">
          <h5 className="card-title d-flex align-items-center">
            <Heart className="me-2 text-success" size={20} />
            Feedback
          </h5>
          <p className="card-text">{result.feedback}</p>
        </div>
      </div>

      {/* Action Button */}
      <button 
        onClick={onStartOver} 
        className="btn btn-primary w-100 py-2"
      >
        <RotateCcw className="me-2" size={16} />
        Try Again
      </button>

      {/* Debug Info (optional - remove in production) */}
      {result.detailedResults && (
        <details className="mt-3">
          <summary className="text-muted small">View Raw Data</summary>
          <pre className="small bg-light p-2 mt-2 rounded">
            {JSON.stringify(result.detailedResults, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
};

export default Results;