'use client';

import { useState } from 'react';
import UserForm from '@/components/UserForm';
import VideoCapture from '@/components/VideoCapture';
import Results from '@/components/Results';
import 'bootstrap/dist/css/bootstrap.min.css';

export default function Home() {
  const [step, setStep] = useState('form');
  const [userData, setUserData] = useState(null);
  const [videoResult, setVideoResult] = useState(null);

  const handleUserSubmit = (data) => {
    setUserData(data);
    setStep('video');
  };

  const handleVideoProcessed = (result) => {
    setVideoResult(result);
    setStep('results');
  };

  const handleStartOver = () => {
    setStep('form');
    setUserData(null);
    setVideoResult(null);
  };

  return (
    <div className="container py-5 min-vh-100 d-flex flex-column">
      <div className="wrapper flex-grow-1 d-flex flex-column justify-content-center">
        {/* Header */}
        <header className="text-center mb-5">
          <h1 className="display-4 fw-bold text-dark">Confidence Analyser</h1>
          <p className="lead text-secondary mb-0" style={{ maxWidth: '600px', margin: 'auto' }}>
            Assess your confidence levels through advanced video analysis. Gain
            insights on body language, vocal tone, and presentation skills.
          </p>
        </header>

        <div className="card shadow-lg border-0 mx-auto rounded-4 card-fixed-width">
          <div className="card-body">
            {step === 'form' && <UserForm onSubmit={handleUserSubmit} />}

            {step === 'video' && userData && (
              <VideoCapture
                userData={userData}
                onVideoProcessed={handleVideoProcessed}
              />
            )}

            {step === 'results' && userData && videoResult && (
              <Results
                userData={userData}
                result={videoResult}
                onStartOver={handleStartOver}
              />
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-5">
          <small className="text-muted">
            Powered by cutting-edge confidence assessment technology
          </small>
        </footer>
      </div>
    </div>
  );
}