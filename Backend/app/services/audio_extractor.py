import whisper
import librosa
import numpy as np
import tempfile
import os
import subprocess
from typing import List, Dict, Any

class AudioFeatureExtractor:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
    
    def extract_audio_from_video_bytes(self, video_bytes: bytes) -> str:
        """Extract audio from video bytes using ffmpeg"""
        # Create temporary video file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name
        
        # Create temporary audio file
        temp_audio_path = temp_video_path.replace('.webm', '_audio.wav')
        
        try:
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', temp_video_path, 
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                temp_audio_path, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
            
            return temp_audio_path
        finally:
            # Clean up temporary video file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    def transcribe_with_confidence(self, audio_path: str):
        """Transcribe audio and get confidence scores - FIXED for new Whisper API"""
        # Use the new API without word_timestamps parameter
        result = self.model.transcribe(audio_path)
        segments = []
        
        for segment in result["segments"]:
            confidence = np.exp(segment.get("avg_logprob", -0.7))
            label = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
            segments.append({
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"].strip(),
                "confidence": round(confidence, 4),
                "label": label
            })
        
        return segments
    
    def extract_audio_features(self, audio_path: str, segments: List[Dict]) -> List[Dict[str, Any]]:
        """Extract audio features using librosa"""
        features_list = []
        y, sr = librosa.load(audio_path, sr=None)
        
        # If no segments from transcription, create one segment for the entire audio
        if not segments:
            segments = [{
                'start_time': 0,
                'end_time': len(y) / sr,
                'text': '',
                'confidence': 0.5,
                'label': 'Medium'
            }]
        
        for seg in segments:
            start = seg['start_time']
            end = seg['end_time']
            text = seg['text']
            label = seg['label']
            
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            y_seg = y[start_sample:end_sample]
            
            # Skip if segment is too short
            if len(y_seg) < 100:
                continue
            
            try:
                # Extract audio features
                mfccs = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)
                
                chroma = librosa.feature.chroma_stft(y=y_seg, sr=sr, n_chroma=12)  # Ensure 12 chroma features
                chroma_mean = np.mean(chroma, axis=1)
                
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_seg, sr=sr))
                spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_seg, sr=sr))
                zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y_seg))
                rms_energy = np.mean(librosa.feature.rms(y=y_seg))
                
                words = text.split()
                num_words = len(words)
                duration = end - start
                speech_rate = num_words / duration if duration > 0 else 0
                
                # Create feature dictionary
                feature_dict = {
                    'start_time': start,
                    'end_time': end,
                    'text': text,
                    'num_words': num_words,
                    'duration': duration,
                    'spectral_centroid': float(spectral_centroid),
                    'spectral_bandwidth': float(spectral_bandwidth),
                    'zero_crossing_rate': float(zero_crossing_rate),
                    'rms_energy': float(rms_energy),
                    'speech_rate': float(speech_rate),
                    'transcription_confidence': seg['confidence'],
                    'transcription_label': label
                }
                
                # Add ALL 13 MFCC features
                for i, val in enumerate(mfccs_mean, 1):
                    feature_dict[f'mfcc_{i}'] = float(val)
                
                # Add ALL 12 chroma features
                for i, val in enumerate(chroma_mean, 1):
                    feature_dict[f'chroma_{i}'] = float(val)
                
                features_list.append(feature_dict)
                
            except Exception as e:
                print(f"⚠️ Error extracting audio features for segment: {e}")
                continue
        
        print(f"✅ Extracted {len(features_list)} audio segments")
        return features_list
    
    def extract_features_from_video_bytes(self, video_bytes: bytes) -> List[Dict[str, Any]]:
        """Main method to extract audio features from video bytes"""
        try:
            # Extract audio to temporary file
            audio_path = self.extract_audio_from_video_bytes(video_bytes)
            
            # Transcribe audio
            segments = self.transcribe_with_confidence(audio_path)
            
            # Extract audio features
            audio_features = self.extract_audio_features(audio_path, segments)
            
            return audio_features
        except Exception as e:
            print(f"❌ Audio extraction failed: {e}")
            return []
        finally:
            # Clean up temporary audio file
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.unlink(audio_path)

# Global instance
audio_extractor = AudioFeatureExtractor()