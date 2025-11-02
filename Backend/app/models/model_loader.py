import pickle
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any  # ADD THIS IMPORT

class ModelLoader:
    def __init__(self):
        self.models_loaded = False
        self.xgb_model = None  # For audio analysis
        self.rf_model = None   # For video analysis
        self.xgb_label_encoder = None
        self.rf_label_encoder = None
        self.xgb_scaler = None
        self.rf_scaler = None
        self.audio_features_config = None
        
    # def load_models(self):
    #     """Load both XGBoost (audio) and Random Forest (video) models"""
    #     try:
    #         models_dir = Path("app/models")
            
    #         # Load XGBoost audio model files
    #         print("📊 Loading XGBoost audio model...")
    #         with open(models_dir / "label_encoder.pkl", "rb") as f:
    #             self.xgb_label_encoder = pickle.load(f)
            
    #         with open(models_dir / "scaler.pkl", "rb") as f:
    #             self.xgb_scaler = pickle.load(f)
            
    #         with open(models_dir / "audio.json", "r") as f:
    #             self.audio_features_config = json.load(f)
            
    #         # Load the actual XGBoost model from JSON file
    #         try:
    #             import xgboost as xgb
    #             self.xgb_model = xgb.Booster()
    #             self.xgb_model.load_model(str(models_dir / "audio.json"))
    #             print("✅ XGBoost model loaded from audio.json")
    #         except Exception as e:
    #             print(f"❌ Failed to load XGBoost model: {e}")
    #             self.xgb_model = None
            
    #         # Load Random Forest video model files
    #         print("🎥 Loading Random Forest video model...")
    #         with open(models_dir / "rf_confidence_model.pkl", "rb") as f:
    #             self.rf_model = joblib.load(f)
            
    #         with open(models_dir / "rf_label_encoder.pkl", "rb") as f:
    #             self.rf_label_encoder = pickle.load(f)
            
    #         with open(models_dir / "rf_scaler.pkl", "rb") as f:
    #             self.rf_scaler = pickle.load(f)
            
    #         self.models_loaded = True
    #         print("✅ All models loaded successfully!")
            
    #     except Exception as e:
    #         print(f"❌ Error loading models: {str(e)}")
    #         raise e

    def debug_feature_extraction(self, audio_features_list: List[Dict]):
        """Debug audio features extraction"""
        if not audio_features_list:
            print("❌ No audio features extracted")
            return
        
        df = pd.DataFrame(audio_features_list)
        print("🔍 AUDIO FEATURE DEBUG INFO:")
        print(f"   Number of segments: {len(audio_features_list)}")
        print(f"   Available columns: {df.columns.tolist()}")
        print(f"   Column count: {len(df.columns)}")
        
        # Check for specific feature categories
        mfcc_cols = [col for col in df.columns if col.startswith('mfcc_')]
        chroma_cols = [col for col in df.columns if col.startswith('chroma_')]
        basic_cols = [col for col in df.columns if not col.startswith(('mfcc_', 'chroma_'))]
        
        print(f"   Basic features: {len(basic_cols)}")
        print(f"   MFCC features: {len(mfcc_cols)}")
        print(f"   Chroma features: {len(chroma_cols)}")
        
        # Show sample values
        if not df.empty:
            sample_row = df.iloc[0]
            print("   Sample feature values:")
            for col in basic_cols[:5]:  # Show first 5 basic features
                print(f"     {col}: {sample_row[col]}")

    def debug_scaler_info(self):
        """Debug scaler information"""
        if self.xgb_scaler is not None:
            print("🔍 SCALER DEBUG INFO:")
            if hasattr(self.xgb_scaler, 'n_features_in_'):
                print(f"   Scaler expects {self.xgb_scaler.n_features_in_} features")
            if hasattr(self.xgb_scaler, 'feature_names_in_'):
                print(f"   Scaler feature names: {self.xgb_scaler.feature_names_in_}")
            else:
                print("   No feature names available in scaler")
        else:
            print("❌ No scaler loaded")


    def load_models(self):
        """Load both XGBoost (audio) and Random Forest (video) models"""
        try:
            models_dir = Path("app/models")
            
            # Load Random Forest video model files FIRST (this should work)
            print("🎥 Loading Random Forest video model...")
            with open(models_dir / "rf_confidence_model.pkl", "rb") as f:
                self.rf_model = joblib.load(f)
            
            with open(models_dir / "rf_label_encoder.pkl", "rb") as f:
                self.rf_label_encoder = joblib.load(f)
            
            with open(models_dir / "rf_scaler.pkl", "rb") as f:
                self.rf_scaler = joblib.load(f)
            print("✅ Random Forest model loaded successfully!")
            
            # Now try to load XGBoost audio model files
            print("📊 Loading XGBoost audio model...")
            try:
                with open(models_dir / "label_encoder.pkl", "rb") as f:
                    self.xgb_label_encoder = joblib.load(f)
                
                with open(models_dir / "scaler.pkl", "rb") as f:
                    self.xgb_scaler = joblib.load(f)
                
                with open(models_dir / "audio.json", "r") as f:
                    self.audio_features_config = json.load(f)
                print("✅ XGBoost preprocessing files loaded!")
                
                # Try to load XGBoost model with better error handling
                self.xgb_model = self._load_xgboost_safe(models_dir)
                
            except Exception as e:
                print(f"⚠️ XGBoost model loading failed: {e}")
                print("🎯 Continuing with Random Forest only for now...")
                self.xgb_model = None
            
            self.models_loaded = True
            print("✅ Model loading completed!")
            
        except Exception as e:
            print(f"❌ Critical error loading models: {str(e)}")
            raise e

    def _load_xgboost_safe(self, models_dir):
        """Safely load XGBoost model with multiple fallback methods"""
        try:
            import xgboost as xgb
            print(f"🔧 XGBoost version: {xgb.__version__}")
            
            # Method 1: Try direct loading
            try:
                model = xgb.Booster()
                model.load_model(str(models_dir / "audio.json"))
                print("✅ XGBoost model loaded with Method 1")
                return model
            except Exception as e1:
                print(f"❌ Method 1 failed: {e1}")
            
            # Method 2: Try with model_file parameter
            try:
                model = xgb.Booster(model_file=str(models_dir / "audio.json"))
                print("✅ XGBoost model loaded with Method 2")
                return model
            except Exception as e2:
                print(f"❌ Method 2 failed: {e2}")
            
            # Method 3: Check if file is actually a pickle
            try:
                with open(models_dir / "audio.json", "rb") as f:
                    # Check if it's a pickle file by reading magic bytes
                    magic = f.read(4)
                    f.seek(0)
                    
                    if magic.startswith(b'\x80\x03') or magic.startswith(b'\x80\x04') or magic.startswith(b'\x80\x05'):
                        print("🔍 File appears to be pickle format, loading as pickle...")
                        model = pickle.load(f)
                        print("✅ XGBoost model loaded as pickle")
                        return model
                    else:
                        print("❓ File format not recognized")
            except Exception as e3:
                print(f"❌ Method 3 failed: {e3}")
            
            return None
            
        except ImportError:
            print("❌ XGBoost not installed, skipping audio model")
            return None
        except Exception as e:
            print(f"❌ All XGBoost loading methods failed: {e}")
            return None
    
    def prepare_audio_features_for_xgboost(self, audio_features_list: List[Dict]) -> np.ndarray:
        """Prepare audio features for XGBoost model - FIXED for 32 features"""
        if not audio_features_list:
            return None
        
        df = pd.DataFrame(audio_features_list)
        
        # Debug what we have vs what scaler expects
        print(f"🔍 XGBOOST FEATURE DEBUG:")
        print(f"   Extracted {len(df.columns)} columns")
        if hasattr(self.xgb_scaler, 'feature_names_in_'):
            print(f"   Scaler expects: {list(self.xgb_scaler.feature_names_in_)}")
        
        # Use the EXACT feature names from the scaler
        if hasattr(self.xgb_scaler, 'feature_names_in_'):
            expected_features = list(self.xgb_scaler.feature_names_in_)
        else:
            # Fallback to the 32 features the scaler expects
            expected_features = [
                'num_words', 'duration', 'spectral_centroid', 'spectral_bandwidth',
                'zero_crossing_rate', 'rms_energy', 'speech_rate', 'mfcc_1', 'mfcc_2',
                'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 
                'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1', 'chroma_2', 
                'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 
                'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12'
            ]
        
        print(f"📊 EXPECTED {len(expected_features)} XGBoost features")
        
        # Aggregate features across all audio segments
        aggregated_features = {}
        
        for feature in expected_features:
            if feature in df.columns:
                aggregated_features[feature] = float(df[feature].mean())
            else:
                aggregated_features[feature] = 0.0
                print(f"   ⚠️  Using default 0 for missing: {feature}")
        
        # Convert to array in EXACT order expected by scaler
        feature_array = [aggregated_features[feat] for feat in expected_features]
        
        print(f"✅ FINAL: Prepared {len(feature_array)} features for XGBoost")
        print(f"   Sample values: {feature_array[:5]}...")  # Show first 5 values
        
        return np.array(feature_array)
    
    def prepare_video_features_for_random_forest(self, video_features_list: List[Dict]) -> np.ndarray:
        """Prepare video frame features for Random Forest model - USE ALL 22 FEATURES"""
        if not video_features_list:
            return None
        
        df = pd.DataFrame(video_features_list)
        
        print(f"🔍 VIDEO FEATURES DEBUG:")
        print(f"   Extracted {len(df.columns)} columns: {df.columns.tolist()}")
        
        # Define ALL 22 features that you're actually extracting
        expected_video_features = [
            'img_width', 'img_height', 'eye_contact', 'eye_aspect_ratio', 
            'face_touching', 'genuine_smile', 'smile_ratio', 'head_upright', 
            'head_yaw', 'head_pitch', 'head_nod', 'brow_raised', 'brow_distance', 
            'jaw_relaxed', 'jaw_openness', 'arms_uncrossed', 'upright_torso', 
            'open_posture', 'shoulder_width', 'torso_angle', 
            'hand_gestures_expressive', 'hand_movement'
        ]
        
        print(f"📊 EXPECTED {len(expected_video_features)} video features")
        
        # Check for missing features
        available_features = df.columns.tolist()
        missing_features = [f for f in expected_video_features if f not in available_features]
        
        if missing_features:
            print(f"❌ MISSING video features: {missing_features}")
        
        # Aggregate features across frames
        aggregated_features = []
        for feature in expected_video_features:
            if feature in df.columns:
                # Categorical features (0/1 values)
                if feature in ['eye_contact', 'genuine_smile', 'head_upright', 'face_touching',
                            'brow_raised', 'jaw_relaxed', 'arms_uncrossed', 'head_nod',
                            'upright_torso', 'open_posture', 'hand_gestures_expressive']:
                    # Use mode (most common value) for categorical features
                    value = df[feature].mode()[0] if not df[feature].empty else 0
                else:
                    # Continuous features - use mean
                    value = df[feature].mean()
                aggregated_features.append(float(value))
            else:
                # Feature not found, use default value
                aggregated_features.append(0.0)
                print(f"   ⚠️  Using default 0 for missing: {feature}")
        
        print(f"✅ FINAL: Prepared {len(aggregated_features)} video features for Random Forest")
        return np.array(aggregated_features)
    
    def debug_rf_scaler_info(self):
        """Debug Random Forest scaler information"""
        if self.rf_scaler is not None:
            print("🔍 RF SCALER DEBUG INFO:")
            if hasattr(self.rf_scaler, 'n_features_in_'):
                print(f"   RF Scaler expects {self.rf_scaler.n_features_in_} features")
            if hasattr(self.rf_scaler, 'feature_names_in_'):
                print(f"   RF Scaler feature names: {self.rf_scaler.feature_names_in_}")
            else:
                print("   No feature names available in RF scaler")
        else:
            print("❌ No RF scaler loaded")
    
    def predict_audio_confidence(self, audio_features_list: List[Dict]):
        """Predict confidence using XGBoost on audio features"""
        if not self.models_loaded:
            self.load_models()
        
        features = self.prepare_audio_features_for_xgboost(audio_features_list)
        if features is None:
            return {"error": "No audio features extracted"}
        
        print(f"🔍 Final feature vector shape: {features.shape}")
        
        # Preprocess features with detailed error handling
        try:
            features_scaled = self.xgb_scaler.transform([features])
            print(f"✅ Features scaled successfully: {features_scaled.shape}")
        except Exception as e:
            print(f"❌ Scaling failed: {e}")
            # Try to get more specific error info
            if hasattr(self.xgb_scaler, 'n_features_in_'):
                print(f"   Scaler expects: {self.xgb_scaler.n_features_in_} features")
            print(f"   We provided: {features.shape[1]} features")
            return {"error": f"Feature scaling failed: {str(e)}"}
        
        # Rest of your prediction code remains the same...
        # Use actual XGBoost model if available
        if self.xgb_model is not None:
            try:
                import xgboost as xgb
                # Convert to DMatrix for XGBoost
                dmatrix = xgb.DMatrix(features_scaled)
                
                # Make prediction
                prediction_proba = self.xgb_model.predict(dmatrix)
                print(f"✅ Raw prediction: {prediction_proba}")
                
                # Handle prediction output (binary or multi-class)
                if len(prediction_proba.shape) > 1 and prediction_proba.shape[1] > 1:
                    # Multi-class: get highest probability class
                    predicted_class = np.argmax(prediction_proba, axis=1)[0]
                    confidence_score = np.max(prediction_proba)
                else:
                    # Binary classification: use threshold of 0.5
                    predicted_class = 1 if prediction_proba[0] > 0.5 else 0
                    confidence_score = prediction_proba[0] if predicted_class == 1 else 1 - prediction_proba[0]
                
                # Convert encoded prediction to label
                prediction_label = self.xgb_label_encoder.inverse_transform([predicted_class])[0]
                
                return {
                    "prediction": prediction_label,
                    "confidence_score": float(confidence_score),
                    "model": "XGBoost (Audio)",
                    "audio_segments_analyzed": len(audio_features_list),
                    "features_used": features.shape[0]
                }
                
            except Exception as e:
                print(f"❌ XGBoost prediction error: {e}")
                # Fall back to mock response
                return self._get_mock_audio_prediction(audio_features_list)
        else:
            print("⚠️ XGBoost model not available, using mock prediction")
            return self._get_mock_audio_prediction(audio_features_list)
    
    def _get_mock_audio_prediction(self, audio_features_list: List[Dict]):
        """Fallback mock prediction when XGBoost model fails"""
        # Simple heuristic based on audio features
        if audio_features_list:
            avg_confidence = np.mean([f.get('transcription_confidence', 0) for f in audio_features_list])
            speech_rate_avg = np.mean([f.get('speech_rate', 0) for f in audio_features_list])
            mock_confidence = min(1.0, (avg_confidence * 0.7 + (speech_rate_avg / 5) * 0.3))
        else:
            mock_confidence = 0.5
        
        return {
            "prediction": "High_Confidence" if mock_confidence > 0.7 else "Medium_Confidence" if mock_confidence > 0.5 else "Low_Confidence",
            "confidence_score": round(mock_confidence, 3),
            "model": "XGBoost (Audio) - Mock",
            "audio_segments_analyzed": len(audio_features_list),
            "features_used": 0
        }
    
    def predict_video_confidence(self, video_features_list: List[Dict]):
        """Predict confidence using Random Forest on video features"""
        if not self.models_loaded:
            self.load_models()
        
        # Debug RF scaler info
        self.debug_rf_scaler_info()
        
        features = self.prepare_video_features_for_random_forest(video_features_list)
        if features is None:
            return {"error": "No video features extracted"}
        
        print(f"🔍 Final video feature vector shape: {features.shape}")
        
        # Preprocess features with error handling
        try:
            features_scaled = self.rf_scaler.transform([features])
            print(f"✅ Video features scaled successfully: {features_scaled.shape}")
        except Exception as e:
            print(f"❌ Video feature scaling failed: {e}")
            if hasattr(self.rf_scaler, 'n_features_in_'):
                print(f"   RF Scaler expects: {self.rf_scaler.n_features_in_} features")
            print(f"   We provided: {features.shape[0]} features")
            return {"error": f"Video feature scaling failed: {str(e)}"}
        
        # Make actual prediction with Random Forest
        prediction_encoded = self.rf_model.predict(features_scaled)
        prediction = self.rf_label_encoder.inverse_transform(prediction_encoded)
        confidence = np.max(self.rf_model.predict_proba(features_scaled))
        
        return {
            "prediction": prediction[0],
            "confidence_score": float(confidence),
            "model": "Random Forest (Video)",
            "frames_analyzed": len(video_features_list),
            "features_used": features.shape[0]
        }

# Global instance
model_loader = ModelLoader()