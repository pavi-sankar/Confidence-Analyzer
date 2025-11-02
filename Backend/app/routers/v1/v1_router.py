from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from app.models.model_loader import model_loader
from app.services.feature_extractor import feature_extractor
from app.services.audio_extractor import audio_extractor
import asyncio
from datetime import datetime

frontend_router = APIRouter(
    prefix="/confidence-analyzer",
    tags=["Analyzing Confidence Score"]
)

@frontend_router.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    try:
        model_loader.load_models()
        print("✅ All models and extractors ready!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load models on startup: {e}")

@frontend_router.post("/confidence-score")
async def analyze_confidence_score(video: UploadFile = File(...)):
    """
    Main endpoint that analyzes both video and audio for confidence scoring
    """
    try:
        # Read video bytes
        video_bytes = await video.read()
        
        print(f"✅ Video received: {video.filename} ({len(video_bytes)} bytes)")
        
        # Run both analyses
        print("🔄 Starting video and audio analysis...")
        
        video_features, audio_features = await asyncio.gather(
            asyncio.to_thread(feature_extractor.extract_features_from_video_bytes, video_bytes),
            asyncio.to_thread(audio_extractor.extract_features_from_video_bytes, video_bytes),
            return_exceptions=True
        )
        
        # Handle exceptions properly
        video_result = None
        audio_result = None
        
        # Process video analysis
        if isinstance(video_features, Exception):
            print(f"❌ Video analysis failed: {video_features}")
            video_result = {"error": f"Video analysis failed: {str(video_features)}"}
        else:
            video_result = model_loader.predict_video_confidence(video_features)
        
        # Process audio analysis  
        if isinstance(audio_features, Exception):
            print(f"❌ Audio analysis failed: {audio_features}")
            audio_result = {"error": f"Audio analysis failed: {str(audio_features)}"}
        else:
            audio_result = model_loader.predict_audio_confidence(audio_features)
        
        print(f"✅ Video: {len(video_features) if not isinstance(video_features, Exception) else 0} frames analyzed")
        print(f"✅ Audio: {len(audio_features) if not isinstance(audio_features, Exception) else 0} segments analyzed")
        
        # Calculate combined score only if both succeeded
        if "error" not in video_result and "error" not in audio_result:
            video_weight = 0.4
            audio_weight = 0.6
            
            video_score = video_result.get("confidence_score", 0)
            audio_score = audio_result.get("confidence_score", 0)
            
            combined_score = (video_score * video_weight) + (audio_score * audio_weight)
            final_level = get_confidence_level(combined_score)
        else:
            # Use video score only if audio failed
            if "error" not in video_result:
                combined_score = video_result.get("confidence_score", 0)
                final_level = get_confidence_level(combined_score)
            else:
                combined_score = 0
                final_level = "Analysis Failed"
        
        # Prepare final result
        result = {
            "video_info": {
                "filename": video.filename,
                "size_bytes": len(video_bytes),
                "received_at": datetime.now().isoformat()
            },
            "analysis_summary": {
                "video_frames_analyzed": len(video_features) if not isinstance(video_features, Exception) else 0,
                "audio_segments_analyzed": len(audio_features) if not isinstance(audio_features, Exception) else 0
            },
            "predictions": {
                "video_analysis": video_result,
                "audio_analysis": audio_result
            },
            "combined_confidence_score": round(combined_score, 3),
            "final_confidence_level": final_level
        }
        
        return JSONResponse({
            "status": "success", 
            "result": result
        })

    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()  # This will show the full stack trace
        raise HTTPException(status_code=500, detail=str(e))

@frontend_router.get("/analyze-video-only")
async def analyze_video_only(video: UploadFile = File(...)):
    """Analyze only video frames using Random Forest"""
    try:
        video_bytes = await video.read()
        video_features = feature_extractor.extract_features_from_video_bytes(video_bytes)
        result = model_loader.predict_video_confidence(video_features)
        
        return {
            "status": "success",
            "analysis_type": "video_only",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@frontend_router.get("/analyze-audio-only")
async def analyze_audio_only(video: UploadFile = File(...)):
    """Analyze only audio using XGBoost"""
    try:
        video_bytes = await video.read()
        audio_features = audio_extractor.extract_features_from_video_bytes(video_bytes)
        result = model_loader.predict_audio_confidence(audio_features)
        
        return {
            "status": "success",
            "analysis_type": "audio_only",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_confidence_level(score):
    """Convert numerical score to confidence level"""
    if score >= 0.8:
        return "High Confidence"
    elif score >= 0.6:
        return "Medium Confidence"
    else:
        return "Low Confidence"