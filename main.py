from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from audio_analyzer import AudioAnalyzer
import traceback
import librosa

app = FastAPI(title="AI Music Tutor Backend")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
analyzer = AudioAnalyzer()

class ComparisonResponse(BaseModel):
    test_analysis: dict
    reference_analysis: dict
    comparison: dict
    success: bool

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Music Tutor"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze uploaded audio file and return pitch, spectrogram, and rhythm metrics"""
    try:
        # Read audio file
        audio_bytes = await file.read()
        audio, sr = analyzer.load_audio(audio_bytes)
        
        # Perform analysis
        analysis = analyzer.analyze_audio(audio)
        
        return JSONResponse({
            "success": True,
            "analysis": analysis,
            "sample_rate": sr
        })
    except Exception as e:
        print(f"Error analyzing audio: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=400)

@app.post("/compare")
async def compare_audio(
    test_file: UploadFile = File(..., description="User's recorded audio"),
    reference_file: UploadFile = File(..., description="Reference audio to compare against")
):
    """Compare test audio with reference audio and return feedback"""
    try:
        # Read both files
        test_bytes = await test_file.read()
        ref_bytes = await reference_file.read()
        
        # Load audio
        test_audio, test_sr = analyzer.load_audio(test_bytes)
        ref_audio, ref_sr = analyzer.load_audio(ref_bytes)
        
        # Analyze both
        test_analysis = analyzer.analyze_audio(test_audio)
        ref_analysis = analyzer.analyze_audio(ref_audio)
        
        # Compare pitch
        pitch_comparison = analyzer.calculate_pitch_accuracy(ref_audio, test_audio)
        
        # Compare rhythm
        test_rhythm = test_analysis["rhythm"]
        ref_rhythm = ref_analysis["rhythm"]
        
        # Calculate rhythm accuracy (based on tempo and regularity)
        ref_tempo = ref_rhythm.get("tempo_estimate", 0)
        test_tempo = test_rhythm.get("tempo_estimate", 0)
        
        if ref_tempo > 0:
            tempo_accuracy = min(1.0, test_tempo / ref_tempo) * 100
        else:
            tempo_accuracy = 0
        
        rhythm_accuracy = (test_rhythm.get("onset_regularity", 0) * 100)
        
        # Compile comparison
        comparison = {
            "pitch_feedback": {
                "accuracy": round(pitch_comparison["overall_accuracy"], 2),
                "mean_error_cents": round(pitch_comparison["mean_pitch_error_cents"], 2),
                "range_match": round(pitch_comparison["pitch_range_match"], 2)
            },
            "rhythm_feedback": {
                "tempo_accuracy": round(tempo_accuracy, 2),
                "regularity": round(rhythm_accuracy, 2),
                "reference_tempo": round(ref_tempo, 2),
                "your_tempo": round(test_tempo, 2)
            },
            "overall_score": round(
                (pitch_comparison["overall_accuracy"] + rhythm_accuracy) / 2, 2
            )
        }
        
        return JSONResponse({
            "success": True,
            "test_analysis": test_analysis,
            "reference_analysis": ref_analysis,
            "comparison": comparison
        })
    
    except Exception as e:
        print(f"Error comparing audio: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=400)

@app.post("/analyze-simple")
async def analyze_simple(file: UploadFile = File(...)):
    """Simple analysis endpoint that returns minimal data for faster response"""
    try:
        audio_bytes = await file.read()
        audio, sr = analyzer.load_audio(audio_bytes)
        
        # Quick analysis
        f0, times = analyzer.extract_pitch(audio)
        S_db, freqs, spec_times = analyzer.extract_spectrogram(audio)
        
        # Only return essential data
        return JSONResponse({
            "success": True,
            "pitch": f0[:500].tolist() if len(f0) > 0 else [],  # Limit data
            "times": times[:500].tolist() if len(times) > 0 else [],
            "spectrogram": S_db[:, :500].tolist() if S_db.shape[1] > 0 else [],
            "frequencies": freqs.tolist(),
            "duration": float(librosa.get_duration(y=audio, sr=sr))
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
