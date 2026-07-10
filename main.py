import os
# Yeh 2 lines librosa ke load hone se pehle RAM spike ko rokengi
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp"

from fastapi import FastAPI, UploadFile, File, HTTPException
import gc # Garbage collection ke liye
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
    allow_origins=["https://swar-samved.vercel.app"],
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
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Music Tutor"}

# ⚠️ Yahan se 'async' hata diya gaya hai
@app.post("/analyze")
def analyze_audio(file: UploadFile = File(...)):
    """Analyze uploaded audio file and return pitch, spectrogram, and rhythm metrics"""
    try:
        # ⚠️ 'await' hatakar '.file.read()' use kiya hai
        audio_bytes = file.file.read()
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

# ⚠️ Yahan se bhi 'async' hata diya gaya hai thread blocking rokne ke liye
@app.post("/compare")
def compare_audio(
    test_file: UploadFile = File(..., description="User's recorded audio"),
    reference_file: UploadFile = File(..., description="Reference audio to compare against")
):
    try:
        # ⚠️ '.file.read()' use kiya hai
        test_bytes = test_file.file.read()
        ref_bytes = reference_file.file.read()
        
        # Dono audios ek saath load karein
        test_audio, test_sr = analyzer.load_audio(test_bytes)
        ref_audio, ref_sr = analyzer.load_audio(ref_bytes)
        
        # Analyze both
        test_analysis = analyzer.analyze_audio(test_audio)
        ref_analysis = analyzer.analyze_audio(ref_audio)
        
        # Compare pitch (Ab variable delete nahi hua hai, toh yeh safely chalega)
        pitch_comparison = analyzer.calculate_pitch_accuracy(ref_audio, test_audio)
        
        # Compare rhythm
        test_rhythm = test_analysis["rhythm"]
        ref_rhythm = ref_analysis["rhythm"]
        
        # Calculate rhythm accuracy
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
        
        # 🧹 Memory cleanup sabse last mein kiya hai taaki koi error na aaye
        del test_audio, ref_audio, test_bytes, ref_bytes
        gc.collect() 
        
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

# ⚠️ Yahan se bhi 'async' hata diya gaya hai
@app.post("/analyze-simple")
def analyze_simple(file: UploadFile = File(...)):
    """Simple analysis endpoint that returns minimal data for faster response"""
    try:
        audio_bytes = file.file.read()
        audio, sr = analyzer.load_audio(audio_bytes)
        
        # Quick analysis
        f0, times = analyzer.extract_pitch(audio)
        S_db, freqs, spec_times = analyzer.extract_spectrogram(audio)
        
        return JSONResponse({
            "success": True,
            "pitch": f0[:500].tolist() if len(f0) > 0 else [],  
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
