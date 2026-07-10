import librosa
import numpy as np
from scipy import signal
from typing import Dict, Tuple, List
import io

class AudioAnalyzer:
    # 1. Yahan default 8000 set kar diya taaki pooray code mein consistency rahe
    # hop_length 512 se badha kar 2048 kar diya hai
    def __init__(self, sr: int = 8000, hop_length: int = 2048):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = 80  
        self.fmax = 400

    def load_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load audio from bytes and return (audio_data, sample_rate)"""
        try:
            audio_file = io.BytesIO(audio_bytes)
            
            try:
                # 2. Yahan duration=15.0 aur sr=self.sr add kiya (Sabse zaroori step RAM bachane ke liye!)
                audio, sr = librosa.load(audio_file, sr=self.sr, mono=True, duration=15.0)
            except Exception:
                audio_file.seek(0)
                import soundfile as sf
                audio_data, sr_original = sf.read(audio_file)
                
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # 3. Soundfile fallback ke liye bhi 15 second ki limit
                max_samples = int(15.0 * sr_original)
                if len(audio_data) > max_samples:
                    audio_data = audio_data[:max_samples]
                
                if sr_original != self.sr:
                    audio = librosa.resample(audio_data, orig_sr=sr_original, target_sr=self.sr)
                else:
                    audio = audio_data
                sr = self.sr
            
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio: {str(e)}")

    # def extract_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     # Har jagah 8000 hata kar self.sr kar diya
    #     f0, voiced_flag, voiced_probs = librosa.pyin(
    #         audio,
    #         fmin=self.fmin,
    #         fmax=self.fmax,
    #         sr=self.sr,
    #         hop_length=self.hop_length
    #     )
    #     f0 = np.nan_to_num(f0, nan=0.0)
    #     times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sr, hop_length=self.hop_length)
    #     return f0, times
    def extract_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch contour using YIN algorithm (Faster for Free Cloud limits)
        Returns: (pitch_values, time_frames)
        """
        # PYIN ko hata kar sirf YIN use kar rahe hain
        # Dhyan dein: YIN sirf f0 return karta hai, voiced_flag nahi
        f0 = librosa.yin(
            y=audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Convert NaN values to 0 (agar koi aaye toh)
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Time frames
        times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sr, hop_length=self.hop_length)
        
        return f0, times

    def extract_spectrogram(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        D = librosa.stft(audio, hop_length=self.hop_length)
        S_db = librosa.power_to_db(np.abs(D) ** 2, ref=np.max)
        freqs = librosa.fft_frequencies(sr=self.sr)
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=self.sr, hop_length=self.hop_length)
        return S_db, freqs, times

    def extract_onset_frames(self, audio: np.ndarray) -> np.ndarray:
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=self.sr, 
            hop_length=self.hop_length
        )
        return onset_frames

    def calculate_rhythm_metrics(self, audio: np.ndarray) -> Dict:
        if len(audio) == 0 or np.max(np.abs(audio)) < 1e-4:
            return {
                "onset_times": [],
                "onset_count": 0,
                "tempo_estimate": 0,
                "onset_regularity": 0
            }
        
        onset_frames = self.extract_onset_frames(audio)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        
        # Calculate regularity (consistency of timing)
        if len(onset_times) > 1:
            intervals = np.diff(onset_times)
            # ⚠️ CHANGE: Harsh subtraction ki jagah Exponential Decay (Soft scaling) use karein
            cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
            regularity = float(np.exp(-cv)) # Isse score kabhi direct 0 nahi hoga
        else:
            regularity = 0.0
            
        return {
            "onset_times": onset_times.tolist(),
            "onset_count": int(len(onset_frames)),
            "tempo_estimate": float(tempo),
            "onset_regularity": regularity
        }
    # calculate_pitch_accuracy bilkul theek hai, usme change ki zaroorat nahi
    def calculate_pitch_accuracy(self, reference_audio: np.ndarray, test_audio: np.ndarray) -> Dict:
        ref_f0, ref_times = self.extract_pitch(reference_audio)
        test_f0, test_times = self.extract_pitch(test_audio)
        
        ref_voiced = ref_f0[ref_f0 > 0]
        test_voiced = test_f0[test_f0 > 0]
        
        if len(ref_voiced) == 0 or len(test_voiced) == 0:
            return {
                "overall_accuracy": 0,
                "mean_pitch_error_cents": 0,
                "pitch_range_match": 0
            }
        
        ref_mean_pitch = np.mean(ref_voiced)
        test_mean_pitch = np.mean(test_voiced)
        
        if ref_mean_pitch > 0:
            cents_error = 1200 * np.log2(test_mean_pitch / ref_mean_pitch)
        else:
            cents_error = 0
        
        # ⚠️ CHANGE: Tolerance ko 100 (1 semitone) se badha kar 600 (half-octave) kar dein
        max_error = 600  
        pitch_accuracy = max(0, 1.0 - abs(cents_error) / max_error)
        
        ref_range = np.max(ref_voiced) - np.min(ref_voiced)
        test_range = np.max(test_voiced) - np.min(test_voiced)
        
        if ref_range > 0:
            range_match = min(test_range, ref_range) / max(test_range, ref_range)
        else:
            range_match = 0
        
        return {
            "overall_accuracy": float(pitch_accuracy * 100),
            "mean_pitch_error_cents": float(cents_error),
            "pitch_range_match": float(range_match * 100)
        }

    def analyze_audio(self, audio: np.ndarray) -> Dict:
        f0, times = self.extract_pitch(audio)
        S_db, freqs, spec_times = self.extract_spectrogram(audio)
        rhythm = self.calculate_rhythm_metrics(audio)
        
        S = np.abs(librosa.stft(audio))
        loudness_val = np.mean(librosa.power_to_db(S**2))
        loudness = float(loudness_val) 
        return {
            "pitch": f0.tolist(),
            "times": times.tolist(),
            "spectrogram": S_db.tolist(),
            "frequencies": freqs.tolist(),
            "spec_times": spec_times.tolist(),
            "rhythm": rhythm,
            "loudness": loudness,
            "duration": float(librosa.get_duration(y=audio, sr=self.sr))
        }
