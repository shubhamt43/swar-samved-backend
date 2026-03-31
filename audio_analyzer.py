import librosa
import numpy as np
from scipy import signal
from typing import Dict, Tuple, List
import io

class AudioAnalyzer:
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = 80  # Minimum frequency for vocal analysis (Hz)
        self.fmax = 400  # Maximum frequency for vocal analysis (Hz)

    def load_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load audio from bytes and return (audio_data, sample_rate)"""
        try:
            # Create BytesIO object with proper format detection
            audio_file = io.BytesIO(audio_bytes)
            
            # Try to load with format inference
            try:
                audio, sr = librosa.load(audio_file, sr=self.sr, mono=True)
            except Exception:
                # Reset file pointer and try with explicit format detection
                audio_file.seek(0)
                import soundfile as sf
                audio_data, sr_original = sf.read(audio_file)
                
                # Ensure mono
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample if needed
                if sr_original != self.sr:
                    audio = librosa.resample(audio_data, orig_sr=sr_original, target_sr=self.sr)
                else:
                    audio = audio_data
                sr = self.sr
            
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio: {str(e)}")

    def extract_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch contour using PYIN algorithm
        Returns: (pitch_values, time_frames)
        """
        # Use PYIN for robust pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Convert NaN values to 0
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Time frames
        times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sr, hop_length=self.hop_length)
        
        return f0, times

    def extract_spectrogram(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract STFT spectrogram
        Returns: (spectrogram, frequencies, times)
        """
        D = librosa.stft(audio, hop_length=self.hop_length)
        S_db = librosa.power_to_db(np.abs(D) ** 2, ref=np.max)
        
        freqs = librosa.fft_frequencies(sr=self.sr)
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=self.sr, hop_length=self.hop_length)
        
        return S_db, freqs, times

    def extract_onset_frames(self, audio: np.ndarray) -> np.ndarray:
        """Detect note onset frames"""
        # 1. Pehle onset strength calculate karein
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        
        # 2. Phir onset_detect ko 'onset_envelope' explicitly pass karein
        # Purane code mein shayad aapne sirf onset_env likha tha bina keyword ke
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=self.sr, 
            hop_length=self.hop_length
        )
        return onset_frames

    def calculate_rhythm_metrics(self, audio: np.ndarray) -> Dict:
        # Check if audio is empty or all zeros
        if len(audio) == 0 or np.max(np.abs(audio)) < 1e-4:
            return {
                "onset_times": [],
                "onset_count": 0,
                "tempo_estimate": 0,
                "onset_regularity": 0
            }
        
        onset_frames = self.extract_onset_frames(audio)
        # ... rest of your code
        
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        
        # Estimate tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        
        # Calculate regularity (consistency of timing)
        if len(onset_times) > 1:
            intervals = np.diff(onset_times)
            regularity = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-6))
            regularity = max(0, min(1, regularity))  # Clamp to [0, 1]
        else:
            regularity = 0
        
        return {
            "onset_times": onset_times.tolist(),
            "onset_count": int(len(onset_frames)),
            "tempo_estimate": float(tempo),
            "onset_regularity": float(regularity)
        }

    def calculate_pitch_accuracy(self, reference_audio: np.ndarray, test_audio: np.ndarray) -> Dict:
        """
        Compare test audio pitch with reference audio pitch
        Returns accuracy metrics
        """
        ref_f0, ref_times = self.extract_pitch(reference_audio)
        test_f0, test_times = self.extract_pitch(test_audio)
        
        # Filter out unvoiced parts (f0 = 0)
        ref_voiced = ref_f0[ref_f0 > 0]
        test_voiced = test_f0[test_f0 > 0]
        
        if len(ref_voiced) == 0 or len(test_voiced) == 0:
            return {
                "overall_accuracy": 0,
                "mean_pitch_error_cents": 0,
                "pitch_range_match": 0
            }
        
        # Calculate mean pitch error in cents (100 cents = 1 semitone)
        ref_mean_pitch = np.mean(ref_voiced)
        test_mean_pitch = np.mean(test_voiced)
        
        if ref_mean_pitch > 0:
            cents_error = 1200 * np.log2(test_mean_pitch / ref_mean_pitch)
        else:
            cents_error = 0
        
        # Pitch accuracy (penalize large deviations)
        max_error = 100  # 1 semitone tolerance
        pitch_accuracy = max(0, 1.0 - abs(cents_error) / max_error)
        
        # Range match (check if both cover similar pitch ranges)
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
        """Complete audio analysis returning all features"""
        f0, times = self.extract_pitch(audio)
        S_db, freqs, spec_times = self.extract_spectrogram(audio)
        rhythm = self.calculate_rhythm_metrics(audio)
        
        # Calculate overall loudness
        S = np.abs(librosa.stft(audio))
        loudness_val = np.mean(librosa.power_to_db(S**2))
        loudness = float(loudness_val) # Ensure it's a single value        
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
