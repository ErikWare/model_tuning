import whisper
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from pathlib import Path

class VoiceToText:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.sample_rate = 16000
        self.channels = 1
        
    def record_audio(self, duration=5):
        """Record audio from microphone for given duration"""
        print("Recording...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        sd.wait()  # Wait until recording is finished
        print("Done recording.")
        return audio_data
    
    def transcribe(self, audio_data=None, duration=5):
        """Convert speech to text. If no audio_data provided, record from mic"""
        if audio_data is None:
            audio_data = self.record_audio(duration)
        
        # Ensure audio is the right shape
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # Transcribe using whisper
        result = self.model.transcribe(audio_data)
        return result["text"].strip()

    def save_audio(self, audio_data, filename="recording.wav"):
        """Utility method to save audio for debugging"""
        wavfile.write(filename, self.sample_rate, audio_data)
