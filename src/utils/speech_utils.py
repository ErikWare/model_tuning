import os
import time
import logging
import warnings
import tempfile
import numpy as np
import sounddevice as sd
import whisper
import pyttsx3

from typing import Optional
from pathlib import Path
from scipy.io import wavfile

class VoiceToText:
    """
    Handles recording from microphone and transcribing via Whisper, all offline.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.sample_rate: int = 16000
        self.channels: int = 1
        self.recording: bool = False
        self.timeout: int = 30  # Max recording time in seconds

        # Suppress future warnings from Whisper
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # Load a small or base Whisper model for offline STT
            self.model = whisper.load_model("base")

    def record_audio(self, duration: int = 5) -> np.ndarray:
        """
        Record audio from the microphone for a given duration (in seconds).
        Returns a NumPy array containing the audio samples.
        """
        try:
            self.logger.info("Recording audio...")
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            sd.wait()  # Block until recording is finished
            self.logger.info("Recording complete.")
            # Squeeze out extra dimensions (e.g., shape [x,1] -> [x])
            return np.squeeze(audio)
        except Exception as e:
            self.logger.error(f"Error recording audio: {e}")
            raise

    def transcribe(self, audio_data: Optional[np.ndarray] = None, duration: int = 5) -> Optional[str]:
        """
        Convert speech to text using Whisper. 
        If audio_data is None, a fresh recording is made.
        """
        try:
            if audio_data is None:
                audio_data = self.record_audio(duration)
            # Transcribe the raw NumPy array
            result = self.model.transcribe(audio_data)
            text = result["text"].strip()
            self.logger.info(f"Transcribed text: {text}")
            return text
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return None

    def save_audio(self, audio_data: np.ndarray, filename: str = "recording.wav") -> None:
        """
        Save audio data to a WAV file (for debugging or logging).
        """
        try:
            wavfile.write(filename, self.sample_rate, audio_data)
            self.logger.info(f"Audio saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving audio: {e}")
            raise

    def listen(self, duration: int = 5) -> Optional[str]:
        """
        Continuously record and transcribe audio until a valid transcription is found,
        or until timeout is reached.
        """
        self.recording = True
        start_time = time.time()
        try:
            while time.time() - start_time < self.timeout:
                audio_data = self.record_audio(duration)
                text = self.transcribe(audio_data)
                if text and text.strip():
                    return text

            raise TimeoutError("Recording timeout reached.")
        except Exception as e:
            self.logger.error(f"Error during listen: {e}")
            return None
        finally:
            self.recording = False


class TextToSpeech:
    """
    Offline text-to-speech using pyttsx3.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self.logger = logging.getLogger(__name__)

        # Initialize pyttsx3 once at startup (offline TTS engine)
        try:
            self.engine = pyttsx3.init()
            self.logger.info("TTS engine initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize offline TTS engine: {e}")
            self.engine = None

    def speak(self, text: str) -> bool:
        """
        Convert text to speech (using pyttsx3) and play it offline.
        Returns True if successful, else False.
        """
        if not self.enabled or not text:
            self.logger.debug("TTS is disabled or text is empty.")
            return False

        if not self.engine:
            self.logger.error("No valid TTS engine initialized.")
            return False

        try:
            self.engine.say(text)
            self.engine.runAndWait()
            self.logger.info("Speech synthesis completed successfully.")
            return True
        except Exception as e:
            self.logger.error(f"TTS Error: {e}")
            return False

    def toggle(self) -> bool:
        """
        Toggle TTS on/off (used by the GUI).
        Returns the new state (True = enabled, False = disabled).
        """
        self.enabled = not self.enabled
        self.logger.info(f"Text-to-speech {'enabled' if self.enabled else 'disabled'}")
        return self.enabled