import logging
import numpy as np
import sounddevice as sd
from typing import Optional
from scipy.io import wavfile
import os
import tempfile
import speech_recognition as sr  # New import for offline transcription using PocketSphinx

# ...existing code (if any) ...

class VoiceToText:
    def __init__(self, max_record_seconds: int = 30):
        self.logger = logging.getLogger(__name__)
        self.sample_rate: int = 16000
        self.channels: int = 1
        self.max_record_seconds: int = max_record_seconds
        self._recording: bool = False
        self._recorded_audio: Optional[np.ndarray] = None
        # Initialize SpeechRecognition recognizer
        self.recognizer = sr.Recognizer()

    def start_recording(self) -> None:
        if self._recording:
            self.logger.warning("start_recording() called but we are already recording.")
            return
        self._recording = True
        self._recorded_audio = None
        max_frames = int(self.max_record_seconds * self.sample_rate)
        self.logger.info(f"Starting microphone recording for up to {self.max_record_seconds} s.")
        try:
            self._recorded_audio = sd.rec(
                frames=max_frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16'  # change dtype for compatibility with SpeechRecognition
            )
        except Exception as e:
            self._recording = False
            self.logger.error(f"start_recording() failed: {e}")
            self._recorded_audio = None

    def stop_recording(self) -> Optional[str]:
        if not self._recording:
            self.logger.warning("stop_recording() called but was not recording.")
            return None
        self.logger.info("Stopping microphone recording.")
        self._recording = False
        try:
            sd.stop()
            audio_data = self._recorded_audio
            if audio_data is None or len(audio_data) == 0:
                self.logger.warning("No audio data captured.")
                return None
            text = self.transcribe(audio_data)
            return text
        except Exception as e:
            self.logger.error(f"stop_recording() error: {e}")
            return None
        finally:
            self._recorded_audio = None

    def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        # Save audio_data to a temporary WAV file
        temp_wav = tempfile.mktemp(suffix=".wav")
        try:
            wavfile.write(temp_wav, self.sample_rate, audio_data)
        except Exception as e:
            self.logger.error(f"Error saving temporary audio file: {e}")
            return None

        try:
            with sr.AudioFile(temp_wav) as source:
                audio = self.recognizer.record(source)
            self.logger.info("Transcribing audio via PocketSphinx...")
            text = self.recognizer.recognize_sphinx(audio)
            if text:
                self.logger.info(f"Transcribed text: {text}")
            else:
                self.logger.warning("PocketSphinx returned empty transcription.")
            return text
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return None
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def save_audio(self, audio_data: np.ndarray, filename: str = "recording.wav") -> None:
        try:
            wavfile.write(filename, self.sample_rate, audio_data)
            self.logger.info(f"Audio saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving audio: {e}")
