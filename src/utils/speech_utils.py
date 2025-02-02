import logging
import warnings
import numpy as np
import sounddevice as sd
import whisper
from typing import Optional
from scipy.io import wavfile
import pyttsx3
import threading
import queue
import os


class VoiceToText:
    """
    Offline voice-to-text using Whisper in a press-and-hold (push-to-talk) style.
    Safely handles recording in a separate thread to avoid blocking the UI.

    Usage:
      1) Create instance:  vtt = VoiceToText()
      2) On button press:   vtt.start_recording()
      3) On button release: text = vtt.stop_recording()
         -> 'text' is your transcribed speech.
    """

    def __init__(self, model_name: str = "base", max_record_seconds: int = 30):
        """
        :param model_name: Name of the Whisper model to load (e.g., 'base', 'small', etc.).
        :param max_record_seconds: Max recording duration before forced stop, in seconds.
        """
        self.logger = logging.getLogger(__name__)

        # Audio settings
        self.sample_rate: int = 16000
        self.channels: int = 1
        self.max_record_seconds: int = max_record_seconds
        
        # Internal state
        self._recording: bool = False
        self._recorded_audio: Optional[np.ndarray] = None
        
        # Load Whisper model (suppress deprecation warnings).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.logger.info(f"Loading Whisper model: {model_name}")
            self.model = whisper.load_model(model_name)
            self.logger.info("Whisper model loaded successfully.")

    def start_recording(self) -> None:
        """
        Begin recording audio. Intended to be called when a user presses (and holds) a button.
        Begins async capture of audio using sounddevice.
        Ensures no double-recording occurs at once.
        """
        if self._recording:
            self.logger.warning("start_recording() called but we are already recording.")
            return

        self._recording = True
        self._recorded_audio = None

        # Calculate how many frames we can record at most (for safety).
        max_frames = int(self.max_record_seconds * self.sample_rate)

        self.logger.info(f"Starting microphone recording for up to {self.max_record_seconds} s.")
        try:
            # Start recording. We'll let the user release the button to stop.
            self._recorded_audio = sd.rec(
                frames=max_frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            # NOTE: We do NOT call sd.wait() here, we wait until stop_recording().
        except Exception as e:
            self._recording = False
            self.logger.error(f"start_recording() failed: {e}")
            self._recorded_audio = None

    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and transcribe the captured audio.
        Intended to be called when the user releases the button.
        This method is safe to call in a background thread 
        to keep the UI responsive and resources free.
        :return: Transcribed text or None if any error occurs.
        """
        if not self._recording:
            self.logger.warning("stop_recording() called but was not recording.")
            return None

        self.logger.info("Stopping microphone recording.")
        self._recording = False
        
        try:
            # Stop the recording immediately.
            sd.stop()
            # Now we have audio data in self._recorded_audio up to this point.
            audio_data = np.squeeze(self._recorded_audio) if self._recorded_audio is not None else None

            if audio_data is None or len(audio_data) == 0:
                self.logger.warning("No audio data captured.")
                return None

            # Transcribe using Whisper
            text = self.transcribe(audio_data)
            return text
        except Exception as e:
            self.logger.error(f"stop_recording() error: {e}")
            return None
        finally:
            self._recorded_audio = None  # Clear buffer

    def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe a NumPy array of audio data into text using Whisper.
        :param audio_data: float32 NumPy array from sounddevice.
        :return: Transcribed text or None if an error occurs.
        """
        # Validate audio data
        if not isinstance(audio_data, np.ndarray):
            self.logger.error("Audio data is not a NumPy array.")
            return None

        if audio_data.dtype != np.float32:
            self.logger.error(f"Audio data dtype is {audio_data.dtype}, expected float32.")
            return None

        if np.isnan(audio_data).any():
            self.logger.error("Audio data contains NaN values.")
            return None

        if np.isinf(audio_data).any():
            self.logger.error("Audio data contains Inf values.")
            return None

        if audio_data.shape[0] == 0:
            self.logger.error("Audio data is empty.")
            return None

        # Optionally, normalize audio data
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        try:
            self.logger.info("Transcribing audio data via Whisper...")
            result = self.model.transcribe(audio_data, fp16=False)
            text = result.get("text", "").strip()
            # Remove text prior to '[thinking]' to limit logic or reasoning
            text = self.remove_pre_thinking(text)
            if not text:
                self.logger.warning("Whisper returned empty transcription.")
            else:
                self.logger.info(f"Transcribed text: {text}")
            return text
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return None

    def remove_pre_thinking(self, text: str) -> str:
        """
        Removes text prior to any recognized 'thinking' tags, making it more flexible.
        """
        thinking_tags = ["[thinking]", "[internal]", "[chain-of-thought]", "(consideration)"]
        earliest_index = -1
        chosen_tag = None

        for tag in thinking_tags:
            idx = text.find(tag)
            if idx != -1 and (earliest_index == -1 or idx < earliest_index):
                earliest_index = idx
                chosen_tag = tag

        if earliest_index != -1 and chosen_tag is not None:
            return text[earliest_index + len(chosen_tag):].strip()
        return text

    def save_audio(self, audio_data: np.ndarray, filename: str = "recording.wav") -> None:
        """
        (Optional) Save audio data to a .wav file for debugging or logging.
        :param audio_data: float32 NumPy array from sounddevice.
        :param filename: Filename to save.
        """
        try:
            wavfile.write(filename, self.sample_rate, audio_data)
            self.logger.info(f"Audio saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving audio: {e}")



class TextToSpeech:
    def __init__(self, lang='en'):
        # Initialize pyttsx3 TTS engine for offline usage
        self.engine = pyttsx3.init()
        self.enabled = True

    def speak(self, text):
        if self.enabled:
            self.engine.say(text)
            self.engine.runAndWait()

    def stop(self):
        # pyttsx3 does not support true interruption; we simulate stop by ending any queued commands.
        self.engine.stop()
