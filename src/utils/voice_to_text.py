import speech_recognition as sr
import logging
from typing import Optional

class VoiceToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.logger = logging.getLogger(__name__)
        
    def listen(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for voice input and convert to text.
        
        Args:
            timeout: Number of seconds to listen for
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            with sr.Microphone() as source:
                self.logger.info("Listening for voice input...")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=timeout)
                
            try:
                # Use Google's speech recognition
                text = self.recognizer.recognize_google(audio)
                self.logger.info(f"Recognized text: {text}")
                return text
            except sr.UnknownValueError:
                self.logger.warning("Could not understand audio")
                return None
            except sr.RequestError as e:
                self.logger.error(f"Could not request results; {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing audio: {e}")
            return None