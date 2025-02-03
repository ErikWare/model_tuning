import logging
import pyttsx3

# ...existing code (if any) ...

class TextToSpeech:
    def __init__(self, lang='en'):
        self.lang = lang
        self.enabled = True
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)

    def speak(self, text, speaker='default'):
        if not self.enabled:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logging.error(f"Error in speak(): {e}")

    def stop(self):
        self.engine.stop()
