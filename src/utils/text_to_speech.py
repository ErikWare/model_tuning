import logging
import sys
import subprocess
import pyttsx3

class TextToSpeech:
    def __init__(self, lang='en'):
        self.lang = lang
        self.enabled = True
        # Initialize engine only for non-macOS systems
        if sys.platform == "darwin":
            self.engine = None
        else:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)

    def speak(self, text, speaker='default'):
        if not self.enabled:
            return
        try:
            if sys.platform == "darwin":
                # Use the system's "say" command on macOS
                subprocess.run(["say", text])
            else:
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as e:
            logging.error(f"Error in speak(): {e}")

    def stop(self):
        if sys.platform == "darwin":
            # There's no stop for the "say" command, so do nothing
            pass
        else:
            self.engine.stop()
