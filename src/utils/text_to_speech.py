import logging
import sys
import subprocess
import pyttsx3

class TextToSpeech:
    def __init__(self, lang='en'):
        self.lang = lang
        self.enabled = True
        # For macOS, store a reference to the subprocess
        if sys.platform == "darwin":
            self.engine = None
            self.process = None
        else:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)

    def speak(self, text, speaker='default'):
        if not self.enabled:
            return
        try:
            if sys.platform == "darwin":
                # Terminate any previous TTS process
                if self.process is not None:
                    self.stop()
                # Launch "say" command with Popen to allow termination
                self.process = subprocess.Popen(["say", text])
            else:
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as e:
            logging.error(f"Error in speak(): {e}")

    def stop(self):
        if sys.platform == "darwin":
            if self.process is not None:
                self.process.terminate()
                self.process = None
        else:
            self.engine.stop()
