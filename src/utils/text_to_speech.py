import pyttsx3

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def speak(self, text):
        if self.enabled:
            self.engine.say(text)
            self.engine.runAndWait()
