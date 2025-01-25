import sys
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add project root and src to Python path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from src.models import generate_text
from src.gui.model_test_window import ModelTestWindow  # Updated import path
from src.utils.text_to_speech import TextToSpeech

def main():
    # Load model and tokenizer directly from huggingface hub
    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2')  # Using default GPT-2 model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Initialize TTS engine
        tts_engine = TextToSpeech()
        
        # Create and run GUI with TTS
        app = ModelTestWindow(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_fn=generate_text,
            tts_engine=tts_engine
        )
        app.run()
        
    except Exception as e:
        print(f"Error launching application: {e}")
        raise

if __name__ == "__main__":
    main()
