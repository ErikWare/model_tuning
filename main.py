import sys
from pathlib import Path
from src.models.model_controller import ModelController
from src.gui.chat_interface import ChatInterface  # Updated import
from src.utils.logging_utils import setup_logging
from src.utils.speech_utils import TextToSpeech  # Add TTS import

# Add project root to Python path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

def main():
    controller = None
    try:
        # Setup logging
        log_dir = ROOT_DIR / "logs"
        logger = setup_logging(log_dir)
        
        # Initialize TTS
        tts_engine = TextToSpeech()
        
        # Initialize controller
        model_dir = ROOT_DIR / "outputs" / "2025-01-15" / "19-56-41" / "outputs" / "math_model"
        controller = ModelController(model_dir, logger)
        
        logger.info("Initializing application...")
        
        def on_closing():
            """Handle window closing"""
            try:
                if controller:
                    controller.cleanup()
                app.root.destroy()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
                
        # Create and verify GUI initialization
        try:
            app = ChatInterface(
                model=controller.model,
                tokenizer=controller.tokenizer,
                device=controller.device,
                generate_fn=controller.generate_text,
                tts_engine=tts_engine
            )
            
            # Set up closing handler
            app.root.protocol("WM_DELETE_WINDOW", on_closing)
            
            logger.info("GUI initialized successfully")
            app.run()
        except Exception as e:
            logger.error(f"GUI initialization failed: {e}")
            raise
            
    except Exception as e:
        print(f"Error launching application: {e}")
        logger.error(f"Application error: {e}", exc_info=True)
        if controller:
            controller.cleanup()
        raise

if __name__ == "__main__":
    main()
