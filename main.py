import sys
from pathlib import Path
import torch
import os
from src.models.model_controller import ModelController
from src.gui.chat_interface import ChatInterface
from src.utils.logging_utils import setup_logging

ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Initialize centralized logger
logger = setup_logging(ROOT_DIR / "logs")

def setup_environment():
    """Configure environment for M1 Mac."""
    logger.info("Configuring environment...")
    
    # Configure PyTorch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    # Set default dtype for better memory handling
    torch.set_default_dtype(torch.float32)
    
    return device

def main():
    try:
        # Setup environment
        device = setup_environment()
        
        # Initialize model controller
        model_path = ROOT_DIR / "models" / "deepseek"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        controller = ModelController(
            model_path=model_path,
            device=device
        )
        
        # Initialize GUI
        logger.info("Initializing chat interface")
        app = ChatInterface(
            model=controller.model,
            tokenizer=controller.tokenizer,
            device=device,
            generate_fn=controller.generate
        )
        
        logger.info("Starting application")
        app.run()

    except Exception as e:
        logger.critical("Fatal application error", exc_info=True)
        raise
    finally:
        logger.info("Application shutting down")

if __name__ == "__main__":
    main()
