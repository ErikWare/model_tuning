import sys
from pathlib import Path
import torch
import os
import json  # New import

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.model_controller import ModelController
from src.user_interface import ChatInterface
from src.utils.logging_utils import setup_logging

# Initialize centralized logger
logger = setup_logging(PROJECT_ROOT / "logs")

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

def load_model_list():
    models_json = PROJECT_ROOT / "models" / "light_weight_models.json"
    if models_json.exists():
        with open(models_json, "r") as f:
            # Expecting a dict in the form { "ModelName": "relative/path/to/model", ... }
            return json.load(f)
    else:
        logger.warning("Model list JSON not found. Using default model.")
        return {}

def main():
    try:
        # Setup environment
        device = setup_environment()
        models_list = load_model_list()
        # Select default model (if available, first entry; else default to 'deepseek')
        if models_list:
            default_model_name, model_rel_path = next(iter(models_list.items()))
            model_path = PROJECT_ROOT / model_rel_path
        else:
            logger.error("No models found in the model list. Please download models first.")

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
            generate_fn=controller.generate,
            models_list=models_list,  # Pass the models mapping to the UI
            controller_class=ModelController,  # Pass controller for later reloads
            default_model_path=model_path
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
