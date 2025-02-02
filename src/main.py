import sys
from pathlib import Path
import torch
import os
import json  # New import
import yaml

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

def load_model_config():
    config_path = PROJECT_ROOT / "models" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Filter out models with blank repo or snapshot
    available_models = {
        model["name"]: model["full_path"]
        for model in config["models"].values()
        if model.get("repo") and model.get("snapshot")
    }
    return available_models

def main():
    try:
        # Setup environment
        device = setup_environment()
        # Load models from config.yaml instead of JSON
        models_list = load_model_config()
        # Use "gpt-neo-1.3B" as default model if available, else default to first available model.
        if "gpt-neo-1.3B" in models_list:
            default_model_name = "gpt-neo-1.3B"
        else:
            default_model_name = next(iter(models_list)) if models_list else None
        if not default_model_name:
            logger.error("No valid models found in config.yaml.")
            return
        model_rel_path = models_list[default_model_name]
        # Prepend "models" folder; remove leading "./" if present.
        if model_rel_path.startswith("./"):
            model_rel_path = model_rel_path[2:]
        model_path = PROJECT_ROOT / "models" / model_rel_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        controller = ModelController(
            model_path=model_path,
            device=device
        )
        
        # Initialize GUI with models mapping from YAML
        logger.info("Initializing chat interface")
        app = ChatInterface(
            model=controller.model,
            tokenizer=controller.tokenizer,
            device=device,
            generate_fn=controller.generate,
            models_list=models_list,  # Pass the models mapping from config.yaml
            controller_class=ModelController,
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
