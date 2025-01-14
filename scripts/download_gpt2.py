import os
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.utils.download_utils import ensure_directory, setup_logging

# Get project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
logger = setup_logging()

def download_gpt2_model(model_name: str = "gpt2", output_dir: str = None):
    """
    Download GPT-2 model and tokenizer, saving them locally.
    
    Args:
        model_name: Name of the model to download ("gpt2" for 117M parameter version)
        output_dir: Directory to save the model and tokenizer
    """
    try:
        # Use default path if none provided
        if output_dir is None:
            output_dir = ROOT_DIR / "models" / "gpt2"
            
        # Create output directory
        model_dir = ensure_directory(output_dir)
        
        logger.info(f"Downloading {model_name} model and tokenizer...")
        
        # Download and save tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=False
        )
        tokenizer_path = model_dir / "tokenizer"
        tokenizer.save_pretrained(tokenizer_path)
        
        # Download and save model
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            local_files_only=False
        )
        model_path = model_dir / "model"
        model.save_pretrained(model_path)
        
        # Save config file
        config_path = model_dir / "config.json"
        model.config.to_json_file(config_path)
        
        logger.info(f"Model and tokenizer successfully saved to {model_dir}")
        
        # Verify files exist
        if not (model_path / "pytorch_model.bin").exists():
            raise FileNotFoundError("Model file not saved correctly")
        if not (tokenizer_path / "vocab.json").exists():
            raise FileNotFoundError("Tokenizer files not saved correctly")
            
        return {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "config_path": config_path
        }
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    paths = download_gpt2_model()
    logger.info("Download complete. Saved files:")
    for key, path in paths.items():
        logger.info(f"{key}: {path}")
