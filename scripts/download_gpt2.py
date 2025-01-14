import os
import sys
import torch
from pathlib import Path

# Hugging Face Transformers imports
from transformers import (
    GPT2LMHeadModel,     # GPT-2 model architecture
    GPT2Tokenizer,       # GPT-2 tokenizer
    AutoTokenizer,       # Generic tokenizer for future models
    AutoModelForCausalLM # Generic model loader for future models
)

# Optional imports for future model support
try:
    from transformers import (
        LlamaTokenizer,        # Meta's LLaMA tokenizer
        LlamaForCausalLM,      # Meta's LLaMA model
        OPTForCausalLM,        # Meta's OPT model series
        BloomTokenizerFast,    # Bloom series tokenizer
        BloomForCausalLM,      # Bloom model architecture
    )
    EXTRA_MODELS_AVAILABLE = True
except ImportError:
    EXTRA_MODELS_AVAILABLE = False

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.utils.download_utils import ensure_directory, setup_logging

# Initialize logger
logger = setup_logging()

# Model registry for supported architectures
SUPPORTED_MODELS = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    # Stub for future model support - uncomment and modify as needed
    # "llama": (LlamaForCausalLM, LlamaTokenizer),
    # "opt": (OPTForCausalLM, AutoTokenizer),
    # "bloom": (BloomForCausalLM, BloomTokenizerFast),
}

def setup_cache_dirs(model_dir: Path) -> dict:
    """Setup cache directories for model and tokenizer downloads."""
    cache_dirs = {
        "model": ensure_directory(model_dir / "model" / "cache"),
        "tokenizer": ensure_directory(model_dir / "tokenizer" / "cache"),
    }
    return cache_dirs

def check_model_files(model_dir: Path) -> bool:
    """Check if model files already exist and are valid."""
    model_path = model_dir / "model"
    tokenizer_path = model_dir / "tokenizer"
    
    # Check both main directory and cache
    model_exists = any([
        (model_path / "pytorch_model.bin").exists(),
        (model_path / "model.safetensors").exists(),
        (model_path / "cache" / "pytorch_model.bin").exists(),
        (model_path / "cache" / "model.safetensors").exists()
    ])
    
    tokenizer_exists = all([
        (tokenizer_path / "vocab.json").exists(),
        (tokenizer_path / "merges.txt").exists()
    ])
    
    return model_exists and tokenizer_exists

def download_gpt2_model(model_name: str = "gpt2", output_dir: str = None, force_download: bool = False):
    """
    Download GPT-2 model and tokenizer, saving them locally with caching.
    
    Args:
        model_name: Name of the model to download ("gpt2" for 117M parameter version)
        output_dir: Directory to save the model and tokenizer
        force_download: If True, redownload even if files exist
    
    Notes:
        Modern HF models use safetensors format by default (model.safetensors)
        Legacy models may use PyTorch format (pytorch_model.bin)
        Both formats are valid and supported
    """
    try:
        if output_dir is None:
            output_dir = ROOT_DIR / "models" / "gpt2"
        
        model_dir = ensure_directory(output_dir)
        cache_dirs = setup_cache_dirs(model_dir)
        
        # Check if files already exist
        if not force_download and check_model_files(model_dir):
            logger.info(f"Model files already exist in {model_dir}")
            return {
                "model_path": model_dir / "model",
                "tokenizer_path": model_dir / "tokenizer",
                "config_path": model_dir / "config.json"
            }
        
        logger.info(f"Downloading {model_name} model and tokenizer...")
        
        # Download and save tokenizer with cache
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=False,
            cache_dir=cache_dirs["tokenizer"]
        )
        tokenizer_path = model_dir / "tokenizer"
        tokenizer.save_pretrained(tokenizer_path)
        
        # Download and save model with cache
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            local_files_only=False,
            cache_dir=cache_dirs["model"]
        )
        model_path = model_dir / "model"
        model.save_pretrained(model_path)
        
        # Save config file
        config_path = model_dir / "config.json"
        model.config.to_json_file(config_path)
        
        logger.info(f"Model and tokenizer successfully saved to {model_dir}")
        
        # Verify downloaded files - check both formats
        if not ((model_path / "pytorch_model.bin").exists() or 
                (model_path / "model.safetensors").exists()):
            raise FileNotFoundError("No valid model file found after download")
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
    paths = download_gpt2_model(force_download=False)
    logger.info("Download complete. Saved files:")
    for key, path in paths.items():
        logger.info(f"{key}: {path}")
