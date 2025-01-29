import os
import argparse
import sys
import logging
import torch
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)

# Assuming you have a similar utility for setting up logging
# from src.utils.download_utils import setup_logging

# If no custom logger setup, just use basicConfig:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define available model variants
DEEPSEEK_MODELS = {
    #https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    'small': {
        'name': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'size': '1.5B',
        'description': 'Smallest variant, good for testing'
    },
}

def ensure_directory(path: Path) -> Path:
    """Simple utility to create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def download_deepseek_model(
    model_name: str = None,
    output_dir: str = None,
    force_download: bool = False,
    model_variant: str = 'light'
):
    """
    Download the DeepSeek model and tokenizer.
    
    Args:
        model_name: Override default model name if provided
        output_dir: Where to save model and tokenizer
        force_download: If True, always download
        model_variant: 'light' (1.3B) or 'heavy' (6.7B)
    """
    try:
        # Setup directory
        if output_dir is None:
            script_dir = Path(__file__).resolve().parent.parent
            output_dir = script_dir / "models" / "deepseek"
        
        model_dir = ensure_directory(Path(output_dir))

        # Select model variant
        if not model_name:
            if model_variant not in DEEPSEEK_MODELS:
                raise ValueError(f"Invalid model variant. Choose from: {list(DEEPSEEK_MODELS.keys())}")
            model_info = DEEPSEEK_MODELS[model_variant]
            model_name = model_info['name']
            logger.info(f"Selected {model_variant} model: {model_name} ({model_info['size']} parameters)")

        # Configure model loading for M1 Mac
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto"

        logger.info(f"Downloading {model_name} to {model_dir}")
        logger.info(f"Using dtype: {torch_dtype}, device_map: {device_map}")

        # Download and save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
            cache_dir=model_dir
        )

        # Download and save model with appropriate configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            cache_dir=model_dir,
            low_cpu_mem_usage=True
        )

        # Save everything to the directory
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(
            model_dir,
            safe_serialization=True  # Use safetensors format
        )

        # Save configuration
        config_path = model_dir / "config.json"
        model.config.to_json_file(config_path)

        logger.info(f"Model successfully downloaded to {model_dir}")
        return {
            "model_path": model_dir,
            "tokenizer_path": model_dir,
            "config_path": config_path
        }

    except Exception as e:
        logger.error(f"Error downloading DeepSeek model: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download DeepSeek Coder model variants')
    parser.add_argument(
        '--variant', 
        choices=['tiny', 'small', 'medium', 'large'],
        default='small',
        help='Model variant to download'
    )
    parser.add_argument('--force', action='store_true',
                       help='Force redownload even if files exist')
    parser.add_argument('--list', action='store_true',
                       help='List available model variants and exit')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable DeepSeek Coder models:")
        print("-" * 50)
        for variant, info in DEEPSEEK_MODELS.items():
            print(f"{variant:6} : {info['size']:6} - {info['description']}")
        print("-" * 50)
        sys.exit(0)
    
    paths = download_deepseek_model(
        model_variant=args.variant,
        force_download=args.force
    )
    logger.info("Download complete. Saved files:")
    for key, path in paths.items():
        logger.info(f"{key}: {path}")