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

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define available model variants
DEEPSEEK_MODELS = {
    'small': {
        'name': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'size': '1.5B',
        'description': 'Smallest variant, good for testing'
    },
    # You can add more variants here (e.g., 'medium', 'large') if available
}

def download_model(model_variant: str, model_dir: Path):
    if model_variant not in DEEPSEEK_MODELS:
        logger.error(f"Model variant '{model_variant}' is not defined.")
        sys.exit(1)
    
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
        use_fast=True,               # Utilize fast tokenizer
        trust_remote_code=True,      # Trust remote code if required by the model
        cache_dir=model_dir          # Specify cache directory
    )

    # Download and save model with appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,      # Trust remote code if required by the model
        cache_dir=model_dir,
        low_cpu_mem_usage=True
    )

    logger.info(f"Model '{model_name}' and tokenizer downloaded successfully.")
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DeepSeek Models")
    parser.add_argument('--variant', type=str, default='small', choices=DEEPSEEK_MODELS.keys(),
                        help='Model variant to download')
    parser.add_argument('--model_dir', type=str, default='models/',
                        help='Directory to cache the downloaded model and tokenizer')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = download_model(args.variant, model_dir)