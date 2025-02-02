import os
import argparse
import sys
import logging
import torch
from pathlib import Path
import json

from transformers import (
    AutoModelForCausalLM,  # for generative models
    AutoTokenizer,
    AutoConfig,
    AutoModel  # for non-generative models
)

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model configurations from a JSON file
CONFIG_FILE = Path(__file__).parent / "light_weight_models.json"
with open(CONFIG_FILE, 'r') as f:
    MODEL_CONFIGS = json.load(f)

def download_model(model_variant: str, model_dir: Path):
    """
    Universal download utility to retrieve models from Hugging Face.
    This script supports various lightweight LLMs for scientific analysis and testing on a 16GB M1 MacBook Pro.
    
    Args:
        model_variant (str): Key of the model to download from the configuration.
        model_dir (Path): Directory where the model will be cached and stored.
        
    Returns:
        (model, tokenizer): Loaded model and its tokenizer.
    """
    if model_variant not in MODEL_CONFIGS:
        logger.error(f"Model variant '{model_variant}' is not defined in the configuration.")
        sys.exit(1)
    
    model_info = MODEL_CONFIGS[model_variant]
    model_name = model_info['name']
    logger.info(f"Selected {model_variant} model: {model_name} ({model_info['size']} parameters)")
    
    # Settings optimized for M1 Mac testing environments
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto"

    models_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    model_variant_dir = os.path.join(models_root, model_variant)
    os.makedirs(model_variant_dir, exist_ok=True)  # Ensure the model folder exists

    logger.info(f"Downloading {model_name} into {model_variant_dir}")
    logger.info(f"Using torch_dtype: {torch_dtype} and device_map: {device_map}")

    # Download and cache the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=model_variant_dir
    )

    # Modern model class selection based on the model's category
    category = model_info.get("category", "").lower()
    if "generative" in category or "seq2seq" in category:
        model_class = AutoModelForCausalLM
    else:
        model_class = AutoModel

    # Download the model with parameters optimized for low CPU memory usage
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=model_variant_dir,
        low_cpu_mem_usage=True
    )

    logger.info(f"Successfully downloaded {model_name} and its tokenizer.")
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal LLM download script for scientific analysis")
    parser.add_argument('--variant', type=str, required=True, choices=MODEL_CONFIGS.keys(),
                        help='Select the model variant to download (e.g., "gpt-neo-1.3B")')
    parser.add_argument('--model_dir', type=str, default='models/',
                        help='Local directory to store the downloaded model and tokenizer')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = download_model(args.variant, model_dir)