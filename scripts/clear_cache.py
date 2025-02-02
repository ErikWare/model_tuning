import os
import shutil
import argparse
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_cache(model_dir: Path):
    if model_dir.exists() and model_dir.is_dir():
        print(f"Clearing cache in directory: {model_dir}")
        logger.info(f"Clearing cache in directory: {model_dir}")
        try:
            shutil.rmtree(model_dir)
            print("Cache cleared successfully.")
            logger.info("Cache cleared successfully.")
        except Exception as e:
            print(f"Failed to clear cache: {e}")
            logger.error(f"Failed to clear cache: {e}")
    else:
        print(f"Directory {model_dir} does not exist or is not a directory.")
        logger.warning(f"Directory {model_dir} does not exist or is not a directory.")

def clear_huggingface_cache():
    hf_cache_dir = Path.home() / ".cache" / "huggingface"
    if hf_cache_dir.exists() and hf_cache_dir.is_dir():
        print(f"Clearing Hugging Face cache in directory: {hf_cache_dir}")
        logger.info(f"Clearing Hugging Face cache in directory: {hf_cache_dir}")
        try:
            shutil.rmtree(hf_cache_dir)
            print("Hugging Face cache cleared successfully.")
            logger.info("Hugging Face cache cleared successfully.")
        except Exception as e:
            print(f"Failed to clear Hugging Face cache: {e}")
            logger.error(f"Failed to clear Hugging Face cache: {e}")
    else:
        print(f"Hugging Face cache directory {hf_cache_dir} does not exist or is not a directory.")
        logger.warning(f"Hugging Face cache directory {hf_cache_dir} does not exist or is not a directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear Model Cache")
    parser.add_argument('--model_dir', type=str, default='models/',
                        help='Directory to clear the cached model and tokenizer')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    clear_cache(model_dir)
    clear_huggingface_cache()
