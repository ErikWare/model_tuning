import os
import logging
from pathlib import Path
from typing import Union

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        directory (Union[str, Path]): The path to the directory to ensure.

    Returns:
        Path: The Path object of the directory.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info(f"Directory ensured at: {path}")
    return path

# Remove duplicate setup_logging function as it's now in logging_utils.py
