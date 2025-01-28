import os
import logging
from pathlib import Path
from typing import Union

def ensure_directory(directory: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

# Remove duplicate setup_logging function as it's now in logging_utils.py
