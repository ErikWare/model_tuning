import os
import logging
from pathlib import Path

def ensure_directory(directory: str) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)
