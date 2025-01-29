import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import sys

def setup_logging(log_path=Path("logs")):
    """Setup centralized logging."""
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "app.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)
