import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import sys
import os  # Added import

def setup_logging(log_path=Path("logs")):
    """Setup centralized logging."""
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "app.log"
    
    logging.basicConfig(
        level=logging.DEBUG,  # Updated log level to DEBUG
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Initialize dedicated logger for speak_text_thread
    speak_logger = logging.getLogger('speak_text_thread')
    if not speak_logger.handlers:
        speak_logger.setLevel(logging.ERROR)
        speak_log_file = log_path / "speak_text.log"
        handler = RotatingFileHandler(speak_log_file, maxBytes=5*1024*1024, backupCount=2)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        speak_logger.addHandler(handler)
    return logging.getLogger(__name__)
