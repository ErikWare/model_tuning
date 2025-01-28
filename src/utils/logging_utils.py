import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: Path = None) -> logging.Logger:
    """Setup logging configuration with rotation."""
    if log_dir is None:
        log_dir = Path(__file__).parents[2] / "logs"
    
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"
    
    # Create formatter with simpler format
    formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Setup rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10_485_760,  # 10MB
        backupCount=5,
        mode='a'
    )
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('model_tuning')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger
