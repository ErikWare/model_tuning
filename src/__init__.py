"""
AI Assistant Application
Core package initialization
"""
from src.models.model_controller import ModelController
from src.gui.chat_interface import ChatInterface
from src.utils import setup_logging, ensure_directory

__version__ = '1.0.0'

__all__ = [
    'ModelController',
    'ChatInterface',
    'setup_logging',
    'ensure_directory'
]