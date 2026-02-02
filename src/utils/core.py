"""Core utilities for transfer reinforcement learning project."""

import random
import numpy as np
import torch
from typing import Any, Dict, Optional, Union
import logging
from pathlib import Path


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional log file path.
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("transfer_rl")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_directories(base_path: Union[str, Path]) -> None:
    """Create necessary directories for the project.
    
    Args:
        base_path: Base project path.
    """
    base_path = Path(base_path)
    directories = [
        "assets/plots",
        "assets/videos", 
        "assets/models",
        "data",
        "checkpoints",
        "logs"
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)


def safe_dict_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with default.
    
    Args:
        dictionary: Dictionary to get value from.
        key: Key to look for.
        default: Default value if key not found.
        
    Returns:
        Value from dictionary or default.
    """
    return dictionary.get(key, default)


class ConfigValidator:
    """Configuration validator for ensuring required parameters are present."""
    
    def __init__(self, required_keys: list):
        """Initialize validator with required keys.
        
        Args:
            required_keys: List of required configuration keys.
        """
        self.required_keys = required_keys
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        missing_keys = [key for key in self.required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        return True
