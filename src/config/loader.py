"""
Configuration loader for YAML settings
"""

import os
import yaml
from pathlib import Path

_config = None

def load_config(config_path=None):
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config file. If None, uses default src/config/settings.yaml
    
    Returns:
        dict: Configuration dictionary
    """
    global _config
    
    if config_path is None:
        # Find project root (where this file is located: src/config/loader.py)
        current_dir = Path(__file__).resolve().parent  # src/config/
        src_dir = current_dir.parent  # src/
        project_root = src_dir.parent  # project root
        config_path = project_root / "src" / "config" / "settings.yaml"
    
    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)
    
    return _config

def get_config():
    """
    Get the loaded configuration. Loads default if not already loaded.
    
    Returns:
        dict: Configuration dictionary
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config

if __name__ == "__main__":
    # Smoke test: print a few config values
    cfg = load_config()
    print(f"Config loaded successfully from {Path(__file__).resolve().parent / 'settings.yaml'}")
    print(f"Model name: {cfg['model']['name']}")
    print(f"Sequence length: {cfg['model']['max_seq_len']}")