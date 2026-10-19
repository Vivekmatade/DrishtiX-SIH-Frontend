import os
from typing import Any, Dict
import yaml

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration dictionary from a YAML file."""
    if not os.path.exists(config_path):
        # Fallback search if path is relative to repo root
        root_config = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
        if os.path.exists(root_config):
            config_path = root_config
        else:
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
