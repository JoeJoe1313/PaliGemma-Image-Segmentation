import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "model": {"default": "google/paligemma2-3b-mix-448", "cache_dir": "/models"},
    "api": {"host": "0.0.0.0", "port": 8000},
}


def load_config():
    """Load configuration from file if available, otherwise use defaults."""
    config = DEFAULT_CONFIG.copy()

    config_paths = [
        "/app/config.yaml",
        "/app/config.yml",
    ]
    for path in config_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    file_config = yaml.safe_load(f)
                    if file_config and isinstance(file_config, dict):
                        if "model" in file_config and isinstance(
                            file_config["model"], dict
                        ):
                            config["model"].update(file_config["model"])
                        if "api" in file_config and isinstance(
                            file_config["api"], dict
                        ):
                            config["api"].update(file_config["api"])
                break
            except Exception as e:
                print(f"Error loading config from {path}: {e}")

    return config


# Load configuration once at import time
CONFIG = load_config()


def get_default_model():
    """Get the default model path."""
    return CONFIG["model"]["model_name"]


def get_model_cache_dir():
    """Get the model cache directory."""
    return CONFIG["model"]["cache_dir"]


def ensure_dirs_exist():
    """Ensure all required directories exist."""
    Path(get_model_cache_dir()).mkdir(parents=True, exist_ok=True)

    # Create a subdirectory for the default model
    default_model = get_default_model().replace("/", "_")
    Path(os.path.join(get_model_cache_dir(), default_model)).mkdir(
        parents=True, exist_ok=True
    )
