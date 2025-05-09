import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "model": {"model_path": "google/paligemma2-3b-mix-448", "cache_dir": "/models"},
    "api": {"host": "0.0.0.0", "port": 8000},
}


def load_config():
    """Load configuration from file if available, otherwise use defaults."""
    config = DEFAULT_CONFIG.copy()
    # config = {}

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

    # # Override with environment variables (highest priority)
    # if os.environ.get("MODEL_PATH"):
    #     config["model"]["default"] = os.environ.get("MODEL_PATH")

    # if os.environ.get("MODEL_CACHE_DIR"):
    #     config["model"]["cache_dir"] = os.environ.get("MODEL_CACHE_DIR")

    return config


# Load configuration once at import time
CONFIG = load_config()


def get_model_path():
    """Get the default model path."""
    return CONFIG["model"]["model_path"]


def get_model_cache_dir():
    """Get the model cache directory."""
    return CONFIG["model"]["cache_dir"]


def ensure_dirs_exist():
    """Ensure all required directories exist."""
    Path(get_model_cache_dir()).mkdir(parents=True, exist_ok=True)

    default_model = f"models--{get_model_path().replace('/', '--')}"
    Path(os.path.join(get_model_cache_dir(), default_model)).mkdir(
        parents=True, exist_ok=True
    )
