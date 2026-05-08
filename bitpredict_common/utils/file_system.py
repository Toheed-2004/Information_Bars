import os
import yaml
from pathlib import Path
from typing import Optional

def read_yaml_config(path_config):
    """
    Read a YAML configuration file and return it as a dictionary.

    Parameters:
    - path_config: Path to the YAML configuration file

    Returns:
    - Dictionary containing the configuration
    """
    if not os.path.exists(path_config):
        print(f"Configuration file not found: {path_config}")
        return {}

    try:
        with open(path_config, "r") as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {}


def ensure_directory(base_path: Optional[str], folder_name: str) -> Path:
        """
        Ensure a folder exists at base_path/folder_name.
        If base_path is None, use current working directory.
        Returns the Path object to the folder.
        """
        root = Path(base_path) if base_path else Path.cwd()
        folder_path = root / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path