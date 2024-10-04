from pydantic import BaseModel, Field, ValidationError
from pathlib import Path
from typing import List, Optional
import yaml
import importlib.resources

USER_CONFIG_PATH = Path.home() / ".cala" / "config.yaml"
"""
USER_CONFIG_PATH (Path): Default path to the user's Cala configuration file, located in the home directory.
"""


class Config(BaseModel):
    video_directory: Path
    data_directory: Path
    video_files: Optional[List[Path]] = Field(default_factory=list)
    data_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "Config":
        """
        Loads configuration from the provided path, falling back to the default configuration paths.
        - Custom path > User config in ~/.cala/config.yaml > Default package config in src/cala/config/config.yaml.
        """
        if path:
            config_path = Path(path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}.")
        else:
            if not USER_CONFIG_PATH.exists():
                # Load default config from the package resources
                try:
                    config_content = importlib.resources.read_text(
                        "cala.config", "config.yaml"
                    )
                    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(USER_CONFIG_PATH, "w") as f:
                        f.write(config_content)

                except FileNotFoundError:
                    raise FileNotFoundError(
                        "Default config not found in package resources."
                    )

            config_path = USER_CONFIG_PATH

        # Load the configuration from the determined path
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """
        Converts a dictionary to a Config model instance.
        """
        try:
            return cls(**config_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration format: {e}")


CONFIG = Config.from_yaml()
