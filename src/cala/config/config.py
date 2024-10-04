from pydantic import BaseModel, Field, ValidationError
from pathlib import Path
from typing import List, Optional
import yaml
import importlib.resources


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
            # Check for user-specific config in ~/.cala/config.yaml
            user_config_path = Path.home() / ".cala" / "config.yaml"
            if user_config_path.exists():
                config_path = user_config_path
            else:
                # Load default config from the package resources
                try:
                    config_content = importlib.resources.read_text(
                        "cala.config", "config.yaml"
                    )
                    return cls.from_dict(yaml.safe_load(config_content))
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "Default config not found in package resources."
                    )

        # Load the configuration file from the specified or default location
        return cls.from_dict(yaml.safe_load(config_path.read_text()))

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
