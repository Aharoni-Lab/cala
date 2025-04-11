from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, TypeAdapter, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from cala.config.base import _dirs, _global_config_path
from cala.config.mixin import YAMLMixin
from cala.config.pipe import StreamingConfig


class Config(BaseSettings, YAMLMixin):
    user_dir: Path = Field(
        _dirs.user_config_dir,
        description="Base project directory containing the config file.",
    )

    input_files: list[Path] = Field(default_factory=list)

    output_dir: Path = Field(
        Path(_dirs.user_data_dir) / "output",
        description="Location of output data. "
        "If a relative path that doesn't exist relative to cwd, "
        "interpreted as a relative to ``user_dir``",
    )

    pipeline: StreamingConfig

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_prefix="cala_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        nested_model_default_partial_update=True,
        yaml_file="cala_config.yaml",
        pyproject_toml_table_header=("tool", "cala", "config_examples"),
    )

    @field_validator("user_dir", mode="after")
    @classmethod
    def dir_exists(cls, v: Path) -> Path:
        """Ensure user_dir exists, make it otherwise"""
        v.mkdir(exist_ok=True, parents=True)
        assert v.exists(), f"{v} does not exist!"
        return v

    @model_validator(mode="after")
    def paths_relative_to_basedir(self) -> "Config":
        """
        If path is relative, make it absolute underneath user_dir
        """
        paths = ["output_dir"]
        for path_name in paths:
            path = getattr(self, path_name)
            if not path.is_absolute():
                if not path.exists():
                    path = self.user_dir / path
                    path.mkdir(parents=True, exist_ok=True)
                setattr(self, path_name, path.resolve())

        return self

    @model_validator(mode="after")
    def files_exist(self) -> "Config":
        files = self.input_files
        missing_files = [file for file in files if not Path(file).exists()]
        if missing_files:
            raise ValueError(f"The following files do not exist: {', '.join(missing_files)}")

        return self

    @model_validator(mode="after")
    def validate_pipeline(self) -> "Config":
        """Validate pipeline config"""
        # additional validation logic
        return self

    @classmethod
    def settings_customize_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Read config_examples settings from, in order of priority from high to low, where
        high priorities override lower priorities:

        * in the arguments passed to the class constructor (not user configurable)
        * in environment variables like ``export CALA_LOG_DIR=~/``
        * in a ``.env`` file in the working directory
        * in a ``config.yaml`` file in the working directory
        * in the ``tool.cala.config_examples`` table in a ``pyproject.toml`` file in the working directory
        * in the global ``config.yaml`` file in the platform-specific data directory
          (use ``cala config_examples get config_file`` to find its location)
        * the default values in the :class:`.Config` model
        """
        _create_default_global_config()

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            PyprojectTomlConfigSettingsSource(settings_cls),
            _GlobalYamlConfigSource(settings_cls),
        )


def _create_default_global_config(path: Path = _global_config_path, force: bool = False) -> None:
    """
    Create a default global `config.yaml` file.

    Args:
        force (bool): Override any existing global config
    """
    if path.exists() and not force:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"user_dir": str(path.parent)}
    with open(path, "w") as f:
        yaml.safe_dump(config, f)


class _GlobalYamlConfigSource(YamlConfigSettingsSource):
    """Yaml config_examples source that gets the location of the global settings file from the prior sources"""

    def __init__(self, *args: Any, **kwargs: Any):
        self._global_config: Any = None
        super().__init__(*args, **kwargs)

    @property
    def global_config_path(self) -> Path:
        """
        Location of the global ``config.yaml`` file,
        given the current state of prior config_examples sources
        """
        current_state = self.current_state
        config_file = Path(current_state.get("config_file", "cala_config.yaml"))
        user_dir = Path(current_state.get("user_dir", _dirs.user_config_dir))
        if not config_file.is_absolute():
            config_file = (user_dir / config_file).resolve()
        return config_file

    @property
    def global_config(self) -> Any:
        """
        Contents of the global config_examples file
        """
        if self._global_config is None:
            if self.global_config_path.exists():
                self._global_config = self._read_files(self.global_config_path)
            else:
                self._global_config = {}
        return self._global_config

    def __call__(self) -> dict[str, Any]:
        return dict(
            TypeAdapter(dict[str, Any]).dump_python(self.global_config)
            if self.nested_model_default_partial_update
            else self.global_config
        )
