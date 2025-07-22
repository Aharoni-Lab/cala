from pathlib import Path
from typing import Literal

from platformdirs import PlatformDirs
from pydantic import Field, field_validator, model_validator, BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from cala.config.yaml import YAMLMixin

_default_userdir = Path().home() / ".config" / "cala"
_dirs = PlatformDirs("cala", "cala")


LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class LogConfig(BaseModel):
    """
    Configuration for logging
    """

    level: LOG_LEVELS = "INFO"
    """
    Severity of log messages to process.
    """
    level_file: LOG_LEVELS | None = None
    """
    Severity for file-based logging. If unset, use ``level``
    """
    level_stdout: LOG_LEVELS | None = None
    """
    Severity for stream-based logging. If unset, use ``level``
    """
    dir: Path = _dirs.user_log_dir
    """
    Directory where logs are stored.
    """
    file_n: int = 5
    """
    Number of log files to rotate through
    """
    file_size: int = 2**22  # roughly 4MB
    """
    Maximum size of log files (bytes)
    """

    @field_validator("level", "level_file", "level_stdout", mode="before")
    @classmethod
    def uppercase_levels(cls, value: str | None = None) -> str | None:
        """
        Ensure log level strings are uppercased
        """
        if value is not None:
            value = value.upper()
        return value

    @field_validator("dir", mode="after")
    def create_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value


class Config(BaseSettings, YAMLMixin):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="cala_",
        env_nested_delimiter="__",
        extra="ignore",
        nested_model_default_partial_update=True,
        yaml_file="cala_config.yaml",
        pyproject_toml_table_header=("tool", "cala", "config"),
    )

    logs: LogConfig = LogConfig()

    user_dir: Path = Field(default=Path(_dirs.user_data_dir))

    config_dir: Path = Field(
        default=Path(_dirs.user_data_dir) / "config",
        description="Directory where config yaml files are stored",
    )

    input_files: list[Path] = Field(default_factory=list)

    output_dir: Path = Field(
        default=Path(_dirs.user_data_dir) / "output",
        description="Location of output data. "
        "If a relative path that doesn't exist relative to cwd, "
        "interpreted as a relative to ``user_dir``",
    )

    @field_validator("user_dir", "config_dir", mode="after")
    @classmethod
    def create_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value

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
    def validate_input_files(self) -> "Config":
        inputs_relative_to_user_dir = []
        missing_files = []
        for filepath in self.input_files:
            resolved_path = (self.user_dir / filepath).resolve()
            if resolved_path.exists():
                inputs_relative_to_user_dir.append(resolved_path)
            else:
                missing_files.append(str(resolved_path))
        if missing_files:
            raise ValueError(f"The following files do not exist: {', '.join(missing_files)}")

        self.input_files = inputs_relative_to_user_dir

        return self

    @model_validator(mode="after")
    def validate_pipeline(self) -> "Config":
        """Validate pipeline config"""
        # additional validation logic
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Read config settings from, in order of priority from high to low, where
        high priorities override lower priorities:
        * in the arguments passed to the class constructor (not user configurable)
        * in environment variables like ``export CALA_LOGS__DIR=~/``
        * in a ``.env`` file in the working directory
        * in a ``cala_config.yaml`` file in the working directory
        * in the ``tool.cala.config`` table in a ``pyproject.toml`` file
          in the working directory
        * the default values in the :class:`.Config` model
        """

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            PyprojectTomlConfigSettingsSource(settings_cls),
        )


config = Config()
