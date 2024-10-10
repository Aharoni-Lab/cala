from pathlib import Path
from typing import List, Optional, Type, Tuple, Dict, Any

from pydantic import Field, TypeAdapter, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from platformdirs import PlatformDirs

_dirs = PlatformDirs("cala", "cala")


class Config(BaseSettings):
    user_dir: Path = Field(
        _dirs.user_config_dir, description="Directory containing cala config files"
    )
    config_file: Path = Field(
        Path("cala_config.yaml"),
        description="Location of global cala config file. "
        "If a relative path that doesn't exist relative to cwd, "
        "interpreted as a relative to ``user_dir``",
    )
    video_directory: Path = Path(_dirs.user_data_dir) / "videos"
    video_files: Optional[List[Path]] = Field(default_factory=list)
    data_directory: Path = Path(_dirs.user_data_dir)
    data_name: Optional[str] = "cala"

    @property
    def video_paths(self) -> List[Path]:
        return [
            self.video_directory.joinpath(video_file) for video_file in self.video_files
        ]

    model_config = SettingsConfigDict(
        env_prefix="cala_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        nested_model_default_partial_update=True,
        yaml_file="cala_config.yaml",
        pyproject_toml_table_header=("tool", "cala", "config"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        Read config settings from, in order of priority from high to low, where
        high priorities override lower priorities:

        * in the arguments passed to the class constructor (not user configurable)
        * in environment variables like ``export CALA_LOG_DIR=~/``
        * in a ``.env`` file in the working directory
        * in a ``cala_config.yaml`` file in the working directory
        * in the ``tool.cala.config`` table in a ``pyproject.toml`` file in the working directory
        * in the global ``cala_config.yaml`` file in the platform-specific data directory
          (use ``cala config get config_file`` to find its location)
        * the default values in the :class:`.Config` model
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            PyprojectTomlConfigSettingsSource(settings_cls),
            _GlobalYamlConfigSource(settings_cls),
        )

    @field_validator("user_dir", "data_directory", mode="after")
    @classmethod
    def dir_exists(cls, v: Path) -> Path:
        """Ensure user_dir exists, make it otherwise"""
        v = Path(v)
        v.mkdir(exist_ok=True, parents=True)
        assert v.exists(), f"{v} does not exist!"
        return v

    @model_validator(mode="after")
    def config_file_is_absolute(self) -> "Config":
        """
        If ``config_file`` is relative, make it absolute underneath user_dir
        """
        if not self.config_file.is_absolute():
            if self.config_file.exists():
                self.config_file = self.config_file.resolve()
            else:
                self.config_file = self.user_dir / self.config_file
        return self

    @field_validator("video_files", mode="before")
    @classmethod
    def split_video_files(cls, v):
        if isinstance(v, str):
            # Split the string by commas and strip any whitespace
            items = [item.strip() for item in v.split(",") if item.strip()]
            return items
        return v

    @model_validator(mode="after")
    def check_video_files_exist(self) -> "Config":
        missing_files = []
        for video_file in self.video_files or []:
            full_path = self.video_directory / video_file
            if not full_path.exists():
                missing_files.append(str(full_path))

        if missing_files:
            raise ValueError(
                f"The following video files do not exist in {self.video_directory}: {', '.join(missing_files)}"
            )

        return self


class _GlobalYamlConfigSource(YamlConfigSettingsSource):
    """Yaml config source that gets the location of the global settings file from the prior sources"""

    def __init__(self, *args, **kwargs):
        self._global_config = None
        super().__init__(*args, **kwargs)

    @property
    def global_config_path(self) -> Path:
        """
        Location of the global ``cala_config.yaml`` file,
        given the current state of prior config sources
        """
        current_state = self.current_state
        config_file = Path(current_state.get("config_file", "cala_config.yaml"))
        user_dir = Path(current_state.get("user_dir", _dirs.user_config_dir))
        if not config_file.is_absolute():
            config_file = (user_dir / config_file).resolve()
        return config_file

    @property
    def global_config(self) -> Dict[str, Any]:
        """
        Contents of the global config file
        """
        if self._global_config is None:
            if self.global_config_path.exists():
                self._global_config = self._read_files(self.global_config_path)
            else:
                self._global_config = {}
        return self._global_config

    def __call__(self) -> Dict[str, Any]:
        return (
            TypeAdapter(Dict[str, Any]).dump_python(self.global_config)
            if self.nested_model_default_partial_update
            else self.global_config
        )
