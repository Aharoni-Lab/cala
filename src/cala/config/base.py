from pathlib import Path

from platformdirs import PlatformDirs

_default_userdir = Path().home() / ".config" / "cala"
_dirs = PlatformDirs("cala", "cala")
_global_config_path = Path(_dirs.user_config_path) / "config.yaml"
