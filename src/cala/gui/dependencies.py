import os
import tempfile
from pathlib import Path

from cala.config import Config
from cala.gui.socket_manager import SocketManager

stream_dir = Path(tempfile.TemporaryDirectory().name)
socket_manager = SocketManager()
config = Config.from_yaml(os.getenv("CALA_CONFIG_PATH", "cala_config.yaml"))


async def get_config() -> Config:
    return config


async def get_socket_manager() -> SocketManager:
    return socket_manager


async def get_stream_dir() -> Path:
    return stream_dir
