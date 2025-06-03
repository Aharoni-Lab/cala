import os
from pathlib import Path

from cala.config import Config
from cala.gui.socket_manager import SocketManager

socket_manager = SocketManager()
root_path = Path(__file__).parents[3]
config = Config.from_yaml(root_path / os.getenv("CALA_CONFIG_PATH"))


async def get_config() -> Config:
    return config


def get_socket_manager() -> SocketManager:
    return socket_manager
