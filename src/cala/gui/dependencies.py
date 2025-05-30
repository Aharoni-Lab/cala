import os

from cala.config import Config
from cala.gui.socket_manager import SocketManager

socket_manager = SocketManager()
config = Config.from_yaml(os.getenv("CALA_CONFIG_PATH"))


async def get_config() -> Config:
    return config


def get_socket_manager() -> SocketManager:
    return socket_manager
