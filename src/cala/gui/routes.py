import importlib
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cala.config import Config
from cala.gui.dependencies import get_config, get_socket_manager, get_stream_dir
from cala.gui.socket_manager import SocketManager


def get_frontend_dir() -> Path:
    env = os.getenv("NODE_ENV", "production")

    if env == "development":
        root_dir = Path(__file__).parents[3]
        frontend_dir = root_dir / "frontend"

        if not frontend_dir.exists():
            raise FileNotFoundError(f"Frontend build directory not found at {frontend_dir}. ")

        return frontend_dir
    elif env == "production":
        package_name = __name__.split(".")[0]  # Get the top-level package name
        with importlib.resources.files(package_name) as package_path:
            frontend_path = package_path / "gui"
            if frontend_path.exists():
                return frontend_path
            else:
                raise FileNotFoundError(
                    f"Could not find frontend assets in the installed package {package_name}"
                )
    else:
        raise ValueError(f"Invalid NODE_ENV value: {env}")


frontend_dir = get_frontend_dir()
templates = Jinja2Templates(directory=frontend_dir / "templates")

router = APIRouter()
router.mount(path="/dist", app=StaticFiles(directory=frontend_dir / "dist"), name="dist")


@router.get("/", response_class=HTMLResponse)
async def get(request: Request, config: Annotated[Config, Depends(get_config)]) -> HTMLResponse:
    """Serve the dashboard page"""
    return templates.TemplateResponse(request, "index.html", {"config": config})


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, socket_manager: Annotated[SocketManager, Depends(get_socket_manager)]
) -> None:
    """Handle WebSocket connection and run streamers"""
    await socket_manager.connect(websocket)
    try:
        # some kind of waiting mechanism here to keep the connection open until the client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print(
            f"websocket disconnected, still have {len(socket_manager.active_connections)} active connections"
        )
        if websocket in socket_manager.active_connections:
            socket_manager.disconnect(websocket)


@router.get("/stream/{filename}")
async def stream(
    filename: str, stream_dir: Annotated[Path, Depends(get_stream_dir)]
) -> FileResponse:
    """Serve HLS stream files"""
    stream_path = stream_dir / filename
    if stream_path.exists():
        return FileResponse(str(stream_path))
    raise FileNotFoundError({"AppStreamError": "Playlist not found"})
