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

frontend_dir = Path(__file__).parents[2] / "frontend"

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
