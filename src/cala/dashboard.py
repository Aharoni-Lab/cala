import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.requests import Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cala.config import Config

from cala.connection_manager import ConnectionManager  # isort:skip
from cala.main import run_pipeline

frontend_dir = Path(__file__).parents[2] / "frontend"
templates = Jinja2Templates(directory=frontend_dir / "templates")

manager = ConnectionManager()

temp_dir = Path(tempfile.TemporaryDirectory().name)
config_path = os.getenv("CALA_CONFIG_PATH", "cala_config.yaml")
config = Config.from_yaml(config_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    temp_dir.mkdir(exist_ok=True, parents=True)

    # Start the streamers in the background
    # modify runner.run_streamers to not block the event loop or run it in a background task
    asyncio.create_task(run_pipeline(config, manager, temp_dir))

    yield


app = FastAPI(lifespan=lifespan)

app.mount(path="/dist", app=StaticFiles(directory=frontend_dir / "dist"), name="dist")


@app.get("/")
async def get(request: Request) -> dict[str, str] | None:
    """Serve the dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request, "config": config})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connection and run streamers"""
    await manager.connect(websocket)
    # Keep the connection alive
    try:
        # some kind of waiting mechanism here to keep the connection open until the client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print(f"Num Connections: {len(manager.active_connections)}")
        if websocket in manager.active_connections:
            manager.disconnect(websocket)


@app.get("/stream/{filename}")
async def stream(filename: str) -> FileResponse:
    """Serve HLS stream files"""
    stream_path = temp_dir / filename
    if stream_path.exists():
        return FileResponse(str(stream_path))
    raise FileNotFoundError({"AppStreamError": "Playlist not found"})
