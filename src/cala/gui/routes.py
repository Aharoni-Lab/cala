from pathlib import Path
from queue import Queue
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, BackgroundTasks
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from noob import SynchronousRunner

from cala.gui.dependencies import get_frontend_dir, Spec
from cala.models.gui import GUISpec

templates = Jinja2Templates(directory=get_frontend_dir() / "templates")

router = APIRouter()

_thread = None


class QManager:
    _qs: dict[str, Queue] = {}

    @classmethod
    def get_queue(cls, name: str) -> Queue:
        if cls._qs.get(name) is None:
            cls._qs[name] = Queue()
        return cls._qs.get(name)


@router.get("/", response_class=HTMLResponse)
async def get(request: Request, spec: Annotated[GUISpec, Depends(Spec)]) -> HTMLResponse:
    """Serve the dashboard page"""
    return templates.TemplateResponse(request, "index.html", {"config": spec.model_dump()})


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connection and run streamers"""
    q = QManager.get_queue("lineplot")

    await websocket.accept()
    # some kind of waiting mechanism here
    # to keep the connection open until the client disconnects
    while True:
        event = q.get()
        if event is None:
            break
        await websocket.send_json(event)


@router.get("/{node_id}/{filename}")
async def stream(node_id: str, filename: str) -> FileResponse:
    """Serve HLS stream files"""
    stream_path = Path("output_dir") / node_id / filename
    if stream_path.exists():
        return FileResponse(str(stream_path))
    raise FileNotFoundError({"AppStreamError": f"Playlist not found: {stream_path}"})


@router.post("/start")
def start(background: BackgroundTasks):
    global _thread
    tube = Tube.from_specification()
    runner = SynchronousRunner(tube=tube)

    def _cb(event):
        q = QManager.get_queue("lineplot")
        if event.condition == "what i want":
            q.put(event)

    runner.add_callback(_cb)
    background.add_task(runner.run)


@router.post("/stop")
def stop():
    global _thread
    q = QManager.get_queue("lineplot")
    q.put(None)
    _thread.terminate()
    _thread.join()
