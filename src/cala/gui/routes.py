import base64
from asyncio import sleep
from queue import Empty

import yaml
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from noob.event import Event
from noob.tube import TubeSpecification, Tube
from starlette.websockets import WebSocket

from cala.config import config
from cala.gui.const import TEMPLATES_DIR
from cala.gui.deps import Spec
from cala.gui.runner import BackgroundRunner
from cala.gui.util import QManager
from cala.logging import init_logger

logger = init_logger(__name__)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter()

_running = False
_thread = None
_tube_config = None
_runner = None


@router.get("/", response_class=HTMLResponse)
async def get(request: Request, spec: Spec) -> HTMLResponse:
    response = templates.TemplateResponse(request, "index.html", {"config": spec.model_dump()})
    config = base64.b64encode(bytes(spec.model_dump_json(), "utf-8")).decode()
    response.set_cookie(key="config", value=config, samesite="lax")

    return response


@router.post("/start")
def start(gui_spec: Spec, request: Request) -> HTMLResponse:
    try:
        global _running
        if _running:
            raise HTTPException(400, f"Already running.")
        spec = TubeSpecification(**_tube_config)
        spec.assets = {**spec.assets, **gui_spec.assets}
        spec.nodes = {**spec.nodes, **gui_spec.nodes}
        tube = Tube.from_specification(spec)

        global _runner
        _runner = BackgroundRunner(tube=tube)

        def _cb(event):
            for node_id, _ in gui_spec.nodes.items():
                if event["node_id"] == node_id:
                    q = QManager.get_queue(node_id)
                    q.put(event)

        _runner.add_callback(_cb)
        _runner.run()

    except Exception as e:
        raise HTTPException(500, str(e))

    _running = True
    return templates.TemplateResponse(
        request, "partials/grids.html", {"grids": list(gui_spec.nodes.values())}
    )


@router.post("/stop")
def stop():
    global _runner
    _runner.shutdown()


@router.post("/submit-tube")
def submit_tube(file: UploadFile, request: Request) -> HTMLResponse:
    global _tube_config
    try:
        tube_config = yaml.safe_load(file.file)
    except Exception as e:
        raise HTTPException(422, f"Failed to load Tube specification. {e}")
    _tube_config = tube_config
    return templates.TemplateResponse(
        request, "partials/tube-config.html", {"msg": f"noob_id: {tube_config['noob_id']}"}
    )


@router.post("/player/{node_id}")
async def player(node_id: str, request: Request) -> HTMLResponse:
    """Serve video player DOM"""
    stream_path = config.runtime_dir / node_id / "stream.m3u8"
    if stream_path.exists():
        return templates.TemplateResponse(
            request,
            "partials/player.html",
            {"id": node_id, "path": f"/stream/{node_id}/stream.m3u8"},
            headers={"Content-Control": "no-store"},
        )
    else:
        raise HTTPException(404)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connection and run streamers"""
    await websocket.accept()
    while True:
        await sleep(0.01)
        qs = [QManager.get_queue(q) for q in QManager().queues.keys()]
        events: list[Event] = []
        for q in qs:
            try:
                events.append(q.get(False))
            except Empty:
                pass
        for event in events:
            await websocket.send_json({"node_id": event["node_id"], "value": event["value"]})
