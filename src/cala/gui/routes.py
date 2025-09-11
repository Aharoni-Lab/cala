import base64

import yaml
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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
            for name, node in gui_spec.nodes.items():
                for depend in node.depends:
                    if isinstance(depend, dict):
                        depend = str(depend.values())
                    src_node_id, signal = depend.split(".")
                    if (event["node_id"] == src_node_id) and (event["signal"] == signal):
                        q = QManager.get_queue(node.id)
                        q.put(event)
                        # logger.warning(msg=q.get())
                        # raise NotImplementedError()

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
    return HTMLResponse(f"Loaded: {tube_config["noob_id"]}")


@router.post("/player/{node_id}")
async def player(node_id: str, request: Request) -> HTMLResponse:
    """Serve video player DOM"""
    stream_path = config.runtime_dir / node_id / "stream.m3u8"
    print(f"stream_path={stream_path}")
    if stream_path.exists():  # later change this to "has chunk ext files"
        return templates.TemplateResponse(
            request,
            "partials/player.html",
            {"id": node_id, "path": f"/stream/{node_id}/stream.m3u8"},
        )
    else:
        raise HTTPException(404)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connection and run streamers"""
    q = QManager.get_queue()

    await websocket.accept()
    # some kind of waiting mechanism here
    # to keep the connection open until the client disconnects
    while True:
        event = q.get()
        if event is None:
            break
        await websocket.send_json(event)
