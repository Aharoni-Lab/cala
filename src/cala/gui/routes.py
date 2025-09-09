import base64
from datetime import datetime
from queue import Queue
from typing import Annotated

import yaml
from fastapi import APIRouter, WebSocket, BackgroundTasks, HTTPException, UploadFile, Header
from fastapi.encoders import jsonable_encoder
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse

from cala.config import config
from cala.gui.const import TEMPLATES_DIR
from cala.gui.deps import Spec

templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter()

_running = False
_thread = None
_tube_config = None


class QManager:
    _qs: dict[str, Queue] = {}

    @classmethod
    def get_queue(cls, name: str) -> Queue:
        if cls._qs.get(name) is None:
            cls._qs[name] = Queue()
        return cls._qs.get(name)


@router.get("/now", response_class=HTMLResponse)
async def metadata(request: Request):
    return templates.TemplateResponse(request, "includes/metadata.html", {"now": datetime.now()})


@router.get("/open-time", response_class=HTMLResponse)
async def list_todos(request: Request, hx_request: Annotated[str | None, Header()] = None):
    if hx_request:
        return templates.TemplateResponse(
            request=request, name="includes/metadata.html", context={"open_time": datetime.now()}
        )
    return JSONResponse(content=jsonable_encoder(datetime.now()))


@router.get("/", response_class=HTMLResponse)
async def get(request: Request, spec: Spec) -> HTMLResponse:
    response = templates.TemplateResponse(request, "index.html", {"config": spec.model_dump()})
    config = base64.b64encode(bytes(spec.model_dump_json(), "utf-8")).decode()
    response.set_cookie(key="config", value=config, samesite="lax")

    return response


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


@router.get("/{node_id}/stream.m3u8")
async def stream(node_id: str, filename: str) -> FileResponse:
    """Serve HLS stream files"""
    stream_path = config.runtime_dir / node_id / filename
    if stream_path.exists():
        return FileResponse(str(stream_path))
    else:
        return HTMLResponse("File not found")
    raise HTTPException(404, detail={"AppStreamError": f"Playlist not found: {stream_path}"})


@router.post("/start")
def start(
    background: BackgroundTasks,
    spec: Spec,
    request: Request,
    hx_request: Annotated[str | None, Header()] = None,
) -> HTMLResponse:
    # try:
    #     global _running
    #     if _running:
    #         raise HTTPException(400, f"Already running.")
    #     global _thread
    #     spec = TubeSpecification(**_tube_config)
    #     tube = Tube.from_specification(spec)
    #     runner = SynchronousRunner(tube=tube)
    #
    #     def _cb(event):
    #         q = QManager.get_queue("lineplot")
    #         if event.condition == "what i want":
    #             q.put(event)
    #
    #     runner.add_callback(_cb)
    #     background.add_task(runner.run)
    # except Exception as e:
    #     raise HTTPException(500, str(e))
    #
    # _running = True
    if hx_request:
        print(spec.grids.values())
        grid = templates.TemplateResponse(
            request, "partials/grids.html", {"grids": list(spec.grids.values())}
        )
        print("grid", grid.body.decode())
    return grid


@router.post("/stop")
def stop():
    global _thread
    q = QManager.get_queue("lineplot")
    q.put(None)
    _thread.terminate()
    _thread.join()


@router.post("/submit-tube")
def submit_tube(file: UploadFile, request: Request) -> HTMLResponse:
    global _tube_config
    try:
        tube_config = yaml.safe_load(file.file)
    except Exception as e:
        raise HTTPException(422, f"Failed to load Tube specification. {e}")
    _tube_config = tube_config
    return HTMLResponse(f"Loaded: {tube_config["noob_id"]}")
    # return templates.TemplateResponse(
    #     request, "partials/tube-config.html", {"tube_config": tube_config}
    # )
