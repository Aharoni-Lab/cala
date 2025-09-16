from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from cala.config import config
from cala.gui.const import STATIC_DIR
from cala.gui.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    yield


def get_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan, debug=True)

    app.mount(path="/static", app=StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/stream", app=StaticFiles(directory=config.runtime_dir), name="stream")
    app.include_router(router)

    return app
