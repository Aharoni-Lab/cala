from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from cala.gui.const import STATIC_DIR
from cala.gui.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    yield


def get_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan, debug=True)

    app.mount(path="/static", app=StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(router)

    return app
